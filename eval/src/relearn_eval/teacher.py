"""Frozen teacher, used only as a judge.

The teacher scores `(prompt, candidate)` pairs on an OpenAI-compatible HTTP API
the operator stands up (wire id `glm-5.3` by default). It never serves the
scored model: the artifact under test is loaded inside this image, and a
payload that looks like weights is refused here the same way
`relearn_eval::teacher_judge_guard` refuses it on the control plane.

There is no endpoint, host, or key in this repo, and no offline judge. An
unconfigured teacher ends the run: a judge that invents scores when the real
one is unreachable is exactly the sim number this image must never produce.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .contract import ContractError

log = logging.getLogger(__name__)

#: Default wire id. `relearn_challenge_task::TEACHER_MODEL_ID`.
TEACHER_MODEL_ID = "glm-5.3"

#: Wire ids the operator may configure. Anything else is refused, so a miner
#: artifact can never be passed off as the judge.
CONFIGURED_TEACHER_MODELS = (
    TEACHER_MODEL_ID,
    "zai-org/GLM-5.3",
    "kimi-k3",
    "moonshotai/Kimi-K3",
)

#: Tokens that mean "these are weights", not "this is an answer to judge".
_WEIGHT_TOKENS = ("safetensors", "gguf", "nvfp4", "ckpt")

#: Judge calls in flight. Bounded so the pod does not look like a client
#: flooding the operator's teacher.
DEFAULT_JUDGE_CONCURRENCY = 8

#: Attempts per judge call, and the backoff step between them. Both are
#: bounded: retries are spent out of the harvest's run timeout.
DEFAULT_JUDGE_ATTEMPTS = 3
DEFAULT_JUDGE_RETRY_SECS = 5.0

#: Completion budget for one judge call. GLM-5.3 thinking is mandatory — the
#: chat template ignores `enable_thinking=false` — so a 32-token cap is spent
#: on the think block and vLLM returns `content: null` with
#: `finish_reason=length`. 1024 leaves room for a short think plus
#: `{"score": 0.x}`. Override with `RELEARN_TEACHER_MAX_TOKENS`.
DEFAULT_TEACHER_MAX_TOKENS = 1024

_JUDGE_SYSTEM = (
    "You grade one candidate answer. Reply with JSON only, exactly "
    '{"score": <number between 0 and 1>}. 1.0 is a fully correct, complete, '
    "on-topic answer; 0.0 is wrong, empty, or evasive. Do not explain."
)

_SCORE = re.compile(r"-?\d+(?:\.\d+)?")


class TeacherError(ContractError):
    """The teacher could not judge, so the run has no numbers."""


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def teacher_api_url() -> str:
    """`RELEARN_TEACHER_API_URL`. No baked host."""
    return _env("RELEARN_TEACHER_API_URL").rstrip("/")


def teacher_api_key() -> str:
    """`RELEARN_TEACHER_API_KEY`. Never logged, never echoed."""
    return _env("RELEARN_TEACHER_API_KEY")


def judge_attempts() -> int:
    """`RELEARN_JUDGE_ATTEMPTS`. Retries cost run time, so they are bounded."""
    return _positive_int_env("RELEARN_JUDGE_ATTEMPTS", DEFAULT_JUDGE_ATTEMPTS)


def judge_retry_secs() -> float:
    raw = _env("RELEARN_JUDGE_RETRY_SECS")
    if not raw:
        return DEFAULT_JUDGE_RETRY_SECS
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise TeacherError(f"RELEARN_JUDGE_RETRY_SECS {raw!r} is not a number") from exc
    if seconds < 0:
        raise TeacherError("RELEARN_JUDGE_RETRY_SECS must not be negative")
    return seconds


def _positive_int_env(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise TeacherError(f"{name} {raw!r} is not an integer") from exc
    if value <= 0:
        raise TeacherError(f"{name} must be positive")
    return value


def judge_concurrency() -> int:
    """`RELEARN_JUDGE_CONCURRENCY`, defaulting to [`DEFAULT_JUDGE_CONCURRENCY`]."""
    return _positive_int_env("RELEARN_JUDGE_CONCURRENCY", DEFAULT_JUDGE_CONCURRENCY)


def teacher_max_tokens() -> int:
    """`RELEARN_TEACHER_MAX_TOKENS`, defaulting to [`DEFAULT_TEACHER_MAX_TOKENS`]."""
    return _positive_int_env("RELEARN_TEACHER_MAX_TOKENS", DEFAULT_TEACHER_MAX_TOKENS)


def looks_like_digest(model: str) -> bool:
    candidate = model.strip().lower().removeprefix("0x")
    return len(candidate) == 64 and all(char in "0123456789abcdef" for char in candidate)


def is_configured_teacher_model(model: str) -> bool:
    """Whether `model` is a teacher wire id this pod may call."""
    trimmed = model.strip()
    if not trimmed:
        return False
    override = _env("RELEARN_TEACHER_MODEL")
    if override and trimmed == override:
        return True
    return any(
        trimmed == known or trimmed.lower() == known.lower()
        for known in CONFIGURED_TEACHER_MODELS
    )


def guard_judge_call(model: str, candidate: str) -> None:
    """Refuse anything that would make the teacher serve miner weights.

    # Raises
    [`TeacherError`] when the model is an artifact digest, is not a configured
    teacher wire id, or when the candidate body is a weights blob.
    """
    if looks_like_digest(model):
        raise TeacherError("an artifact digest is not a teacher model")
    if not is_configured_teacher_model(model):
        raise TeacherError(f"model {model!r} is not a configured teacher")
    lowered = candidate.lower()
    if any(token in lowered for token in _WEIGHT_TOKENS):
        raise TeacherError("miner weights are not a teacher payload")


def extract_judge_text(message: object) -> str:
    """Pick the text that may carry `{"score": …}` out of a chat message.

    GLM-5.3 thinking is mandatory. With a short `max_tokens` the completion
    is `finish_reason=length`, `content` is JSON `null`, and the tokens land
    in `reasoning` / `reasoning_content`. `str(None)` is `"None"` and must
    never be treated as judge text.
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def parse_score(content: str) -> float:
    """Read a `[0, 1]` score out of a judge reply.

    # Raises
    [`TeacherError`] when the reply carries no number. A judge that answered
    prose is a failed item, not a zero. The string `"None"` — `str(None)` —
    is not a score.
    """
    if not isinstance(content, str):
        raise TeacherError("teacher reply carried no score")
    body = content.strip()
    if not body or body in {"None", "null"}:
        raise TeacherError("teacher reply carried no score")
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and "score" in parsed:
            return min(1.0, max(0.0, float(parsed["score"])))
        if isinstance(parsed, int | float):
            return min(1.0, max(0.0, float(parsed)))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = _SCORE.search(body)
    if not match:
        raise TeacherError("teacher reply carried no score")
    return min(1.0, max(0.0, float(match.group())))


@dataclass
class HttpTeacher:
    """OpenAI-compatible judge client."""

    api_url: str
    model: str
    api_key: str = ""
    timeout_secs: float = 120.0
    attempts: int = DEFAULT_JUDGE_ATTEMPTS
    retry_secs: float = DEFAULT_JUDGE_RETRY_SECS

    def judge(self, prompt: str, candidate: str) -> float:
        """Score one candidate answer in `[0, 1]`."""
        guard_judge_call(self.model, candidate)
        # Do not send enable_thinking / thinking=false: GLM-5.3's template
        # ignores them, and the vLLM parser can then dump the scratchpad
        # into content. reasoning_effort=low keeps the mandatory think block
        # short so {"score": 0.x} still fits in the completion budget.
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": teacher_max_tokens(),
                "reasoning_effort": "low",
                "messages": [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Question:\n{prompt}\n\nCandidate answer:\n{candidate}",
                    },
                ],
            }
        ).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(  # noqa: S310 - operator-configured https endpoint
            f"{self.api_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        last: Exception | None = None
        for attempt in range(1, max(1, self.attempts) + 1):
            try:
                with urllib.request.urlopen(  # noqa: S310
                    request, timeout=self.timeout_secs
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                # Never log the prompt or the key; the attempt number and the
                # error class are the whole diagnosis.
                log.warning("teacher judge attempt %d failed: %s", attempt, type(exc).__name__)
                if attempt >= max(1, self.attempts):
                    raise TeacherError(f"teacher unreachable after {attempt} attempts") from last
                time.sleep(self.retry_secs * attempt)
        finish_reason = None
        message = None
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
            raw_message = choices[0].get("message")
            if isinstance(raw_message, dict):
                message = raw_message
        content = extract_judge_text(message)
        if not content:
            raise TeacherError(
                f"teacher reply carried no score (finish_reason={finish_reason!r})"
            )
        return parse_score(content)

    def judge_all(self, pairs: Iterable[tuple[str, str]]) -> list[float]:
        """Judge a batch, in order.

        Concurrent because a live run judges every holdout item twice plus the
        public split: one round trip at a time is minutes of the harvest's
        timeout spent waiting on a socket. `map` keeps the results positional,
        so the concurrency cannot reorder scores onto the wrong items, and the
        first failure propagates — a partially judged slice is not a slice.
        """
        work = list(pairs)
        if not work:
            return []
        workers = min(judge_concurrency(), len(work))
        if workers <= 1:
            return [self.judge(prompt, candidate) for prompt, candidate in work]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda pair: self.judge(*pair), work))


def build_teacher(teacher_model: str) -> HttpTeacher:
    """Build the judge for this run from the operator's environment.

    # Raises
    [`TeacherError`] when the teacher is not configured. Without a judge the
    holdout cannot be scored, and the run must fail rather than guess.
    """
    url = teacher_api_url()
    if not url:
        raise TeacherError(
            "RELEARN_TEACHER_API_URL is unset; the pod has no judge and cannot score"
        )
    model = _env("RELEARN_TEACHER_MODEL") or teacher_model.strip() or TEACHER_MODEL_ID
    guard_judge_call(model, "configuration probe")
    return HttpTeacher(
        api_url=url,
        model=model,
        api_key=teacher_api_key(),
        attempts=judge_attempts(),
        retry_secs=judge_retry_secs(),
    )
