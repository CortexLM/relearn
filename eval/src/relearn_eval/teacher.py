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


def parse_score(content: str) -> float:
    """Read a `[0, 1]` score out of a judge reply.

    # Raises
    [`TeacherError`] when the reply carries no number. A judge that answered
    prose is a failed item, not a zero.
    """
    body = content.strip()
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
    attempts: int = 3
    retry_secs: float = 5.0

    def judge(self, prompt: str, candidate: str) -> float:
        """Score one candidate answer in `[0, 1]`."""
        guard_judge_call(self.model, candidate)
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 32,
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
        content = ""
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = str(message.get("content", ""))
        if not content:
            raise TeacherError("teacher reply carried no message content")
        return parse_score(content)

    def judge_all(self, pairs: Iterable[tuple[str, str]]) -> list[float]:
        """Judge a batch, in order."""
        return [self.judge(prompt, candidate) for prompt, candidate in pairs]


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
    return HttpTeacher(api_url=url, model=model, api_key=teacher_api_key())
