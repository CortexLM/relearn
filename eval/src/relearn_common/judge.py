"""The judge transport: an OpenAI-compatible endpoint the operator stands up.

Judges in this subnet are **judge-only**. The artifact under test is always
loaded inside the eval image; nothing scored is ever served through the judge
API. That is enforced here rather than by convention: a model id that looks
like an artifact digest, a model id that is not one this challenge accepts, or
a payload that looks like weights is refused before any request leaves the pod.

There is no endpoint, hostname, or key in this repository, and there is no
offline judge. An unconfigured judge ends the run, because a judge that
invents scores when the real one is unreachable is exactly the simulated number
these images exist to eliminate.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .errors import ContractError

log = logging.getLogger(__name__)

#: Tokens that mean "these are weights", not "this is something to judge".
WEIGHT_TOKENS = ("safetensors", "gguf", "nvfp4", "ckpt")


class JudgeError(ContractError):
    """The judge could not be reached, or refused to be used as one."""


def looks_like_artifact_digest(model: str) -> bool:
    """Whether a model id is really a submitted artifact's digest."""
    candidate = model.strip().lower().removeprefix("0x").removeprefix("sha256:")
    return len(candidate) == 64 and all(char in "0123456789abcdef" for char in candidate)


def model_matches(declared: str, pinned: str) -> bool:
    """Compare two model ids the way the control plane's `base_matches_pin` does.

    Case-insensitive on the repo id, ignoring an optional `@revision` suffix.
    The revision is pinned separately, so a stale card cannot pass as the pin.
    """

    def strip(value: str) -> str:
        return value.strip().split("@")[0].strip("/").lower()

    return bool(declared.strip()) and strip(declared) == strip(pinned)


def guard_judge_call(model: str, allowed: Sequence[str], payload: str) -> None:
    """Refuse anything that would make the judge serve or score miner weights.

    # Raises
    [`JudgeError`] when the model is an artifact digest, is not one of
    `allowed`, or when the payload body is a weights blob.
    """
    if looks_like_artifact_digest(model):
        raise JudgeError("an artifact digest is not a judge model")
    if not any(model_matches(model, known) for known in allowed):
        raise JudgeError(f"model {model!r} is not a judge this challenge accepts")
    lowered = payload.lower()
    if any(token in lowered for token in WEIGHT_TOKENS):
        raise JudgeError("miner weights are not a judge payload")


def image_data_url(body: bytes, media_type: str = "image/png") -> str:
    """A generated image, inline, as an OpenAI-compatible image part.

    Images are passed inline rather than by URL: the pod must not require the
    judge to be able to reach it, and a URL the judge fetches later would not
    be the bytes that were scored.
    """
    return f"data:{media_type};base64,{base64.b64encode(body).decode('ascii')}"


def text_part(body: str) -> dict[str, object]:
    return {"type": "text", "text": body}


def image_part(body: bytes, media_type: str = "image/png") -> dict[str, object]:
    return {"type": "image_url", "image_url": {"url": image_data_url(body, media_type)}}


@dataclass
class ChatJudge:
    """One OpenAI-compatible judge endpoint, called synchronously.

    `extra_body` carries the challenge's frozen inference parameters. They are
    part of the contract, not tuning knobs: a judge run at a different
    temperature is not comparable with the champion's recorded run, so they are
    sent on every call rather than left to the server's defaults.
    """

    api_url: str
    model: str
    allowed_models: tuple[str, ...]
    api_key: str = ""
    timeout_secs: float = 180.0
    attempts: int = 3
    retry_secs: float = 5.0
    extra_body: Mapping[str, object] = field(default_factory=dict)

    def complete(self, messages: Sequence[Mapping[str, object]], *, max_tokens: int) -> str:
        """One completion. Returns the reply text, or raises.

        # Raises
        [`JudgeError`] when the guard refuses the call, when the endpoint is
        unreachable after `attempts`, or when the reply carries no content.
        """
        guard_judge_call(self.model, self.allowed_models, _payload_text(messages))
        body = json.dumps(
            {
                "model": self.model,
                "messages": list(messages),
                "max_tokens": max_tokens,
                **dict(self.extra_body),
            }
        ).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(  # noqa: S310 - operator-configured endpoint
            f"{self.api_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )

        payload: object = None
        for attempt in range(1, max(1, self.attempts) + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_secs) as response:  # noqa: S310
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                # Never log the prompt, the image, or the key. The attempt
                # number and the error class are the whole diagnosis.
                log.warning("judge attempt %d failed: %s", attempt, type(exc).__name__)
                if attempt >= max(1, self.attempts):
                    raise JudgeError(f"judge unreachable after {attempt} attempts") from exc
                time.sleep(self.retry_secs * attempt)

        content = _reply_content(payload)
        if not content:
            raise JudgeError("judge reply carried no message content")
        return content


def _payload_text(messages: Sequence[Mapping[str, object]]) -> str:
    """The text a guard inspects: message text, never the image bytes."""
    chunks: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, Sequence):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
    return "\n".join(chunks)


def _reply_content(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if not isinstance(message, Mapping):
        return ""
    return str(message.get("content", "") or "")
