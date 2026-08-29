"""HTTP teacher / judge.

OpenAI-compatible `/chat/completions`. Never send miner weights /
safetensors / GGUF as the `model` or as a payload to be served.

Operator sets `RELEARN_TEACHER_API_URL`, `RELEARN_TEACHER_MODEL`, and
`RELEARN_TEACHER_API_KEY`. No default API base in this repo.
Optional GLM pin: set `RELEARN_TEACHER_MODEL=zai-org/GLM-5.3`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "kimi-k3"


class TeacherError(RuntimeError):
    pass


def teacher_model() -> str:
    return os.environ.get("RELEARN_TEACHER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def teacher_api_url() -> str:
    return os.environ.get("RELEARN_TEACHER_API_URL", "").strip()


def teacher_api_key() -> str:
    for name in ("RELEARN_TEACHER_API_KEY", "MODAL_TOKEN"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return ""


def _looks_like_digest(model: str) -> bool:
    t = model.strip().lower().removeprefix("0x")
    return len(t) == 64 and all(c in "0123456789abcdef" for c in t)


def refuse_miner_weights(model: str, candidate: str) -> None:
    if _looks_like_digest(model):
        raise TeacherError("miner artifact digest is not a teacher model")
    lower = candidate.lower()
    if any(tok in lower for tok in ("safetensors", "gguf", "nvfp4", "ckpt")):
        raise TeacherError("miner weights are not a teacher payload")


def judge(prompt: str, candidate: str, api_url: str | None = None) -> dict:
    model = teacher_model()
    refuse_miner_weights(model, candidate)
    key = teacher_api_key()
    url = (api_url if api_url is not None else teacher_api_url()).strip().rstrip("/")
    if not key or not url:
        # No host/key in-tree. Missing env → sim (fail closed for live HTTP).
        score = 1.0 if candidate.strip() else 0.0
        return {"model": model, "score": score, "backend": "sim"}
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Score the candidate 0..1. JSON only."},
                {"role": "user", "content": f"prompt={prompt}\ncandidate={candidate}"},
            ],
        }
    ).encode()
    headers = {"content-type": "application/json", "authorization": f"Bearer {key}"}
    req = urllib.request.Request(
        url + "/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise TeacherError(f"teacher HTTP {e.code}") from e
    return {"model": model, "raw": payload, "backend": "http_api"}


TEACHER_MODEL = teacher_model()
