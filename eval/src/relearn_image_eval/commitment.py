"""Commitment over a frozen prompt set.

Byte-for-byte port of `relearn_t2i_task::frozen_prompt_commitment`.

The holdout is **not in git**: `config/relearn-t2i-pin.toml` carries only
`holdout_commitment` and `holdout_size`, and the records come from an operator
file. The image recomputes the commitment over the prompts the request
delivered and refuses to score when it disagrees with the declared one, so a
request that was edited in flight is a failed run rather than a verdict on a
different split.

Domain-separated, id-sorted, and length-prefixed: neither reordering the
prompts nor splicing two prompt bodies together can collide.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence
from typing import Protocol

from .pins import HOLDOUT_DOMAIN


class CommittedPrompt(Protocol):
    """The fields the commitment covers."""

    id: int
    text: str
    upsampled_json: str


def _u32_le(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFF_FFFF)


def _u64_le(value: int) -> bytes:
    return struct.pack("<Q", value & 0xFFFF_FFFF_FFFF_FFFF)


def frozen_prompt_commitment(prompts: Iterable[CommittedPrompt]) -> str:
    """Hex commitment over `prompts`."""
    ordered: Sequence[CommittedPrompt] = sorted(prompts, key=lambda prompt: prompt.id)
    digest = hashlib.sha256()
    digest.update(HOLDOUT_DOMAIN)
    digest.update(b"\xff")
    digest.update(_u64_le(len(ordered)))
    for prompt in ordered:
        digest.update(_u32_le(int(prompt.id)))
        for field in (prompt.text, prompt.upsampled_json):
            body = field.encode("utf-8")
            digest.update(_u64_le(len(body)))
            digest.update(body)
    return digest.hexdigest()


def commitments_match(left: str, right: str) -> bool:
    """Compare two commitments the way the control plane does."""
    return left.strip().lower() == right.strip().lower()
