"""Commitment over a holdout set.

Byte-for-byte port of `relearn_challenge_task::holdout_commitment`. The image
recomputes the commitment over the items the request delivered and refuses to
score when it disagrees with the commitment the request declared, so a request
that was edited in flight is a failed run rather than a verdict on a different
split.

Domain-separated, id-sorted, and length-prefixed: neither reordering the items
nor splicing two prompt bodies together can collide.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence
from typing import Protocol

#: `relearn_challenge_task::HOLDOUT_DOMAIN`.
HOLDOUT_DOMAIN = b"base-relearn-holdout-v1"


class CommittedItem(Protocol):
    """The fields the commitment covers."""

    id: int
    prompt: str
    dataset_id: str
    task: str
    image_hash: str


def _u32_le(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFF_FFFF)


def _u64_le(value: int) -> bytes:
    return struct.pack("<Q", value & 0xFFFF_FFFF_FFFF_FFFF)


def holdout_commitment(items: Iterable[CommittedItem]) -> str:
    """Hex commitment over `items`."""
    ordered: Sequence[CommittedItem] = sorted(items, key=lambda item: item.id)
    digest = hashlib.sha256()
    digest.update(HOLDOUT_DOMAIN)
    digest.update(b"\xff")
    digest.update(_u64_le(len(ordered)))
    for item in ordered:
        digest.update(_u32_le(int(item.id)))
        digest.update(item.task.encode("utf-8"))
        digest.update(b"\xff")
        for value in (item.prompt, item.dataset_id, item.image_hash.strip()):
            body = value.encode("utf-8")
            digest.update(_u64_le(len(body)))
            digest.update(body)
    return digest.hexdigest()


def commitments_match(left: str, right: str) -> bool:
    """Compare two commitments the way the control plane does."""
    return left.strip().lower() == right.strip().lower()
