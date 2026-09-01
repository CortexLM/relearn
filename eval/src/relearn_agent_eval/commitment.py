"""Commitment over a holdout trace set.

Same construction as `relearn_challenge_task::holdout_commitment` — domain
separated, id sorted, length prefixed — over the fields that make a trace what
it is. The image recomputes it over the traces the request delivered and
refuses to score when it disagrees with the declared commitment, so a request
edited in flight is a failed run rather than a verdict on a different split.

A trace commits to more than a prompt does, and every part of it has to be
covered. Editing an observation would change what the correct next action is;
editing a recorded argument would change what counts as correct; reordering
steps would change what counts as in order. Any of those left out of the
commitment would be an editable knob on a "verified" holdout.

The domain tag is the agent challenge's own, so the same records committed
under the language challenge produce a different digest and cannot be replayed
across challenges.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence
from typing import Protocol

from .pins import HOLDOUT_DOMAIN


class CommittedStep(Protocol):
    """The fields of one recorded step that the commitment covers."""

    tool: str
    arguments_json: str
    observation: str
    observation_image_hash: str


class CommittedTrace(Protocol):
    """The fields of one recorded trace that the commitment covers."""

    id: int
    goal: str
    dataset_id: str
    final_answer: str
    tool_names: tuple[str, ...]
    steps: Sequence[CommittedStep]


def _u32_le(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFF_FFFF)


def _u64_le(value: int) -> bytes:
    return struct.pack("<Q", value & 0xFFFF_FFFF_FFFF_FFFF)


def _field(digest: hashlib._Hash, value: str) -> None:
    body = value.encode("utf-8")
    digest.update(_u64_le(len(body)))
    digest.update(body)


def trace_commitment(traces: Iterable[CommittedTrace]) -> str:
    """Hex commitment over `traces`."""
    ordered: Sequence[CommittedTrace] = sorted(traces, key=lambda trace: trace.id)
    digest = hashlib.sha256()
    digest.update(HOLDOUT_DOMAIN)
    digest.update(b"\xff")
    digest.update(_u64_le(len(ordered)))
    for trace in ordered:
        digest.update(_u32_le(int(trace.id)))
        for value in (trace.goal, trace.dataset_id, trace.final_answer):
            _field(digest, value)
        digest.update(_u64_le(len(trace.tool_names)))
        for name in trace.tool_names:
            _field(digest, name)
        digest.update(_u64_le(len(trace.steps)))
        for step in trace.steps:
            for value in (
                step.tool,
                step.arguments_json,
                step.observation,
                step.observation_image_hash.strip().lower(),
            ):
                _field(digest, value)
    return digest.hexdigest()


def commitments_match(left: str, right: str) -> bool:
    """Compare two commitments the way the control plane does."""
    return left.strip().lower() == right.strip().lower()
