"""The harvest client's side of the contract, re-implemented in the image.

Mirror of `relearn_lium_harvest::extract_metrics` and the marker check in its
`LiveScorer::score`. The image uses it to prove — in its own tests, and in
`relearn-eval verify` — that a transcript it produced is one the control plane
would accept, and that transcripts it must never produce (another run's
numbers, a replay, silence) are refused.

This is the consumer's logic, so it stays deliberately literal: the same first
matching line, the same bare-document fallback, the same "no OK marker, no
score" rule.
"""

from __future__ import annotations

from .contract import (
    METRICS_MARKER,
    OK_MARKER,
    ContractError,
    EvalDocument,
    decode_document,
)
from .request import HarvestRequest
from .verify import verify_document


class TranscriptError(ContractError):
    """A pod transcript the control plane would refuse."""


def extract_metrics(stdout: str) -> EvalDocument:
    """Pull the metrics document out of a pod transcript.

    The document is one line and can be far larger than any log tail, so it is
    found by prefix rather than scraped from the end. A bare document is
    accepted too, matching the control plane.
    """
    for line in stdout.splitlines():
        if line.startswith(METRICS_MARKER):
            return decode_document(line[len(METRICS_MARKER) :])
    trimmed = stdout.strip()
    if trimmed.startswith("{"):
        return decode_document(trimmed)
    raise TranscriptError(f"eval image printed no {METRICS_MARKER} document")


def has_ok_marker(stdout: str) -> bool:
    """Whether the transcript carries the completion marker on its own line."""
    return any(line.rstrip() == OK_MARKER for line in stdout.splitlines())


def accept(stdout: str, request: HarvestRequest) -> EvalDocument:
    """Accept a transcript as a score, or raise.

    Order matters and matches the control plane: silence is refused before the
    document is parsed, so a pod that printed numbers but never finished is
    never read as a verdict.

    # Raises
    [`TranscriptError`] when the completion marker or the document is missing,
    and [`VerificationError`] when the document is not bound to `request`.
    """
    if not has_ok_marker(stdout):
        raise TranscriptError(f"eval image did not print {OK_MARKER}")
    document = extract_metrics(stdout)
    verify_document(document, request)
    return document
