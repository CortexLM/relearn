"""The harvest client's side of the contract, re-implemented in the image.

Mirror of `relearn_lium_harvest::extract_metrics` and the marker check in its
`LiveScorer::score`. Each image uses it to prove — in its own tests, and in its
`verify` subcommand — that a transcript it produced is one the control plane
would accept, and that transcripts it must never produce (another run's
numbers, a replay, silence) are refused.

This is the consumer's logic, so it stays deliberately literal: the same first
matching line, the same bare-document fallback, the same "no OK marker, no
score" rule.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .errors import ContractError
from .wire import METRICS_MARKER, OK_MARKER, decode_line

Document = TypeVar("Document")


class TranscriptError(ContractError):
    """A pod transcript the control plane would refuse."""


def extract_document(stdout: str, decode: Callable[[dict[str, object]], Document]) -> Document:
    """Pull the metrics document out of a pod transcript.

    The document is one line and can be far larger than any log tail, so it is
    found by prefix rather than scraped from the end. A bare document is
    accepted too, matching the control plane.
    """
    for line in stdout.splitlines():
        if line.startswith(METRICS_MARKER):
            return decode(decode_line(line[len(METRICS_MARKER) :]))
    trimmed = stdout.strip()
    if trimmed.startswith("{"):
        return decode(decode_line(trimmed))
    raise TranscriptError(f"eval image printed no {METRICS_MARKER} document")


def has_ok_marker(stdout: str) -> bool:
    """Whether the transcript carries the completion marker on its own line."""
    return any(line.rstrip() == OK_MARKER for line in stdout.splitlines())


def accept(
    stdout: str,
    decode: Callable[[dict[str, object]], Document],
    verify: Callable[[Document], None],
) -> Document:
    """Accept a transcript as a score, or raise.

    Order matters and matches the control plane: silence is refused before the
    document is parsed, so a pod that printed numbers but never finished is
    never read as a verdict.

    # Raises
    [`TranscriptError`] when the completion marker or the document is missing,
    and whatever `verify` raises when the document is not bound to the run.
    """
    if not has_ok_marker(stdout):
        raise TranscriptError(f"eval image did not print {OK_MARKER}")
    document = extract_document(stdout, decode)
    verify(document)
    return document
