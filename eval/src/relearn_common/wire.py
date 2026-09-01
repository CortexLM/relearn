"""The marker protocol every Relearn eval image speaks.

One contract, three challenges. The control plane's harvest client boots a
digest-pinned image, stages `request.json`, runs the challenge's scorer, and
reads back exactly two things from stdout:

    RELEARN_METRICS=<one-line JSON document>
    RELEARN_EVAL_OK

`relearn_lium_harvest::{METRICS_MARKER, OK_MARKER}` on the control plane are
these strings. Reusing them for the Image and Agent challenges is the point: a
cortex harvest client for either one is `relearn-lium-harvest` with a different
document type, not a second transport to review and keep in step.

What keeps two challenges from being confused for each other is not a different
marker, it is the binding inside the document. Every document carries
`challenge_id`, every image refuses a document that is not its own challenge,
and the series keys of the three challenges do not overlap. A digest pinned
into the wrong challenge's config fails closed at verification rather than
scoring something nobody asked for.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping

from .errors import ContractError

#: Prefix the image prints before its one-line metrics document.
#:
#: The document can be far larger than any log tail, so the harvest greps for
#: this prefix rather than scraping the end of the transcript.
METRICS_MARKER = "RELEARN_METRICS="

#: Printed last, and only after the document passed the image's own copy of the
#: control plane's acceptance checks.
OK_MARKER = "RELEARN_EVAL_OK"

#: Scores are rounded before publication so two runs of the same model on the
#: same items agree bit-for-bit, and the paired test is never decided by float
#: noise in the last few digits.
SCORE_DECIMALS = 6


def clamp_score(value: float, *, what: str) -> float:
    """Round a measured score onto the published `[0, 1]` grid.

    A non-finite score is a scoring bug, not a zero. Publishing it as a number
    would turn a broken run into a verdict, so it ends the run instead.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{what} is not finite")
    return round(min(1.0, max(0.0, number)), SCORE_DECIMALS)


def series_wire(name: str, values: Mapping[str, float]) -> dict[str, float]:
    """Encode one series: sorted keys, every value clamped and rounded."""
    return {
        key: clamp_score(value, what=f"{name}[{key}]") for key, value in sorted(values.items())
    }


def read_series(body: Mapping[str, object], name: str) -> dict[str, float]:
    """Decode one series, refusing anything that is not an object of numbers."""
    raw = body.get(name) or {}
    if not isinstance(raw, Mapping):
        raise ContractError(f"{name} is not an object")
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{name}[{key}] is not a number") from exc
    return out


def encode_line(payload: Mapping[str, object]) -> str:
    """Encode a document as the single line the harvest reads.

    The harvest reconstructs the marker line with `printf 'RELEARN_METRICS=';
    cat metrics.json`, so an embedded or trailing newline would truncate the
    document mid-JSON and turn a finished run into a 503.
    """
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    if "\n" in line or "\r" in line:
        raise ContractError("encoded document contains a newline")
    return line


def decode_line(line: str) -> dict[str, object]:
    """Parse one encoded document into its wire mapping."""
    try:
        body = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ContractError(f"metrics document is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ContractError("metrics document is not an object")
    return body


def marker_line(payload: Mapping[str, object]) -> str:
    """The `RELEARN_METRICS=<document>` line, without its newline."""
    return f"{METRICS_MARKER}{encode_line(payload)}"


def mean(values: Iterable[float]) -> float | None:
    """Mean of a series, or `None` when it is empty."""
    collected = list(values)
    if not collected:
        return None
    return sum(collected) / len(collected)
