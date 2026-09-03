"""The control plane's acceptance checks, re-implemented in the image.

Mirror of `RelearnEvalMetrics::verify` and `BaselineMeasurement::verify`. The
image runs them on its own document before it prints [`OK_MARKER`], so a
document the control plane would refuse ends the run as a failure here, where
the log tail can say why, instead of as an unexplained 503 after the pod is
already gone.

Keep this in step with `crates/relearn-eval` on the control plane. It is a
mirror, not a second protocol: when the two disagree, the control plane wins.
"""

from __future__ import annotations

import math
import re

from .contract import (
    CANARY_KEY_PREFIX,
    GENERAL_CANARY_KEY_PREFIX,
    HOLDOUT_KEY_PREFIX,
    METRICS_SCHEMA_VERSION,
    PERTURBED_KEY_PREFIX,
    PUBLIC_KEY_PREFIX,
    VISION_TASKS,
    ContractError,
    EvalDocument,
)
from .request import HarvestRequest

#: A series key is a prefix plus an item id. Anything else risks carrying
#: holdout text into a document that is logged and stored off the pod.
_KEY = re.compile(r"^[a-z][0-9]+$")

_PREFIXES = (
    HOLDOUT_KEY_PREFIX,
    PERTURBED_KEY_PREFIX,
    PUBLIC_KEY_PREFIX,
    CANARY_KEY_PREFIX,
    GENERAL_CANARY_KEY_PREFIX,
)


class VerificationError(ContractError):
    """A document that would not be accepted as a score."""


def _same(left: str, right: str) -> bool:
    return left.strip() == right.strip()


def _same_ignore_case(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def verify_document(document: EvalDocument, request: HarvestRequest) -> None:
    """Check a document against the run that asked for it.

    # Raises
    [`VerificationError`] on a schema, run-identity, image, holdout, series, or
    key-shape problem.
    """
    if document.schema_version != METRICS_SCHEMA_VERSION:
        raise VerificationError(
            f"metrics schema_version {document.schema_version}, "
            f"expected {METRICS_SCHEMA_VERSION}"
        )
    if not _same(document.submission_digest, request.submission_digest):
        raise VerificationError("metrics submission_digest is not the frozen run")
    if not _same_ignore_case(document.artifact_digest, request.artifact_digest):
        raise VerificationError("metrics artifact_digest is not the scored artifact")

    measured = document.measurement
    if not _same(measured.eval_image_digest, request.eval_image_digest):
        raise VerificationError("metrics eval_image_digest is not the pinned image")
    if not _same_ignore_case(measured.holdout_commitment, request.holdout_commitment):
        raise VerificationError("metrics holdout_commitment does not match the request")

    if len(measured.holdout) != len(request.holdout):
        raise VerificationError(
            f"{len(measured.holdout)} holdout scores for {len(request.holdout)} items"
        )
    if not measured.general_canary:
        raise VerificationError("no general-bench canary; every challenger would fail closed")
    if not measured.public:
        raise VerificationError("no public split; the memorization gap gate cannot run")
    if not math.isfinite(measured.agent_trace) or not 0.0 <= measured.agent_trace <= 1.0:
        raise VerificationError(f"agent_trace {measured.agent_trace} outside [0, 1]")

    expected_holdout = {
        f"{HOLDOUT_KEY_PREFIX}{item.id}" for item in request.holdout
    }
    if set(measured.holdout) != expected_holdout:
        raise VerificationError("holdout series keys are not the requested item ids")

    series = {
        "holdout": measured.holdout,
        "public": measured.public,
        "perturbed": measured.perturbed,
        "canaries": measured.canaries,
        "general_canary": measured.general_canary,
    }
    for name, values in series.items():
        for key, value in values.items():
            if not _KEY.match(key) or not key.startswith(_PREFIXES):
                raise VerificationError(f"{name} key {key!r} is not a series key")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise VerificationError(f"{name}[{key}] = {value} outside [0, 1]")

    families = set(request.vision_families())
    for task, evidence in measured.vision_shuffle.items():
        if task not in VISION_TASKS:
            raise VerificationError(f"vision_shuffle carries unknown family {task!r}")
        if task not in families:
            raise VerificationError(
                f"vision_shuffle claims {task!r}, which is not in this holdout"
            )
        if evidence.items <= 0:
            raise VerificationError(f"vision_shuffle[{task}] scored no items")
    missing = families - set(measured.vision_shuffle)
    if missing:
        raise VerificationError(
            "vision families in the holdout without a pixel-shuffle control: "
            + ", ".join(sorted(missing))
        )
