"""The control plane's acceptance checks, re-implemented in the image.

The image runs them on its own document before it prints the completion marker,
so a document the control plane would refuse ends the run as a failure here —
where the log tail can say why — instead of as an unexplained 503 after the pod
is already gone.

The check that matters most is the last one. An episode set with screenshots
that comes back without a pixel-shuffle control, or any holdout at all that
comes back without a tool-blind control, is refused: those controls are the
only thing standing between this challenge and rewarding a model that ignores
its tools, and a document that quietly omits them would look like a clean run.

It is a mirror, not a second protocol: when the two disagree, the control plane
wins.
"""

from __future__ import annotations

import math
import re

from relearn_common.errors import ContractError

from .contract import AgentDocument
from .pins import (
    CANARY_KEY_PREFIX,
    HOLDOUT_KEY_PREFIX,
    IMAGE_MODALITY,
    MAX_INVALID_ACTION_RATE,
    ORDER_KEY_PREFIX,
    PUBLIC_KEY_PREFIX,
    SCHEMA_VERSION,
    TOOL_CALL_KEY_PREFIX,
)
from .request import AgentHarvestRequest

#: A series key is a prefix plus an episode id. Anything else risks carrying a
#: goal, an argument, or an observation into a document that is logged and
#: stored off the pod.
_KEY = re.compile(r"^[a-z][0-9]+$")

_PREFIXES = (
    HOLDOUT_KEY_PREFIX,
    TOOL_CALL_KEY_PREFIX,
    ORDER_KEY_PREFIX,
    PUBLIC_KEY_PREFIX,
    CANARY_KEY_PREFIX,
)

#: Modalities a document may claim a shuffle control for.
_MODALITIES = (IMAGE_MODALITY,)


class VerificationError(ContractError):
    """A document that would not be accepted as a score."""


def verify_document(document: AgentDocument, request: AgentHarvestRequest) -> None:
    """Check a document against the run that asked for it.

    # Raises
    [`VerificationError`] on a schema, run-identity, series, key-shape, rate,
    or control problem.
    """
    if document.schema_version != SCHEMA_VERSION:
        raise VerificationError(
            f"metrics schema_version {document.schema_version}, expected {SCHEMA_VERSION}"
        )
    try:
        request.identity.check_document(document.identity)
    except ContractError as exc:
        raise VerificationError(str(exc)) from exc

    measured = document.measurement
    if measured.base_model.strip() != request.base_model.strip():
        raise VerificationError("document base_model is not the pinned base")
    if measured.teacher_model.strip() != request.teacher_model.strip():
        raise VerificationError("document teacher_model is not the pinned teacher")

    _verify_series(document, request)
    _verify_rates(document)
    _verify_controls(document, request)


def _verify_series(document: AgentDocument, request: AgentHarvestRequest) -> None:
    measured = document.measurement
    expected = {trace.id for trace in request.holdout}

    for name, values, prefix in (
        ("holdout", measured.holdout, HOLDOUT_KEY_PREFIX),
        ("tool_call", measured.tool_call, TOOL_CALL_KEY_PREFIX),
        ("order", measured.order, ORDER_KEY_PREFIX),
    ):
        if {key for key in values} != {f"{prefix}{trace_id}" for trace_id in expected}:
            raise VerificationError(
                f"{name} series keys are not the requested episodes "
                f"({len(values)} scored for {len(expected)} episodes)"
            )

    if not measured.public:
        raise VerificationError("no public split; the memorization gap gate cannot run")
    if not measured.canaries:
        raise VerificationError(
            "no canaries; a model that cannot emit a tool call at all would go unnoticed"
        )

    for name, values in (
        ("holdout", measured.holdout),
        ("tool_call", measured.tool_call),
        ("order", measured.order),
        ("public", measured.public),
        ("canaries", measured.canaries),
    ):
        for key, value in values.items():
            if not _KEY.match(key) or not key.startswith(_PREFIXES):
                raise VerificationError(f"{name} key {key!r} is not a series key")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise VerificationError(f"{name}[{key}] = {value} outside [0, 1]")


def _verify_rates(document: AgentDocument) -> None:
    rate = document.measurement.invalid_action_rate
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise VerificationError(f"invalid_action_rate {rate} outside [0, 1]")
    if rate > MAX_INVALID_ACTION_RATE:
        raise VerificationError(
            f"invalid_action_rate {rate:.3f} above the {MAX_INVALID_ACTION_RATE:.3f} ceiling"
        )


def _verify_controls(document: AgentDocument, request: AgentHarvestRequest) -> None:
    measured = document.measurement

    blind = measured.tool_blind
    if blind.traces <= 0:
        raise VerificationError(
            "no tool-blind control; a model that ignores its tools would be indistinguishable "
            "from one that reads them"
        )
    if blind.traces != len(request.holdout):
        raise VerificationError(
            f"tool-blind control covers {blind.traces} of {len(request.holdout)} episodes"
        )
    for name, value in (("score", blind.score), ("degraded_score", blind.degraded_score)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise VerificationError(f"tool_blind.{name} = {value} outside [0, 1]")

    families = set(request.modalities())
    for modality, evidence in measured.observation_shuffle.items():
        if modality not in _MODALITIES:
            raise VerificationError(
                f"observation_shuffle carries unknown modality {modality!r}"
            )
        if modality not in families:
            raise VerificationError(
                f"observation_shuffle claims {modality!r}, which is not in this holdout"
            )
        if evidence.traces <= 0:
            raise VerificationError(f"observation_shuffle[{modality}] scored no episodes")
    missing = families - set(measured.observation_shuffle)
    if missing:
        raise VerificationError(
            "observation modalities in the holdout without a shuffle control: "
            + ", ".join(sorted(missing))
        )
