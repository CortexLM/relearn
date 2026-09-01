"""The control plane's acceptance checks, re-implemented in the image.

Mirror of what `relearn_t2i_score` and the pin need to be true before a
document can become a verdict. The image runs them on its own document before
it prints the completion marker, so a document the control plane would refuse
ends the run as a failure here — where the log tail can say why — instead of as
an unexplained 503 after the pod is already gone.

It is a mirror, not a second protocol: when the two disagree, the control plane
wins.
"""

from __future__ import annotations

import math
import re

from relearn_common.errors import ContractError

from .contract import ImageDocument, known_pillar
from .pins import MAX_NA_RATE, SCHEMA_VERSION
from .request import ImageHarvestRequest
from .seeds import cell_key

#: A cell key is `p<prompt id>#v<variation index>`. Anything else risks
#: carrying prompt text into a document that is logged and stored off the pod.
_CELL_KEY = re.compile(r"^p[0-9]+#v[0-9]+$")


class VerificationError(ContractError):
    """A document that would not be accepted as a score."""


def _expected_keys(cells: list[tuple[int, int, int]]) -> set[str]:
    return {cell_key(prompt_id, variation) for prompt_id, variation, _seed in cells}


def verify_document(document: ImageDocument, request: ImageHarvestRequest) -> None:
    """Check a document against the run that asked for it.

    # Raises
    [`VerificationError`] on a schema, run-identity, series, key-shape, pillar,
    or evidence problem.
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
        raise VerificationError("document base_model is not the pinned generator")
    if measured.judge_model.strip() != request.judge_model.strip():
        raise VerificationError("document judge_model is not the pinned judge")

    _verify_series(measured.holdout, "holdout", _expected_keys(request.holdout_cells()))
    _verify_series(measured.public, "public", _expected_keys(request.public_cells()))
    _verify_pillars(document)
    _verify_rates_and_evidence(document)


def _verify_series(values: dict[str, float], name: str, expected: set[str]) -> None:
    if not values:
        raise VerificationError(f"{name} series is empty")
    for key, value in values.items():
        if not _CELL_KEY.match(key):
            raise VerificationError(f"{name} key {key!r} is not a cell key")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise VerificationError(f"{name}[{key}] = {value} outside [0, 1]")
    if set(values) != expected:
        raise VerificationError(
            f"{name} series keys are not the requested cells "
            f"({len(values)} scored for {len(expected)} cells)"
        )


def _verify_pillars(document: ImageDocument) -> None:
    measured = document.measurement
    if not measured.holdout_by_pillar:
        # Without per-pillar series the anti-hidden-regression gate cannot run,
        # and a challenger whose Quality collapsed would be judged on its total
        # alone.
        raise VerificationError("no per-pillar holdout series; the pillar gate cannot run")
    holdout_keys = set(measured.holdout)
    for name, values in measured.holdout_by_pillar.items():
        if not known_pillar(name):
            raise VerificationError(f"holdout_by_pillar carries unknown pillar {name!r}")
        if not values:
            raise VerificationError(f"holdout_by_pillar[{name}] scored no cells")
        unexpected = set(values) - holdout_keys
        if unexpected:
            raise VerificationError(
                f"holdout_by_pillar[{name}] scores cells that are not in the holdout"
            )
        for key, value in values.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise VerificationError(
                    f"holdout_by_pillar[{name}][{key}] = {value} outside [0, 1]"
                )


def _verify_rates_and_evidence(document: ImageDocument) -> None:
    measured = document.measurement
    if not math.isfinite(measured.na_rate) or not 0.0 <= measured.na_rate <= 1.0:
        raise VerificationError(f"na_rate {measured.na_rate} outside [0, 1]")
    if measured.na_rate > MAX_NA_RATE:
        raise VerificationError(
            f"na_rate {measured.na_rate:.3f} above the {MAX_NA_RATE:.3f} ceiling"
        )

    replay = measured.replay
    if replay.cells_checked <= 0:
        raise VerificationError("no seed-replay evidence")
    if replay.exact_hash_matches > replay.cells_checked:
        raise VerificationError("more exact replay matches than cells checked")
    if not math.isfinite(replay.max_embedding_drift) or not (
        0.0 <= replay.max_embedding_drift <= 1.0
    ):
        raise VerificationError("replay drift outside [0, 1]")

    faithfulness = measured.faithfulness
    if faithfulness.agreements > faithfulness.checks:
        raise VerificationError("more faithfulness agreements than checks")
