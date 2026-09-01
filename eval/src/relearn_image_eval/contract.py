"""The Relearn Image metrics document.

Field names here are wire names. They are chosen to drop straight into
`relearn_t2i_score::T2iSliceScores` plus the run identity, so a cortex harvest
client for this challenge is `relearn-lium-harvest` with a different document
type — not a second protocol to review and keep in step.

Cell keys are `p{prompt_id}#v{variation_index}`
(`relearn_t2i_task::cell_key`): an id and a variation index, never prompt
text. The private holdout does not leave the pod in the document, in a log, or
in the sidecar.

Nothing in this module computes a score. It only carries and encodes one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from relearn_common.errors import ContractError
from relearn_common.identity import RunIdentity
from relearn_common.wire import clamp_score, read_series, series_wire

from .pillars import PILLAR_WIRE_NAMES
from .pins import SCHEMA_VERSION


@dataclass(frozen=True)
class ReplayEvidence:
    """Seed-replay evidence. `relearn_t2i_score::ReplayEvidence`.

    Exact hashes are the fast path and are not required: pixel determinism does
    not survive a driver change, so a small descriptor distance is accepted as
    the same weights. Both numbers are reported; the control plane decides.
    """

    cells_checked: int = 0
    exact_hash_matches: int = 0
    #: Worst descriptor distance across the replayed cells. Defaults to the
    #: maximum, so an evidence object nobody filled in fails the gate rather
    #: than passing it.
    max_embedding_drift: float = 1.0

    def to_wire(self) -> dict[str, object]:
        drift = float(self.max_embedding_drift)
        if not math.isfinite(drift):
            raise ContractError("replay max_embedding_drift is not finite")
        return {
            "cells_checked": int(self.cells_checked),
            "exact_hash_matches": int(self.exact_hash_matches),
            "max_embedding_drift": round(min(1.0, max(0.0, drift)), 6),
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object] | None) -> ReplayEvidence:
        if not isinstance(body, Mapping):
            raise ContractError("replay evidence is not an object")
        return cls(
            cells_checked=int(body.get("cells_checked", 0) or 0),
            exact_hash_matches=int(body.get("exact_hash_matches", 0) or 0),
            max_embedding_drift=float(body.get("max_embedding_drift", 1.0)),
        )


@dataclass(frozen=True)
class FaithfulnessEvidence:
    """Agentic prompt-faithfulness evidence. `relearn_t2i_score::FaithfulnessEvidence`."""

    checks: int = 0
    agreements: int = 0

    def to_wire(self) -> dict[str, object]:
        return {"checks": int(self.checks), "agreements": int(self.agreements)}

    @classmethod
    def from_wire(cls, body: Mapping[str, object] | None) -> FaithfulnessEvidence:
        if not isinstance(body, Mapping):
            raise ContractError("faithfulness evidence is not an object")
        return cls(
            checks=int(body.get("checks", 0) or 0),
            agreements=int(body.get("agreements", 0) or 0),
        )


@dataclass(frozen=True)
class ImageMeasurement:
    """Everything measured about one artifact on one holdout.

    The same shape an operator installs as the recorded champion, so a boot
    baseline is literally this image's output for the base checkpoint.
    """

    base_model: str
    judge_model: str
    holdout: dict[str, float] = field(default_factory=dict)
    public: dict[str, float] = field(default_factory=dict)
    holdout_by_pillar: dict[str, dict[str, float]] = field(default_factory=dict)
    na_rate: float = 0.0
    replay: ReplayEvidence = field(default_factory=ReplayEvidence)
    faithfulness: FaithfulnessEvidence = field(default_factory=FaithfulnessEvidence)
    contaminated_prompt_ids: tuple[int, ...] = ()

    def to_wire(self) -> dict[str, object]:
        rate = float(self.na_rate)
        if not math.isfinite(rate):
            raise ContractError("na_rate is not finite")
        return {
            "base_model": self.base_model,
            "judge_model": self.judge_model,
            "holdout": series_wire("holdout", self.holdout),
            "public": series_wire("public", self.public),
            "holdout_by_pillar": {
                pillar: series_wire(f"holdout_by_pillar[{pillar}]", values)
                for pillar, values in sorted(self.holdout_by_pillar.items())
            },
            "na_rate": clamp_score(rate, what="na_rate"),
            "replay": self.replay.to_wire(),
            "faithfulness": self.faithfulness.to_wire(),
            "contaminated_prompt_ids": sorted(int(x) for x in self.contaminated_prompt_ids),
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> ImageMeasurement:
        raw_pillars = body.get("holdout_by_pillar") or {}
        if not isinstance(raw_pillars, Mapping):
            raise ContractError("holdout_by_pillar is not an object")
        raw_ids = body.get("contaminated_prompt_ids") or []
        if not isinstance(raw_ids, list):
            raise ContractError("contaminated_prompt_ids is not a list")
        return cls(
            base_model=str(body.get("base_model", "") or ""),
            judge_model=str(body.get("judge_model", "") or ""),
            holdout=read_series(body, "holdout"),
            public=read_series(body, "public"),
            holdout_by_pillar={
                str(name): read_series({"series": values}, "series")
                for name, values in raw_pillars.items()
            },
            na_rate=float(body.get("na_rate", 0.0) or 0.0),
            replay=ReplayEvidence.from_wire(body.get("replay")),  # type: ignore[arg-type]
            faithfulness=FaithfulnessEvidence.from_wire(
                body.get("faithfulness")  # type: ignore[arg-type]
            ),
            contaminated_prompt_ids=tuple(int(value) for value in raw_ids),
        )


@dataclass(frozen=True)
class ImageDocument:
    """One scored artifact, bound to the run that asked for it."""

    identity: RunIdentity
    measurement: ImageMeasurement
    schema_version: int = SCHEMA_VERSION

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            **self.identity.to_wire(),
            **self.measurement.to_wire(),
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> ImageDocument:
        if not isinstance(body, Mapping):
            raise ContractError("metrics document is not an object")
        return cls(
            schema_version=int(body.get("schema_version", 0) or 0),
            identity=RunIdentity.from_wire(body),
            measurement=ImageMeasurement.from_wire(body),
        )


def known_pillar(name: str) -> bool:
    """Whether a `holdout_by_pillar` key is one of the five pillars."""
    return name in PILLAR_WIRE_NAMES
