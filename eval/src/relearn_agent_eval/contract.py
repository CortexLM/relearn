"""The Relearn Agent metrics document.

Field names here are wire names, so a cortex harvest client for this challenge
is `relearn-lium-harvest` with a different document type rather than a second
protocol.

| Series | Key | What it is |
|---|---|---|
| `holdout` | `t<id>` | the composite score for one private episode |
| `tool_call` | `s<id>` | mean action score across that episode's steps |
| `order` | `o<id>` | whether the episode was acted in the recorded order |
| `public` | `q<id>` | the same composite on the published episodes |
| `canaries` | `c<id>` | shipped single-call episodes, matched not judged |

Plus the two controls, which are the point of the challenge:

* `tool_blind` — the action score with every observation withheld. A model that
  read the tool results loses ground here; one that pattern-matched the goal
  does not move, and not moving is what fails on the control plane.
* `observation_shuffle` — the action score with observation pixels destroyed,
  one entry per modality present in the holdout.

Keys are ids. An episode's goal, tool schemas, arguments, and observations
never leave the pod in the document, in a log, or in the sidecar.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from relearn_common.errors import ContractError
from relearn_common.identity import RunIdentity
from relearn_common.wire import clamp_score, read_series, series_wire

from .pins import SCHEMA_VERSION


@dataclass(frozen=True)
class ControlEvidence:
    """A scored pass and the same pass with something taken away.

    Same shape as `relearn_mm_score::AgenticEvidence`, because it is the same
    measurement: score, score-with-the-evidence-destroyed, and the gap between
    them. `drop` is what the control plane gates.
    """

    traces: int
    score: float
    degraded_score: float

    @property
    def drop(self) -> float:
        return self.score - self.degraded_score

    def to_wire(self) -> dict[str, object]:
        return {
            "traces": int(self.traces),
            "score": clamp_score(self.score, what="control score"),
            "degraded_score": clamp_score(self.degraded_score, what="degraded score"),
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> ControlEvidence:
        if not isinstance(body, Mapping):
            raise ContractError("control evidence is not an object")
        return cls(
            traces=int(body.get("traces", 0) or 0),
            score=float(body.get("score", 0.0) or 0.0),
            degraded_score=float(body.get("degraded_score", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class AgentMeasurement:
    """Everything measured about one artifact on one holdout of episodes."""

    base_model: str
    teacher_model: str
    holdout: dict[str, float] = field(default_factory=dict)
    tool_call: dict[str, float] = field(default_factory=dict)
    order: dict[str, float] = field(default_factory=dict)
    public: dict[str, float] = field(default_factory=dict)
    canaries: dict[str, float] = field(default_factory=dict)
    invalid_action_rate: float = 0.0
    tool_blind: ControlEvidence = field(
        default_factory=lambda: ControlEvidence(0, 0.0, 0.0)
    )
    observation_shuffle: dict[str, ControlEvidence] = field(default_factory=dict)

    def to_wire(self) -> dict[str, object]:
        rate = float(self.invalid_action_rate)
        if not math.isfinite(rate):
            raise ContractError("invalid_action_rate is not finite")
        return {
            "base_model": self.base_model,
            "teacher_model": self.teacher_model,
            "holdout": series_wire("holdout", self.holdout),
            "tool_call": series_wire("tool_call", self.tool_call),
            "order": series_wire("order", self.order),
            "public": series_wire("public", self.public),
            "canaries": series_wire("canaries", self.canaries),
            "invalid_action_rate": clamp_score(rate, what="invalid_action_rate"),
            "tool_blind": self.tool_blind.to_wire(),
            "observation_shuffle": {
                modality: evidence.to_wire()
                for modality, evidence in sorted(self.observation_shuffle.items())
            },
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> AgentMeasurement:
        shuffle_raw = body.get("observation_shuffle") or {}
        if not isinstance(shuffle_raw, Mapping):
            raise ContractError("observation_shuffle is not an object")
        blind_raw = body.get("tool_blind")
        if not isinstance(blind_raw, Mapping):
            raise ContractError("tool_blind is not an object")
        return cls(
            base_model=str(body.get("base_model", "") or ""),
            teacher_model=str(body.get("teacher_model", "") or ""),
            holdout=read_series(body, "holdout"),
            tool_call=read_series(body, "tool_call"),
            order=read_series(body, "order"),
            public=read_series(body, "public"),
            canaries=read_series(body, "canaries"),
            invalid_action_rate=float(body.get("invalid_action_rate", 0.0) or 0.0),
            tool_blind=ControlEvidence.from_wire(blind_raw),
            observation_shuffle={
                str(modality): ControlEvidence.from_wire(evidence)
                for modality, evidence in shuffle_raw.items()
                if isinstance(evidence, Mapping)
            },
        )


@dataclass(frozen=True)
class AgentDocument:
    """One scored artifact, bound to the run that asked for it."""

    identity: RunIdentity
    measurement: AgentMeasurement
    schema_version: int = SCHEMA_VERSION

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            **self.identity.to_wire(),
            **self.measurement.to_wire(),
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> AgentDocument:
        if not isinstance(body, Mapping):
            raise ContractError("metrics document is not an object")
        return cls(
            schema_version=int(body.get("schema_version", 0) or 0),
            identity=RunIdentity.from_wire(body),
            measurement=AgentMeasurement.from_wire(body),
        )
