"""The harvest request for the Relearn Agent challenge.

Mirrors the shape of `relearn_lium_harvest::HarvestRequest`, carrying recorded
tool-use episodes instead of prompts. Unknown fields are tolerated so a control
plane that grows the request does not need a new image digest, but every field
the image acts on is validated: an unusable request is a failed run, never a
scored one.

**The request carries the private holdout.** A trace is more sensitive than a
prompt — it contains the goal, the tool schemas, every recorded argument, and
every observation — so nothing in this image writes any part of one to stdout
or to a persisted path. Series keys are ids.

Pixels stay out of the request. A step whose observation is a screenshot names
it by `sha256` and the operator mounts the store on the pod, exactly as the
language challenge does for its vision items.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from relearn_common.clitools import read_text
from relearn_common.errors import ContractError
from relearn_common.identity import RunIdentity, is_hex
from relearn_common.judge import model_matches

from .commitment import commitments_match, trace_commitment
from .pins import (
    BASE_CHAMPION_ARTIFACT,
    BASE_MODEL_ID,
    CHALLENGE_IDS,
    IMAGE_MODALITY,
    MIN_HOLDOUT_TRACES,
    SCHEMA_VERSION,
)


class RequestError(ContractError):
    """A request the image refuses to score."""


def _canonical_arguments(value: object) -> str:
    """Recorded arguments, canonically encoded.

    Sorted keys and no whitespace, so the commitment does not depend on how the
    operator's exporter happened to serialize the same arguments.
    """
    if value in (None, ""):
        return "{}"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RequestError(f"step arguments are not JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RequestError("step arguments are not an object")
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class ToolSchema:
    """One tool the episode made available."""

    name: str
    description: str = ""
    parameters: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> ToolSchema:
        if not isinstance(body, Mapping):
            raise RequestError("tool schema is not an object")
        name = str(body.get("name", "") or "").strip()
        if not name:
            raise RequestError("tool schema has no name")
        parameters = body.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise RequestError(f"tool {name!r} parameters are not an object")
        return cls(
            name=name,
            description=str(body.get("description", "") or ""),
            parameters=dict(parameters),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class TraceStep:
    """One recorded turn: the action that was taken, and what came back."""

    index: int
    tool: str
    arguments_json: str
    observation: str = ""
    observation_image_hash: str = ""

    @property
    def arguments(self) -> dict[str, object]:
        parsed = json.loads(self.arguments_json)
        return parsed if isinstance(parsed, dict) else {}

    @property
    def has_image(self) -> bool:
        return bool(self.observation_image_hash.strip())

    @classmethod
    def from_wire(cls, index: int, body: Mapping[str, object]) -> TraceStep:
        if not isinstance(body, Mapping):
            raise RequestError("trace step is not an object")
        return cls(
            index=index,
            tool=str(body.get("tool", "") or "").strip(),
            arguments_json=_canonical_arguments(body.get("arguments")),
            observation=str(body.get("observation", "") or ""),
            observation_image_hash=str(body.get("observation_image_hash", "") or ""),
        )

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "tool": self.tool,
            "arguments": json.loads(self.arguments_json),
            "observation": self.observation,
        }
        if self.observation_image_hash:
            wire["observation_image_hash"] = self.observation_image_hash
        return wire


@dataclass(frozen=True)
class Trace:
    """One recorded tool-use episode to replay."""

    id: int
    goal: str
    steps: tuple[TraceStep, ...]
    tools: tuple[ToolSchema, ...] = ()
    dataset_id: str = ""
    final_answer: str = ""

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    @property
    def has_image_observation(self) -> bool:
        return any(step.has_image for step in self.steps)

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> Trace:
        if not isinstance(body, Mapping):
            raise RequestError("trace is not an object")
        if "id" not in body:
            raise RequestError("trace has no id")
        try:
            trace_id = int(body["id"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise RequestError("trace id is not an integer") from exc

        raw_steps = body.get("steps") or []
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, str | bytes):
            raise RequestError(f"trace {trace_id} steps are not a list")
        raw_tools = body.get("tools") or []
        if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, str | bytes):
            raise RequestError(f"trace {trace_id} tools are not a list")

        return cls(
            id=trace_id,
            goal=str(body.get("goal", "") or ""),
            steps=tuple(
                TraceStep.from_wire(index, step) for index, step in enumerate(raw_steps)
            ),
            tools=tuple(ToolSchema.from_wire(tool) for tool in raw_tools),
            dataset_id=str(body.get("dataset_id", "") or ""),
            final_answer=str(body.get("final_answer", "") or ""),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "goal": self.goal,
            "dataset_id": self.dataset_id,
            "tools": [tool.to_wire() for tool in self.tools],
            "steps": [step.to_wire() for step in self.steps],
            "final_answer": self.final_answer,
        }

    def validate(self, *, where: str) -> None:
        if not self.goal.strip():
            raise RequestError(f"{where} trace {self.id} has an empty goal")
        if not self.steps:
            raise RequestError(f"{where} trace {self.id} records no steps")
        if not self.tools:
            raise RequestError(f"{where} trace {self.id} offers no tools")
        names = self.tool_names
        if len(set(names)) != len(names):
            raise RequestError(f"{where} trace {self.id} declares a tool twice")
        for step in self.steps:
            if not step.tool:
                raise RequestError(f"{where} trace {self.id} step {step.index} names no tool")
            if step.tool not in names:
                # A recorded action outside the trace's own schema would be
                # ungradeable: the model is shown the schema, so it could never
                # produce that action, and every model would score zero there.
                raise RequestError(
                    f"{where} trace {self.id} step {step.index} calls {step.tool!r}, "
                    "which is not in its tool schema"
                )
            if step.has_image and not is_hex(step.observation_image_hash, 64):
                raise RequestError(
                    f"{where} trace {self.id} step {step.index} has a malformed image hash"
                )
        if not self.final_answer.strip():
            raise RequestError(f"{where} trace {self.id} records no final answer")


@dataclass(frozen=True)
class AgentHarvestRequest:
    """What the image was asked to score."""

    schema_version: int
    challenge_id: str
    submission_digest: str
    artifact_digest: str
    base_model: str
    teacher_model: str
    eval_image_digest: str
    holdout_commitment: str
    holdout: tuple[Trace, ...] = ()
    artifact_uri: str = ""

    @property
    def identity(self) -> RunIdentity:
        return RunIdentity(
            challenge_id=self.challenge_id,
            submission_digest=self.submission_digest,
            artifact_digest=self.artifact_digest,
            eval_image_digest=self.eval_image_digest,
            holdout_commitment=self.holdout_commitment,
        )

    @property
    def is_base_champion_run(self) -> bool:
        """True when the control plane is measuring the un-post-trained base."""
        return self.artifact_digest.strip() == BASE_CHAMPION_ARTIFACT

    def modalities(self) -> tuple[str, ...]:
        """Observation modalities present in this holdout that take a control."""
        if any(trace.has_image_observation for trace in self.holdout):
            return (IMAGE_MODALITY,)
        return ()

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> AgentHarvestRequest:
        if not isinstance(body, Mapping):
            raise RequestError("request is not an object")
        raw_holdout = body.get("holdout") or []
        if not isinstance(raw_holdout, Sequence) or isinstance(raw_holdout, str | bytes):
            raise RequestError("request holdout is not a list")
        try:
            schema_version = int(body.get("schema_version", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise RequestError("schema_version is not an integer") from exc
        return cls(
            schema_version=schema_version,
            challenge_id=str(body.get("challenge_id", "") or ""),
            submission_digest=str(body.get("submission_digest", "") or ""),
            artifact_digest=str(body.get("artifact_digest", "") or ""),
            base_model=str(body.get("base_model", "") or ""),
            teacher_model=str(body.get("teacher_model", "") or ""),
            eval_image_digest=str(body.get("eval_image_digest", "") or ""),
            holdout_commitment=str(body.get("holdout_commitment", "") or ""),
            holdout=tuple(Trace.from_wire(trace) for trace in raw_holdout),
            artifact_uri=str(body.get("artifact_uri", "") or ""),
        )

    @classmethod
    def from_json(cls, body: str) -> AgentHarvestRequest:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RequestError(f"request is not JSON: {exc}") from exc
        return cls.from_wire(parsed)

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "schema_version": self.schema_version,
            "challenge_id": self.challenge_id,
            "submission_digest": self.submission_digest,
            "artifact_digest": self.artifact_digest,
            "base_model": self.base_model,
            "teacher_model": self.teacher_model,
            "eval_image_digest": self.eval_image_digest,
            "holdout_commitment": self.holdout_commitment,
            "holdout": [trace.to_wire() for trace in self.holdout],
        }
        if self.artifact_uri:
            wire["artifact_uri"] = self.artifact_uri
        return wire

    def validate(self) -> None:
        """Refuse a request the image cannot honour.

        # Raises
        [`RequestError`] on a schema, identity, base, trace, or commitment
        problem. Every one of these is fail-closed: the run ends without a
        document rather than scoring something other than what was asked for.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise RequestError(
                f"request schema_version {self.schema_version}, expected {SCHEMA_VERSION}"
            )
        try:
            self.identity.validate(challenge_ids=CHALLENGE_IDS)
        except ContractError as exc:
            raise RequestError(str(exc)) from exc

        if not model_matches(self.base_model, BASE_MODEL_ID):
            raise RequestError(f"base must be {BASE_MODEL_ID!r}, got {self.base_model!r}")
        if not self.teacher_model.strip():
            raise RequestError("request has no teacher_model")

        if not self.holdout:
            raise RequestError("request carries no holdout traces")
        if len(self.holdout) < MIN_HOLDOUT_TRACES:
            raise RequestError(
                f"holdout carries {len(self.holdout)} traces, below the "
                f"{MIN_HOLDOUT_TRACES} floor the paired test needs"
            )

        seen: set[int] = set()
        for trace in self.holdout:
            if trace.id in seen:
                raise RequestError(f"duplicate holdout trace id {trace.id}")
            seen.add(trace.id)
            trace.validate(where="holdout")

        measured = trace_commitment(self.holdout)
        if not commitments_match(measured, self.holdout_commitment):
            # Hex only. The traces themselves never reach a log.
            raise RequestError(
                f"holdout commitment mismatch: traces hash to {measured}, "
                f"request declared {self.holdout_commitment.strip().lower()}"
            )


def read_request(source: str | Path) -> AgentHarvestRequest:
    """Read a request from a path, or from stdin when `source` is `-`."""
    return AgentHarvestRequest.from_json(read_text(source))
