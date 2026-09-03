"""The harvest request the control plane stages into `/tmp/relearn_eval`.

Mirrors `relearn_lium_harvest::HarvestRequest`. Unknown fields are tolerated so
a control plane that grows the request (an `artifact_uri`, say) does not need a
new image, but every field the image acts on is validated: an unusable request
is a failed run, never a scored one.

**The request carries the private holdout.** Nothing in this module — or
anywhere else in the image — writes a holdout prompt to stdout or to a
persisted path.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .commitment import commitments_match, holdout_commitment
from .contract import (
    BASE_CHAMPION_ARTIFACT,
    METRICS_SCHEMA_VERSION,
    TASKS,
    VISION_TASKS,
    ContractError,
)


class RequestError(ContractError):
    """A request the image refuses to score."""


def _is_hex(value: str, length: int) -> bool:
    trimmed = value.strip()
    return len(trimmed) == length and all(char in "0123456789abcdefABCDEF" for char in trimmed)


@dataclass(frozen=True)
class HoldoutItem:
    """One item to score. `relearn_challenge_task::HoldoutItem`."""

    id: int
    prompt: str
    dataset_id: str = ""
    task: str = "text"
    image_hash: str = ""

    @property
    def is_vision(self) -> bool:
        return self.task in VISION_TASKS

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> HoldoutItem:
        if not isinstance(body, Mapping):
            raise RequestError("holdout item is not an object")
        if "id" not in body:
            raise RequestError("holdout item has no id")
        try:
            item_id = int(body["id"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise RequestError("holdout item id is not an integer") from exc
        return cls(
            id=item_id,
            prompt=str(body.get("prompt", "")),
            dataset_id=str(body.get("dataset_id", "") or ""),
            task=str(body.get("task", "text") or "text"),
            image_hash=str(body.get("image_hash", "") or ""),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "dataset_id": self.dataset_id,
            "task": self.task,
            "image_hash": self.image_hash,
        }


@dataclass(frozen=True)
class HarvestRequest:
    """What the image was asked to score."""

    schema_version: int
    submission_digest: str
    artifact_digest: str
    base_model: str
    teacher_model: str
    eval_image_digest: str
    holdout_commitment: str
    holdout: tuple[HoldoutItem, ...] = field(default_factory=tuple)
    artifact_uri: str = ""

    @property
    def is_base_champion_run(self) -> bool:
        """True when the control plane is measuring the un-post-trained base.

        `boot_base_champion` asks for [`BASE_CHAMPION_ARTIFACT`]; there is no
        artifact to fetch, and the base model itself is the subject.
        """
        return self.artifact_digest.strip() == BASE_CHAMPION_ARTIFACT

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> HarvestRequest:
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
            submission_digest=str(body.get("submission_digest", "") or ""),
            artifact_digest=str(body.get("artifact_digest", "") or ""),
            base_model=str(body.get("base_model", "") or ""),
            teacher_model=str(body.get("teacher_model", "") or ""),
            eval_image_digest=str(body.get("eval_image_digest", "") or ""),
            holdout_commitment=str(body.get("holdout_commitment", "") or ""),
            holdout=tuple(HoldoutItem.from_wire(item) for item in raw_holdout),
            artifact_uri=str(body.get("artifact_uri", "") or ""),
        )

    @classmethod
    def from_json(cls, body: str) -> HarvestRequest:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RequestError(f"request is not JSON: {exc}") from exc
        return cls.from_wire(parsed)

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "schema_version": self.schema_version,
            "submission_digest": self.submission_digest,
            "artifact_digest": self.artifact_digest,
            "base_model": self.base_model,
            "teacher_model": self.teacher_model,
            "eval_image_digest": self.eval_image_digest,
            "holdout_commitment": self.holdout_commitment,
            "holdout": [item.to_wire() for item in self.holdout],
        }
        if self.artifact_uri:
            wire["artifact_uri"] = self.artifact_uri
        return wire

    def validate(self) -> None:
        """Refuse a request the image cannot honour.

        # Raises
        [`RequestError`] on a schema, identity, image-digest, holdout, or
        commitment problem. Every one of these is fail-closed: the run ends
        without a document rather than scoring something other than what was
        asked for.
        """
        if self.schema_version != METRICS_SCHEMA_VERSION:
            raise RequestError(
                f"request schema_version {self.schema_version}, "
                f"expected {METRICS_SCHEMA_VERSION}"
            )
        if not self.submission_digest.strip():
            raise RequestError("request has no submission_digest")
        if not self.artifact_digest.strip():
            raise RequestError("request has no artifact_digest")
        if not self.base_model.strip():
            raise RequestError("request has no base_model")
        if not self.teacher_model.strip():
            raise RequestError("request has no teacher_model")

        digest = self.eval_image_digest.strip()
        # `RelearnPin::can_rent` only rents a `sha256:`-pinned image, and the
        # document echoes this value back to be checked against the pin.
        if not digest.startswith("sha256:") or len(digest) < 71:
            raise RequestError("request eval_image_digest is not a sha256: pin")

        if not self.holdout:
            raise RequestError("request carries no holdout items")

        seen: set[int] = set()
        for item in self.holdout:
            if item.id in seen:
                raise RequestError(f"duplicate holdout id {item.id}")
            seen.add(item.id)
            if not item.prompt.strip():
                raise RequestError(f"holdout id {item.id} has an empty prompt")
            if item.task not in TASKS:
                raise RequestError(f"holdout id {item.id} has unknown task {item.task!r}")
            if item.is_vision and not _is_hex(item.image_hash, 64):
                raise RequestError(
                    f"holdout id {item.id} is a vision task without an image hash"
                )

        declared = self.holdout_commitment.strip()
        if not _is_hex(declared, 64):
            raise RequestError("request holdout_commitment is not 64 hex chars")
        measured = holdout_commitment(self.holdout)
        if not commitments_match(measured, declared):
            # Hex only. The items themselves never reach a log.
            raise RequestError(
                f"holdout commitment mismatch: items hash to {measured}, "
                f"request declared {declared.lower()}"
            )

    def vision_families(self) -> tuple[str, ...]:
        """Vision families present in this holdout, in contract order."""
        present = {item.task for item in self.holdout if item.is_vision}
        return tuple(task for task in VISION_TASKS if task in present)


def read_request(source: str | Path) -> HarvestRequest:
    """Read a request from a path, or from stdin when `source` is `-`."""
    if str(source) == "-":
        body = sys.stdin.read()
    else:
        body = Path(source).read_text(encoding="utf-8")
    return HarvestRequest.from_json(body)
