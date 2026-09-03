"""The eval image contract: markers, schema, and the metrics document.

Normative source: `docs/RELEARN.md` § Eval image contract in
[`CortexLM/cortex`](https://github.com/CortexLM/cortex), consumed by
`crates/relearn-lium-harvest`. Field names here are the wire names that crate's
`RelearnEvalMetrics` (a `BaselineMeasurement` envelope plus the run identity)
deserializes, so renaming anything in this module breaks live scoring.

Nothing in this module computes a score. It only carries and encodes one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

#: Prefix the image prints before the one-line metrics document.
METRICS_MARKER = "RELEARN_METRICS="

#: Marker the image prints, last, on a completed and self-verified run.
OK_MARKER = "RELEARN_EVAL_OK"

#: Directory the control plane stages `request.json` into, over stdin. Fixed by
#: the harvest client, and deliberately not a persisted path: the request
#: carries the private holdout, so it lives somewhere the post-run scrub and the
#: pod's own teardown both reach.
POD_WORKDIR = "/tmp/relearn_eval"  # noqa: S108 - the control plane's staging path

#: Must equal `relearn_eval::RELEARN_METRICS_SCHEMA` on the control plane.
METRICS_SCHEMA_VERSION = 1

#: Artifact id the control plane asks for when it measures the base model.
BASE_CHAMPION_ARTIFACT = "base-relearn-champion"

#: Run id bound into that boot baseline measurement.
BASE_CHAMPION_RUN = "boot-baseline"

#: Task families that carry an image and take the pixel-shuffle control.
VISION_TASKS = ("captioning", "vqa", "ocr", "spatial")

#: Every task family a holdout item may declare.
TASKS = ("text", *VISION_TASKS)

#: Series keys are `<prefix><item id>`: an id, never a prompt body.
HOLDOUT_KEY_PREFIX = "h"
PERTURBED_KEY_PREFIX = "x"
PUBLIC_KEY_PREFIX = "p"
CANARY_KEY_PREFIX = "c"
GENERAL_CANARY_KEY_PREFIX = "g"

#: Scores are rounded before they are published so two runs of the same model
#: on the same items agree bit-for-bit and the paired test is not decided by
#: float noise.
SCORE_DECIMALS = 6


class ContractError(RuntimeError):
    """A document, request, or pod transcript that the contract refuses."""


def series_key(prefix: str, item_id: int) -> str:
    """Key for one item in a published series."""
    return f"{prefix}{item_id}"


def clamp_score(value: float, *, what: str) -> float:
    """Round a measured score into the published `[0, 1]` grid.

    A non-finite score is a scoring bug, not a zero: publishing it as a number
    would turn a broken run into a verdict.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{what} is not finite")
    return round(min(1.0, max(0.0, number)), SCORE_DECIMALS)


@dataclass(frozen=True)
class ShuffleEvidence:
    """Pixel-shuffle control for one vision family."""

    items: int
    score: float
    shuffled_score: float

    def to_wire(self) -> dict[str, object]:
        return {
            "items": int(self.items),
            "score": clamp_score(self.score, what="shuffle score"),
            "shuffled_score": clamp_score(self.shuffled_score, what="shuffled score"),
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> ShuffleEvidence:
        return cls(
            items=int(body.get("items", 0) or 0),
            score=float(body.get("score", 0.0) or 0.0),
            shuffled_score=float(body.get("shuffled_score", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class Measurement:
    """`BaselineMeasurement`: the series plus what produced them.

    The same shape the operator installs as `RELEARN_BASE_CHAMPION_FILE`, so a
    recorded champion baseline is literally this image's output for the base
    model — one format, not two.
    """

    eval_image_digest: str
    holdout_commitment: str
    holdout: dict[str, float] = field(default_factory=dict)
    public: dict[str, float] = field(default_factory=dict)
    perturbed: dict[str, float] = field(default_factory=dict)
    canaries: dict[str, float] = field(default_factory=dict)
    general_canary: dict[str, float] = field(default_factory=dict)
    agent_trace: float = 0.0
    vision_shuffle: dict[str, ShuffleEvidence] = field(default_factory=dict)

    def to_wire(self) -> dict[str, object]:
        def series(name: str, values: Mapping[str, float]) -> dict[str, float]:
            return {
                key: clamp_score(value, what=f"{name}[{key}]")
                for key, value in sorted(values.items())
            }

        return {
            "eval_image_digest": self.eval_image_digest,
            "holdout_commitment": self.holdout_commitment,
            "holdout": series("holdout", self.holdout),
            "public": series("public", self.public),
            "perturbed": series("perturbed", self.perturbed),
            "canaries": series("canaries", self.canaries),
            "general_canary": series("general_canary", self.general_canary),
            "agent_trace": clamp_score(self.agent_trace, what="agent_trace"),
            "vision_shuffle": {
                task: evidence.to_wire()
                for task, evidence in sorted(self.vision_shuffle.items())
            },
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> Measurement:
        def series(name: str) -> dict[str, float]:
            raw = body.get(name) or {}
            if not isinstance(raw, Mapping):
                raise ContractError(f"{name} is not an object")
            return {str(key): float(value) for key, value in raw.items()}

        shuffle_raw = body.get("vision_shuffle") or {}
        if not isinstance(shuffle_raw, Mapping):
            raise ContractError("vision_shuffle is not an object")
        return cls(
            eval_image_digest=str(body.get("eval_image_digest", "")),
            holdout_commitment=str(body.get("holdout_commitment", "")),
            holdout=series("holdout"),
            public=series("public"),
            perturbed=series("perturbed"),
            canaries=series("canaries"),
            general_canary=series("general_canary"),
            agent_trace=float(body.get("agent_trace", 0.0) or 0.0),
            vision_shuffle={
                str(task): ShuffleEvidence.from_wire(evidence)
                for task, evidence in shuffle_raw.items()
                if isinstance(evidence, Mapping)
            },
        )


@dataclass(frozen=True)
class EvalDocument:
    """`RelearnEvalMetrics`: one scored artifact, bound to the run that asked."""

    submission_digest: str
    artifact_digest: str
    measurement: Measurement
    schema_version: int = METRICS_SCHEMA_VERSION

    def to_wire(self) -> dict[str, object]:
        # `measurement` is `#[serde(flatten)]` on the control plane, so its
        # fields sit beside the identity fields rather than under a key.
        return {
            "schema_version": int(self.schema_version),
            "submission_digest": self.submission_digest,
            "artifact_digest": self.artifact_digest,
            **self.measurement.to_wire(),
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> EvalDocument:
        if not isinstance(body, Mapping):
            raise ContractError("metrics document is not an object")
        return cls(
            schema_version=int(body.get("schema_version", 0) or 0),
            submission_digest=str(body.get("submission_digest", "")),
            artifact_digest=str(body.get("artifact_digest", "")),
            measurement=Measurement.from_wire(body),
        )


def encode_document(document: EvalDocument) -> str:
    """Encode a document as the single line the harvest reads.

    The control plane harvests the sidecar with `printf '<marker>'; cat
    metrics.json`, so an embedded or trailing newline would truncate the
    document mid-JSON and turn a finished run into a 503.
    """
    line = json.dumps(document.to_wire(), separators=(",", ":"), sort_keys=True)
    if "\n" in line or "\r" in line:
        raise ContractError("encoded document contains a newline")
    return line


def decode_document(line: str) -> EvalDocument:
    """Parse one encoded document."""
    try:
        body = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ContractError(f"metrics document is not JSON: {exc}") from exc
    return EvalDocument.from_wire(body)


def marker_line(document: EvalDocument) -> str:
    """The `RELEARN_METRICS=<document>` line, without its newline."""
    return f"{METRICS_MARKER}{encode_document(document)}"


def is_marker_line(line: str) -> bool:
    return line.startswith(METRICS_MARKER)


def mean(values: Iterable[float]) -> float | None:
    """Mean of a series, or `None` when it is empty."""
    collected = list(values)
    if not collected:
        return None
    return sum(collected) / len(collected)
