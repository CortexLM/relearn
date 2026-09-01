"""The harvest request for the Relearn Image challenge.

Mirrors the shape of `relearn_lium_harvest::HarvestRequest`, carrying what a
text-to-image run needs instead of what a language run needs: frozen prompts
rather than holdout items, the seed lattice the whole subnet shares, the frozen
sampler recipe, and the miner's declared manifest.

Unknown fields are tolerated so a control plane that grows the request does not
need a new image digest, but every field the image acts on is validated. An
unusable request is a failed run, never a scored one.

**The request carries the private holdout.** Nothing in this module — or
anywhere else in the image — writes a prompt to stdout or to a persisted path.

Two refusals here are the challenge's product rules rather than hygiene: a
Flux-family base is rejected outright, and any judge but Q-Judger is rejected
outright. Both are checked before a single image is generated, so a submission
that should never have been accepted costs a refusal rather than a GPU hour.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from relearn_common.clitools import read_text
from relearn_common.errors import ContractError
from relearn_common.identity import RunIdentity
from relearn_common.judge import model_matches

from .commitment import commitments_match, frozen_prompt_commitment
from .pins import (
    BASE_CHAMPION_ARTIFACT,
    BASE_MODEL_ID,
    BASE_MODEL_LICENSE,
    CHALLENGE_IDS,
    JUDGE_MODEL_ID,
    MIN_SCORED_CELLS,
    SCHEMA_VERSION,
    base_is_rejected,
    is_bench_prompt_id,
    normalize_license,
)
from .seeds import seed_cells


class RequestError(ContractError):
    """A request the image refuses to score."""


@dataclass(frozen=True)
class FrozenPrompt:
    """One frozen eval prompt. `relearn_t2i_task::FrozenPrompt`.

    NVIDIA recommends upsampling a short prompt into a JSON document before
    handing it to Cosmos3. That is fine for a miner's own training and fatal
    for a benchmark, so whichever string will be sent to the generator is
    frozen in the pin and replayed verbatim. The image never runs an upsampler.
    """

    id: int
    text: str
    upsampled_json: str = ""

    @property
    def generator_input(self) -> str:
        """The exact string handed to the generator for this prompt."""
        return self.upsampled_json or self.text

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> FrozenPrompt:
        if not isinstance(body, Mapping):
            raise RequestError("frozen prompt is not an object")
        if "id" not in body:
            raise RequestError("frozen prompt has no id")
        try:
            prompt_id = int(body["id"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise RequestError("frozen prompt id is not an integer") from exc
        return cls(
            id=prompt_id,
            text=str(body.get("text", "") or ""),
            upsampled_json=str(body.get("upsampled_json", "") or ""),
        )

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {"id": self.id, "text": self.text}
        if self.upsampled_json:
            wire["upsampled_json"] = self.upsampled_json
        return wire


@dataclass(frozen=True)
class Sampler:
    """The frozen sampler recipe. `relearn_t2i_task::SamplerConfig`.

    Defaults are the Cosmos3-Super-Text2Image card's text-to-image recipe, so a
    request that omits the block still generates what the pin describes.
    """

    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    flow_shift: float = 3.0
    negative_prompt: str = ""
    num_frames: int = 1
    dtype: str = "bfloat16"
    scheduler: str = "UniPCMultistepScheduler"

    @classmethod
    def from_wire(cls, body: Mapping[str, object] | None) -> Sampler:
        if not body:
            return cls()
        if not isinstance(body, Mapping):
            raise RequestError("sampler is not an object")
        default = cls()
        try:
            return cls(
                width=int(body.get("width", default.width)),
                height=int(body.get("height", default.height)),
                num_inference_steps=int(
                    body.get("num_inference_steps", default.num_inference_steps)
                ),
                guidance_scale=float(body.get("guidance_scale", default.guidance_scale)),
                flow_shift=float(body.get("flow_shift", default.flow_shift)),
                negative_prompt=str(body.get("negative_prompt", "") or ""),
                num_frames=int(body.get("num_frames", default.num_frames)),
                dtype=str(body.get("dtype", default.dtype) or default.dtype),
                scheduler=str(body.get("scheduler", default.scheduler) or default.scheduler),
            )
        except (TypeError, ValueError) as exc:
            raise RequestError(f"sampler is not well formed: {exc}") from exc

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise RequestError("sampler resolution must be positive")
        if self.num_inference_steps <= 0:
            raise RequestError("sampler num_inference_steps must be positive")
        if self.num_frames != 1:
            raise RequestError("this challenge scores single images: num_frames must be 1")


@dataclass(frozen=True)
class ArtifactManifest:
    """What the miner declared about the artifact.

    `relearn_t2i_store::ArtifactManifest`. The image needs it for three things
    the request cannot answer on its own: whether the declared base and license
    attest to the pin, which eval prompt ids the submission admits to training
    on, and which output hashes the seed-replay check compares against.
    """

    base: str = ""
    base_license: str = ""
    train_prompt_ids: tuple[int, ...] = ()
    claimed_output_hashes: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, body: Mapping[str, object] | None) -> ArtifactManifest:
        if not body:
            return cls()
        if not isinstance(body, Mapping):
            raise RequestError("manifest is not an object")
        raw_ids = body.get("train_prompt_ids") or []
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, str | bytes):
            raise RequestError("manifest train_prompt_ids is not a list")
        raw_hashes = body.get("claimed_output_hashes") or {}
        if not isinstance(raw_hashes, Mapping):
            raise RequestError("manifest claimed_output_hashes is not an object")
        try:
            ids = tuple(int(value) for value in raw_ids)
        except (TypeError, ValueError) as exc:
            raise RequestError("manifest train_prompt_ids are not integers") from exc
        return cls(
            base=str(body.get("base", "") or ""),
            base_license=str(body.get("base_license", "") or ""),
            train_prompt_ids=ids,
            claimed_output_hashes={
                str(key): str(value).strip().lower() for key, value in raw_hashes.items()
            },
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "base": self.base,
            "base_license": self.base_license,
            "train_prompt_ids": list(self.train_prompt_ids),
            "claimed_output_hashes": dict(self.claimed_output_hashes),
        }


@dataclass(frozen=True)
class ImageHarvestRequest:
    """What the image was asked to score."""

    schema_version: int
    challenge_id: str
    submission_digest: str
    artifact_digest: str
    base_model: str
    judge_model: str
    eval_image_digest: str
    holdout_commitment: str
    pin_salt: str
    variations_per_prompt: int
    holdout: tuple[FrozenPrompt, ...] = ()
    public: tuple[FrozenPrompt, ...] = ()
    sampler: Sampler = field(default_factory=Sampler)
    manifest: ArtifactManifest = field(default_factory=ArtifactManifest)
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
        """True when the control plane is measuring the un-fine-tuned base."""
        return self.artifact_digest.strip() == BASE_CHAMPION_ARTIFACT

    def holdout_ids(self) -> list[int]:
        return sorted(prompt.id for prompt in self.holdout)

    def public_ids(self) -> list[int]:
        return sorted(prompt.id for prompt in self.public)

    def holdout_cells(self) -> list[tuple[int, int, int]]:
        return list(seed_cells(self.holdout_ids(), self.variations_per_prompt, self.pin_salt))

    def public_cells(self) -> list[tuple[int, int, int]]:
        return list(seed_cells(self.public_ids(), self.variations_per_prompt, self.pin_salt))

    def prompt_text(self, prompt_id: int) -> str:
        """The frozen generator input for one prompt id."""
        for prompt in (*self.holdout, *self.public):
            if prompt.id == prompt_id:
                return prompt.generator_input
        raise RequestError(f"no frozen prompt for id {prompt_id}")

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> ImageHarvestRequest:
        if not isinstance(body, Mapping):
            raise RequestError("request is not an object")

        def prompts(name: str) -> tuple[FrozenPrompt, ...]:
            raw = body.get(name) or []
            if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
                raise RequestError(f"request {name} is not a list")
            return tuple(FrozenPrompt.from_wire(item) for item in raw)

        try:
            schema_version = int(body.get("schema_version", 0) or 0)
            variations = int(body.get("variations_per_prompt", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise RequestError("request has a non-integer field") from exc

        return cls(
            schema_version=schema_version,
            challenge_id=str(body.get("challenge_id", "") or ""),
            submission_digest=str(body.get("submission_digest", "") or ""),
            artifact_digest=str(body.get("artifact_digest", "") or ""),
            base_model=str(body.get("base_model", "") or ""),
            judge_model=str(body.get("judge_model", "") or ""),
            eval_image_digest=str(body.get("eval_image_digest", "") or ""),
            holdout_commitment=str(body.get("holdout_commitment", "") or ""),
            pin_salt=str(body.get("pin_salt", "") or ""),
            variations_per_prompt=variations,
            holdout=prompts("holdout"),
            public=prompts("public"),
            sampler=Sampler.from_wire(body.get("sampler")),  # type: ignore[arg-type]
            manifest=ArtifactManifest.from_wire(body.get("manifest")),  # type: ignore[arg-type]
            artifact_uri=str(body.get("artifact_uri", "") or ""),
        )

    @classmethod
    def from_json(cls, body: str) -> ImageHarvestRequest:
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
            "judge_model": self.judge_model,
            "eval_image_digest": self.eval_image_digest,
            "holdout_commitment": self.holdout_commitment,
            "pin_salt": self.pin_salt,
            "variations_per_prompt": self.variations_per_prompt,
            "holdout": [prompt.to_wire() for prompt in self.holdout],
            "public": [prompt.to_wire() for prompt in self.public],
            "sampler": {
                "width": self.sampler.width,
                "height": self.sampler.height,
                "num_inference_steps": self.sampler.num_inference_steps,
                "guidance_scale": self.sampler.guidance_scale,
                "flow_shift": self.sampler.flow_shift,
                "negative_prompt": self.sampler.negative_prompt,
                "num_frames": self.sampler.num_frames,
                "dtype": self.sampler.dtype,
                "scheduler": self.sampler.scheduler,
            },
            "manifest": self.manifest.to_wire(),
        }
        if self.artifact_uri:
            wire["artifact_uri"] = self.artifact_uri
        return wire

    def validate(self) -> None:
        """Refuse a request the image cannot honour.

        # Raises
        [`RequestError`] on a schema, identity, base, judge, split, seed, or
        commitment problem. Every one of these is fail-closed: the run ends
        without a document rather than scoring something other than what was
        asked for.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise RequestError(
                f"request schema_version {self.schema_version}, expected {SCHEMA_VERSION}"
            )
        try:
            self.identity.validate(challenge_ids=CHALLENGE_IDS)
        except ContractError as exc:
            raise RequestError(str(exc)) from exc

        self._validate_models()
        self._validate_splits()
        self._validate_seed_lattice()
        self.sampler.validate()
        self._validate_commitment()

    def _validate_models(self) -> None:
        if base_is_rejected(self.base_model):
            raise RequestError(
                f"base {self.base_model!r} is a refused family for this challenge"
            )
        if not model_matches(self.base_model, BASE_MODEL_ID):
            raise RequestError(f"base must be {BASE_MODEL_ID!r}, got {self.base_model!r}")
        if not model_matches(self.judge_model, JUDGE_MODEL_ID):
            raise RequestError(f"judge must be {JUDGE_MODEL_ID!r}, got {self.judge_model!r}")

        # The manifest is the miner's own declaration. It is attested against
        # the pin here, before any GPU time is spent, exactly as
        # `RelearnT2iPin::attest_artifact_base` does on the control plane.
        if self.is_base_champion_run:
            return
        declared_base = self.manifest.base
        if base_is_rejected(declared_base):
            raise RequestError(
                f"artifact base {declared_base!r} is a refused family for this challenge"
            )
        if not model_matches(declared_base, BASE_MODEL_ID):
            raise RequestError(
                f"artifact base must be {BASE_MODEL_ID!r}, got {declared_base!r}"
            )
        if normalize_license(self.manifest.base_license) != normalize_license(
            BASE_MODEL_LICENSE
        ):
            raise RequestError(
                f"artifact license must be {BASE_MODEL_LICENSE!r}, "
                f"got {self.manifest.base_license!r}"
            )

    def _validate_splits(self) -> None:
        if not self.holdout:
            raise RequestError("request carries no holdout prompts")
        if not self.public:
            # The public split is what the memorization gap gate compares
            # against. Without it the gate silently does not run, which looks
            # like a pass.
            raise RequestError("request carries no public split; the gap gate cannot run")

        holdout_ids = set()
        for prompt in self.holdout:
            self._validate_prompt(prompt, holdout_ids, "holdout")
        public_ids = set()
        for prompt in self.public:
            self._validate_prompt(prompt, public_ids, "public")

        overlap = sorted(holdout_ids & public_ids)
        if overlap:
            raise RequestError(
                f"holdout prompt ids are also in the public split: {overlap[:8]}"
            )

    @staticmethod
    def _validate_prompt(prompt: FrozenPrompt, seen: set[int], split: str) -> None:
        if prompt.id in seen:
            raise RequestError(f"duplicate {split} prompt id {prompt.id}")
        seen.add(prompt.id)
        if not is_bench_prompt_id(prompt.id):
            raise RequestError(
                f"{split} prompt id {prompt.id} outside Qwen-Image-Bench range 1..=1000"
            )
        if not prompt.generator_input.strip():
            raise RequestError(f"{split} prompt id {prompt.id} has empty text")

    def _validate_seed_lattice(self) -> None:
        if not self.pin_salt.strip():
            raise RequestError("pin_salt must not be empty; the seed lattice is not pinned")
        if self.variations_per_prompt <= 0:
            raise RequestError("variations_per_prompt must be positive")
        for split, count in (
            ("holdout", len(self.holdout)),
            ("public", len(self.public)),
        ):
            cells = count * self.variations_per_prompt
            if cells < MIN_SCORED_CELLS:
                raise RequestError(
                    f"{split} yields {cells} scored cells, below the "
                    f"{MIN_SCORED_CELLS} floor the paired test needs"
                )

    def _validate_commitment(self) -> None:
        measured = frozen_prompt_commitment(self.holdout)
        if not commitments_match(measured, self.holdout_commitment):
            # Hex only. The prompts themselves never reach a log.
            raise RequestError(
                f"holdout commitment mismatch: prompts hash to {measured}, "
                f"request declared {self.holdout_commitment.strip().lower()}"
            )


def read_request(source: str | Path) -> ImageHarvestRequest:
    """Read a request from a path, or from stdin when `source` is `-`."""
    return ImageHarvestRequest.from_json(read_text(source))
