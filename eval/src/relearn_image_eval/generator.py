"""Generating the scored images with the pinned Cosmos3 checkpoint.

The subject of a run is the miner artifact loaded **inside this image** on top
of `nvidia/Cosmos3-Super-Text2Image`. There is no offline stand-in and no
second generator: when the pipeline cannot be loaded the run fails, because
scores computed without the model would be made-up numbers wearing a real run's
identity.

Every cell is generated at the frozen seed the whole subnet shares, from the
frozen prompt string, with the pin's sampler recipe. Nothing about generation
is left to the pod: two artifacts scored on two pods are compared on the same
prompts, the same seeds, and the same sampler.

Heavy imports stay inside the backend so the contract layer can be tested and
linted on a machine with no CUDA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from relearn_common.env import env
from relearn_common.errors import ContractError

from .pins import BASE_MODEL_ID, ENV_PREFIX
from .request import Sampler

log = logging.getLogger(__name__)


class GeneratorError(ContractError):
    """The generator could not be loaded, or could not produce an image."""


def resolve_base_checkpoint(base_model: str) -> str:
    """Where to load the Cosmos3 weights from.

    A local directory wins so a pod can be primed with the 65B checkpoint
    instead of pulling it per run.
    """
    local = env(f"{ENV_PREFIX}BASE_MODEL_DIR")
    if local:
        path = Path(local)
        if not path.is_dir():
            raise GeneratorError(f"{ENV_PREFIX}BASE_MODEL_DIR {local} is not a directory")
        return str(path)
    if not base_model.strip():
        raise GeneratorError(
            f"no base checkpoint: set {ENV_PREFIX}BASE_MODEL_DIR or pin base_model"
        )
    return base_model.strip()


def is_adapter(artifact_dir: Path | None) -> bool:
    """Whether the artifact is a LoRA adapter rather than a full checkpoint."""
    if artifact_dir is None:
        return False
    return any(
        (artifact_dir / name).is_file()
        for name in ("adapter_config.json", "pytorch_lora_weights.safetensors")
    )


@dataclass
class CosmosGenerator:
    """Diffusers backend for the pinned Cosmos3 text-to-image pipeline."""

    base_checkpoint: str
    artifact_dir: Path | None
    sampler: Sampler
    _pipeline: object | None = field(default=None, repr=False)
    _torch: object | None = field(default=None, repr=False)

    #: Pipeline classes to try, in order. The card names `Cosmos3OmniPipeline`;
    #: `DiffusionPipeline` is the portable fallback that reads the class out of
    #: the checkpoint's own `model_index.json`, so a diffusers release that has
    #: not landed the named class yet still loads the right pipeline.
    PIPELINE_CLASSES = ("Cosmos3OmniPipeline", "DiffusionPipeline")

    def __post_init__(self) -> None:  # pragma: no cover - needs torch + diffusers
        try:
            import diffusers
            import torch
        except ImportError as exc:
            raise GeneratorError("torch / diffusers are not installed in this image") from exc

        self._torch = torch
        dtype = getattr(torch, self.sampler.dtype, None)
        if dtype is None:
            raise GeneratorError(f"sampler dtype {self.sampler.dtype!r} is not a torch dtype")

        weights = (
            self.base_checkpoint
            if is_adapter(self.artifact_dir) or self.artifact_dir is None
            else str(self.artifact_dir)
        )
        self._pipeline = self._load_pipeline(diffusers, weights, dtype)
        if is_adapter(self.artifact_dir) and self.artifact_dir is not None:
            loader = getattr(self._pipeline, "load_lora_weights", None)
            if loader is None:
                raise GeneratorError(
                    "artifact is a LoRA adapter but this pipeline cannot load one"
                )
            loader(str(self.artifact_dir))
        self._configure_scheduler(diffusers)

    @classmethod
    def _load_pipeline(cls, diffusers, weights: str, dtype):  # pragma: no cover
        last: Exception | None = None
        for name in cls.PIPELINE_CLASSES:
            pipeline_class = getattr(diffusers, name, None)
            if pipeline_class is None:
                continue
            try:
                pipeline = pipeline_class.from_pretrained(
                    weights, torch_dtype=dtype, trust_remote_code=False
                )
            except TypeError:
                pipeline = pipeline_class.from_pretrained(weights, torch_dtype=dtype)
            except (OSError, ValueError, KeyError) as exc:
                log.info("%s could not load these weights: %s", name, type(exc).__name__)
                last = exc
                continue
            mover = getattr(pipeline, "to", None)
            if mover is not None:
                pipeline = mover("cuda") if _cuda_available() else pipeline
            return pipeline
        raise GeneratorError(f"no diffusers pipeline could load {weights}") from last

    def _configure_scheduler(self, diffusers) -> None:  # pragma: no cover
        """Install the pin's scheduler, with the pin's flow shift.

        The recipe is part of the contract. A pod that quietly kept the
        checkpoint's default scheduler would produce images that are not
        comparable with the champion's.
        """
        scheduler_class = getattr(diffusers, self.sampler.scheduler, None)
        if scheduler_class is None:
            raise GeneratorError(
                f"sampler scheduler {self.sampler.scheduler!r} is not in this diffusers build"
            )
        current = getattr(self._pipeline, "scheduler", None)
        if current is None:
            raise GeneratorError("pipeline exposes no scheduler to pin")
        config = dict(getattr(current, "config", {}))
        config["flow_shift"] = self.sampler.flow_shift
        try:
            self._pipeline.scheduler = scheduler_class.from_config(config)  # type: ignore[union-attr]
        except (TypeError, ValueError) as exc:
            raise GeneratorError(f"could not pin the scheduler: {exc}") from exc

    def generate(self, prompt: str, seed: int) -> bytes:  # pragma: no cover - needs GPU
        """One PNG for one `(prompt, seed)` cell."""
        if self._pipeline is None or self._torch is None:
            raise GeneratorError("generator is not initialised")
        from io import BytesIO

        generator = self._torch.Generator(  # type: ignore[attr-defined]
            device="cuda" if _cuda_available() else "cpu"
        ).manual_seed(seed)
        result = self._pipeline(  # type: ignore[operator]
            prompt=prompt,
            negative_prompt=self.sampler.negative_prompt or None,
            width=self.sampler.width,
            height=self.sampler.height,
            num_inference_steps=self.sampler.num_inference_steps,
            guidance_scale=self.sampler.guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
        )
        images = getattr(result, "images", None)
        if not images:
            raise GeneratorError("the pipeline returned no image")
        buffer = BytesIO()
        images[0].save(buffer, format="PNG")
        return buffer.getvalue()

    def close(self) -> None:  # pragma: no cover
        self._pipeline = None
        self._torch = None


def _cuda_available() -> bool:  # pragma: no cover - needs torch
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def build_generator(
    base_model: str, artifact_dir: Path | None, sampler: Sampler
) -> CosmosGenerator:
    """Load the pinned checkpoint plus artifact.

    # Raises
    [`GeneratorError`] when the pipeline cannot be loaded. There is no fallback
    that produces images without it.
    """
    checkpoint = resolve_base_checkpoint(base_model or BASE_MODEL_ID)
    return CosmosGenerator(checkpoint, artifact_dir, sampler)
