"""Loading the scored language model and generating from it.

The subject of a run is the miner artifact loaded **inside this image** on top
of the pinned base checkpoint. Nothing here reaches a judge API for generation,
and there is no offline stand-in: when the runtime cannot load the model the
run fails, because a document produced without the model would be a made-up
number wearing a real run's identity.

Two backends, both driven greedily so a rerun of the same model on the same
items reproduces the same scores. The paired test compares per item, so decode
drift would show up as skill.

* `vllm` — preferred on a GPU pod; LoRA adapters ride along as a LoRA request.
* `transformers` — the portable path, with PEFT for adapters.

Heavy imports stay inside the backends so the contract layer can be tested and
linted on a machine with no CUDA.

This is a sibling of `relearn_eval.runner`, not a refactor of it. That module
ships inside an image the control plane already pins by digest, so it is left
alone; the difference here is that every environment name is prefixed by the
challenge, which is what lets one host run more than one challenge.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .env import env, positive_int
from .errors import ContractError

log = logging.getLogger(__name__)

#: Backends the image will select. Nothing else is accepted, so a stray
#: environment value can never route generation somewhere that fabricates text.
BACKENDS = ("auto", "vllm", "transformers")

#: Deterministic decode. Overridable in width, never in temperature.
DEFAULT_MAX_NEW_TOKENS = 512


class RunnerError(ContractError):
    """The scored model could not be loaded, or could not generate."""


@dataclass(frozen=True)
class Prompt:
    """One generation request."""

    key: str
    text: str
    image: bytes | None = None


@runtime_checkable
class ModelRunner(Protocol):
    """Generation over the loaded base checkpoint plus artifact."""

    def generate(self, prompts: Sequence[Prompt]) -> list[str]:
        """Answer every prompt, in order."""

    def close(self) -> None:
        """Release the model."""


def max_new_tokens(prefix: str) -> int:
    return positive_int(f"{prefix}MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)


def resolve_base_model(base_model: str, *, prefix: str) -> str:
    """Where to load the base weights from.

    A local directory wins, so a pod can be primed with the weights instead of
    pulling them per run. Otherwise the pinned id from the request is used,
    which needs a warm cache or network on the pod.
    """
    local = env(f"{prefix}BASE_MODEL_DIR")
    if local:
        path = Path(local)
        if not path.is_dir():
            raise RunnerError(f"{prefix}BASE_MODEL_DIR {local} is not a directory")
        return str(path)
    if not base_model.strip():
        raise RunnerError(f"no base model: set {prefix}BASE_MODEL_DIR or pin base_model")
    return base_model.strip()


def is_adapter(artifact_dir: Path | None) -> bool:
    """Whether the artifact is a LoRA/PEFT adapter rather than full weights."""
    return artifact_dir is not None and (artifact_dir / "adapter_config.json").is_file()


def decode_image(body: bytes):  # pragma: no cover - needs pillow
    from io import BytesIO

    from PIL import Image

    return Image.open(BytesIO(body)).convert("RGB")


@dataclass
class VllmRunner:
    """vLLM backend."""

    base_model: str
    artifact_dir: Path | None
    prefix: str
    _llm: object | None = field(default=None, repr=False)
    _sampling: object | None = field(default=None, repr=False)
    _lora: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - needs vllm + GPU
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RunnerError("vllm is not installed in this image") from exc

        adapter = is_adapter(self.artifact_dir)
        model = (
            self.base_model
            if adapter or self.artifact_dir is None
            else str(self.artifact_dir)
        )
        kwargs: dict[str, object] = {
            "model": model,
            "seed": 0,
            "trust_remote_code": False,
            "enable_lora": adapter,
        }
        tensor_parallel = env(f"{self.prefix}TENSOR_PARALLEL")
        if tensor_parallel:
            kwargs["tensor_parallel_size"] = int(tensor_parallel)
        self._llm = LLM(**kwargs)
        self._sampling = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens(self.prefix)
        )
        if adapter and self.artifact_dir is not None:
            from vllm.lora.request import LoRARequest

            self._lora = LoRARequest("miner-artifact", 1, str(self.artifact_dir))

    def generate(self, prompts: Sequence[Prompt]) -> list[str]:  # pragma: no cover
        if self._llm is None:
            raise RunnerError("vllm runner is not initialised")
        requests = []
        for prompt in prompts:
            body: dict[str, object] = {"prompt": prompt.text}
            if prompt.image is not None:
                body["multi_modal_data"] = {"image": decode_image(prompt.image)}
            requests.append(body)
        outputs = self._llm.generate(  # type: ignore[attr-defined]
            requests, self._sampling, lora_request=self._lora
        )
        answers = [output.outputs[0].text.strip() for output in outputs]
        if len(answers) != len(prompts):
            raise RunnerError("vllm returned a different number of answers")
        return answers

    def close(self) -> None:  # pragma: no cover
        self._llm = None


@dataclass
class TransformersRunner:
    """transformers backend, with PEFT for adapters."""

    base_model: str
    artifact_dir: Path | None
    prefix: str
    _model: object | None = field(default=None, repr=False)
    _processor: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    #: Auto classes to try, in order. The base is a native VLM, so an
    #: image-text-to-text class comes first; the name changed across
    #: transformers majors (`AutoModelForVision2Seq` →
    #: `AutoModelForImageTextToText`), so the image tries what the installed
    #: version actually has rather than a name pinned at authoring time.
    AUTO_CLASSES = (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    )

    def __post_init__(self) -> None:  # pragma: no cover - needs torch
        try:
            import torch
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RunnerError("torch / transformers are not installed in this image") from exc

        torch.manual_seed(0)
        adapter = is_adapter(self.artifact_dir)
        weights = (
            self.base_model
            if adapter or self.artifact_dir is None
            else str(self.artifact_dir)
        )
        self._model = self._load_weights(weights)
        if adapter and self.artifact_dir is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RunnerError("artifact is a LoRA adapter but peft is missing") from exc
            self._model = PeftModel.from_pretrained(self._model, str(self.artifact_dir))
        self._model.eval()  # type: ignore[attr-defined]
        try:
            self._processor = AutoProcessor.from_pretrained(weights)
        except (OSError, ValueError):
            self._processor = None
        self._tokenizer = AutoTokenizer.from_pretrained(weights)

    @classmethod
    def _load_weights(cls, weights: str):  # pragma: no cover - needs torch
        """Load the weights with the first auto class that can take them."""
        import torch
        import transformers

        def load(auto_class):
            try:
                return auto_class.from_pretrained(weights, dtype=torch.bfloat16, device_map="auto")
            except TypeError:
                # The dtype keyword was renamed (`torch_dtype` → `dtype`).
                return auto_class.from_pretrained(
                    weights, torch_dtype=torch.bfloat16, device_map="auto"
                )

        last: Exception | None = None
        for name in cls.AUTO_CLASSES:
            auto_class = getattr(transformers, name, None)
            if auto_class is None:
                continue
            try:
                return load(auto_class)
            except (OSError, ValueError, KeyError) as exc:
                log.info("%s could not load these weights: %s", name, type(exc).__name__)
                last = exc
        raise RunnerError(f"no transformers auto class could load {weights}") from last

    def generate(self, prompts: Sequence[Prompt]) -> list[str]:  # pragma: no cover
        import torch

        if self._model is None or self._tokenizer is None:
            raise RunnerError("transformers runner is not initialised")
        answers: list[str] = []
        width = max_new_tokens(self.prefix)
        for prompt in prompts:
            if prompt.image is not None:
                if self._processor is None:
                    raise RunnerError("an image item was scored but the model has no processor")
                inputs = self._processor(
                    text=prompt.text, images=decode_image(prompt.image), return_tensors="pt"
                )
            else:
                inputs = self._tokenizer(prompt.text, return_tensors="pt")
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            with torch.no_grad():
                generated = self._model.generate(  # type: ignore[attr-defined]
                    **inputs, do_sample=False, max_new_tokens=width
                )
            trimmed = generated[0][inputs["input_ids"].shape[-1] :]
            answers.append(self._tokenizer.decode(trimmed, skip_special_tokens=True).strip())
        return answers

    def close(self) -> None:  # pragma: no cover
        self._model = None
        self._processor = None
        self._tokenizer = None


def selected_backend(prefix: str) -> str:
    """The configured backend, validated."""
    backend = env(f"{prefix}EVAL_BACKEND", "auto").lower()
    if backend not in BACKENDS:
        raise RunnerError(f"{prefix}EVAL_BACKEND {backend!r} is not one of {BACKENDS}")
    return backend


def build_runner(base_model: str, artifact_dir: Path | None, *, prefix: str) -> ModelRunner:
    """Load the base checkpoint plus artifact on the best available backend.

    # Raises
    [`RunnerError`] when no backend can load the model. There is no fallback
    that answers without one.
    """
    weights = resolve_base_model(base_model, prefix=prefix)
    backend = selected_backend(prefix)
    if backend == "vllm":
        return VllmRunner(weights, artifact_dir, prefix)
    if backend == "transformers":
        return TransformersRunner(weights, artifact_dir, prefix)
    try:
        return VllmRunner(weights, artifact_dir, prefix)
    except (RunnerError, ImportError, OSError, RuntimeError, ValueError) as exc:
        # `auto` means "whichever backend this pod can actually load". If
        # transformers cannot load it either, that error is the run's failure.
        log.warning(
            "vllm backend unavailable (%s: %s); falling back to transformers",
            type(exc).__name__,
            exc,
        )
        return TransformersRunner(weights, artifact_dir, prefix)
