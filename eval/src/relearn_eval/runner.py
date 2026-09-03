"""Loading the scored model and generating answers.

The subject of a run is the miner artifact loaded **inside this image** on top
of the pinned base model. Nothing here reaches the teacher API for generation,
and there is no offline stand-in: when the runtime cannot load the model the
run fails, because a document produced without the model would be a made-up
number wearing a real run's identity.

Two backends, both driven greedily so a rerun of the same model on the same
items reproduces the same scores (the paired test compares per item, so drift
would show up as skill):

* `vllm` — preferred on a GPU pod; LoRA adapters ride along as a LoRA request.
* `transformers` — the portable path, with PEFT for adapters.

Heavy imports stay inside the backends so the contract layer can be tested and
linted on a machine with no CUDA.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .contract import ContractError

log = logging.getLogger(__name__)

#: Backends the image will select. Nothing else is accepted, so a stray
#: environment value can never route generation somewhere that fabricates text.
BACKENDS = ("auto", "vllm", "transformers")

#: Deterministic decode. Overridable only in width, never in temperature.
#:
#: A live run generates for every item in the holdout twice, plus the slices
#: the image owns, so the width is a direct multiplier on whether the run fits
#: the harvest's timeout. Individual slices ask for less (a multiple-choice
#: answer is one letter); this is the ceiling for open-ended answers.
DEFAULT_MAX_NEW_TOKENS = 256

#: Prompts per forward pass. Batching is the difference between one generation
#: at a time and a run that finishes: a 120-item holdout is 350-odd
#: generations once every slice is counted.
DEFAULT_BATCH_SIZE = 8


class RunnerError(ContractError):
    """The scored model could not be loaded or could not generate."""


@dataclass(frozen=True)
class Prompt:
    """One generation request.

    `max_new_tokens` is per prompt so a slice that needs a letter does not pay
    for the open-ended ceiling. `None` means the run's default.
    """

    key: str
    text: str
    image: bytes | None = None
    max_new_tokens: int | None = None

    def width(self, default: int) -> int:
        return self.max_new_tokens or default


@runtime_checkable
class ModelRunner(Protocol):
    """Generation over the loaded base model plus artifact."""

    def generate(self, prompts: Sequence[Prompt]) -> list[str]:
        """Answer every prompt, in order."""

    def close(self) -> None:
        """Release the model."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _positive_env(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RunnerError(f"{name} {raw!r} is not an integer") from exc
    if value <= 0:
        raise RunnerError(f"{name} must be positive")
    return value


def max_new_tokens() -> int:
    return _positive_env("RELEARN_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)


def batch_size() -> int:
    return _positive_env("RELEARN_BATCH_SIZE", DEFAULT_BATCH_SIZE)


def batches(prompts: Sequence[Prompt], size: int, default_width: int) -> list[list[Prompt]]:
    """Group prompts into batches that share a decode width and a modality.

    Order is preserved within and across batches, so the answers come back in
    the order the caller asked for them. Vision prompts go one at a time:
    processors differ in how (and whether) they batch images.
    """
    grouped: list[list[Prompt]] = []
    current: list[Prompt] = []
    current_width: int | None = None
    for prompt in prompts:
        width = prompt.width(default_width)
        if prompt.image is not None:
            if current:
                grouped.append(current)
                current, current_width = [], None
            grouped.append([prompt])
            continue
        if current and (current_width != width or len(current) >= size):
            grouped.append(current)
            current, current_width = [], None
        current.append(prompt)
        current_width = width
    if current:
        grouped.append(current)
    return grouped


def resolve_base_model(base_model: str) -> str:
    """Where to load the base weights from.

    A local directory (`RELEARN_BASE_MODEL_DIR`) wins, so a pod can be primed
    with the weights instead of pulling them per run. Otherwise the pinned id
    from the request is used, which needs a warm cache or network on the pod.
    """
    local = _env("RELEARN_BASE_MODEL_DIR")
    if local:
        path = Path(local)
        if not path.is_dir():
            raise RunnerError(f"RELEARN_BASE_MODEL_DIR {local} is not a directory")
        return str(path)
    if not base_model.strip():
        raise RunnerError("no base model: set RELEARN_BASE_MODEL_DIR or pin base_model")
    return base_model.strip()


def is_adapter(artifact_dir: Path | None) -> bool:
    """Whether the artifact is a LoRA/PEFT adapter rather than full weights."""
    return artifact_dir is not None and (artifact_dir / "adapter_config.json").is_file()


def _decode_image(image: bytes):  # pragma: no cover - needs pillow
    from io import BytesIO

    from PIL import Image

    return Image.open(BytesIO(image)).convert("RGB")


@dataclass
class VllmRunner:
    """vLLM backend."""

    base_model: str
    artifact_dir: Path | None
    _llm: object | None = field(default=None, repr=False)
    _sampling: object | None = field(default=None, repr=False)
    _lora: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - needs vllm + GPU
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RunnerError("vllm is not installed in this image") from exc

        adapter = is_adapter(self.artifact_dir)
        model = self.base_model if adapter or self.artifact_dir is None else str(self.artifact_dir)
        kwargs: dict[str, object] = {
            "model": model,
            "seed": 0,
            "trust_remote_code": False,
            "enable_lora": adapter,
        }
        tensor_parallel = _env("RELEARN_TENSOR_PARALLEL")
        if tensor_parallel:
            kwargs["tensor_parallel_size"] = int(tensor_parallel)
        self._llm = LLM(**kwargs)
        self._sampling = SamplingParams
        if adapter and self.artifact_dir is not None:
            from vllm.lora.request import LoRARequest

            self._lora = LoRARequest("miner-artifact", 1, str(self.artifact_dir))

    def generate(self, prompts: Sequence[Prompt]) -> list[str]:  # pragma: no cover
        if self._llm is None or self._sampling is None:
            raise RunnerError("vllm runner is not initialised")
        default_width = max_new_tokens()
        requests = []
        sampling = []
        for prompt in prompts:
            body: dict[str, object] = {"prompt": prompt.text}
            if prompt.image is not None:
                body["multi_modal_data"] = {"image": _decode_image(prompt.image)}
            requests.append(body)
            # Greedy, and only as wide as the slice needs. vLLM batches the
            # whole list itself, so no manual batching here.
            sampling.append(
                self._sampling(  # type: ignore[operator]
                    temperature=0.0, top_p=1.0, max_tokens=prompt.width(default_width)
                )
            )
        outputs = self._llm.generate(  # type: ignore[attr-defined]
            requests, sampling, lora_request=self._lora
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
    _model: object | None = field(default=None, repr=False)
    _processor: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

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
        except (OSError, ValueError, KeyError):
            self._processor = None
        tokenizer = AutoTokenizer.from_pretrained(weights)
        # Left padding so every row in a batch ends at the same position and
        # the new tokens can be sliced off by prompt length.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer

    #: Auto classes to try, in order. The base is a native VLM, so an
    #: image-text-to-text class comes first; the name changed across
    #: transformers majors (`AutoModelForVision2Seq` →
    #: `AutoModelForImageTextToText`) and a text-only artifact needs neither, so
    #: the image tries what the installed version actually has.
    AUTO_CLASSES = (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    )

    @classmethod
    def _load_weights(cls, weights: str):  # pragma: no cover - needs torch
        """Load the weights with the first auto class that can take them."""
        import torch
        import transformers

        def load(auto_class):
            try:
                return auto_class.from_pretrained(
                    weights, dtype=torch.bfloat16, device_map="auto"
                )
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

    def generate(self, prompts: Sequence[Prompt]) -> list[str]:
        if self._model is None or self._tokenizer is None:
            raise RunnerError("transformers runner is not initialised")
        default_width = max_new_tokens()
        answers: list[str] = []
        for batch in batches(prompts, batch_size(), default_width):
            answers.extend(self._generate_batch(batch, batch[0].width(default_width)))
        if len(answers) != len(prompts):
            raise RunnerError("transformers returned a different number of answers")
        return answers

    def _generate_batch(self, batch: list[Prompt], width: int) -> list[str]:
        import torch

        if batch[0].image is not None:
            if self._processor is None:
                raise RunnerError("a vision item was scored but the model has no processor")
            inputs = self._processor(
                text=batch[0].text,
                images=_decode_image(batch[0].image),
                return_tensors="pt",
            )
        else:
            inputs = self._tokenizer(  # type: ignore[misc]
                [prompt.text for prompt in batch],
                return_tensors="pt",
                padding=True,
            )
        device = getattr(self._model, "device", None)
        if device is not None:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        prompt_length = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            generated = self._model.generate(  # type: ignore[attr-defined]
                **inputs, do_sample=False, max_new_tokens=width
            )
        return [
            self._tokenizer.decode(  # type: ignore[union-attr]
                row[prompt_length:], skip_special_tokens=True
            ).strip()
            for row in generated
        ]

    def close(self) -> None:  # pragma: no cover
        self._model = None
        self._processor = None
        self._tokenizer = None


def selected_backend() -> str:
    """`RELEARN_EVAL_BACKEND`, validated."""
    backend = _env("RELEARN_EVAL_BACKEND", "auto").lower()
    if backend not in BACKENDS:
        raise RunnerError(f"RELEARN_EVAL_BACKEND {backend!r} is not one of {BACKENDS}")
    return backend


def build_runner(base_model: str, artifact_dir: Path | None) -> ModelRunner:
    """Load the base model plus artifact on the best available backend.

    # Raises
    [`RunnerError`] when no backend can load the model. There is no fallback
    that answers without one.
    """
    weights = resolve_base_model(base_model)
    backend = selected_backend()
    if backend == "vllm":
        return VllmRunner(weights, artifact_dir)
    if backend == "transformers":
        return TransformersRunner(weights, artifact_dir)
    try:
        return VllmRunner(weights, artifact_dir)
    except (RunnerError, ImportError, OSError, RuntimeError, ValueError) as exc:
        # `auto` means "whichever backend this pod can actually load". If
        # transformers cannot load it either, that error is the run's failure.
        log.warning(
            "vllm backend unavailable (%s: %s); falling back to transformers",
            type(exc).__name__,
            exc,
        )
        return TransformersRunner(weights, artifact_dir)
