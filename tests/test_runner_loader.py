"""Weight loading across transformers versions, without torch.

The published image ships transformers 5.x, where the image-text-to-text auto
class has a different name than it had in 4.x and the dtype keyword was
renamed. The base model is a native VLM, so getting this wrong means the image
loads it as a text-only model or not at all. Both shapes are exercised here
with fake modules — the real backends need a GPU.
"""

from __future__ import annotations

import sys
import types

import pytest

from relearn_eval.runner import RunnerError, TransformersRunner


class FakeModel:
    def __init__(self, source: str, dtype_keyword: str) -> None:
        self.source = source
        self.dtype_keyword = dtype_keyword


def auto_class(name: str, *, keyword: str = "dtype", raises: type[Exception] | None = None):
    class Auto:
        @staticmethod
        def from_pretrained(weights, **kwargs):
            if raises is not None:
                raise raises("cannot load")
            if keyword not in kwargs:
                raise TypeError(f"unexpected keyword; this version wants {keyword}")
            return FakeModel(f"{name}:{weights}", keyword)

    return Auto


@pytest.fixture
def fake_transformers(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.manual_seed = lambda _seed: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    module = types.ModuleType("transformers")
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


def test_the_image_text_to_text_class_is_preferred(fake_transformers):
    fake_transformers.AutoModelForImageTextToText = auto_class("itt")
    fake_transformers.AutoModelForCausalLM = auto_class("lm")
    model = TransformersRunner._load_weights("/weights")
    assert model.source == "itt:/weights"


def test_the_older_class_name_still_works(fake_transformers):
    fake_transformers.AutoModelForVision2Seq = auto_class("v2s")
    fake_transformers.AutoModelForCausalLM = auto_class("lm")
    assert TransformersRunner._load_weights("/weights").source == "v2s:/weights"


def test_a_text_only_artifact_falls_through_to_the_language_model(fake_transformers):
    fake_transformers.AutoModelForImageTextToText = auto_class("itt", raises=ValueError)
    fake_transformers.AutoModelForCausalLM = auto_class("lm")
    assert TransformersRunner._load_weights("/weights").source == "lm:/weights"


def test_the_renamed_dtype_keyword_is_handled(fake_transformers):
    fake_transformers.AutoModelForCausalLM = auto_class("lm", keyword="torch_dtype")
    model = TransformersRunner._load_weights("/weights")
    assert model.dtype_keyword == "torch_dtype"


def test_weights_nothing_can_load_end_the_run(fake_transformers):
    fake_transformers.AutoModelForCausalLM = auto_class("lm", raises=OSError)
    with pytest.raises(RunnerError, match="no transformers auto class"):
        TransformersRunner._load_weights("/weights")
