"""Backend selection and weight resolution.

The backends themselves need a GPU, so what is testable here is everything
around them: which backend the pod will use, where the weights come from, how
an adapter is recognised, and that a misconfigured pod fails instead of
answering.
"""

from __future__ import annotations

import pytest

from relearn_eval.runner import (
    BACKENDS,
    DEFAULT_MAX_NEW_TOKENS,
    RunnerError,
    is_adapter,
    max_new_tokens,
    resolve_base_model,
    selected_backend,
)


def test_the_backends_are_the_real_ones_only():
    assert BACKENDS == ("auto", "vllm", "transformers")


def test_the_backend_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("RELEARN_EVAL_BACKEND", raising=False)
    assert selected_backend() == "auto"
    monkeypatch.setenv("RELEARN_EVAL_BACKEND", "VLLM")
    assert selected_backend() == "vllm"


def test_an_unknown_backend_is_refused(monkeypatch):
    # A stray value must not route generation anywhere unexpected.
    monkeypatch.setenv("RELEARN_EVAL_BACKEND", "sim")
    with pytest.raises(RunnerError, match="is not one of"):
        selected_backend()


def test_local_weights_win_over_the_pinned_id(tmp_path, monkeypatch):
    monkeypatch.setenv("RELEARN_BASE_MODEL_DIR", str(tmp_path))
    assert resolve_base_model("Qwen/Qwen3.8-27B") == str(tmp_path)

    monkeypatch.delenv("RELEARN_BASE_MODEL_DIR")
    assert resolve_base_model("Qwen/Qwen3.8-27B") == "Qwen/Qwen3.8-27B"


def test_a_pod_with_no_weights_at_all_cannot_score(tmp_path, monkeypatch):
    monkeypatch.setenv("RELEARN_BASE_MODEL_DIR", str(tmp_path / "missing"))
    with pytest.raises(RunnerError, match="is not a directory"):
        resolve_base_model("Qwen/Qwen3.8-27B")

    monkeypatch.delenv("RELEARN_BASE_MODEL_DIR")
    with pytest.raises(RunnerError, match="no base model"):
        resolve_base_model("  ")


def test_an_adapter_is_recognised_by_its_config(tmp_path):
    assert not is_adapter(None)
    assert not is_adapter(tmp_path)
    (tmp_path / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    assert is_adapter(tmp_path)


def test_the_decode_width_is_bounded(monkeypatch):
    monkeypatch.delenv("RELEARN_MAX_NEW_TOKENS", raising=False)
    assert max_new_tokens() == DEFAULT_MAX_NEW_TOKENS
    monkeypatch.setenv("RELEARN_MAX_NEW_TOKENS", "64")
    assert max_new_tokens() == 64
    for bad in ("0", "-1", "wide"):
        monkeypatch.setenv("RELEARN_MAX_NEW_TOKENS", bad)
        with pytest.raises(RunnerError):
            max_new_tokens()
