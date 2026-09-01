"""Preflight: name the missing dependency in seconds, not minutes.

The live failure was a pod that booted, ran the scorer, and returned no
`RELEARN_EVAL_OK`. Each of these cases used to be discovered only after the
model had been loaded — or not at all, because the run was killed first.
"""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from relearn_eval.preflight import (
    PreflightError,
    check_artifact,
    check_base_weights,
    check_teacher,
    preflight,
)

from .conftest import holdout_items, make_request


@pytest.fixture(autouse=True)
def bare_pod(monkeypatch):
    for name in (
        "RELEARN_TEACHER_API_URL",
        "RELEARN_TEACHER_API_KEY",
        "RELEARN_TEACHER_MODEL",
        "RELEARN_BASE_MODEL_DIR",
        "RELEARN_ALLOW_MODEL_DOWNLOAD",
        "RELEARN_ARTIFACT_DIR",
        "RELEARN_ARTIFACT_URL_TEMPLATE",
    ):
        monkeypatch.delenv(name, raising=False)


def model_dir(root, name: str = "weights"):
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    return path


def test_no_judge_is_reported_as_no_judge():
    with pytest.raises(PreflightError, match="no judge: RELEARN_TEACHER_API_URL"):
        check_teacher(make_request())


def test_a_judge_that_does_not_answer_is_reported_before_the_model_loads(monkeypatch):
    monkeypatch.setenv("RELEARN_TEACHER_API_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("RELEARN_JUDGE_ATTEMPTS", "1")
    request = make_request()
    with pytest.raises(PreflightError, match=r"no judge: glm-5.3 did not answer"):
        check_teacher(request)


def test_primed_local_weights_are_accepted(tmp_path, monkeypatch):
    weights = model_dir(tmp_path)
    monkeypatch.setenv("RELEARN_BASE_MODEL_DIR", str(weights))
    assert check_base_weights(make_request()) == str(weights)


def test_a_mount_that_is_not_a_model_directory_is_reported(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("RELEARN_BASE_MODEL_DIR", str(empty))
    with pytest.raises(PreflightError, match=r"no model:.*has no config.json"):
        check_base_weights(make_request())


def test_a_pod_that_would_have_to_download_the_base_refuses(monkeypatch):
    """Pulling tens of gibibytes inside the run timeout is how a pod returns nothing."""
    monkeypatch.setattr("relearn_eval.preflight._cached_locally", lambda _repo: False)
    with pytest.raises(PreflightError, match=r"no model:.*is not on this pod"):
        check_base_weights(make_request())


def test_the_download_can_be_opted_into(monkeypatch):
    monkeypatch.setattr("relearn_eval.preflight._cached_locally", lambda _repo: False)
    monkeypatch.setenv("RELEARN_ALLOW_MODEL_DOWNLOAD", "1")
    assert check_base_weights(make_request()) == "Qwen/Qwen3.8-27B"


def test_a_warm_cache_counts_as_primed(monkeypatch):
    monkeypatch.setattr("relearn_eval.preflight._cached_locally", lambda _repo: True)
    assert check_base_weights(make_request()) == "Qwen/Qwen3.8-27B"


def test_the_boot_baseline_needs_no_artifact(tmp_path):
    request = make_request(artifact_digest="base-relearn-champion")
    assert check_artifact(request, tmp_path) is None


def test_an_artifact_that_is_not_what_was_asked_for_is_reported(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo("adapter/adapter_config.json")
        body = b'{"r": 8}'
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    payload = archive.getvalue()
    wanted = hashlib.sha256(payload).hexdigest()
    store = tmp_path / "store"
    store.mkdir()
    (store / f"{wanted}.tar").write_bytes(payload + b"tampered")
    monkeypatch.setenv("RELEARN_ARTIFACT_DIR", str(store))

    request = make_request(artifact_digest=wanted)
    with pytest.raises(PreflightError, match=r"artifact:.*hashes to"):
        check_artifact(request, tmp_path / "work")


def test_a_ready_pod_reports_what_it_will_use(tmp_path, monkeypatch):
    weights = model_dir(tmp_path)
    monkeypatch.setenv("RELEARN_BASE_MODEL_DIR", str(weights))
    monkeypatch.setattr(
        "relearn_eval.preflight.check_teacher",
        lambda _request: type("T", (), {"model": "glm-5.3"})(),
    )
    request = make_request(holdout_items(4), artifact_digest="base-relearn-champion")
    ready = preflight(request, tmp_path / "work")
    assert ready.base_model == str(weights)
    assert ready.is_base_baseline
