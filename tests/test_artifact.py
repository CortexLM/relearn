"""Resolving the artifact by digest, and refusing anything else."""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from relearn_eval.artifact import ArtifactError, resolve_artifact, sha256_tree
from relearn_eval.contract import BASE_CHAMPION_ARTIFACT


def adapter_tar(body: bytes = b'{"r": 8}') -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        info = tarfile.TarInfo("adapter/adapter_config.json")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    return out.getvalue()


def test_the_boot_baseline_run_has_no_artifact_to_fetch(tmp_path):
    assert resolve_artifact(BASE_CHAMPION_ARTIFACT, tmp_path) is None


def test_an_artifact_is_unpacked_after_its_digest_is_checked(tmp_path, monkeypatch):
    archive = adapter_tar()
    digest = hashlib.sha256(archive).hexdigest()
    store = tmp_path / "store"
    store.mkdir()
    (store / f"{digest}.tar").write_bytes(archive)
    monkeypatch.setenv("RELEARN_ARTIFACT_DIR", str(store))

    resolved = resolve_artifact(digest, tmp_path / "work")
    assert resolved is not None
    assert (resolved / "adapter_config.json").is_file()


def test_a_store_serving_other_bytes_is_refused(tmp_path, monkeypatch):
    archive = adapter_tar()
    digest = hashlib.sha256(archive).hexdigest()
    store = tmp_path / "store"
    store.mkdir()
    (store / f"{digest}.tar").write_bytes(adapter_tar(b'{"r": 16}'))
    monkeypatch.setenv("RELEARN_ARTIFACT_DIR", str(store))

    with pytest.raises(ArtifactError, match="hashes to"):
        resolve_artifact(digest, tmp_path / "work")


def test_an_unpacked_artifact_is_checked_against_its_tree_digest(tmp_path, monkeypatch):
    store = tmp_path / "store"
    weights = store / "placeholder"
    weights.mkdir(parents=True)
    (weights / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    digest = sha256_tree(weights)
    weights.rename(store / digest)
    monkeypatch.setenv("RELEARN_ARTIFACT_DIR", str(store))

    assert resolve_artifact(digest, tmp_path / "work") == store / digest

    # The same tree filed under someone else's digest is not that artifact.
    mislabelled = "cd" * 32
    (store / digest).rename(store / mislabelled)
    with pytest.raises(ArtifactError, match="hashes to"):
        resolve_artifact(mislabelled, tmp_path / "work")


def test_a_pod_with_no_store_configured_cannot_score(tmp_path, monkeypatch):
    monkeypatch.delenv("RELEARN_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("RELEARN_ARTIFACT_URL_TEMPLATE", raising=False)
    with pytest.raises(ArtifactError, match="no artifact source"):
        resolve_artifact("ab" * 32, tmp_path / "work")


def test_a_digest_that_is_not_a_digest_is_refused(tmp_path):
    with pytest.raises(ArtifactError, match="not a sha256"):
        resolve_artifact("latest", tmp_path)


def test_an_artifact_url_must_be_a_supported_scheme(tmp_path, monkeypatch):
    monkeypatch.delenv("RELEARN_ARTIFACT_DIR", raising=False)
    with pytest.raises(ArtifactError, match="http"):
        resolve_artifact("ab" * 32, tmp_path / "work", "ftp://store.invalid/artifact.tar")


def test_an_archive_that_escapes_its_directory_is_refused(tmp_path, monkeypatch):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        info = tarfile.TarInfo("../escaped.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    archive = out.getvalue()
    digest = hashlib.sha256(archive).hexdigest()
    store = tmp_path / "store"
    store.mkdir()
    (store / f"{digest}.tar").write_bytes(archive)
    monkeypatch.setenv("RELEARN_ARTIFACT_DIR", str(store))

    with pytest.raises(ArtifactError, match="escapes its directory"):
        resolve_artifact(digest, tmp_path / "work")


def test_the_tree_digest_covers_names_and_contents(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        root.mkdir()
        (root / "a.json").write_text("{}", encoding="utf-8")
    assert sha256_tree(left) == sha256_tree(right)
    (right / "b.json").write_text("{}", encoding="utf-8")
    assert sha256_tree(left) != sha256_tree(right)
