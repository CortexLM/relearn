"""Resolving the submitted artifact, by digest.

The request names an artifact by its `sha256`, not by a location: the pod
resolves it from an operator-configured store and **verifies the bytes hash to
the digest that was asked for** before anything is loaded. A store that serves
different bytes fails the run, so the identity in the metrics document always
describes what actually ran.

There is no baked host here. The operator sets `RELEARN_ARTIFACT_DIR` (a
content-addressed directory, checked first) and/or
`RELEARN_ARTIFACT_URL_TEMPLATE` (`https://…/{digest}.tar`). A request that
carries its own `artifact_uri` is honoured, and still digest-checked.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .contract import BASE_CHAMPION_ARTIFACT, ContractError

log = logging.getLogger(__name__)

#: Read size for hashing and downloads.
_CHUNK = 1024 * 1024

#: Refuse an artifact larger than this (bytes). A pod that fills its disk is a
#: failed run either way; failing early leaves the log readable.
DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024


class ArtifactError(ContractError):
    """The artifact could not be resolved, verified, or unpacked."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def max_artifact_bytes() -> int:
    raw = _env("RELEARN_MAX_ARTIFACT_BYTES")
    if not raw:
        return DEFAULT_MAX_ARTIFACT_BYTES
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ArtifactError(f"RELEARN_MAX_ARTIFACT_BYTES {raw!r} is not an integer") from exc
    if limit <= 0:
        raise ArtifactError("RELEARN_MAX_ARTIFACT_BYTES must be positive")
    return limit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Digest of a directory: sorted relative paths and their contents.

    Used when the store holds an already-unpacked artifact. Length-prefixed
    names so two different layouts cannot collide.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        name = str(path.relative_to(root)).encode("utf-8")
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _looks_like_digest(value: str) -> bool:
    trimmed = value.strip().lower()
    return len(trimmed) == 64 and all(char in "0123456789abcdef" for char in trimmed)


def _safe_extract_tar(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ArtifactError(f"artifact archive escapes its directory: {member.name}")
            if member.issym() or member.islnk():
                raise ArtifactError("artifact archive carries a link")
        tar.extractall(target, filter="data")


def _safe_extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ArtifactError(f"artifact archive escapes its directory: {name}")
        zipped.extractall(target)  # noqa: S202 - members checked above


def unpack(archive: Path, target: Path) -> Path:
    """Unpack an artifact archive under `target`, or return it as weights.

    A single unpacked directory becomes the model directory, so a tarball with
    one top-level folder behaves like one without.
    """
    target.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        _safe_extract_tar(archive, target)
    elif zipfile.is_zipfile(archive):
        _safe_extract_zip(archive, target)
    else:
        raise ArtifactError("artifact is neither a tar nor a zip archive")
    entries = [path for path in target.iterdir() if not path.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return target


def _download(url: str, target: Path, limit: int) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not url.lower().startswith(("https://", "http://", "file://")):
        raise ArtifactError("artifact url must be http(s) or file")
    log.info("fetching artifact from the configured store")
    written = 0
    with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310 - scheme checked
        with target.open("wb") as handle:
            while chunk := response.read(_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise ArtifactError("artifact exceeds RELEARN_MAX_ARTIFACT_BYTES")
                handle.write(chunk)
    return target


def _store_candidates(digest: str) -> list[Path]:
    directory = _env("RELEARN_ARTIFACT_DIR")
    if not directory:
        return []
    root = Path(directory)
    return [
        root / digest,
        root / f"{digest}.tar",
        root / f"{digest}.tar.gz",
        root / f"{digest}.zip",
        root / digest[:2] / digest,
    ]


def resolve_artifact(artifact_digest: str, workdir: Path, artifact_uri: str = "") -> Path | None:
    """Return the directory holding the artifact's weights.

    `None` means "no artifact": the control plane asked for
    [`BASE_CHAMPION_ARTIFACT`], the un-post-trained base model, when it records
    the champion baseline.

    # Raises
    [`ArtifactError`] when the artifact cannot be found, does not hash to the
    requested digest, or cannot be unpacked.
    """
    digest = artifact_digest.strip()
    if digest == BASE_CHAMPION_ARTIFACT:
        log.info("scoring the base model: no artifact to fetch")
        return None
    if not _looks_like_digest(digest):
        raise ArtifactError("artifact_digest is not a sha256 hex digest")

    digest = digest.lower()
    limit = max_artifact_bytes()
    for candidate in _store_candidates(digest):
        if candidate.is_dir():
            measured = sha256_tree(candidate)
            if measured != digest:
                raise ArtifactError(
                    f"artifact directory hashes to {measured}, request asked for {digest}"
                )
            log.info("artifact resolved from the local store")
            return candidate
        if candidate.is_file():
            measured = sha256_file(candidate)
            if measured != digest:
                raise ArtifactError(
                    f"artifact file hashes to {measured}, request asked for {digest}"
                )
            return unpack(candidate, workdir / "artifact")

    template = _env("RELEARN_ARTIFACT_URL_TEMPLATE")
    url = artifact_uri.strip()
    if not url and template:
        url = template.replace("{digest}", digest)
    if not url:
        raise ArtifactError(
            "no artifact source: set RELEARN_ARTIFACT_DIR or RELEARN_ARTIFACT_URL_TEMPLATE"
        )

    downloaded = _download(url, workdir / "artifact.bin", limit)
    measured = sha256_file(downloaded)
    if measured != digest:
        raise ArtifactError(
            f"fetched artifact hashes to {measured}, request asked for {digest}"
        )
    unpacked = unpack(downloaded, workdir / "artifact")
    downloaded.unlink(missing_ok=True)
    return unpacked


def scrub(path: Path) -> None:
    """Best-effort removal of run state from the pod."""
    shutil.rmtree(path, ignore_errors=True)
