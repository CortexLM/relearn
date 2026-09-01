"""Check the pod can score *before* it spends the run doing it.

The live failure this exists for: the pod booted, the harvest ran the scorer,
and the run produced no `RELEARN_EVAL_OK`. Everything the run depends on — a
reachable judge, base weights on the pod, an artifact that matches its digest —
is knowable in seconds, but was previously discovered minutes in, or not at
all, because the run was killed by the harvest's timeout first while pulling
tens of gibibytes of weights.

So each dependency is probed up front and each failure names itself in one
line: no judge, no model, digest mismatch. A pod that cannot score says so
immediately instead of timing out.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .artifact import ArtifactError, resolve_artifact
from .contract import ContractError
from .request import HarvestRequest
from .runner import RunnerError, resolve_base_model
from .teacher import HttpTeacher, TeacherError, build_teacher

log = logging.getLogger(__name__)

#: Files that make a directory a loadable model rather than an empty mount.
_WEIGHT_MARKERS = ("config.json", "params.json")


class PreflightError(ContractError):
    """The pod cannot score this run. The message says which dependency."""


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _cached_locally(repo_id: str) -> bool:
    """Whether the base weights are already in this pod's Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    try:
        hit = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
    except (OSError, ValueError):
        return False
    return isinstance(hit, str)


def check_teacher(request: HarvestRequest) -> HttpTeacher:
    """Build the judge and prove it answers.

    A probe costs one judge call and rules out an unset endpoint, a rejected
    credential, an unroutable host, and a reply this image cannot read — each of
    which would otherwise surface as a failed run after the model was loaded.

    # Raises
    [`PreflightError`] naming the judge as the missing dependency.
    """
    try:
        teacher = build_teacher(request.teacher_model)
    except TeacherError as exc:
        raise PreflightError(f"no judge: {exc}") from exc
    try:
        teacher.judge(
            "Reply with a score of 1.0. This is a connectivity probe.",
            "1.0",
        )
    except TeacherError as exc:
        raise PreflightError(f"no judge: {teacher.model} did not answer: {exc}") from exc
    log.info("judge reachable: model=%s", teacher.model)
    return teacher


def check_base_weights(request: HarvestRequest) -> str:
    """Resolve the base weights and refuse to start a run that must download them.

    Pulling the base model inside the harvest's run timeout is how a pod spends
    its whole budget and returns nothing. Prime the pod
    (`RELEARN_BASE_MODEL_DIR`, or a warm Hugging Face cache) or opt in
    explicitly with `RELEARN_ALLOW_MODEL_DOWNLOAD=1`.

    # Raises
    [`PreflightError`] naming the model as the missing dependency.
    """
    try:
        weights = resolve_base_model(request.base_model)
    except RunnerError as exc:
        raise PreflightError(f"no model: {exc}") from exc

    local = Path(weights)
    if local.is_dir():
        if not any((local / marker).is_file() for marker in _WEIGHT_MARKERS):
            raise PreflightError(
                f"no model: {weights} has no {' or '.join(_WEIGHT_MARKERS)}; "
                "RELEARN_BASE_MODEL_DIR is not a model directory"
            )
        log.info("base weights: local directory")
        return weights

    if _cached_locally(weights):
        log.info("base weights: warm Hugging Face cache")
        return weights
    if _truthy("RELEARN_ALLOW_MODEL_DOWNLOAD"):
        log.warning(
            "base weights %s are not on this pod; downloading inside the run "
            "timeout because RELEARN_ALLOW_MODEL_DOWNLOAD is set",
            weights,
        )
        return weights
    raise PreflightError(
        f"no model: {weights} is not on this pod. Point "
        "RELEARN_BASE_MODEL_DIR at primed weights, warm the Hugging Face "
        "cache, or set RELEARN_ALLOW_MODEL_DOWNLOAD=1 to pull them inside the "
        "run timeout"
    )


def check_artifact(request: HarvestRequest, workdir: Path) -> Path | None:
    """Resolve and digest-verify the artifact, or `None` for the base baseline.

    # Raises
    [`PreflightError`] naming the artifact and, on a mismatch, both digests.
    """
    try:
        return resolve_artifact(request.artifact_digest, workdir, request.artifact_uri)
    except ArtifactError as exc:
        raise PreflightError(f"artifact: {exc}") from exc


@dataclass
class Ready:
    """Everything the run needs, proven present."""

    teacher: HttpTeacher
    base_model: str
    artifact_dir: Path | None

    @property
    def is_base_baseline(self) -> bool:
        return self.artifact_dir is None


def preflight(request: HarvestRequest, workdir: Path) -> Ready:
    """Prove the pod can score this run, in seconds rather than minutes."""
    teacher = check_teacher(request)
    base_model = check_base_weights(request)
    artifact_dir = check_artifact(request, workdir)
    log.info(
        "preflight ok: judge=%s weights=%s artifact=%s",
        teacher.model,
        base_model,
        "base model (no artifact)" if artifact_dir is None else "verified",
    )
    return Ready(teacher=teacher, base_model=base_model, artifact_dir=artifact_dir)
