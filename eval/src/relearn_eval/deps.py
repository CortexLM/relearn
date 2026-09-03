"""What the scoring runtime needs, and whether this image has it.

The live failure this exists for: a pod booted `sha256:cbc4bbb8`, preflight
passed, the run reached the model load, and the base — a native VLM — died on

    ImportError: Qwen3VLVideoProcessor requires Torchvision

after vLLM had already been skipped, because it was not installed either. The
image was built without the `vllm` extra and without `torchvision`, and nothing
between the build and the rented GPU said so. The run exited 1 with no
`RELEARN_EVAL_OK`, and the champion was not recorded.

So the dependencies a run needs are named here, in one place, probed by import,
and checked in three:

* the build (`eval/Dockerfile.scoring` imports them, and fails the layer);
* `relearn-eval selftest`, which CI runs against the *pushed digest* through
  the harvest's own PATH, before a pin is reported;
* `preflight`, so a pod that is missing one says which in seconds rather than
  minutes into a model load.

Importing is the only honest probe. Distribution metadata says a wheel was
unpacked, not that the extension module it ships can load against the CUDA
libraries on this pod, and it is the import that fails live.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass

from .contract import ContractError


class DependencyError(ContractError):
    """A dependency the scoring run needs is not importable in this image."""


@dataclass(frozen=True)
class Dependency:
    """One import the scorer needs, and what breaks without it."""

    module: str
    #: The distribution to name in the message, when it differs from `module`.
    distribution: str = ""
    #: Why the run needs it, in the failure line an operator reads.
    why: str = ""

    @property
    def package(self) -> str:
        return self.distribution or self.module


#: Needed by every backend: without these the image cannot load a model at all.
CORE = (
    Dependency("torch", why="the model runtime"),
    Dependency("transformers", why="the portable backend and the processors"),
)

#: Needed to load a native VLM base. `Qwen3VLVideoProcessor` reaches for
#: torchvision while the processor is being built, so this is a hard
#: requirement of loading the pinned base for a holdout with vision items —
#: not an optional extra that only the vision slices pay for.
VISION = (
    Dependency("torchvision", why="Qwen3VLVideoProcessor, on the native VLM base"),
    Dependency("PIL", distribution="pillow", why="decoding holdout images"),
)

#: The GPU backend. `auto` falls back to transformers without it, so it is only
#: fatal to a run when the operator asked for vLLM by name — but an image that
#: ships without it is not the image the control plane means to pin, which is
#: what [`SCORING`] is for.
GPU = (Dependency("vllm", why="the GPU backend the pod is rented for"),)

#: Everything a published *scoring* image must be able to import.
SCORING = CORE + VISION + GPU


def version(dependency: Dependency) -> str | None:
    """Import `dependency` and return its version, or `None` when it is absent.

    Any exception from the import counts as absent: a torchvision that raises
    against a mismatched torch ABI is exactly as unusable as one that was never
    installed, and it ends the run the same way.
    """
    try:
        module = importlib.import_module(dependency.module)
    except Exception:
        return None
    installed = getattr(module, "__version__", "")
    if isinstance(installed, str) and installed:
        return installed
    try:
        return importlib.metadata.version(dependency.package)
    except importlib.metadata.PackageNotFoundError:
        return "present"


def missing(dependencies: tuple[Dependency, ...]) -> tuple[Dependency, ...]:
    """The subset of `dependencies` this image cannot import."""
    return tuple(item for item in dependencies if version(item) is None)


def report(dependencies: tuple[Dependency, ...] = SCORING) -> list[str]:
    """One `name version` — or `name MISSING` — line per dependency, in order."""
    lines = []
    for item in dependencies:
        found = version(item)
        lines.append(f"{item.module} {found}" if found else f"{item.module} MISSING")
    return lines


def describe(dependencies: tuple[Dependency, ...]) -> str:
    """`torchvision (Qwen3VLVideoProcessor…), vllm (…)`, for a failure line."""
    return ", ".join(
        f"{item.package} ({item.why})" if item.why else item.package for item in dependencies
    )


def require(dependencies: tuple[Dependency, ...], context: str) -> None:
    """Refuse to continue when any of `dependencies` cannot be imported.

    # Raises
    [`DependencyError`] naming every missing package and what needs it, so the
    8 KiB of log the harvest keeps carries the whole diagnosis.
    """
    absent = missing(dependencies)
    if absent:
        raise DependencyError(f"{context}: this image cannot import {describe(absent)}")
