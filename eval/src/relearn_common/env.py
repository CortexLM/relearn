"""Reading pod environment.

Every eval image is configured entirely by pod environment the operator sets.
Nothing below is baked into an image, and no value read here is a secret this
repository knows: there is no endpoint, hostname, key, or holdout record in
git.

Each challenge owns an environment prefix (`RELEARN_`, `RELEARN_IMAGE_`,
`RELEARN_AGENT_`) so one host can run more than one challenge without a
variable set for one silently steering another.
"""

from __future__ import annotations

import os

from .errors import ContractError

#: Values that mean "yes" in a pod environment flag.
_TRUE = ("1", "true", "yes", "on")


def env(name: str, default: str = "") -> str:
    """One environment value, trimmed."""
    return os.environ.get(name, default).strip()


def env_first(*names: str, default: str = "") -> str:
    """The first of `names` that is set and non-empty.

    Lets a challenge-specific variable (`RELEARN_AGENT_TEACHER_API_URL`) fall
    back to a shared one (`RELEARN_TEACHER_API_URL`) so an operator running two
    challenges against one judge deployment configures it once. Falling back to
    nothing is still nothing: an unset judge is a refusal, never a default.
    """
    for name in names:
        value = env(name)
        if value:
            return value
    return default


def flag(name: str, *, default: bool = False) -> bool:
    """A boolean pod flag."""
    raw = env(name).lower()
    if not raw:
        return default
    return raw in _TRUE


def positive_int(name: str, default: int) -> int:
    """A positive integer pod setting, or `default` when unset."""
    raw = env(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ContractError(f"{name} {raw!r} is not an integer") from exc
    if value <= 0:
        raise ContractError(f"{name} must be positive")
    return value
