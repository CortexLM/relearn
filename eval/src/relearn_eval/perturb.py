"""The pinned holdout perturbation.

The perturbation gate asks whether a model's holdout answers survive a
surface-level rewrite of the question. The rewrite must therefore be
meaning-preserving, deterministic, and independent of the model and the
artifact: champion and challenger are perturbed identically, so the drop
between `holdout` and `perturbed` is a property of the model rather than of the
draw.

Pinned here rather than configurable: a host that perturbed differently from
the host that measured the champion would compare two different tests.
"""

from __future__ import annotations

import re

#: Bumped only with `challenge_scoring_version`: changing the rewrite changes
#: every recorded champion baseline.
PERTURBATION_VERSION = 1

_WHITESPACE = re.compile(r"\s+")

_FRAME = (
    "Read the request below carefully, then answer it completely.\n"
    "Request: {body}\n"
    "Answer the request as stated, without restating it."
)


def perturb_prompt(prompt: str) -> str:
    """Rewrite one prompt for the perturbed pass."""
    body = _WHITESPACE.sub(" ", prompt.strip())
    return _FRAME.format(body=body)
