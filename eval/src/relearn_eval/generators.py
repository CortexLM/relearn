"""Disjoint train / eval synthetic factory (miner-side helper).

Not part of the scoring path: the scored items come from the request (holdout)
and the shipped slices. This is the generator miners use to build their own
training and self-eval sets without letting the two overlap, and without
picking up official bench material.
"""

from __future__ import annotations

import hashlib

from .decontam import is_contaminated

KINDS = ("train", "eval")


def generate(kind: str, seed: str, n: int) -> list[str]:
    """`n` deterministic, decontaminated prompts for one split."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if n < 0:
        raise ValueError("n must not be negative")
    out: list[str] = []
    index = 0
    while len(out) < n:
        digest = hashlib.sha256(f"{kind}:{seed}:{index}".encode()).hexdigest()
        prompt = f"relearn-{kind}-{digest[:16]}"
        if not is_contaminated(prompt):
            out.append(prompt)
        index += 1
    return out


def disjoint(seed: str, n: int) -> tuple[list[str], list[str]]:
    """A train split and an eval split that provably do not overlap."""
    train = generate("train", f"{seed}-train", n)
    evaluation = generate("eval", f"{seed}-eval", n)
    if set(train) & set(evaluation):
        raise RuntimeError("train/eval generators collided")
    return train, evaluation
