"""Official-bench decontamination.

A pinned blocklist of well-known public benchmark stems. Two uses:

* the slices this image ships are checked against it, so the general-bench
  canary is MMLU / MMMU-*style* rather than the benches themselves — a canary
  lifted from a bench cannot detect regression on that bench;
* miner-side generators drop generated items that match.

Stems only, never official items: the list is a filter, not a copy of a bench.
"""

from __future__ import annotations

BLOCKLIST = (
    "hellaswag",
    "mmlu",
    "mmmu",
    "gsm8k",
    "arc-challenge",
    "winogrande",
    "truthfulqa",
    "gpqa",
    "aime",
    "humaneval",
    "bigbench",
    "big-bench",
    "drop benchmark",
    "agieval",
    "mathvista",
    "docvqa",
    "textvqa",
)


def is_contaminated(text: str) -> bool:
    """Whether `text` mentions an official bench by name."""
    lowered = text.lower()
    return any(stem in lowered for stem in BLOCKLIST)


def contaminated_stems(text: str) -> tuple[str, ...]:
    """Every blocklisted stem in `text`, for a diagnosable failure."""
    lowered = text.lower()
    return tuple(stem for stem in BLOCKLIST if stem in lowered)
