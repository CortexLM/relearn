"""Fixtures for the image contract tests.

The runner and the judge are the two things this image cannot carry an offline
version of, so the test doubles for them live **here**, in the test tree, and
are never installed into the image. That is the point of the arrangement: a
fixture that could produce scores inside the package would be the sim harness
this challenge refuses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from relearn_eval import HarvestRequest, HoldoutItem
from relearn_eval.commitment import holdout_commitment
from relearn_eval.contract import METRICS_SCHEMA_VERSION
from relearn_eval.runner import Prompt
from relearn_eval.scoring import Slices

IMAGE_DIGEST = f"sha256:{'ab' * 32}"


def holdout_items(n: int = 120, *, vision: bool = False) -> tuple[HoldoutItem, ...]:
    """`n` holdout records, optionally covering every vision family."""
    families = ("captioning", "vqa", "ocr", "spatial")
    items: list[HoldoutItem] = []
    for index in range(1, n + 1):
        task = "text"
        image_hash = ""
        if vision and index % 5 != 0:
            task = families[index % 4]
            image_hash = f"{index:064x}"
        items.append(
            HoldoutItem(
                id=800 + index,
                prompt=f"private holdout question {index} about a specific thing",
                dataset_id="dev",
                task=task,
                image_hash=image_hash,
            )
        )
    return tuple(items)


def make_request(
    items: Sequence[HoldoutItem] | None = None,
    *,
    submission_digest: str = "frozen-1",
    artifact_digest: str = "ab" * 32,
    eval_image_digest: str = IMAGE_DIGEST,
) -> HarvestRequest:
    holdout = tuple(items if items is not None else holdout_items())
    return HarvestRequest(
        schema_version=METRICS_SCHEMA_VERSION,
        submission_digest=submission_digest,
        artifact_digest=artifact_digest,
        base_model="Qwen/Qwen3.8-27B",
        teacher_model="glm-5.3",
        eval_image_digest=eval_image_digest,
        holdout_commitment=holdout_commitment(holdout),
        holdout=holdout,
    )


@dataclass
class FakeRunner:
    """A model stand-in. Answers are fixed text, not scores."""

    answer: str = "the answer"
    canned: dict[str, str] = field(default_factory=dict)
    seen: list[Prompt] = field(default_factory=list)
    closed: bool = False

    def generate(self, prompts: Sequence[Prompt]) -> list[str]:
        self.seen.extend(prompts)
        return [self.canned.get(prompt.key, self.answer) for prompt in prompts]

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeTeacher:
    """A judge stand-in. Returns a fixed level so tests can assert plumbing."""

    level: float = 0.61
    calls: list[tuple[str, str]] = field(default_factory=list)

    def judge(self, prompt: str, candidate: str) -> float:
        self.calls.append((prompt, candidate))
        return self.level


@pytest.fixture
def request_fixture() -> HarvestRequest:
    return make_request()


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def teacher() -> FakeTeacher:
    return FakeTeacher()


@pytest.fixture
def slices() -> Slices:
    return Slices.load()
