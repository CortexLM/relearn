"""Scoring one artifact into one metrics document.

Every series in the document is measured here, from the model's own answers:

| Series | Items | Graded by |
|--------|-------|-----------|
| `holdout` | the request's private split | frozen teacher |
| `perturbed` | the same items, pinned rewrite | frozen teacher |
| `public` | the published split | frozen teacher |
| `canaries` | shipped known-answer items | reference match |
| `general_canary` | shipped MMLU / MMMU-style choices | choice letter |
| `agent_trace` | shipped ordered-plan tasks | rubric |
| `vision_shuffle` | vision families in the holdout | teacher, real vs shuffled pixels |

The holdout and the public split are judged on the same scale by the same
judge, because the gap between them is itself a gate: measuring one with a
reference matcher and the other with a judge would make the gap an artefact of
the graders.

Nothing here has a path that produces a number without the model. An item the
model could not be asked about — a missing image, an unreachable judge — ends
the run. A hole in the document would be read as a low score for that item,
which is a verdict nobody measured.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from . import catalog
from .contract import (
    CANARY_KEY_PREFIX,
    GENERAL_CANARY_KEY_PREFIX,
    HOLDOUT_KEY_PREFIX,
    PERTURBED_KEY_PREFIX,
    PUBLIC_KEY_PREFIX,
    ContractError,
    EvalDocument,
    Measurement,
    ShuffleEvidence,
    mean,
    series_key,
)
from .grading import grade_choice, grade_reference, grade_rubric
from .images import load_image, shuffle_pixels
from .perturb import perturb_prompt
from .request import HarvestRequest, HoldoutItem
from .runner import ModelRunner, Prompt
from .teacher import HttpTeacher

log = logging.getLogger(__name__)


class ScoringError(ContractError):
    """The run could not be scored end to end."""


@dataclass(frozen=True)
class Slices:
    """The slices the image owns, resolved once per run."""

    public: tuple[HoldoutItem, ...]
    canaries: tuple[catalog.GradedItem, ...]
    general_canary: tuple[catalog.ChoiceItem, ...]
    agent_trace: tuple[catalog.TraceItem, ...]

    @classmethod
    def load(cls) -> Slices:
        return cls(
            public=catalog.public_items(),
            canaries=catalog.canary_items(),
            general_canary=catalog.general_canary_items(),
            agent_trace=catalog.agent_trace_items(),
        )


def _answers(runner: ModelRunner, prompts: Sequence[Prompt], what: str) -> list[str]:
    if not prompts:
        return []
    log.info("generating %d answers for %s", len(prompts), what)
    answers = runner.generate(prompts)
    if len(answers) != len(prompts):
        raise ScoringError(
            f"{what}: model returned {len(answers)} answers for {len(prompts)} prompts"
        )
    return answers


def _judged_series(
    items: Sequence[HoldoutItem],
    prefix: str,
    runner: ModelRunner,
    teacher: HttpTeacher,
    what: str,
    *,
    perturbed: bool = False,
    shuffled_images: bool = False,
) -> dict[str, float]:
    """Generate answers for `items` and have the teacher score each one."""
    prompts: list[Prompt] = []
    for item in items:
        text = perturb_prompt(item.prompt) if perturbed else item.prompt
        image: bytes | None = None
        if item.is_vision:
            body = load_image(item.image_hash)
            image = shuffle_pixels(body, item.image_hash) if shuffled_images else body
        prompts.append(Prompt(key=series_key(prefix, item.id), text=text, image=image))

    answers = _answers(runner, prompts, what)
    scores: dict[str, float] = {}
    for item, prompt, answer in zip(items, prompts, answers, strict=True):
        scores[prompt.key] = teacher.judge(item.prompt, answer)
    return scores


def _canary_series(
    items: Sequence[catalog.GradedItem], runner: ModelRunner
) -> dict[str, float]:
    prompts = [
        Prompt(key=series_key(CANARY_KEY_PREFIX, item.id), text=item.prompt) for item in items
    ]
    answers = _answers(runner, prompts, "canaries")
    return {
        prompt.key: grade_reference(answer, item.answer, item.aliases)
        for item, prompt, answer in zip(items, prompts, answers, strict=True)
    }


def _general_canary_series(
    items: Sequence[catalog.ChoiceItem], runner: ModelRunner
) -> dict[str, float]:
    prompts = [
        Prompt(key=series_key(GENERAL_CANARY_KEY_PREFIX, item.id), text=item.rendered())
        for item in items
    ]
    answers = _answers(runner, prompts, "general canary")
    return {
        prompt.key: grade_choice(answer, item.answer)
        for item, prompt, answer in zip(items, prompts, answers, strict=True)
    }


def _agent_trace(items: Sequence[catalog.TraceItem], runner: ModelRunner) -> float:
    prompts = [Prompt(key=f"a{item.id}", text=item.prompt) for item in items]
    answers = _answers(runner, prompts, "agent trace")
    scored = [
        grade_rubric(answer, item.must_include)
        for item, answer in zip(items, answers, strict=True)
    ]
    value = mean(scored)
    if value is None:
        raise ScoringError("agent-trace slice produced no scores")
    return value


def _vision_shuffle(
    request: HarvestRequest,
    holdout: dict[str, float],
    runner: ModelRunner,
    teacher: HttpTeacher,
) -> dict[str, ShuffleEvidence]:
    """Pixel-shuffle control for each vision family in this holdout."""
    evidence: dict[str, ShuffleEvidence] = {}
    for family in request.vision_families():
        items = tuple(item for item in request.holdout if item.task == family)
        real = mean(holdout[series_key(HOLDOUT_KEY_PREFIX, item.id)] for item in items)
        shuffled = _judged_series(
            items,
            HOLDOUT_KEY_PREFIX,
            runner,
            teacher,
            f"{family} pixel shuffle",
            shuffled_images=True,
        )
        shuffled_mean = mean(shuffled.values())
        if real is None or shuffled_mean is None:
            raise ScoringError(f"pixel-shuffle control for {family} scored no items")
        evidence[family] = ShuffleEvidence(
            items=len(items), score=real, shuffled_score=shuffled_mean
        )
    return evidence


def score_request(
    request: HarvestRequest,
    runner: ModelRunner,
    teacher: HttpTeacher,
    slices: Slices | None = None,
) -> EvalDocument:
    """Measure every series and bind them to the run that was requested.

    # Raises
    [`ScoringError`], [`TeacherError`], [`ImageStoreError`], or
    [`RunnerError`] — all of which end the run without a document.
    """
    resolved = slices or Slices.load()

    holdout = _judged_series(
        request.holdout, HOLDOUT_KEY_PREFIX, runner, teacher, "holdout"
    )
    perturbed = _judged_series(
        request.holdout,
        PERTURBED_KEY_PREFIX,
        runner,
        teacher,
        "perturbed holdout",
        perturbed=True,
    )
    public = _judged_series(
        resolved.public, PUBLIC_KEY_PREFIX, runner, teacher, "public split"
    )
    canaries = _canary_series(resolved.canaries, runner)
    general_canary = _general_canary_series(resolved.general_canary, runner)
    agent_trace = _agent_trace(resolved.agent_trace, runner)
    vision_shuffle = _vision_shuffle(request, holdout, runner, teacher)

    log.info(
        "measured holdout=%d perturbed=%d public=%d canaries=%d general=%d shuffle=%d",
        len(holdout),
        len(perturbed),
        len(public),
        len(canaries),
        len(general_canary),
        len(vision_shuffle),
    )
    return EvalDocument(
        submission_digest=request.submission_digest,
        artifact_digest=request.artifact_digest,
        measurement=Measurement(
            eval_image_digest=request.eval_image_digest,
            holdout_commitment=request.holdout_commitment,
            holdout=holdout,
            public=public,
            perturbed=perturbed,
            canaries=canaries,
            general_canary=general_canary,
            agent_trace=agent_trace,
            vision_shuffle=vision_shuffle,
        ),
    )
