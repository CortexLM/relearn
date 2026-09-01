"""Scoring one artifact into one Relearn Agent metrics document.

Every recorded episode is replayed three times against the model under test:

1. **as recorded** — the scored pass;
2. **tool-blind** — identical, with every observation replaced by a
   content-free placeholder;
3. **pixel-shuffled** — identical, with observation screenshots destroyed,
   for the episodes that have them.

Passes 2 and 3 are why this challenge exists as something separate from the
language challenge. A model that never reads what a tool returned can still
write a confident, well-formed answer, and prose grading rewards it. Here the
same model is measured against the recorded argument that could only have come
from the observation, and then measured again with that observation taken away.
If nothing changes, it was not using the tool results, and the control plane
fails it — a low score is not enough, because a model can be uniformly mediocre
and still beat a champion on noise.

The controls compare the **action** score only, and involve no judge. That is
deliberate: the action score is deterministic, so the difference between the
two passes is attributable to the withheld information rather than to judge
variance, and running the judge three times over would triple the cost of a run
to measure something the judge is worse at.

Nothing here produces a number without the model. An episode whose pixels are
missing, or whose final answer the teacher could not grade, ends the run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from relearn_common.errors import ContractError
from relearn_common.runner import ModelRunner, Prompt
from relearn_common.wire import mean

from .catalog import CanaryEpisode, canary_episodes, public_episodes
from .contract import AgentDocument, AgentMeasurement, ControlEvidence
from .grading import Action, grade_step, order_fidelity, parse_action
from .pins import (
    CANARY_KEY_PREFIX,
    HOLDOUT_KEY_PREFIX,
    IMAGE_MODALITY,
    MAX_INVALID_ACTION_RATE,
    ORDER_KEY_PREFIX,
    PUBLIC_KEY_PREFIX,
    TOOL_CALL_KEY_PREFIX,
    TRACE_ACTION_WEIGHT,
    TRACE_ANSWER_WEIGHT,
    TRACE_ORDER_WEIGHT,
)
from .replay import Mode, answer_prompt, step_prompts
from .request import AgentHarvestRequest, Trace, TraceStep
from .teacher import AnswerTeacher

log = logging.getLogger(__name__)


class ScoringError(ContractError):
    """The run could not be scored end to end."""


@dataclass(frozen=True)
class TracePass:
    """One episode, replayed once."""

    trace_id: int
    action: float
    order: float
    invalid_steps: int
    total_steps: int


@dataclass(frozen=True)
class Slices:
    """The slices the image owns, resolved once per run."""

    public: tuple[Trace, ...]
    canaries: tuple[CanaryEpisode, ...]

    @classmethod
    def load(cls) -> Slices:
        return cls(public=public_episodes(), canaries=canary_episodes())


def _answers(runner: ModelRunner, prompts: Sequence[Prompt], what: str) -> list[str]:
    if not prompts:
        return []
    log.info("replaying %d steps for %s", len(prompts), what)
    replies = runner.generate(prompts)
    if len(replies) != len(prompts):
        raise ScoringError(
            f"{what}: model returned {len(replies)} replies for {len(prompts)} prompts"
        )
    return replies


def replay_traces(
    traces: Sequence[Trace], runner: ModelRunner, mode: Mode, what: str
) -> dict[int, TracePass]:
    """Replay every step of every episode once, and grade the actions.

    All the steps of all the episodes go to the runner as one batch, so the
    backend can schedule them together; the replies are matched back by
    position, which is safe because the runner contract is order-preserving.
    """
    prompts = [
        step for trace in traces for step in step_prompts(trace, mode, HOLDOUT_KEY_PREFIX)
    ]
    replies = _answers(runner, [item.prompt for item in prompts], what)

    actions: dict[int, list[Action]] = {trace.id: [] for trace in traces}
    for item, reply in zip(prompts, replies, strict=True):
        actions[item.trace_id].append(parse_action(reply))

    passes: dict[int, TracePass] = {}
    for trace in traces:
        proposed = actions[trace.id]
        verdicts = [
            grade_step(step, action, trace.tool_names)
            for step, action in zip(trace.steps, proposed, strict=True)
        ]
        action_score = mean(verdict.score for verdict in verdicts)
        if action_score is None:
            raise ScoringError(f"episode {trace.id} produced no graded steps")
        passes[trace.id] = TracePass(
            trace_id=trace.id,
            action=action_score,
            order=order_fidelity(trace.steps, [verdict.proposed_tool for verdict in verdicts]),
            invalid_steps=sum(1 for verdict in verdicts if verdict.invalid),
            total_steps=len(verdicts),
        )
    return passes


def _answer_scores(
    traces: Sequence[Trace], runner: ModelRunner, teacher: AnswerTeacher, what: str
) -> dict[int, float]:
    """Ask for each episode's final answer, and have the teacher grade it."""
    prompts = [answer_prompt(trace, HOLDOUT_KEY_PREFIX) for trace in traces]
    replies = _answers(runner, prompts, f"{what} final answers")
    scores: dict[int, float] = {}
    for trace, reply in zip(traces, replies, strict=True):
        results = "\n".join(step.observation for step in trace.steps)
        scores[trace.id] = teacher.judge(trace.goal, results, reply)
    return scores


def composite(action: float, order: float, answer: float) -> float:
    """How the three measured parts of an episode make one score."""
    return (
        TRACE_ACTION_WEIGHT * action
        + TRACE_ORDER_WEIGHT * order
        + TRACE_ANSWER_WEIGHT * answer
    )


def _canary_series(
    canaries: Sequence[CanaryEpisode], runner: ModelRunner
) -> dict[str, float]:
    """Score the shipped single-call episodes by matching the recorded call."""
    prompts = [
        Prompt(
            key=f"{CANARY_KEY_PREFIX}{episode.id}",
            text=(
                "You are an agent completing a task with tools. Reply with the single "
                'next tool call as one JSON object: {"tool": "<tool name>", '
                '"arguments": {…}}. Reply with nothing else.\n\n'
                f"Goal:\n{episode.goal}\n\n"
                "Tools:\n"
                + json.dumps(
                    [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": dict(tool.parameters),
                        }
                        for tool in episode.tools
                    ],
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n\nNext action:"
            ),
        )
        for episode in canaries
    ]
    replies = _answers(runner, prompts, "canaries")

    scores: dict[str, float] = {}
    for episode, prompt, reply in zip(canaries, prompts, replies, strict=True):
        recorded = TraceStep(
            index=0,
            tool=episode.tool,
            arguments_json=json.dumps(
                episode.arguments, separators=(",", ":"), sort_keys=True
            ),
        )
        verdict = grade_step(recorded, parse_action(reply), episode.tool_names)
        scores[prompt.key] = verdict.score
    return scores


def _control(
    real: dict[int, TracePass], degraded: dict[int, TracePass]
) -> ControlEvidence:
    """Fold two passes over the same episodes into one piece of evidence."""
    shared = sorted(set(real) & set(degraded))
    if not shared:
        raise ScoringError("a control pass scored none of the episodes it was given")
    real_mean = mean(real[trace_id].action for trace_id in shared)
    degraded_mean = mean(degraded[trace_id].action for trace_id in shared)
    if real_mean is None or degraded_mean is None:
        raise ScoringError("a control pass produced no scores")
    return ControlEvidence(
        traces=len(shared), score=real_mean, degraded_score=degraded_mean
    )


def score_request(
    request: AgentHarvestRequest,
    runner: ModelRunner,
    teacher: AnswerTeacher,
    slices: Slices | None = None,
) -> AgentDocument:
    """Measure every series and bind them to the run that was requested.

    # Raises
    [`ScoringError`], [`TeacherError`], [`ImageError`], or [`RunnerError`] —
    all of which end the run without a document.
    """
    resolved = slices or Slices.load()
    holdout = list(request.holdout)

    real = replay_traces(holdout, runner, Mode.REAL, "holdout")
    answers = _answer_scores(holdout, runner, teacher, "holdout")

    invalid_steps = sum(item.invalid_steps for item in real.values())
    total_steps = sum(item.total_steps for item in real.values())
    invalid_rate = invalid_steps / total_steps if total_steps else 1.0
    if invalid_rate > MAX_INVALID_ACTION_RATE:
        # The model could not emit a well-formed action for most of the split.
        # Whatever the surviving steps scored is not a measurement of tool use.
        raise ScoringError(
            f"invalid action rate {invalid_rate:.3f} above the "
            f"{MAX_INVALID_ACTION_RATE:.3f} ceiling"
        )

    blind = replay_traces(holdout, runner, Mode.BLIND, "holdout, observations withheld")
    tool_blind = _control(real, blind)

    observation_shuffle: dict[str, ControlEvidence] = {}
    if IMAGE_MODALITY in request.modalities():
        with_images = [trace for trace in holdout if trace.has_image_observation]
        shuffled = replay_traces(
            with_images, runner, Mode.SHUFFLED, "holdout, observation pixels shuffled"
        )
        observation_shuffle[IMAGE_MODALITY] = _control(
            {trace.id: real[trace.id] for trace in with_images}, shuffled
        )

    public_real = replay_traces(resolved.public, runner, Mode.REAL, "public split")
    public_answers = _answer_scores(resolved.public, runner, teacher, "public split")

    log.info(
        "measured holdout=%d public=%d canaries=%d invalid_rate=%.4f blind_drop=%.4f",
        len(real),
        len(public_real),
        len(resolved.canaries),
        invalid_rate,
        tool_blind.drop,
    )

    return AgentDocument(
        identity=request.identity,
        measurement=AgentMeasurement(
            base_model=request.base_model,
            teacher_model=request.teacher_model,
            holdout={
                f"{HOLDOUT_KEY_PREFIX}{trace_id}": composite(
                    item.action, item.order, answers[trace_id]
                )
                for trace_id, item in sorted(real.items())
            },
            tool_call={
                f"{TOOL_CALL_KEY_PREFIX}{trace_id}": item.action
                for trace_id, item in sorted(real.items())
            },
            order={
                f"{ORDER_KEY_PREFIX}{trace_id}": item.order
                for trace_id, item in sorted(real.items())
            },
            public={
                f"{PUBLIC_KEY_PREFIX}{trace_id}": composite(
                    item.action, item.order, public_answers[trace_id]
                )
                for trace_id, item in sorted(public_real.items())
            },
            canaries=_canary_series(resolved.canaries, runner),
            invalid_action_rate=invalid_rate,
            tool_blind=tool_blind,
            observation_shuffle=observation_shuffle,
        ),
    )
