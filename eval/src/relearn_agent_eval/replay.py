"""Replaying a recorded episode against the model under test.

Each step is presented on the **recorded** prefix: the goal, the tool schemas,
and every earlier step exactly as it happened, including what the tools
returned. The model is asked only for the next action.

Replaying on the recorded prefix rather than letting the model run free is the
central design choice here, and it is not a convenience:

* Off-policy, the model would call tools this image cannot execute. There is no
  live environment on the pod — only a recording — so an unrecorded action has
  no observation to return, and inventing one would be a simulated number.
* One early mistake would otherwise zero an entire episode, which makes the
  score a measure of the first step and nothing else.
* Every model sees byte-identical context at every step, so the comparison
  between a champion and a challenger is a comparison of decisions.

The same construction is used for the two controls, with one substitution each:
observations withheld, or observation pixels shuffled. Everything else about
the prompt — its length, its turn structure, its tool schemas — is unchanged,
so a drop in score is attributable to the substitution and not to the prompt
having become a different shape.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from relearn_common.imagestore import load_image, shuffle_pixels
from relearn_common.runner import Prompt

from .pins import ENV_PREFIX, WITHHELD_OBSERVATION
from .request import Trace, TraceStep

_SYSTEM = (
    "You are an agent completing a task with tools. You are shown the goal, "
    "the tools you may call, and everything that has happened so far. Reply "
    "with the single next tool call as one JSON object: "
    '{"tool": "<tool name>", "arguments": {…}}. Reply with nothing else.'
)

_ANSWER_INSTRUCTION = (
    "Every tool call is done. Using only what the tools returned, write the "
    "final answer for the goal above. Reply with the answer text and nothing "
    "else."
)


class Mode(Enum):
    """How the recorded observations are presented on this pass."""

    #: What actually came back. This is the scored pass.
    REAL = "real"
    #: Every observation replaced by a content-free placeholder. A model that
    #: used the observations cannot reproduce arguments that came from them.
    BLIND = "blind"
    #: Observation text kept, observation pixels destroyed. A model that read
    #: the screenshot loses ground; one that guessed from the goal does not.
    SHUFFLED = "shuffled"


@dataclass(frozen=True)
class StepPrompt:
    """One prompt to send, and which step of which trace it belongs to."""

    trace_id: int
    step_index: int
    prompt: Prompt


def _tool_block(trace: Trace) -> str:
    return json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            }
            for tool in trace.tools
        ],
        separators=(",", ":"),
        sort_keys=True,
    )


def _observation_text(step: TraceStep, mode: Mode) -> str:
    if mode is Mode.BLIND:
        return WITHHELD_OBSERVATION
    return step.observation


def _history(steps: Sequence[TraceStep], mode: Mode) -> str:
    lines: list[str] = []
    for step in steps:
        call = json.dumps(
            {"tool": step.tool, "arguments": step.arguments},
            separators=(",", ":"),
            sort_keys=True,
        )
        lines.append(f"Action: {call}")
        observation = _observation_text(step, mode)
        if step.has_image:
            note = (
                "<screenshot withheld>"
                if mode is Mode.BLIND
                else "<screenshot attached below>"
            )
            lines.append(f"Observation: {observation} {note}".rstrip())
        else:
            lines.append(f"Observation: {observation}")
    return "\n".join(lines)


def _prefix_image(steps: Sequence[TraceStep], mode: Mode) -> bytes | None:
    """Pixels for the most recent screenshot in the prefix, if any.

    Only the latest is attached: a runner takes one image per prompt, and the
    step a model is about to act on is the one whose screenshot matters. The
    earlier ones are still represented in the history as text, so the turn
    structure is unchanged.
    """
    if mode is Mode.BLIND:
        return None
    for step in reversed(steps):
        if not step.has_image:
            continue
        body = load_image(step.observation_image_hash, prefix=ENV_PREFIX)
        if mode is Mode.SHUFFLED:
            return shuffle_pixels(body, step.observation_image_hash)
        return body
    return None


def step_prompts(trace: Trace, mode: Mode, key_prefix: str) -> list[StepPrompt]:
    """One prompt per recorded step, each on that step's recorded prefix."""
    prompts: list[StepPrompt] = []
    for step in trace.steps:
        prefix = trace.steps[: step.index]
        body = (
            f"{_SYSTEM}\n\n"
            f"Goal:\n{trace.goal}\n\n"
            f"Tools:\n{_tool_block(trace)}\n\n"
            f"So far:\n{_history(prefix, mode) or '(nothing yet)'}\n\n"
            "Next action:"
        )
        prompts.append(
            StepPrompt(
                trace_id=trace.id,
                step_index=step.index,
                prompt=Prompt(
                    key=f"{key_prefix}{trace.id}#{step.index}",
                    text=body,
                    image=_prefix_image(prefix, mode),
                ),
            )
        )
    return prompts


def answer_prompt(trace: Trace, key_prefix: str) -> Prompt:
    """The prompt that asks for the episode's final answer.

    Always on the real, complete recording: the answer is graded against what
    the tools actually returned, so withholding them here would grade the model
    on a task nobody asked it to do.
    """
    body = (
        f"Goal:\n{trace.goal}\n\n"
        f"Tools:\n{_tool_block(trace)}\n\n"
        f"What happened:\n{_history(trace.steps, Mode.REAL)}\n\n"
        f"{_ANSWER_INSTRUCTION}"
    )
    return Prompt(
        key=f"{key_prefix}{trace.id}",
        text=body,
        image=_prefix_image(trace.steps, Mode.REAL),
    )
