"""Grading an action against the action that was recorded.

Everything in this module is deterministic. No judge is involved in scoring an
action, because there is a ground truth: the episode was recorded, so the tool
that was called and the arguments it was called with are known exactly. Asking
a language model whether an action "looks reasonable" would reintroduce the
noise the recording removes, and would make the score depend on the judge's
mood rather than on the model under test.

The judge is used for exactly one thing, in `teacher.py`: the free-text final
answer, which has no ground truth to match against.

Two properties this module is careful about:

* **Naming the right tool is worth something, and not everything.** A model
  that picks `get_oncall` with a wrong service argument is closer than one that
  picks `search_docs`, and the score says so.
* **A hallucinated tool is not the same failure as a wrong tool.** Naming
  something outside the episode's own schema means the model is not playing the
  game at all; it scores zero *and* is counted toward the invalid-action rate,
  which voids the run above a ceiling.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from relearn_common.jsonscan import extract_json_object

from .pins import TOOL_NAME_CREDIT
from .request import TraceStep

#: Collapse whitespace and case before comparing argument values. A model that
#: answers `"Payments"` for `"payments"` chose the same argument.
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Action:
    """One action a model proposed."""

    tool: str
    arguments: Mapping[str, object]

    #: True when the reply could not be read as an action at all.
    malformed: bool = False


MALFORMED = Action(tool="", arguments={}, malformed=True)


def parse_action(reply: str) -> Action:
    """Read the action out of a model reply.

    Tolerant of a thinking preamble and of the three shapes a post-trained
    model is likely to emit — a bare `{"tool": …}`, the `{"name": …}` spelling,
    and the OpenAI `{"function": {"name": …, "arguments": "…"}}` envelope —
    because the challenge is about choosing the right action, not about
    guessing which JSON dialect the harness wanted.

    Anything that cannot be read as one of those is malformed, which scores
    zero and counts toward the invalid-action rate.
    """
    body = extract_json_object(reply or "")
    if body is None:
        return MALFORMED
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return MALFORMED
    if not isinstance(parsed, Mapping):
        return MALFORMED

    envelope = parsed.get("function")
    if isinstance(envelope, Mapping):
        parsed = envelope

    tool = parsed.get("tool", parsed.get("name", ""))
    if not isinstance(tool, str) or not tool.strip():
        return MALFORMED

    raw_arguments = parsed.get("arguments", parsed.get("parameters", {}))
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            raw_arguments = {}
    if not isinstance(raw_arguments, Mapping):
        raw_arguments = {}

    return Action(tool=tool.strip(), arguments=dict(raw_arguments))


def _normalize(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        # 3 and 3.0 are the same argument.
        return f"{float(value):.6g}"
    if isinstance(value, str):
        return _SPACE.sub(" ", value).strip().casefold()
    return json.dumps(value, separators=(",", ":"), sort_keys=True).casefold()


def argument_agreement(
    recorded: Mapping[str, object], proposed: Mapping[str, object]
) -> float:
    """F1 over `(key, value)` pairs, in `[0, 1]`.

    F1 rather than exact match so that both halves of getting arguments wrong
    are measured: omitting a required argument costs recall, and inventing
    extra ones costs precision. Exact agreement is 1.0.
    """
    if not recorded and not proposed:
        return 1.0
    matched = sum(
        1
        for key, value in recorded.items()
        if key in proposed and _normalize(value) == _normalize(proposed[key])
    )
    if matched == 0:
        return 0.0
    precision = matched / len(proposed) if proposed else 0.0
    recall = matched / len(recorded) if recorded else 0.0
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class StepVerdict:
    """What one replayed step was worth, and how it went wrong."""

    score: float
    #: The reply was unreadable, or named a tool outside the episode's schema.
    invalid: bool
    #: The proposed tool, if it was readable. Used for the ordering measure.
    proposed_tool: str


def grade_step(
    recorded: TraceStep, action: Action, available_tools: Sequence[str]
) -> StepVerdict:
    """Score one replayed step against what was recorded."""
    if action.malformed:
        return StepVerdict(score=0.0, invalid=True, proposed_tool="")
    if action.tool not in available_tools:
        return StepVerdict(score=0.0, invalid=True, proposed_tool=action.tool)
    if action.tool != recorded.tool:
        return StepVerdict(score=0.0, invalid=False, proposed_tool=action.tool)
    agreement = argument_agreement(recorded.arguments, action.arguments)
    score = TOOL_NAME_CREDIT + (1.0 - TOOL_NAME_CREDIT) * agreement
    return StepVerdict(score=score, invalid=False, proposed_tool=action.tool)


def order_fidelity(recorded: Sequence[TraceStep], proposed_tools: Sequence[str]) -> float:
    """Did the model act in the recorded order, or skip around?

    Each step is replayed on the recorded prefix, so a model cannot be dragged
    off course by one early mistake — which means ordering has to be measured
    rather than inferred from where the episode ended up.

    A step counts as *in order* when the model names the tool recorded at that
    index, and *out of order* when it names a tool the episode uses somewhere
    else — jumping to the ticket before the lookup, or repeating a lookup it
    was already given the answer to. A tool the episode never uses is neither:
    that is a wrong action, already scored as one by [`grade_step`], and
    counting it twice would let one mistake sink two series.

    A model that never names a tool from the episode scores zero rather than
    one: there is no evidence of ordering, and absent evidence is not credit.
    """
    in_order = 0
    out_of_order = 0
    used = {step.tool for step in recorded}
    for index, proposed in enumerate(proposed_tools):
        if index >= len(recorded):
            break
        if proposed == recorded[index].tool:
            in_order += 1
        elif proposed in used:
            out_of_order += 1
    decided = in_order + out_of_order
    if decided == 0:
        return 0.0
    return in_order / decided


def grade_reference_answer(answer: str, expected: str, aliases: Sequence[str] = ()) -> float:
    """Exact-match grading for the shipped canary traces.

    Canaries have one right answer by construction, so they are matched rather
    than judged. They exist to catch a model that has lost the ability to emit
    a tool call at all — a failure mode that a relative comparison against a
    champion would hide, because both sides would be equally broken.
    """
    candidate = _normalize(answer)
    if not candidate:
        return 0.0
    for wanted in (expected, *aliases):
        normalized = _normalize(wanted)
        if normalized and normalized in candidate:
            return 1.0
    return 0.0
