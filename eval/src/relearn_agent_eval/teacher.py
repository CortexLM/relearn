"""The frozen teacher, used only to grade a free-text final answer.

Every action in this challenge is graded against a recording, deterministically
(`grading.py`). The one thing with no ground truth is the sentence the agent
writes at the end, and that is all the teacher sees.

The teacher never serves the scored model: the artifact under test is loaded
inside this image, and a payload that looks like weights or a model id that
looks like an artifact digest is refused before any request leaves the pod.

There is no endpoint, host, or key in this repository, and no offline judge. An
unconfigured teacher ends the run.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from relearn_common.env import env, env_first
from relearn_common.errors import ContractError
from relearn_common.judge import ChatJudge, JudgeError, guard_judge_call

from .pins import CONFIGURED_TEACHER_MODELS, ENV_PREFIX, TEACHER_MODEL_ID

log = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You grade an agent's final answer against what its tools actually "
    'returned. Reply with JSON only, exactly {"score": <number between 0 and '
    "1>}. 1.0 is correct, complete, and supported by the tool results; 0.0 is "
    "wrong, empty, or asserts something the tool results do not support. "
    "Penalise a confident answer that the tool results do not support as "
    "harshly as a wrong one. Do not explain."
)

_SCORE = re.compile(r"-?\d+(?:\.\d+)?")


class TeacherError(ContractError):
    """The teacher could not judge, so the run has no answer scores."""


def parse_score(content: str) -> float:
    """Read a `[0, 1]` score out of a judge reply.

    A reply that carries no number is a failed item, not a zero: the model may
    well have answered correctly and the judge simply did not say so.
    """
    body = content.strip()
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and "score" in parsed:
            return min(1.0, max(0.0, float(parsed["score"])))
        if isinstance(parsed, int | float):
            return min(1.0, max(0.0, float(parsed)))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = _SCORE.search(body)
    if not match:
        raise TeacherError("teacher reply carried no score")
    return min(1.0, max(0.0, float(match.group())))


@dataclass
class AnswerTeacher:
    """Grades one final answer against the recorded tool results."""

    chat: ChatJudge

    def judge(self, goal: str, tool_results: str, answer: str) -> float:
        content = self.chat.complete(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Goal:\n{goal}\n\n"
                        f"What the tools returned:\n{tool_results}\n\n"
                        f"The agent's final answer:\n{answer}"
                    ),
                },
            ],
            max_tokens=32,
        )
        return parse_score(content)

    def judge_all(self, items: Iterable[tuple[str, str, str]]) -> list[float]:
        return [self.judge(goal, results, answer) for goal, results, answer in items]


def build_teacher(teacher_model: str) -> AnswerTeacher:
    """Build the teacher for this run from the operator's environment.

    Falls back to the shared `RELEARN_TEACHER_*` names so a host already
    running the language challenge against a judge deployment configures it
    once. Falling back to nothing is still nothing.

    # Raises
    [`TeacherError`] when no endpoint is configured.
    """
    url = env_first(f"{ENV_PREFIX}TEACHER_API_URL", "RELEARN_TEACHER_API_URL").rstrip("/")
    if not url:
        raise TeacherError(
            f"{ENV_PREFIX}TEACHER_API_URL is unset; the pod has no judge and cannot score "
            "the final answers"
        )
    model = (
        env_first(f"{ENV_PREFIX}TEACHER_MODEL", "RELEARN_TEACHER_MODEL")
        or teacher_model.strip()
        or TEACHER_MODEL_ID
    )
    allowed = tuple({*CONFIGURED_TEACHER_MODELS, model})
    try:
        guard_judge_call(model, CONFIGURED_TEACHER_MODELS, "configuration probe")
    except JudgeError as exc:
        # An operator may configure a wire id this image has never heard of,
        # but never an artifact digest and never a weights payload, which is
        # what the guard is really for.
        if "is not a judge" not in str(exc):
            raise TeacherError(str(exc)) from exc
        log.warning("teacher wire id %r is not one this image ships a default for", model)
    return AnswerTeacher(
        chat=ChatJudge(
            api_url=url,
            model=model,
            allowed_models=allowed,
            api_key=env_first(f"{ENV_PREFIX}TEACHER_API_KEY", "RELEARN_TEACHER_API_KEY"),
            timeout_secs=float(env(f"{ENV_PREFIX}TEACHER_TIMEOUT_SECS") or 120.0),
            extra_body={"temperature": 0},
        )
    )
