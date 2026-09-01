"""Test doubles for the two new eval images.

The generator, the runner, and the judges are the things these images cannot
carry an offline version of, so their stand-ins live here in the test tree and
are never installed into an image. That is the whole point of the arrangement:
a fixture that could produce scores from inside the package would be the
simulated harness these challenges exist to remove.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from relearn_agent_eval.grading import parse_action
from relearn_agent_eval.pins import WITHHELD_OBSERVATION
from relearn_common.runner import Prompt
from relearn_image_eval.judge import ImageScore
from relearn_image_eval.pillars import ALL_PILLARS

IMAGE_DIGEST = f"sha256:{'ab' * 32}"


# --------------------------------------------------------------------------
# Relearn Image doubles
# --------------------------------------------------------------------------


#: Side of the tiny images the fake generator emits. Real PNGs, because the
#: replay descriptor decodes pixels and a stand-in that returned opaque bytes
#: would skip the code path the evidence depends on.
FAKE_IMAGE_SIDE = 16


@dataclass
class FakeGenerator:
    """A generator stand-in. Produces pixels, never a score.

    Output is a function of the prompt and the seed, so regenerating the same
    cell reproduces the same image — which is what a real deterministic
    pipeline does, and what the seed-replay evidence is measuring.
    `drift_on_replay` turns that off, to exercise the other branch.
    """

    drift_on_replay: bool = False
    calls: list[tuple[str, int]] = field(default_factory=list)
    closed: bool = False

    def generate(self, prompt: str, seed: int) -> bytes:
        from io import BytesIO

        import numpy as np
        from PIL import Image

        self.calls.append((prompt, seed))
        salt = str(self.calls.count((prompt, seed))) if self.drift_on_replay else ""
        digest = hashlib.sha256(f"{prompt}|{seed}|{salt}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        pixels = rng.integers(
            0, 256, size=(FAKE_IMAGE_SIDE, FAKE_IMAGE_SIDE, 3), dtype="uint8"
        )
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        return buffer.getvalue()

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeQJudger:
    """A Q-Judger stand-in. Returns a fixed level so tests assert plumbing."""

    level: float = 72.0
    spot_level: float | None = None
    na_items: int = 1
    scored_items: int = 19
    calls: list[str] = field(default_factory=list)

    def _score(self, level: float) -> ImageScore:
        return ImageScore(
            per_pillar={pillar.wire: level for pillar in ALL_PILLARS},
            total=level,
            scored_items=self.scored_items,
            na_items=self.na_items,
        )

    def score_image(self, prompt: str, image: bytes) -> ImageScore:
        self.calls.append("score")
        return self._score(self.level)

    def spot_check_alignment(self, prompt: str, image: bytes) -> ImageScore:
        self.calls.append("spot")
        return self._score(self.level if self.spot_level is None else self.spot_level)


# --------------------------------------------------------------------------
# Relearn Agent doubles
# --------------------------------------------------------------------------


def _recorded_action(prompt_text: str) -> dict[str, object] | None:
    """What a model that reads the prompt could legitimately answer.

    The fake models below are the point of the agent tests, so they have to
    behave the way real ones would: the only place the correct `owner`
    argument appears is in a prior observation, so a model that is not reading
    observations cannot produce it.
    """
    marker = "So far:\n"
    head, _, history = prompt_text.partition(marker)
    if "\"tool\"" not in prompt_text:
        return None
    return {"history": history, "goal": head}


@dataclass
class ScriptedAgent:
    """A model stand-in that plays a recorded episode back.

    `reads_observations` is the knob the control tests turn. When it is on, the
    model lifts arguments out of the observations it was shown, so withholding
    them costs it the argument. When it is off, it answers from the goal alone
    and withholding changes nothing — which is exactly the model the tool-blind
    control has to catch.
    """

    reads_observations: bool = True
    #: Tool to name at each step, in order. `None` means "whatever the episode
    #: recorded", which the prompt reveals through the history length.
    tool_plan: Sequence[str] | None = None
    malformed_every: int = 0
    seen: list[Prompt] = field(default_factory=list)
    closed: bool = False

    def generate(self, prompts: Sequence[Prompt]) -> list[str]:
        self.seen.extend(prompts)
        return [self._reply(index, prompt) for index, prompt in enumerate(prompts)]

    def _reply(self, index: int, prompt: Prompt) -> str:
        if self.malformed_every and index % self.malformed_every == 0:
            return "I am not going to answer with JSON."
        if "final answer" in prompt.text.lower() and "Next action:" not in prompt.text:
            return "The record is owned by someone and a report was filed."

        history = prompt.text.partition("So far:\n")[2]
        steps_done = history.count("Action: ")
        tool = self._tool_for(prompt, steps_done)
        arguments = self._arguments_for(prompt, history, steps_done)
        call = json.dumps({"tool": tool, "arguments": arguments})
        # A thinking preamble with braces of its own, the way a post-trained
        # model actually replies.
        return f"Let me think {{about it}} first.\n{call}"

    def _tool_for(self, prompt: Prompt, steps_done: int) -> str:
        if self.tool_plan is not None:
            return self.tool_plan[min(steps_done, len(self.tool_plan) - 1)]
        return "lookup_record" if steps_done == 0 else "file_report"

    def _arguments_for(
        self, prompt: Prompt, history: str, steps_done: int
    ) -> dict[str, object]:
        record = _record_id(prompt.text)
        if steps_done == 0:
            return {"record_id": record}
        owner = _owner_from(history) if self.reads_observations else "unknown"
        return {"owner": owner, "record_id": record}

    def close(self) -> None:
        self.closed = True


def _record_id(text: str) -> str:
    marker = "record REC-"
    head = text.find(marker)
    if head < 0:
        return "REC-0000"
    return "REC-" + text[head + len(marker) : head + len(marker) + 4]


def _owner_from(history: str) -> str:
    """Lift the owner out of an observation, the way a reading model would."""
    if WITHHELD_OBSERVATION in history:
        return ""
    marker = '"owner": "'
    head = history.find(marker)
    if head < 0:
        return ""
    tail = history.find('"', head + len(marker))
    return history[head + len(marker) : tail]


@dataclass
class FakeAnswerTeacher:
    """A teacher stand-in for the free-text final answer."""

    level: float = 0.7
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def judge(self, goal: str, tool_results: str, answer: str) -> float:
        self.calls.append((goal, tool_results, answer))
        return self.level


def action_of(reply: str) -> tuple[str, dict[str, object]]:
    """Parse a scripted reply, for assertions."""
    action = parse_action(reply)
    return action.tool, dict(action.arguments)
