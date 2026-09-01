"""Batching, judge concurrency, phases, and the budget.

A 120-item holdout is 280-odd judge calls and 350-odd generations once every
slice is counted. One at a time, that does not fit the harvest's run timeout —
which is how a pod comes up RUNNING and returns no `RELEARN_EVAL_OK`. These are
the mechanics that make a real run finish, and the ordering guarantees that stop
the concurrency from putting a score on the wrong item.
"""

from __future__ import annotations

import time

import pytest

from relearn_eval.progress import Budget, BudgetExceeded, Phase, budget_seconds
from relearn_eval.runner import DEFAULT_MAX_NEW_TOKENS, Prompt, batch_size, batches
from relearn_eval.scoring import CANARY_WIDTH, CHOICE_WIDTH, Slices, score_request
from relearn_eval.teacher import (
    DEFAULT_JUDGE_CONCURRENCY,
    HttpTeacher,
    TeacherError,
    judge_concurrency,
)

from .conftest import FakeRunner, FakeTeacher, holdout_items, make_request


def text(count: int, width: int | None = None) -> list[Prompt]:
    return [
        Prompt(key=f"h{index}", text=f"question {index}", max_new_tokens=width)
        for index in range(count)
    ]


def test_prompts_are_batched_in_order():
    grouped = batches(text(20), 8, DEFAULT_MAX_NEW_TOKENS)
    assert [len(batch) for batch in grouped] == [8, 8, 4]
    assert [prompt.key for batch in grouped for prompt in batch] == [
        prompt.key for prompt in text(20)
    ]


def test_a_batch_never_mixes_decode_widths():
    prompts = [*text(3, 256), *text(2, 8)]
    grouped = batches(prompts, 8, DEFAULT_MAX_NEW_TOKENS)
    assert [len(batch) for batch in grouped] == [3, 2]
    for batch in grouped:
        assert len({prompt.max_new_tokens for prompt in batch}) == 1


def test_a_vision_prompt_is_generated_on_its_own():
    prompts = [
        *text(2),
        Prompt(key="h9", text="caption", image=b"\x89PNG"),
        *text(2),
    ]
    grouped = batches(prompts, 8, DEFAULT_MAX_NEW_TOKENS)
    assert [len(batch) for batch in grouped] == [2, 1, 2]
    assert grouped[1][0].image is not None


def test_the_default_decode_width_leaves_room_for_a_live_run():
    # The width multiplies across every item in every slice, twice for the
    # holdout. 512 was past what a live run could afford.
    assert DEFAULT_MAX_NEW_TOKENS == 256
    assert CANARY_WIDTH < DEFAULT_MAX_NEW_TOKENS
    assert CHOICE_WIDTH < CANARY_WIDTH
    assert batch_size() > 1


def test_the_short_slices_ask_for_short_answers():
    runner = FakeRunner()
    score_request(make_request(holdout_items(2)), runner, FakeTeacher(), Slices.load())
    widths = {prompt.key[0]: prompt.max_new_tokens for prompt in runner.seen}
    assert widths["c"] == CANARY_WIDTH
    assert widths["g"] == CHOICE_WIDTH
    # Open-ended answers take the run default.
    assert widths["h"] is None and widths["p"] is None


def test_judging_is_concurrent_but_positional():
    """Order is what binds a score to an item, so it must survive the pool."""

    class SlowJudge(HttpTeacher):
        def judge(self, prompt: str, candidate: str) -> float:
            time.sleep(0.05)
            return float(candidate)

    judge = SlowJudge(api_url="http://teacher.invalid/v1", model="glm-5.3")
    pairs = [(f"q{index}", str(index / 100)) for index in range(16)]
    started = time.monotonic()
    scores = judge.judge_all(pairs)
    elapsed = time.monotonic() - started

    assert scores == [index / 100 for index in range(16)]
    # 16 calls at 50ms is 800ms sequentially; the pool is well under that.
    assert elapsed < 0.5, elapsed


def test_the_judge_concurrency_is_bounded_and_validated(monkeypatch):
    monkeypatch.delenv("RELEARN_JUDGE_CONCURRENCY", raising=False)
    assert judge_concurrency() == DEFAULT_JUDGE_CONCURRENCY
    monkeypatch.setenv("RELEARN_JUDGE_CONCURRENCY", "1")
    assert judge_concurrency() == 1
    for bad in ("0", "-4", "many"):
        monkeypatch.setenv("RELEARN_JUDGE_CONCURRENCY", bad)
        with pytest.raises(TeacherError):
            judge_concurrency()


def test_the_first_judge_failure_fails_the_slice():
    class FlakyJudge(HttpTeacher):
        def judge(self, prompt: str, candidate: str) -> float:
            if candidate == "bad":
                raise TeacherError("teacher unreachable after 3 attempts")
            return 0.5

    judge = FlakyJudge(api_url="http://teacher.invalid/v1", model="glm-5.3")
    with pytest.raises(TeacherError):
        judge.judge_all([("q", "ok"), ("q", "bad"), ("q", "ok")])


def test_the_phase_is_always_nameable():
    phase = Phase()
    assert phase.name == "starting"
    phase.enter("holdout")
    assert phase.name == "holdout"
    assert phase.elapsed >= 0


def test_a_spent_budget_stops_the_run_between_phases():
    phase = Phase()
    unbounded = Budget(seconds=0, phase=phase)
    unbounded.check("holdout")  # no ceiling, no refusal

    phase.enter("holdout")
    spent = Budget(seconds=0.0001, phase=phase)
    time.sleep(0.01)
    with pytest.raises(BudgetExceeded, match="run budget"):
        spent.check("perturbed holdout")


def test_the_budget_is_unset_by_default_and_validated(monkeypatch):
    monkeypatch.delenv("RELEARN_RUN_BUDGET_SECS", raising=False)
    assert budget_seconds() == 0.0
    monkeypatch.setenv("RELEARN_RUN_BUDGET_SECS", "600")
    assert budget_seconds() == 600.0
    monkeypatch.setenv("RELEARN_RUN_BUDGET_SECS", "soon")
    with pytest.raises(Exception, match="not a number"):
        budget_seconds()
