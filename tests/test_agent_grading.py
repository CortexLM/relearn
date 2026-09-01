"""Grading an action against the action that was recorded.

None of this involves a judge. The episode was recorded, so the tool that was
called and the arguments it was called with are known exactly, and asking a
language model whether an action "looks reasonable" would put the judge's
variance back into a measurement that does not need it.
"""

from __future__ import annotations

import json

import pytest

from relearn_agent_eval.grading import (
    argument_agreement,
    grade_reference_answer,
    grade_step,
    order_fidelity,
    parse_action,
)
from relearn_agent_eval.pins import TOOL_NAME_CREDIT
from relearn_agent_eval.request import TraceStep

TOOLS = ("lookup_record", "file_report", "search_docs")


def step(tool: str = "lookup_record", **arguments: object) -> TraceStep:
    return TraceStep(
        index=0,
        tool=tool,
        arguments_json=json.dumps(arguments, separators=(",", ":"), sort_keys=True),
    )


def test_reads_a_bare_tool_call() -> None:
    action = parse_action('{"tool": "lookup_record", "arguments": {"record_id": "REC-1"}}')
    assert action.tool == "lookup_record"
    assert action.arguments == {"record_id": "REC-1"}


def test_reads_a_call_after_a_thinking_preamble() -> None:
    # A post-trained model reasons before it acts, and the reasoning contains
    # braces of its own. The action is what it emits last.
    reply = (
        'I should look this up first. Something like {"tool": "search_docs"} would be wrong.\n'
        '{"tool": "lookup_record", "arguments": {"record_id": "REC-1"}}'
    )
    assert parse_action(reply).tool == "lookup_record"


def test_reads_the_openai_function_envelope() -> None:
    # The challenge is about choosing the right action, not about guessing
    # which JSON dialect the harness wanted.
    reply = '{"function": {"name": "file_report", "arguments": "{\\"owner\\": \\"ada\\"}"}}'
    action = parse_action(reply)
    assert action.tool == "file_report"
    assert action.arguments == {"owner": "ada"}


def test_reads_the_name_spelling() -> None:
    assert parse_action('{"name": "file_report", "parameters": {}}').tool == "file_report"


def test_braces_inside_an_argument_value_do_not_truncate_the_call() -> None:
    action = parse_action('{"tool": "search_docs", "arguments": {"query": "a {weird} string"}}')
    assert action.arguments == {"query": "a {weird} string"}


@pytest.mark.parametrize(
    "reply",
    ["", "I will look up the record.", "```json\nnot json\n```", "[1, 2, 3]", '{"tool": ""}'],
)
def test_an_unreadable_reply_is_malformed(reply: str) -> None:
    assert parse_action(reply).malformed


def test_an_exact_call_scores_full_marks() -> None:
    verdict = grade_step(
        step("lookup_record", record_id="REC-1"),
        parse_action('{"tool": "lookup_record", "arguments": {"record_id": "REC-1"}}'),
        TOOLS,
    )
    assert verdict.score == pytest.approx(1.0)
    assert not verdict.invalid


def test_the_right_tool_with_wrong_arguments_is_worth_the_name_credit() -> None:
    # Closer than picking the wrong tool entirely, and the score says so.
    verdict = grade_step(
        step("lookup_record", record_id="REC-1"),
        parse_action('{"tool": "lookup_record", "arguments": {"record_id": "REC-9"}}'),
        TOOLS,
    )
    assert verdict.score == pytest.approx(TOOL_NAME_CREDIT)
    assert not verdict.invalid


def test_the_wrong_tool_scores_zero_but_is_not_invalid() -> None:
    verdict = grade_step(
        step("lookup_record", record_id="REC-1"),
        parse_action('{"tool": "search_docs", "arguments": {"query": "REC-1"}}'),
        TOOLS,
    )
    assert verdict.score == 0.0
    assert not verdict.invalid


def test_a_hallucinated_tool_is_invalid() -> None:
    # Naming something outside the episode's own schema is a different failure
    # from picking the wrong one of the tools that exist: the model is not
    # playing the game, and the invalid-action rate has to see it.
    verdict = grade_step(
        step("lookup_record", record_id="REC-1"),
        parse_action('{"tool": "sudo_fix_everything", "arguments": {}}'),
        TOOLS,
    )
    assert verdict.score == 0.0
    assert verdict.invalid


def test_a_malformed_reply_is_invalid() -> None:
    verdict = grade_step(step("lookup_record"), parse_action("no idea"), TOOLS)
    assert verdict.invalid
    assert verdict.proposed_tool == ""


def test_argument_agreement_measures_both_halves_of_being_wrong() -> None:
    recorded = {"record_id": "REC-1", "owner": "ada"}
    assert argument_agreement(recorded, recorded) == pytest.approx(1.0)
    # Omitting one costs recall.
    assert 0.0 < argument_agreement(recorded, {"record_id": "REC-1"}) < 1.0
    # Inventing extras costs precision.
    padded = {**recorded, "extra": 1, "more": 2}
    assert 0.0 < argument_agreement(recorded, padded) < 1.0
    assert argument_agreement(recorded, {}) == 0.0
    assert argument_agreement({}, {}) == pytest.approx(1.0)


def test_argument_values_compare_past_formatting() -> None:
    assert argument_agreement({"service": "payments"}, {"service": " Payments "}) == 1.0
    assert argument_agreement({"quantity": 3}, {"quantity": 3.0}) == 1.0
    assert argument_agreement({"on": True}, {"on": "true"}) == 1.0
    assert argument_agreement({"service": "payments"}, {"service": "search"}) == 0.0


def test_order_fidelity_rewards_acting_in_the_recorded_order() -> None:
    recorded = [step("lookup_record"), step("file_report")]
    assert order_fidelity(recorded, ["lookup_record", "file_report"]) == pytest.approx(1.0)


def test_order_fidelity_catches_jumping_ahead() -> None:
    # Filing the report before looking anything up.
    recorded = [step("lookup_record"), step("file_report")]
    assert order_fidelity(recorded, ["file_report", "file_report"]) == pytest.approx(0.5)


def test_a_tool_the_episode_never_uses_does_not_also_sink_the_order_series() -> None:
    # It is already scored as a wrong action; counting it twice would let one
    # mistake sink two series.
    recorded = [step("lookup_record"), step("file_report")]
    assert order_fidelity(recorded, ["search_docs", "file_report"]) == pytest.approx(1.0)


def test_never_naming_a_recorded_tool_is_zero_not_one() -> None:
    # Absent evidence of ordering is not credit for ordering.
    recorded = [step("lookup_record"), step("file_report")]
    assert order_fidelity(recorded, ["search_docs", "search_docs"]) == 0.0


def test_canary_answers_are_matched_not_judged() -> None:
    assert grade_reference_answer("the answer is 144", "144") == 1.0
    assert grade_reference_answer("Tokyo", "tokyo") == 1.0
    assert grade_reference_answer("Kyoto", "Tokyo") == 0.0
    assert grade_reference_answer("", "Tokyo") == 0.0
    assert grade_reference_answer("H2O", "water", aliases=("h2o",)) == 1.0
