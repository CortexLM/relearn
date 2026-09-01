"""Scoring a Relearn Agent run, and catching a model that ignores its tools.

The two tests that matter most here are
`test_a_model_that_reads_observations_loses_ground_when_they_are_withheld` and
its mirror. They are the reason this challenge is a separate image rather than
a slice of the language one: both models below write well-formed actions and
both would look fine to a prose grader, and only the tool-blind replay
separates them.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from relearn_agent_eval.contract import AgentDocument
from relearn_agent_eval.pins import (
    MAX_INVALID_ACTION_RATE,
    MIN_BLIND_DROP,
    TRACE_ACTION_WEIGHT,
    TRACE_ANSWER_WEIGHT,
    TRACE_ORDER_WEIGHT,
    WITHHELD_OBSERVATION,
)
from relearn_agent_eval.replay import Mode, answer_prompt, step_prompts
from relearn_agent_eval.scoring import ScoringError, Slices, composite, score_request
from relearn_agent_eval.verify import VerificationError, verify_document
from relearn_common.transcript import TranscriptError, accept
from relearn_common.wire import OK_MARKER, decode_line, encode_line, marker_line
from tests.support import FakeAnswerTeacher, ScriptedAgent
from tests.tools.make_agent_request import build

IMAGE_DIGEST = f"sha256:{'ab' * 32}"


def make_request(traces: int = 100, **overrides: object):
    request = build(
        traces, submission="frozen-1", artifact="ef" * 32, image_digest=IMAGE_DIGEST
    )
    return dataclasses.replace(request, **overrides) if overrides else request


def score(agent: ScriptedAgent, request=None, teacher: FakeAnswerTeacher | None = None):
    return score_request(
        request or make_request(),
        agent,
        teacher or FakeAnswerTeacher(),
        Slices.load(),
    )


def test_scores_every_requested_episode_and_binds_the_run() -> None:
    request = make_request()
    document = score(ScriptedAgent(), request)
    measured = document.measurement
    assert len(measured.holdout) == 100
    assert len(measured.tool_call) == 100
    assert len(measured.order) == 100
    assert document.identity == request.identity
    verify_document(document, request)


def test_a_model_that_reads_observations_loses_ground_when_they_are_withheld() -> None:
    # The `owner` argument of the second step exists only in the first step's
    # observation. A model that read it can produce it, and cannot once it is
    # withheld, so the drop is real.
    document = score(ScriptedAgent(reads_observations=True))
    blind = document.measurement.tool_blind
    assert blind.traces == 100
    assert blind.drop >= MIN_BLIND_DROP, blind


def test_a_model_that_ignores_observations_does_not_move_and_is_caught() -> None:
    # Same well-formed actions, same tool names, same order — and no drop when
    # the observations go away, because it was never using them. A low score
    # would not catch this: a uniformly mediocre model can still beat a
    # champion on noise. The flat control is what catches it.
    document = score(ScriptedAgent(reads_observations=False))
    blind = document.measurement.tool_blind
    assert blind.drop == pytest.approx(0.0), blind
    assert blind.drop < MIN_BLIND_DROP


def test_the_blind_replay_withholds_the_observation_and_nothing_else() -> None:
    # The prompt must stay the same shape, or a drop could be attributed to the
    # context having become a different length rather than to the missing
    # information.
    trace = make_request(traces=1).holdout[0]
    real = step_prompts(trace, Mode.REAL, "t")[1].prompt.text
    blind = step_prompts(trace, Mode.BLIND, "t")[1].prompt.text
    assert "Observation:" in real and "Observation:" in blind
    assert WITHHELD_OBSERVATION in blind
    assert '"owner"' in real and '"owner"' not in blind.partition("So far:")[2]
    assert trace.goal in real and trace.goal in blind
    assert real.count("Action: ") == blind.count("Action: ")


def test_each_step_is_replayed_on_the_recorded_prefix() -> None:
    # Not off-policy: there is no live environment on the pod, so an unrecorded
    # action would have no observation to return.
    trace = make_request(traces=1).holdout[0]
    prompts = step_prompts(trace, Mode.REAL, "t")
    assert len(prompts) == len(trace.steps)
    assert "(nothing yet)" in prompts[0].prompt.text
    assert trace.steps[0].observation in prompts[1].prompt.text


def test_the_final_answer_is_asked_on_the_complete_recording() -> None:
    trace = make_request(traces=1).holdout[0]
    prompt = answer_prompt(trace, "t")
    for step in trace.steps:
        assert step.observation in prompt.text


def test_a_model_that_cannot_emit_an_action_voids_the_run() -> None:
    # Whatever the surviving steps scored is not a measurement of tool use.
    with pytest.raises(ScoringError, match="invalid action rate"):
        score(ScriptedAgent(malformed_every=1))
    assert MAX_INVALID_ACTION_RATE == 0.25


def test_series_keys_carry_ids_and_never_episode_text() -> None:
    request = make_request()
    document = score(ScriptedAgent(), request)
    encoded = encode_line(document.to_wire())
    for trace in request.holdout[:5]:
        assert trace.goal not in encoded
        assert trace.final_answer not in encoded
        for step in trace.steps:
            assert step.observation not in encoded
    assert all(key.startswith("t") for key in document.measurement.holdout)
    assert all(key.startswith("s") for key in document.measurement.tool_call)
    assert all(key.startswith("o") for key in document.measurement.order)


def test_the_public_split_and_the_canaries_are_carried() -> None:
    document = score(ScriptedAgent())
    # Without a public split there is no memorization gap to measure, and
    # without canaries a model that has lost the ability to emit a tool call
    # would be invisible to a purely relative comparison.
    assert document.measurement.public
    assert document.measurement.canaries


def test_the_composite_is_the_pinned_weighting() -> None:
    assert TRACE_ACTION_WEIGHT + TRACE_ORDER_WEIGHT + TRACE_ANSWER_WEIGHT == pytest.approx(1.0)
    assert composite(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert composite(0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert composite(1.0, 0.0, 0.0) == pytest.approx(TRACE_ACTION_WEIGHT)


def test_the_document_survives_its_own_encoding() -> None:
    document = score(ScriptedAgent())
    again = AgentDocument.from_wire(decode_line(encode_line(document.to_wire())))
    assert encode_line(again.to_wire()) == encode_line(document.to_wire())


def test_a_transcript_this_image_produces_is_one_the_control_plane_accepts() -> None:
    request = make_request()
    document = score(ScriptedAgent(), request)
    transcript = f"boot ok\n{marker_line(document.to_wire())}\n{OK_MARKER}\n"
    accepted = accept(
        transcript, AgentDocument.from_wire, lambda parsed: verify_document(parsed, request)
    )
    assert accepted.identity == request.identity


def test_a_document_for_another_run_is_refused() -> None:
    request = make_request()
    document = score(ScriptedAgent(), request)
    for field, value in (
        ("submission_digest", "an-earlier-run"),
        ("artifact_digest", "cd" * 32),
        ("eval_image_digest", f"sha256:{'cd' * 32}"),
        ("holdout_commitment", "11" * 32),
        ("challenge_id", "relearn"),
    ):
        impostor = dataclasses.replace(
            document, identity=dataclasses.replace(document.identity, **{field: value})
        )
        with pytest.raises(VerificationError):
            verify_document(impostor, request)


def test_a_document_with_no_tool_blind_control_is_refused() -> None:
    # This is the check that stops the challenge from quietly becoming a
    # prose eval: without the control, a model that ignores its tools is
    # indistinguishable from one that reads them.
    request = make_request()
    document = score(ScriptedAgent(), request)
    stripped = dataclasses.replace(
        document,
        measurement=dataclasses.replace(
            document.measurement,
            tool_blind=dataclasses.replace(document.measurement.tool_blind, traces=0),
        ),
    )
    with pytest.raises(VerificationError, match="tool-blind"):
        verify_document(stripped, request)


def test_silence_is_never_a_score() -> None:
    request = make_request()
    for body in ("", "boot ok\nsegfault\n", f"{OK_MARKER}\n"):
        with pytest.raises(TranscriptError):
            accept(
                body, AgentDocument.from_wire, lambda parsed: verify_document(parsed, request)
            )


def test_the_teacher_only_ever_sees_the_final_answer() -> None:
    teacher = FakeAnswerTeacher()
    request = make_request(traces=100)
    score(ScriptedAgent(), request, teacher)
    # One call per holdout episode plus one per public episode, and never one
    # per step: every action is graded against the recording instead.
    assert len(teacher.calls) == len(request.holdout) + len(Slices.load().public)
    goals = {goal for goal, _results, _answer in teacher.calls}
    assert request.holdout[0].goal in goals


def test_arguments_are_canonical_regardless_of_how_they_were_exported() -> None:
    # The commitment must not depend on the operator's JSON key ordering.
    from relearn_agent_eval.request import TraceStep

    one = TraceStep.from_wire(0, {"tool": "t", "arguments": {"b": 1, "a": 2}})
    two = TraceStep.from_wire(0, {"tool": "t", "arguments": json.dumps({"a": 2, "b": 1})})
    assert one.arguments_json == two.arguments_json
