"""The judge: configured by the operator, guarded against serving weights."""

from __future__ import annotations

import pytest

from relearn_eval.teacher import (
    TEACHER_MODEL_ID,
    HttpTeacher,
    TeacherError,
    build_teacher,
    guard_judge_call,
    is_configured_teacher_model,
    parse_score,
)


def test_the_default_wire_id_is_the_pinned_one():
    assert TEACHER_MODEL_ID == "glm-5.3"
    assert is_configured_teacher_model("glm-5.3")
    assert is_configured_teacher_model("zai-org/GLM-5.3")
    assert not is_configured_teacher_model("")
    assert not is_configured_teacher_model("Qwen/Qwen3.8-27B")


def test_an_unconfigured_pod_has_no_judge(monkeypatch):
    monkeypatch.delenv("RELEARN_TEACHER_API_URL", raising=False)
    with pytest.raises(TeacherError, match="RELEARN_TEACHER_API_URL is unset"):
        build_teacher(TEACHER_MODEL_ID)


def test_the_judge_is_built_from_the_environment(monkeypatch):
    monkeypatch.setenv("RELEARN_TEACHER_API_URL", "http://teacher.invalid/v1/")
    monkeypatch.setenv("RELEARN_TEACHER_API_KEY", "operator-secret")
    teacher = build_teacher(TEACHER_MODEL_ID)
    assert teacher.api_url == "http://teacher.invalid/v1"
    assert teacher.model == TEACHER_MODEL_ID
    assert teacher.api_key == "operator-secret"


def test_an_operator_override_selects_the_wire_id(monkeypatch):
    monkeypatch.setenv("RELEARN_TEACHER_API_URL", "http://teacher.invalid/v1")
    monkeypatch.setenv("RELEARN_TEACHER_MODEL", "glm-5.3-operator-alias")
    assert build_teacher("glm-5.3").model == "glm-5.3-operator-alias"


def test_an_unknown_model_is_never_called(monkeypatch):
    monkeypatch.setenv("RELEARN_TEACHER_API_URL", "http://teacher.invalid/v1")
    monkeypatch.delenv("RELEARN_TEACHER_MODEL", raising=False)
    with pytest.raises(TeacherError, match="not a configured teacher"):
        build_teacher("some-other-model")


def test_the_guard_refuses_miner_weights():
    with pytest.raises(TeacherError, match="not a teacher model"):
        guard_judge_call("ab" * 32, "an answer")
    for payload in ("here is a safetensors blob", "model.gguf", "an nvfp4 checkpoint"):
        with pytest.raises(TeacherError, match="not a teacher payload"):
            guard_judge_call(TEACHER_MODEL_ID, payload)
    guard_judge_call(TEACHER_MODEL_ID, "the capital is paris")


def test_a_judge_reply_is_read_as_a_bounded_score():
    assert parse_score('{"score": 0.75}') == 0.75
    assert parse_score("0.5") == 0.5
    assert parse_score('{"score": 4}') == 1.0
    assert parse_score('{"score": -2}') == 0.0
    assert parse_score("I would say 0.25 out of 1") == 0.25
    with pytest.raises(TeacherError, match="no score"):
        parse_score("cannot judge this")


def test_the_judge_never_serves_a_weights_payload_over_http():
    teacher = HttpTeacher(api_url="http://teacher.invalid/v1", model=TEACHER_MODEL_ID)
    # Guarded before any socket is opened, so an unreachable host is not what
    # this test is asserting.
    with pytest.raises(TeacherError, match="not a teacher payload"):
        teacher.judge("score this", "model-00001-of-00097.safetensors")
