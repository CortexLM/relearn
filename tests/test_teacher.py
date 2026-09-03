"""The judge: configured by the operator, guarded against serving weights."""

from __future__ import annotations

import json
import urllib.request

import pytest

from relearn_eval.teacher import (
    DEFAULT_TEACHER_MAX_TOKENS,
    TEACHER_MODEL_ID,
    HttpTeacher,
    TeacherError,
    build_teacher,
    extract_judge_text,
    guard_judge_call,
    is_configured_teacher_model,
    parse_score,
    teacher_max_tokens,
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


def test_str_none_is_never_parsed_as_a_score():
    # Live 201cc5d2: content was JSON null, the client did str(None)=="None",
    # and parse_score("None") raised "teacher reply carried no score".
    assert extract_judge_text({"content": None}) == ""
    assert extract_judge_text({"content": None, "reasoning": None}) == ""
    with pytest.raises(TeacherError, match="no score"):
        parse_score("None")
    with pytest.raises(TeacherError, match="no score"):
        parse_score(str(None))
    with pytest.raises(TeacherError, match="no score"):
        parse_score("null")


def test_null_content_uses_reasoning_that_carries_a_score():
    text = extract_judge_text(
        {"content": None, "reasoning": '{"score": 0.7}', "reasoning_content": ""}
    )
    assert text != "None"
    assert parse_score(text) == 0.7


def test_null_content_and_empty_reasoning_is_not_a_score():
    assert extract_judge_text({"content": None, "reasoning": "", "reasoning_content": None}) == ""


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def test_judge_reads_a_score_from_reasoning_when_content_is_null(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning": '{"score": 0.7}',
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    teacher = HttpTeacher(api_url="http://teacher.invalid/v1", model=TEACHER_MODEL_ID)
    assert teacher.judge("what is 2+2?", "4") == 0.7
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["max_tokens"] == DEFAULT_TEACHER_MAX_TOKENS
    assert body["reasoning_effort"] == "low"
    assert "enable_thinking" not in body
    assert "thinking" not in body


def test_judge_names_finish_reason_when_reasoning_is_also_empty(monkeypatch):
    def fake_urlopen(_request: urllib.request.Request, timeout: float | None = None):
        del timeout
        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": None,
                            "reasoning": "",
                            "reasoning_content": None,
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    teacher = HttpTeacher(api_url="http://teacher.invalid/v1", model=TEACHER_MODEL_ID)
    with pytest.raises(TeacherError, match=r"finish_reason='length'"):
        teacher.judge("what is 2+2?", "4")


def test_teacher_max_tokens_can_be_overridden(monkeypatch):
    monkeypatch.delenv("RELEARN_TEACHER_MAX_TOKENS", raising=False)
    assert teacher_max_tokens() == 1024
    monkeypatch.setenv("RELEARN_TEACHER_MAX_TOKENS", "2048")
    assert teacher_max_tokens() == 2048
    monkeypatch.setenv("RELEARN_TEACHER_MAX_TOKENS", "0")
    with pytest.raises(TeacherError, match="must be positive"):
        teacher_max_tokens()


def test_the_judge_never_serves_a_weights_payload_over_http():
    teacher = HttpTeacher(api_url="http://teacher.invalid/v1", model=TEACHER_MODEL_ID)
    # Guarded before any socket is opened, so an unreachable host is not what
    # this test is asserting.
    with pytest.raises(TeacherError, match="not a teacher payload"):
        teacher.judge("score this", "model-00001-of-00097.safetensors")
