"""A document is only a score when it is bound to the run that asked for it.

Mirrors `relearn-lium-harvest`'s refusals: another artifact's numbers, a
replayed run, a document from a different image or a different holdout, a
document missing a series the gates read, and silence. Each of those is a 503 on
the control plane, so each of them must be something this image cannot emit.
"""

from __future__ import annotations

import dataclasses

import pytest

from relearn_eval import (
    OK_MARKER,
    ContractError,
    EvalDocument,
    Measurement,
    ShuffleEvidence,
    TranscriptError,
    VerificationError,
    accept,
    marker_line,
    verify_document,
)

from .conftest import holdout_items, make_request


def measurement(request, level: float = 0.61) -> Measurement:
    return Measurement(
        eval_image_digest=request.eval_image_digest,
        holdout_commitment=request.holdout_commitment,
        holdout={f"h{item.id}": level for item in request.holdout},
        public={f"p{index}": level + 0.01 for index in range(1, 41)},
        perturbed={f"x{item.id}": level - 0.01 for item in request.holdout},
        canaries={f"c{index}": 0.98 for index in range(1, 41)},
        general_canary={f"g{index}": 0.97 for index in range(1, 41)},
        agent_trace=0.85,
    )


def document(request, level: float = 0.61, **overrides: object) -> EvalDocument:
    return EvalDocument(
        submission_digest=request.submission_digest,
        artifact_digest=request.artifact_digest,
        measurement=measurement(request, level),
        **overrides,  # type: ignore[arg-type]
    )


def transcript(doc: EvalDocument, *, ok: bool = True) -> str:
    lines = ["boot ok", marker_line(doc)]
    if ok:
        lines.append(OK_MARKER)
    return "\n".join(lines) + "\n"


def test_a_bound_document_is_accepted():
    request = make_request()
    accepted = accept(transcript(document(request)), request)
    assert accepted.measurement.holdout[f"h{request.holdout[0].id}"] == 0.61
    assert len(accepted.measurement.holdout) == len(request.holdout)


def test_another_artifacts_numbers_are_refused():
    request = make_request(artifact_digest="ab" * 32)
    other = document(make_request(artifact_digest="cd" * 32), level=0.99)
    with pytest.raises(VerificationError, match="artifact_digest"):
        accept(transcript(other), request)


def test_an_earlier_runs_document_is_refused():
    request = make_request(submission_digest="frozen-1")
    replay = document(make_request(submission_digest="an-earlier-run"), level=0.99)
    with pytest.raises(VerificationError, match="submission_digest"):
        accept(transcript(replay), request)


def test_a_document_from_another_image_or_holdout_is_refused():
    request = make_request()
    other_image = document(request)
    other_image = dataclasses.replace(
        other_image,
        measurement=dataclasses.replace(
            other_image.measurement, eval_image_digest=f"sha256:{'cd' * 32}"
        ),
    )
    with pytest.raises(VerificationError, match="eval_image_digest"):
        verify_document(other_image, request)

    other_holdout = document(request)
    other_holdout = dataclasses.replace(
        other_holdout,
        measurement=dataclasses.replace(other_holdout.measurement, holdout_commitment="ff" * 32),
    )
    with pytest.raises(VerificationError, match="holdout_commitment"):
        verify_document(other_holdout, request)


def test_a_wrong_schema_version_is_refused():
    request = make_request()
    with pytest.raises(VerificationError, match="schema_version"):
        verify_document(document(request, schema_version=2), request)


def test_a_short_holdout_or_a_relabelled_one_is_refused():
    request = make_request()
    short = document(request)
    trimmed = dict(short.measurement.holdout)
    trimmed.pop(next(iter(trimmed)))
    with pytest.raises(VerificationError, match="holdout scores for"):
        verify_document(
            dataclasses.replace(
                short, measurement=dataclasses.replace(short.measurement, holdout=trimmed)
            ),
            request,
        )

    relabelled = document(request)
    renamed = {f"h{9000 + index}": value for index, value in enumerate(
        relabelled.measurement.holdout.values()
    )}
    with pytest.raises(VerificationError, match="not the requested item ids"):
        verify_document(
            dataclasses.replace(
                relabelled,
                measurement=dataclasses.replace(relabelled.measurement, holdout=renamed),
            ),
            request,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("public", {}, "no public split"),
        ("general_canary", {}, "general-bench canary"),
        ("agent_trace", 7.0, r"outside \[0, 1\]"),
        ("agent_trace", float("nan"), r"outside \[0, 1\]"),
    ],
)
def test_a_document_the_gates_cannot_read_is_refused(field: str, value: object, match: str):
    request = make_request()
    doc = document(request)
    broken = dataclasses.replace(
        doc, measurement=dataclasses.replace(doc.measurement, **{field: value})
    )
    with pytest.raises(VerificationError, match=match):
        verify_document(broken, request)


def test_a_series_key_that_could_carry_holdout_text_is_refused():
    request = make_request()
    doc = document(request)
    leaky = dict(doc.measurement.public)
    leaky[request.holdout[0].prompt] = 0.5
    with pytest.raises(VerificationError, match="is not a series key"):
        verify_document(
            dataclasses.replace(
                doc, measurement=dataclasses.replace(doc.measurement, public=leaky)
            ),
            request,
        )


def test_an_out_of_range_score_is_refused():
    request = make_request()
    doc = document(request)
    hot = dict(doc.measurement.public)
    hot["p1"] = 1.4
    with pytest.raises(VerificationError, match=r"outside \[0, 1\]"):
        verify_document(
            dataclasses.replace(
                doc, measurement=dataclasses.replace(doc.measurement, public=hot)
            ),
            request,
        )


def test_the_pixel_shuffle_control_must_match_the_holdouts_vision_families():
    request = make_request(holdout_items(20, vision=True))
    doc = document(request)

    with pytest.raises(VerificationError, match="without a pixel-shuffle control"):
        verify_document(doc, request)

    complete = dataclasses.replace(
        doc,
        measurement=dataclasses.replace(
            doc.measurement,
            vision_shuffle={
                family: ShuffleEvidence(items=4, score=0.6, shuffled_score=0.45)
                for family in request.vision_families()
            },
        ),
    )
    verify_document(complete, request)

    text_only = make_request()
    invented = dataclasses.replace(
        document(text_only),
        measurement=dataclasses.replace(
            document(text_only).measurement,
            vision_shuffle={"ocr": ShuffleEvidence(items=1, score=0.6, shuffled_score=0.1)},
        ),
    )
    with pytest.raises(VerificationError, match="not in this holdout"):
        verify_document(invented, text_only)


def test_silence_is_never_a_score():
    request = make_request()
    for body in ["", "boot ok\nsegfault\n", f"{OK_MARKER}\n", "cuda out of memory\n"]:
        with pytest.raises(ContractError):
            accept(body, request)


def test_a_document_without_the_ok_marker_is_never_a_score():
    request = make_request()
    with pytest.raises(TranscriptError, match=OK_MARKER):
        accept(transcript(document(request), ok=False), request)


def test_the_ok_marker_alone_is_never_a_score():
    request = make_request()
    with pytest.raises(TranscriptError, match="printed no"):
        accept(f"boot ok\n{OK_MARKER}\n", request)
