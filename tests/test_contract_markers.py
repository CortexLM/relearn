"""The markers and the wire shape the harvest client reads.

`crates/relearn-lium-harvest` finds the document by scanning for the first line
that starts with `RELEARN_METRICS=`, and refuses a run that never printed
`RELEARN_EVAL_OK`. It also harvests the sidecar with

    printf 'RELEARN_METRICS='; cat metrics.json; printf '\\n'

so the encoding rules here are load-bearing, not cosmetic.
"""

from __future__ import annotations

import json

import pytest

from relearn_eval import (
    METRICS_MARKER,
    METRICS_SCHEMA_VERSION,
    OK_MARKER,
    ContractError,
    EvalDocument,
    Measurement,
    ShuffleEvidence,
    decode_document,
    encode_document,
    extract_metrics,
    has_ok_marker,
    marker_line,
)
from relearn_eval.contract import POD_WORKDIR, clamp_score

from .conftest import IMAGE_DIGEST


def document(**overrides: object) -> EvalDocument:
    measurement = Measurement(
        eval_image_digest=IMAGE_DIGEST,
        holdout_commitment="ab" * 32,
        holdout={"h1": 0.5},
        public={"p1": 0.5},
        perturbed={"x1": 0.49},
        canaries={"c1": 1.0},
        general_canary={"g1": 0.97},
        agent_trace=0.85,
        vision_shuffle={"captioning": ShuffleEvidence(items=4, score=0.6, shuffled_score=0.4)},
    )
    return EvalDocument(
        submission_digest="frozen-1",
        artifact_digest="ab" * 32,
        measurement=measurement,
        **overrides,  # type: ignore[arg-type]
    )


def test_the_markers_are_the_control_planes():
    assert METRICS_MARKER == "RELEARN_METRICS="
    assert OK_MARKER == "RELEARN_EVAL_OK"
    assert POD_WORKDIR == "/tmp/relearn_eval"  # noqa: S108 - the staging path
    assert METRICS_SCHEMA_VERSION == 1


def test_the_document_is_one_line_with_the_measurement_flattened():
    encoded = encode_document(document())
    assert "\n" not in encoded and "\r" not in encoded
    body = json.loads(encoded)
    # `#[serde(flatten)]`: the measurement's fields sit beside the identity.
    assert set(body) == {
        "schema_version",
        "submission_digest",
        "artifact_digest",
        "eval_image_digest",
        "holdout_commitment",
        "holdout",
        "public",
        "perturbed",
        "canaries",
        "general_canary",
        "agent_trace",
        "vision_shuffle",
    }
    assert body["schema_version"] == METRICS_SCHEMA_VERSION
    assert body["vision_shuffle"]["captioning"] == {
        "items": 4,
        "score": 0.6,
        "shuffled_score": 0.4,
    }


def test_the_marker_line_is_the_prefix_plus_the_document():
    line = marker_line(document())
    assert line.startswith(METRICS_MARKER)
    assert decode_document(line[len(METRICS_MARKER) :]) == document()


def test_the_document_survives_a_round_trip():
    assert decode_document(encode_document(document())) == document()


def test_a_transcript_is_read_by_prefix_not_by_position():
    transcript = (
        "boot ok\n"
        "loading weights\n"
        f"{marker_line(document())}\n"
        f"{OK_MARKER}\n"
        "tail of the run log\n"
    )
    assert extract_metrics(transcript) == document()
    assert has_ok_marker(transcript)


def test_a_bare_document_is_accepted_like_the_control_plane_does():
    assert extract_metrics(encode_document(document())) == document()


def test_a_transcript_without_a_document_is_refused():
    for body in ["", "boot ok\nsegfault\n", f"{OK_MARKER}\n", "not json at all"]:
        with pytest.raises(ContractError):
            extract_metrics(body)


def test_the_ok_marker_must_be_its_own_line():
    assert not has_ok_marker(f"we did not print {OK_MARKER} really\n")
    assert has_ok_marker(f"{OK_MARKER}   \n")


def test_scores_are_rounded_and_clamped_but_never_invented():
    assert clamp_score(1.5, what="x") == 1.0
    assert clamp_score(-0.2, what="x") == 0.0
    assert clamp_score(0.1234567891, what="x") == 0.123457
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ContractError):
            clamp_score(bad, what="x")


def test_series_keys_are_sorted_so_two_runs_encode_identically():
    first = document()
    unsorted = Measurement(
        eval_image_digest=IMAGE_DIGEST,
        holdout_commitment="ab" * 32,
        holdout={"h2": 0.5, "h1": 0.4},
        public={"p1": 0.5},
        perturbed={},
        canaries={},
        general_canary={"g1": 0.9},
        agent_trace=0.5,
    )
    reordered = Measurement(
        eval_image_digest=IMAGE_DIGEST,
        holdout_commitment="ab" * 32,
        holdout={"h1": 0.4, "h2": 0.5},
        public={"p1": 0.5},
        perturbed={},
        canaries={},
        general_canary={"g1": 0.9},
        agent_trace=0.5,
    )
    assert json.dumps(unsorted.to_wire()) == json.dumps(reordered.to_wire())
    assert encode_document(first) == encode_document(document())
