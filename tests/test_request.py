"""What the image will and will not accept as a run request."""

from __future__ import annotations

import json

import pytest

from relearn_eval import HarvestRequest, HoldoutItem, RequestError
from relearn_eval.commitment import holdout_commitment
from relearn_eval.contract import BASE_CHAMPION_ARTIFACT

from .conftest import IMAGE_DIGEST, holdout_items, make_request


def test_a_well_formed_request_validates(request_fixture: HarvestRequest):
    request_fixture.validate()
    assert len(request_fixture.holdout) == 120
    assert request_fixture.vision_families() == ()


def test_request_round_trips_through_json(request_fixture: HarvestRequest):
    body = json.dumps(request_fixture.to_wire())
    parsed = HarvestRequest.from_json(body)
    parsed.validate()
    assert parsed == request_fixture


def test_unknown_fields_are_tolerated(request_fixture: HarvestRequest):
    # The control plane may grow the request without a new image.
    wire = request_fixture.to_wire()
    wire["something_new"] = {"added": "later"}
    wire["artifact_uri"] = "https://store.example/artifact.tar"
    parsed = HarvestRequest.from_json(json.dumps(wire))
    parsed.validate()
    assert parsed.artifact_uri == "https://store.example/artifact.tar"


def test_a_tampered_holdout_is_refused():
    request = make_request()
    edited = list(request.holdout)
    edited[0] = HoldoutItem(
        id=edited[0].id, prompt="a different question", dataset_id=edited[0].dataset_id
    )
    tampered = HarvestRequest(
        schema_version=request.schema_version,
        submission_digest=request.submission_digest,
        artifact_digest=request.artifact_digest,
        base_model=request.base_model,
        teacher_model=request.teacher_model,
        eval_image_digest=request.eval_image_digest,
        holdout_commitment=request.holdout_commitment,
        holdout=tuple(edited),
    )
    with pytest.raises(RequestError, match="commitment mismatch"):
        tampered.validate()


def test_the_mismatch_message_carries_no_holdout_text():
    request = make_request()
    broken = HarvestRequest(**{**request.__dict__, "holdout_commitment": "aa" * 32})
    with pytest.raises(RequestError) as raised:
        broken.validate()
    message = str(raised.value)
    assert holdout_commitment(request.holdout) in message
    for item in request.holdout:
        assert item.prompt not in message


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "schema_version"),
        ("submission_digest", "", "submission_digest"),
        ("artifact_digest", "", "artifact_digest"),
        ("base_model", "", "base_model"),
        ("teacher_model", "", "teacher_model"),
        ("eval_image_digest", "", "sha256"),
        ("eval_image_digest", "latest", "sha256"),
        ("eval_image_digest", "sha256:short", "sha256"),
        ("holdout", (), "no holdout items"),
        ("holdout_commitment", "not-hex", "64 hex"),
    ],
)
def test_an_unusable_request_is_refused(field: str, value: object, match: str):
    request = make_request()
    broken = HarvestRequest(**{**request.__dict__, field: value})
    with pytest.raises(RequestError, match=match):
        broken.validate()


def test_duplicate_and_empty_holdout_items_are_refused():
    duplicated = (
        HoldoutItem(id=1, prompt="first question here"),
        HoldoutItem(id=1, prompt="second question here"),
    )
    request = make_request(duplicated)
    with pytest.raises(RequestError, match="duplicate holdout id"):
        request.validate()

    blank = (HoldoutItem(id=1, prompt="   "),)
    with pytest.raises(RequestError, match="empty prompt"):
        make_request(blank).validate()


def test_a_vision_item_without_an_image_hash_is_refused():
    items = (HoldoutItem(id=7, prompt="caption this", task="captioning", image_hash=""),)
    with pytest.raises(RequestError, match="without an image hash"):
        make_request(items).validate()

    unknown = (HoldoutItem(id=7, prompt="answer this", task="interpretive-dance"),)
    with pytest.raises(RequestError, match="unknown task"):
        make_request(unknown).validate()


def test_vision_families_are_reported_in_contract_order():
    request = make_request(holdout_items(20, vision=True))
    request.validate()
    assert request.vision_families() == ("captioning", "vqa", "ocr", "spatial")


def test_the_boot_baseline_run_is_recognised():
    request = make_request(artifact_digest=BASE_CHAMPION_ARTIFACT)
    request.validate()
    assert request.is_base_champion_run
    assert not make_request().is_base_champion_run
    assert request.eval_image_digest == IMAGE_DIGEST


def test_garbage_is_not_a_request():
    with pytest.raises(RequestError, match="not JSON"):
        HarvestRequest.from_json("nope")
    with pytest.raises(RequestError, match="not an object"):
        HarvestRequest.from_json("[1, 2, 3]")
    with pytest.raises(RequestError, match="not a list"):
        HarvestRequest.from_json(json.dumps({"holdout": "everything"}))
