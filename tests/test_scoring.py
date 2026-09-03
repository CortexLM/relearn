"""Scoring produces the series the gates read, from the model's answers.

The runner and the judge here are doubles, so what is under test is the
plumbing: every requested item is asked about exactly once, the keys are ids
rather than prompts, the perturbed pass really is perturbed, and the vision
families in the holdout get a pixel-shuffle control measured against the same
items.
"""

from __future__ import annotations

import hashlib
import io

import pytest

from relearn_eval.images import ImageStoreError, shuffle_pixels
from relearn_eval.perturb import perturb_prompt
from relearn_eval.scoring import ScoringError, Slices, score_request
from relearn_eval.verify import verify_document

from .conftest import FakeRunner, FakeTeacher, holdout_items, make_request


def png(colour: tuple[int, int, int], size: int = 8) -> bytes:
    """A non-uniform image: shuffling a flat colour is a no-op."""
    from PIL import Image

    picture = Image.new("RGB", (size, size), colour)
    pixels = picture.load()
    for x in range(size):
        for y in range(size):
            pixels[x, y] = ((colour[0] + 7 * x) % 256, (colour[1] + 11 * y) % 256, colour[2])
    out = io.BytesIO()
    picture.save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def image_store(tmp_path, monkeypatch):
    root = tmp_path / "images"
    root.mkdir()
    monkeypatch.setenv("RELEARN_IMAGE_STORE", str(root))
    return root


def store_image(root, body: bytes) -> str:
    digest = hashlib.sha256(body).hexdigest()
    (root / digest).write_bytes(body)
    return digest


def test_a_text_holdout_is_scored_into_a_verifiable_document(
    runner: FakeRunner, teacher: FakeTeacher, slices: Slices
):
    request = make_request(holdout_items(12))
    document = score_request(request, runner, teacher, slices)
    verify_document(document, request)

    assert document.submission_digest == request.submission_digest
    assert document.artifact_digest == request.artifact_digest
    assert document.measurement.eval_image_digest == request.eval_image_digest
    assert set(document.measurement.holdout) == {f"h{item.id}" for item in request.holdout}
    assert set(document.measurement.perturbed) == {f"x{item.id}" for item in request.holdout}
    assert len(document.measurement.public) == len(slices.public)
    assert len(document.measurement.canaries) == len(slices.canaries)
    assert len(document.measurement.general_canary) == len(slices.general_canary)
    assert document.measurement.vision_shuffle == {}
    assert all(value == 0.61 for value in document.measurement.holdout.values())


def test_every_requested_item_is_asked_about_once_per_pass(
    runner: FakeRunner, teacher: FakeTeacher, slices: Slices
):
    request = make_request(holdout_items(9))
    score_request(request, runner, teacher, slices)

    holdout_prompts = [
        prompt for prompt in runner.seen if prompt.key.startswith("h")
    ]
    perturbed_prompts = [prompt for prompt in runner.seen if prompt.key.startswith("x")]
    assert len(holdout_prompts) == len(request.holdout)
    assert len(perturbed_prompts) == len(request.holdout)
    assert {prompt.text for prompt in holdout_prompts} == {
        item.prompt for item in request.holdout
    }
    assert {prompt.text for prompt in perturbed_prompts} == {
        perturb_prompt(item.prompt) for item in request.holdout
    }
    # The judge sees the original question, never the rewrite, so the perturbed
    # pass is graded on the same task it was drawn from.
    assert all(
        prompt in {item.prompt for item in request.holdout}
        or prompt in {item.prompt for item in slices.public}
        for prompt, _ in teacher.calls
    )


def test_the_canary_slices_are_graded_against_their_references(slices: Slices):
    canned = {f"c{item.id}": item.answer for item in slices.canaries}
    canned.update({f"g{item.id}": item.answer for item in slices.general_canary})
    runner = FakeRunner(answer="no idea", canned=canned)
    request = make_request(holdout_items(4))
    document = score_request(request, runner, FakeTeacher(), slices)

    assert all(value == 1.0 for value in document.measurement.canaries.values())
    assert all(value == 1.0 for value in document.measurement.general_canary.values())

    wrong = score_request(request, FakeRunner(answer="Z"), FakeTeacher(), slices)
    assert all(value == 0.0 for value in wrong.measurement.canaries.values())


def test_the_agent_trace_score_follows_the_rubric(slices: Slices):
    trace = slices.agent_trace[0]
    runner = FakeRunner(answer="", canned={f"a{trace.id}": " then ".join(trace.must_include)})
    document = score_request(make_request(holdout_items(4)), runner, FakeTeacher(), slices)
    assert 0.0 < document.measurement.agent_trace <= 1.0

    empty = score_request(
        make_request(holdout_items(4)), FakeRunner(answer=""), FakeTeacher(), slices
    )
    assert empty.measurement.agent_trace == 0.0


def test_a_vision_holdout_gets_a_shuffle_control_per_family(
    image_store, teacher: FakeTeacher, slices: Slices
):
    body = png((10, 120, 200))
    digest = store_image(image_store, body)
    items = tuple(
        item.__class__(
            id=item.id,
            prompt=item.prompt,
            dataset_id=item.dataset_id,
            task=task,
            image_hash=digest,
        )
        for item, task in zip(
            holdout_items(4), ("captioning", "vqa", "ocr", "spatial"), strict=True
        )
    )
    request = make_request(items)
    runner = FakeRunner()
    document = score_request(request, runner, teacher, slices)
    verify_document(document, request)

    assert set(document.measurement.vision_shuffle) == set(request.vision_families())
    for evidence in document.measurement.vision_shuffle.values():
        assert evidence.items == 1
    destroyed = shuffle_pixels(body, digest)
    shuffled_prompts = [prompt for prompt in runner.seen if prompt.image == destroyed]
    assert len(shuffled_prompts) == len(request.vision_families())


def test_a_vision_item_without_its_pixels_ends_the_run(
    monkeypatch, teacher: FakeTeacher, slices: Slices
):
    monkeypatch.delenv("RELEARN_IMAGE_STORE", raising=False)
    items = holdout_items(6, vision=True)
    request = make_request(items)
    with pytest.raises(ImageStoreError, match="RELEARN_IMAGE_STORE is unset"):
        score_request(request, FakeRunner(), teacher, slices)


def test_a_store_serving_the_wrong_pixels_ends_the_run(image_store, slices: Slices):
    body = png((1, 2, 3))
    digest = hashlib.sha256(body).hexdigest()
    (image_store / digest).write_bytes(png((250, 250, 250)))
    item = holdout_items(1)[0]
    items = (
        item.__class__(
            id=item.id,
            prompt=item.prompt,
            dataset_id=item.dataset_id,
            task="captioning",
            image_hash=digest,
        ),
    )
    with pytest.raises(ImageStoreError, match="hashes to"):
        score_request(make_request(items), FakeRunner(), FakeTeacher(), slices)


def test_a_runner_that_drops_answers_ends_the_run(teacher: FakeTeacher, slices: Slices):
    class ShortRunner(FakeRunner):
        def generate(self, prompts):
            return super().generate(prompts)[:-1]

    with pytest.raises(ScoringError, match="answers for"):
        score_request(make_request(holdout_items(4)), ShortRunner(), teacher, slices)


def test_the_shuffle_control_is_the_same_permutation_for_every_model(image_store):
    body = png((10, 120, 200), size=16)
    digest = store_image(image_store, body)
    once = shuffle_pixels(body, digest)
    assert once == shuffle_pixels(body, digest)
    assert once != body
    other = png((10, 120, 201), size=16)
    other_digest = store_image(image_store, other)
    assert shuffle_pixels(body, digest) != shuffle_pixels(body, other_digest)
