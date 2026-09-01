"""Scoring a Relearn Image run end to end, and refusing to score a broken one.

The generator and the judge are test doubles from `tests/support.py`. Nothing
in the installed package can produce an image or a score without them, which is
the property these tests are really protecting.
"""

from __future__ import annotations

import dataclasses

import pytest

from relearn_common.transcript import TranscriptError, accept
from relearn_common.wire import OK_MARKER, decode_line, encode_line, marker_line
from relearn_image_eval.contract import ImageDocument
from relearn_image_eval.pillars import ALIGNMENT, ALL_PILLARS
from relearn_image_eval.pins import MAX_NA_RATE, MIN_FAITHFULNESS_CHECKS, REPLAY_CELLS
from relearn_image_eval.request import ArtifactManifest
from relearn_image_eval.scoring import (
    ScoringError,
    cells_from,
    contaminated_ids,
    score_request,
    select_cells,
)
from relearn_image_eval.seeds import cell_key
from relearn_image_eval.verify import VerificationError, verify_document
from tests.support import FakeGenerator, FakeQJudger
from tests.tools.make_image_request import build

IMAGE_DIGEST = f"sha256:{'ab' * 32}"


def make_request(**overrides: object):
    request = build(
        25, submission="frozen-1", artifact="cd" * 32, image_digest=IMAGE_DIGEST
    )
    return dataclasses.replace(request, **overrides) if overrides else request


def test_scores_every_requested_cell_and_binds_the_run() -> None:
    request = make_request()
    document = score_request(request, FakeGenerator(), FakeQJudger())

    assert len(document.measurement.holdout) == 100
    assert len(document.measurement.public) == 100
    assert document.identity == request.identity
    assert document.measurement.base_model == request.base_model
    verify_document(document, request)


def test_cell_keys_carry_ids_and_never_prompt_text() -> None:
    request = make_request()
    document = score_request(request, FakeGenerator(), FakeQJudger())
    for key in document.measurement.holdout:
        assert key.startswith("p") and "#v" in key
    assert cell_key(901, 0) in document.measurement.holdout
    encoded = encode_line(document.to_wire())
    for prompt in request.holdout:
        assert prompt.text not in encoded


def test_every_pillar_is_carried_separately() -> None:
    # Without per-pillar series the control plane cannot see a Quality collapse
    # hidden behind an Alignment gain.
    document = score_request(make_request(), FakeGenerator(), FakeQJudger())
    assert set(document.measurement.holdout_by_pillar) == {p.wire for p in ALL_PILLARS}
    for values in document.measurement.holdout_by_pillar.values():
        assert len(values) == 100


def test_the_generator_is_driven_at_the_shared_seeds() -> None:
    request = make_request()
    generator = FakeGenerator()
    score_request(request, generator, FakeQJudger())
    expected = {seed for _pid, _v, seed in request.holdout_cells()}
    generated = {seed for _prompt, seed in generator.calls}
    assert expected <= generated


def test_the_frozen_prompt_is_what_reaches_the_generator() -> None:
    request = make_request()
    generator = FakeGenerator()
    score_request(request, generator, FakeQJudger())
    sent = {prompt for prompt, _seed in generator.calls}
    assert request.holdout[0].generator_input in sent


def test_a_judge_that_declines_almost_everything_voids_the_run() -> None:
    # A run where the judge answered N/A to most items is not a low score, it
    # is no measurement, and publishing the survivors would be a verdict drawn
    # from whichever cells happened to be scorable.
    judge = FakeQJudger(na_items=19, scored_items=1)
    with pytest.raises(ScoringError, match="N/A rate"):
        score_request(make_request(), FakeGenerator(), judge)
    assert MAX_NA_RATE == 0.25


def test_replay_evidence_covers_the_pinned_number_of_cells() -> None:
    document = score_request(make_request(), FakeGenerator(), FakeQJudger())
    assert document.measurement.replay.cells_checked == REPLAY_CELLS


def test_a_deterministic_generator_replays_with_no_drift() -> None:
    document = score_request(make_request(), FakeGenerator(), FakeQJudger())
    assert document.measurement.replay.max_embedding_drift == pytest.approx(0.0)


def test_claimed_output_hashes_that_do_not_match_are_reported_not_hidden() -> None:
    # The miner claimed hashes for cells this run regenerates. None of them
    # match here, so the evidence says zero exact matches and lets the control
    # plane decide, rather than quietly passing the gate.
    request = make_request(
        manifest=ArtifactManifest(
            base="nvidia/Cosmos3-Super-Text2Image",
            base_license="OpenMDW-1.1",
            claimed_output_hashes={"p1#v0": "00" * 32},
        )
    )
    document = score_request(request, FakeGenerator(), FakeQJudger())
    assert document.measurement.replay.exact_hash_matches == 0


def test_faithfulness_agrees_when_both_passes_agree() -> None:
    document = score_request(make_request(), FakeGenerator(), FakeQJudger(level=90.0))
    faithfulness = document.measurement.faithfulness
    assert faithfulness.checks == MIN_FAITHFULNESS_CHECKS
    assert faithfulness.agreements == faithfulness.checks


def test_faithfulness_disagrees_when_the_spot_check_contradicts_the_score() -> None:
    # The scored pass says Alignment is high; the targeted pass says the
    # concrete claims do not hold. That disagreement is what the control plane
    # discards the run on, and it must reach the document rather than be
    # averaged away.
    judge = FakeQJudger(level=90.0, spot_level=10.0)
    document = score_request(make_request(), FakeGenerator(), judge)
    assert document.measurement.faithfulness.checks == MIN_FAITHFULNESS_CHECKS
    assert document.measurement.faithfulness.agreements == 0
    assert ALIGNMENT.wire in document.measurement.holdout_by_pillar


def test_contamination_reports_only_private_ids() -> None:
    # The published split is trainable by design, so training on it is not
    # contamination. Only the holdout ids count.
    assert contaminated_ids([1, 2, 907], [901, 907, 915]) == (907,)
    assert contaminated_ids([1, 26], [901, 907]) == ()


def test_contamination_reaches_the_document() -> None:
    request = make_request(
        manifest=ArtifactManifest(
            base="nvidia/Cosmos3-Super-Text2Image",
            base_license="OpenMDW-1.1",
            train_prompt_ids=(1, 2, 907),
        )
    )
    document = score_request(request, FakeGenerator(), FakeQJudger())
    assert document.measurement.contaminated_prompt_ids == (907,)


def test_evidence_cells_are_deterministic_but_differ_between_submissions() -> None:
    cells = cells_from(make_request().public_cells())
    once = select_cells(cells, 3, "frozen-1", b"domain")
    again = select_cells(cells, 3, "frozen-1", b"domain")
    other_run = select_cells(cells, 3, "frozen-2", b"domain")
    other_domain = select_cells(cells, 3, "frozen-1", b"other-domain")
    assert once == again
    assert once != other_run
    assert once != other_domain


def test_the_document_survives_its_own_encoding() -> None:
    document = score_request(make_request(), FakeGenerator(), FakeQJudger())
    again = ImageDocument.from_wire(decode_line(encode_line(document.to_wire())))
    assert encode_line(again.to_wire()) == encode_line(document.to_wire())


def test_a_transcript_this_image_produces_is_one_the_control_plane_accepts() -> None:
    request = make_request()
    document = score_request(request, FakeGenerator(), FakeQJudger())
    transcript = f"boot ok\n{marker_line(document.to_wire())}\n{OK_MARKER}\n"
    accepted = accept(
        transcript, ImageDocument.from_wire, lambda parsed: verify_document(parsed, request)
    )
    assert accepted.identity == request.identity


def test_a_document_for_another_run_is_refused() -> None:
    request = make_request()
    document = score_request(request, FakeGenerator(), FakeQJudger())
    for field, value in (
        ("submission_digest", "an-earlier-run"),
        ("artifact_digest", "ef" * 32),
        ("eval_image_digest", f"sha256:{'cd' * 32}"),
        ("holdout_commitment", "11" * 32),
        ("challenge_id", "relearn-t2i"),
    ):
        impostor = dataclasses.replace(
            document, identity=dataclasses.replace(document.identity, **{field: value})
        )
        with pytest.raises(VerificationError):
            verify_document(impostor, request)


def test_silence_is_never_a_score() -> None:
    request = make_request()
    for body in ("", "boot ok\nsegfault\n", f"{OK_MARKER}\n"):
        with pytest.raises(TranscriptError):
            accept(
                body,
                ImageDocument.from_wire,
                lambda parsed: verify_document(parsed, request),
            )


def test_a_document_without_the_completion_marker_is_refused() -> None:
    request = make_request()
    document = score_request(request, FakeGenerator(), FakeQJudger())
    with pytest.raises(TranscriptError, match=OK_MARKER):
        accept(
            f"{marker_line(document.to_wire())}\n",
            ImageDocument.from_wire,
            lambda parsed: verify_document(parsed, request),
        )
