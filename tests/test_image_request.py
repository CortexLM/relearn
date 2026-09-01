"""What the Relearn Image request refuses, and why each refusal exists."""

from __future__ import annotations

import dataclasses
import json

import pytest

from relearn_image_eval.commitment import frozen_prompt_commitment
from relearn_image_eval.pins import CHALLENGE_ID, LEGACY_CHALLENGE_ID
from relearn_image_eval.request import (
    ArtifactManifest,
    FrozenPrompt,
    ImageHarvestRequest,
    RequestError,
    Sampler,
)
from tests.tools.make_image_request import build

IMAGE_DIGEST = f"sha256:{'ab' * 32}"


def make(**overrides: object) -> ImageHarvestRequest:
    request = build(
        25, submission="frozen-1", artifact="cd" * 32, image_digest=IMAGE_DIGEST
    )
    return dataclasses.replace(request, **overrides) if overrides else request


def recommit(request: ImageHarvestRequest) -> ImageHarvestRequest:
    """Re-seal a request after editing its holdout, to isolate one refusal."""
    return dataclasses.replace(
        request, holdout_commitment=frozen_prompt_commitment(request.holdout)
    )


def test_a_well_formed_request_validates() -> None:
    make().validate()


def test_round_trips_through_its_wire_form() -> None:
    request = make()
    again = ImageHarvestRequest.from_wire(json.loads(json.dumps(request.to_wire())))
    again.validate()
    assert again.holdout == request.holdout
    assert again.manifest.base == request.manifest.base


def test_the_earlier_t2i_challenge_id_is_still_accepted() -> None:
    # cortex still declares `challenge_id = "relearn-t2i"` in its pin. Refusing
    # it would mean the rename needs a new eval image digest before the control
    # plane can boot anything.
    legacy = make(challenge_id=LEGACY_CHALLENGE_ID)
    legacy.validate()
    assert legacy.identity.challenge_id == LEGACY_CHALLENGE_ID


def test_another_challenge_id_is_refused() -> None:
    for other in ("relearn", "relearn-agent", "relearn-mm", "bounty", ""):
        with pytest.raises(RequestError, match="not scored by this image"):
            make(challenge_id=other).validate()


@pytest.mark.parametrize(
    "flux",
    [
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
        "BLACK-FOREST-LABS/flux.1-pro",
        "someone/flux1-merged-lora",
        "mirror/Flux",
    ],
)
def test_a_flux_base_is_refused_before_anything_is_generated(flux: str) -> None:
    with pytest.raises(RequestError, match="refused family"):
        make(base_model=flux).validate()


def test_a_flux_artifact_manifest_is_refused_even_under_the_pinned_base() -> None:
    # The pin can say Cosmos3 while the miner declares it fine-tuned Flux. The
    # manifest is attested separately for exactly that case.
    request = make(
        manifest=ArtifactManifest(
            base="black-forest-labs/FLUX.1-dev", base_license="OpenMDW-1.1"
        )
    )
    with pytest.raises(RequestError, match="refused family"):
        request.validate()


def test_a_base_other_than_the_pin_is_refused() -> None:
    with pytest.raises(RequestError, match="base must be"):
        make(base_model="stabilityai/sd-3.5").validate()


def test_a_non_commercial_artifact_licence_is_refused() -> None:
    request = make(
        manifest=ArtifactManifest(
            base="nvidia/Cosmos3-Super-Text2Image", base_license="cc-by-nc-4.0"
        )
    )
    with pytest.raises(RequestError, match="artifact license"):
        request.validate()


def test_the_pinned_licence_is_matched_loosely_on_spelling() -> None:
    # `OpenMDW 1.1` and `OpenMDW-1.1` are the same licence; the card and the pin
    # do not agree on the separator.
    make(
        manifest=ArtifactManifest(
            base="nvidia/Cosmos3-Super-Text2Image", base_license="OpenMDW 1.1"
        )
    ).validate()


def test_the_base_champion_run_needs_no_manifest() -> None:
    # The boot baseline has no artifact, so there is nothing to attest.
    baseline = make(artifact_digest="base-relearn-image-champion", manifest=ArtifactManifest())
    baseline.validate()
    assert baseline.is_base_champion_run


def test_any_judge_but_q_judger_is_refused() -> None:
    for other in ("gpt-4o", "google/gemma-3", "openai/some-vlm", ""):
        with pytest.raises(RequestError, match="judge must be"):
            make(judge_model=other).validate()


def test_a_tag_rather_than_a_digest_is_refused() -> None:
    for bad in ("", "latest", "ghcr.io/cortexlm/relearn-image-eval:v1", "sha256:beef"):
        with pytest.raises(RequestError, match="sha256"):
            make(eval_image_digest=bad).validate()


def test_a_thin_split_is_refused_rather_than_holding_the_champion_forever() -> None:
    # One variation per prompt is 25 cells, below the paired test's floor. A
    # request like that could never promote anything, which is a configuration
    # error and not a permanent champion-hold.
    with pytest.raises(RequestError, match="below the 100 floor"):
        make(variations_per_prompt=1).validate()


def test_an_unpinned_seed_lattice_is_refused() -> None:
    with pytest.raises(RequestError, match="pin_salt"):
        make(pin_salt="   ").validate()


def test_a_holdout_prompt_that_is_also_public_is_refused() -> None:
    request = make()
    overlapping = (*request.public[:-1], FrozenPrompt(id=901, text="two cats on a rug"))
    with pytest.raises(RequestError, match="also in the public split"):
        dataclasses.replace(request, public=overlapping).validate()


def test_a_missing_public_split_is_refused() -> None:
    with pytest.raises(RequestError, match="public split"):
        make(public=()).validate()


def test_a_prompt_outside_the_bench_range_is_refused() -> None:
    request = make()
    edited = (*request.holdout[:-1], FrozenPrompt(id=1001, text="out of range"))
    with pytest.raises(RequestError, match="outside Qwen-Image-Bench range"):
        recommit(dataclasses.replace(request, holdout=edited)).validate()


def test_an_empty_prompt_is_refused() -> None:
    request = make()
    edited = (*request.holdout[:-1], FrozenPrompt(id=990, text="   "))
    with pytest.raises(RequestError, match="empty text"):
        recommit(dataclasses.replace(request, holdout=edited)).validate()


def test_a_duplicate_prompt_id_is_refused() -> None:
    request = make()
    edited = (*request.holdout[:-1], request.holdout[0])
    with pytest.raises(RequestError, match="duplicate holdout prompt id"):
        recommit(dataclasses.replace(request, holdout=edited)).validate()


def test_a_holdout_edited_in_flight_is_refused() -> None:
    request = make()
    edited = (
        FrozenPrompt(id=request.holdout[0].id, text="a prompt the commitment does not cover"),
        *request.holdout[1:],
    )
    with pytest.raises(RequestError, match="commitment mismatch") as caught:
        dataclasses.replace(request, holdout=edited).validate()
    # Hex only: the prompts themselves must not reach a log.
    assert "a prompt the commitment does not cover" not in str(caught.value)


def test_an_upsampled_prompt_is_what_reaches_the_generator() -> None:
    plain = FrozenPrompt(id=5, text="a bird")
    assert plain.generator_input == "a bird"
    upsampled = FrozenPrompt(id=5, text="a bird", upsampled_json='{"subject":"a bird"}')
    assert upsampled.generator_input == '{"subject":"a bird"}'
    # And the two commit differently, so a pin cannot be swapped for the other.
    assert frozen_prompt_commitment([plain]) != frozen_prompt_commitment([upsampled])


def test_the_sampler_defaults_to_the_card_recipe() -> None:
    sampler = Sampler.from_wire(None)
    assert (sampler.width, sampler.height) == (1024, 1024)
    assert sampler.num_inference_steps == 50
    assert sampler.guidance_scale == pytest.approx(4.0)
    assert sampler.flow_shift == pytest.approx(3.0)
    assert sampler.dtype == "bfloat16"
    assert sampler.scheduler == "UniPCMultistepScheduler"


def test_a_video_sampler_is_refused() -> None:
    with pytest.raises(RequestError, match="num_frames"):
        make(sampler=Sampler(num_frames=4)).validate()


def test_unknown_request_fields_are_tolerated() -> None:
    # The control plane must be able to grow the request without a new image.
    wire = make().to_wire()
    wire["some_future_field"] = {"nested": True}
    ImageHarvestRequest.from_wire(wire).validate()


def test_the_challenge_id_is_not_the_language_challenge() -> None:
    assert CHALLENGE_ID == "relearn-image"
    assert LEGACY_CHALLENGE_ID == "relearn-t2i"
    assert CHALLENGE_ID != "relearn"
