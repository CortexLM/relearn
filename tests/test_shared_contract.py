"""One harvest contract across three challenges, and three distinct challenges.

`relearn_eval` keeps its own copy of the marker protocol because it is already
published and pinned by digest, and refactoring a live image underneath its pin
is not worth the tidiness. The cost of that decision is that the two copies can
drift, so these tests are the thing that stops them.

The other half is the opposite property: the three challenges must not be
confusable. They share a transport on purpose — a cortex harvest client for
either new challenge is `relearn-lium-harvest` with a different document type —
so what keeps a digest pinned into the wrong challenge's config from scoring
something nobody asked for is the binding inside the document, not a different
marker.
"""

from __future__ import annotations

import pytest

from relearn_agent_eval import pins as agent_pins
from relearn_common import wire
from relearn_common.errors import ContractError
from relearn_common.identity import RunIdentity, is_hex, is_sha256_pin
from relearn_eval import contract as llm_contract
from relearn_image_eval import pins as image_pins


def test_the_marker_protocol_is_one_protocol() -> None:
    assert wire.METRICS_MARKER == llm_contract.METRICS_MARKER == "RELEARN_METRICS="
    assert wire.OK_MARKER == llm_contract.OK_MARKER == "RELEARN_EVAL_OK"
    assert wire.SCORE_DECIMALS == llm_contract.SCORE_DECIMALS == 6


def test_every_challenge_uses_the_same_schema_version() -> None:
    assert (
        llm_contract.METRICS_SCHEMA_VERSION
        == image_pins.SCHEMA_VERSION
        == agent_pins.SCHEMA_VERSION
        == 1
    )


def test_the_three_challenge_ids_are_distinct() -> None:
    ids = {"relearn", image_pins.CHALLENGE_ID, agent_pins.CHALLENGE_ID}
    assert ids == {"relearn", "relearn-image", "relearn-agent"}
    assert agent_pins.CHALLENGE_ID not in image_pins.CHALLENGE_IDS
    assert "relearn" not in image_pins.CHALLENGE_IDS
    assert "relearn" not in agent_pins.CHALLENGE_IDS
    assert image_pins.LEGACY_CHALLENGE_ID not in agent_pins.CHALLENGE_IDS


def test_each_challenge_stages_into_its_own_workdir() -> None:
    # A pod runs one challenge, but leaving a shared path would mean one run's
    # request could be read by another image's scorer.
    paths = {
        llm_contract.POD_WORKDIR,
        image_pins.POD_WORKDIR,
        agent_pins.POD_WORKDIR,
    }
    assert len(paths) == 3


def test_each_challenge_owns_its_environment_prefix() -> None:
    assert image_pins.ENV_PREFIX == "RELEARN_IMAGE_"
    assert agent_pins.ENV_PREFIX == "RELEARN_AGENT_"
    assert image_pins.ENV_PREFIX != agent_pins.ENV_PREFIX


def test_each_challenge_has_its_own_boot_baseline_id() -> None:
    baselines = {
        llm_contract.BASE_CHAMPION_ARTIFACT,
        image_pins.BASE_CHAMPION_ARTIFACT,
        agent_pins.BASE_CHAMPION_ARTIFACT,
    }
    assert len(baselines) == 3


def test_the_agent_challenge_commits_differently_from_the_language_one() -> None:
    # Same records under two challenges must not produce the same commitment,
    # or a verified holdout could be replayed across challenges.
    from relearn_eval.commitment import HOLDOUT_DOMAIN as LLM_DOMAIN

    assert agent_pins.HOLDOUT_DOMAIN != LLM_DOMAIN
    assert agent_pins.HOLDOUT_DOMAIN != image_pins.HOLDOUT_DOMAIN


def test_the_agent_and_language_challenges_share_a_base_and_nothing_else() -> None:
    # This is the fact that makes them look like the same challenge, and the
    # reason the distinction has to be enforced everywhere else.
    assert agent_pins.BASE_MODEL_ID == "Qwen/Qwen3.8-27B"
    assert agent_pins.CHALLENGE_ID != "relearn"
    assert agent_pins.POD_WORKDIR != llm_contract.POD_WORKDIR


def test_only_a_digest_pin_is_accepted_as_an_image_reference() -> None:
    assert is_sha256_pin(f"sha256:{'ab' * 32}")
    for bad in ("", "latest", "sha256:beef", "ghcr.io/cortexlm/relearn-eval:v1", "ab" * 32):
        assert not is_sha256_pin(bad)


def test_hex_checks_are_case_insensitive_and_length_exact() -> None:
    assert is_hex("AB" * 32, 64)
    assert not is_hex("ab" * 31, 64)
    assert not is_hex("zz" * 32, 64)


def identity(**overrides: str) -> RunIdentity:
    base = {
        "challenge_id": "relearn-image",
        "submission_digest": "frozen-1",
        "artifact_digest": "cd" * 32,
        "eval_image_digest": f"sha256:{'ab' * 32}",
        "holdout_commitment": "11" * 32,
    }
    base.update(overrides)
    return RunIdentity(**base)  # type: ignore[arg-type]


def test_a_document_must_echo_every_identity_field() -> None:
    run = identity()
    run.validate(challenge_ids=("relearn-image",))
    run.check_document(identity())
    for field, value in (
        ("challenge_id", "relearn-agent"),
        ("submission_digest", "an-earlier-run"),
        ("artifact_digest", "ef" * 32),
        ("eval_image_digest", f"sha256:{'cd' * 32}"),
        ("holdout_commitment", "22" * 32),
    ):
        with pytest.raises(ContractError):
            run.check_document(identity(**{field: value}))


def test_digest_comparison_ignores_case_but_identity_fields_do_not() -> None:
    run = identity()
    run.check_document(identity(artifact_digest=("cd" * 32).upper()))
    with pytest.raises(ContractError):
        run.check_document(identity(submission_digest="FROZEN-1"))


def test_a_challenge_refuses_an_identity_from_another_challenge() -> None:
    with pytest.raises(ContractError, match="not scored by this image"):
        identity(challenge_id="relearn-agent").validate(challenge_ids=("relearn-image",))


def test_an_encoded_document_is_exactly_one_line() -> None:
    # The harvest reconstructs the marker line with `printf`; a newline
    # anywhere would truncate the document mid-JSON.
    line = wire.encode_line({"a": 1, "b": "two\u00a0three"})
    assert "\n" not in line and "\r" not in line
    assert wire.marker_line({"a": 1}).startswith(wire.METRICS_MARKER)


def test_a_non_finite_score_ends_the_run_rather_than_publishing() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ContractError):
            wire.clamp_score(bad, what="test")


def test_scores_are_rounded_so_two_identical_runs_agree_exactly() -> None:
    assert wire.clamp_score(0.123456789, what="test") == 0.123457
    assert wire.clamp_score(-1.0, what="test") == 0.0
    assert wire.clamp_score(2.0, what="test") == 1.0
