"""Source guards for the two new eval images.

An eval image is only worth pinning if it cannot quietly stop measuring. These
tests read the shipped source and refuse the shapes that would let that happen:
a baked endpoint, an offline scorer, a fallback that answers when the real
thing is unreachable, a licence the subnet cannot redistribute.

They are blunt on purpose. A grep is a poor substitute for a proof, but it
catches the failure mode that matters here — someone adding a convenient
fallback during an incident — and it does so in review rather than in
production.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "eval" / "src"
PACKAGES = ("relearn_common", "relearn_image_eval", "relearn_agent_eval")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sources() -> list[Path]:
    return sorted(
        path
        for package in PACKAGES
        for path in (SOURCE_ROOT / package).rglob("*.py")
    )


def all_source_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sources())


def test_there_are_sources_to_check() -> None:
    assert len(sources()) > 20


@pytest.mark.parametrize("banned", ["modal.com", "modal_stub", "import modal", "modal.App"])
def test_no_modal(banned: str) -> None:
    assert banned not in all_source_text()


@pytest.mark.parametrize("banned", ["DFlash2", "dflash2", "Flash-Next", "flash_next"])
def test_no_dflash2_or_flash_variants(banned: str) -> None:
    # CC BY-NC-ND weights are not redistributable, so they can never be a pin.
    assert banned not in all_source_text()


def test_flux_is_never_a_pinned_base_or_a_default() -> None:
    # Prose may discuss Flux, and the rejection list has to name it. What must
    # never exist is an assignment that makes a Flux checkpoint the base, the
    # default, or a fallback.
    from relearn_image_eval.pins import BASE_MODEL_ID, base_is_rejected

    assert not base_is_rejected(BASE_MODEL_ID)
    assert "flux" not in BASE_MODEL_ID.lower()

    assignment = re.compile(
        r"^\s*(?!REJECTED_BASE_SUBSTRINGS)[A-Za-z_][\w.\[\]\"']*\s*=.*flux", re.IGNORECASE
    )
    for path in sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            assert not assignment.match(line), f"{path}:{number}: {line}"


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
def test_every_flux_spelling_is_refused(flux: str) -> None:
    from relearn_image_eval.pins import base_is_rejected

    assert base_is_rejected(flux)


def test_no_endpoint_or_credential_is_baked_into_a_source_file() -> None:
    text = all_source_text()
    for banned in ("https://api.", "http://10.", "sk-", "Bearer ey"):
        assert banned not in text, banned
    # Endpoints are read from the environment, by name, and never defaulted.
    assert "JUDGE_API_URL" in text
    assert "TEACHER_API_URL" in text


def test_no_offline_scorer_or_fallback_judge() -> None:
    text = all_source_text().lower()
    for banned in ("force_sim", "sim_score", "simulated score", "fallback judge", "def sim_"):
        assert banned not in text, banned


def test_an_unconfigured_judge_is_a_refusal_in_both_images() -> None:
    image = (SOURCE_ROOT / "relearn_image_eval" / "judge.py").read_text(encoding="utf-8")
    agent = (SOURCE_ROOT / "relearn_agent_eval" / "teacher.py").read_text(encoding="utf-8")
    assert "is unset; the pod has no judge and cannot score" in image
    assert "is unset; the pod has no judge and cannot score" in agent


def test_the_image_challenge_never_ships_a_second_judge() -> None:
    judge = (SOURCE_ROOT / "relearn_image_eval" / "judge.py").read_text(encoding="utf-8")
    assert 'allowed_models=(JUDGE_MODEL_ID,)' in judge


def test_the_shipped_agent_catalog_carries_no_official_benchmark_name() -> None:
    # A canary lifted from a benchmark cannot detect regression on that
    # benchmark, and a public split lifted from one is already in every
    # candidate's training mix.
    catalog = SOURCE_ROOT / "relearn_agent_eval" / "catalog"
    text = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in sorted(catalog.glob("*.jsonl"))
    )
    for bench in ("mmlu", "gsm8k", "humaneval", "swe-bench", "toolbench", "gaia", "webarena"):
        assert bench not in text, bench


def test_the_shipped_agent_catalog_is_well_formed() -> None:
    from relearn_agent_eval.catalog import canary_episodes, public_episodes

    public = public_episodes()
    canaries = canary_episodes()
    assert len(public) >= 10
    assert len(canaries) >= 10
    # Every public episode has to be replayable, which is what `validate`
    # checks: a goal, tools, steps inside the schema, and a final answer.
    for trace in public:
        trace.validate(where="public")
        assert len(trace.steps) >= 2


def test_the_readme_and_docs_name_all_three_images() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("relearn-eval", "relearn-image-eval", "relearn-agent-eval"):
        assert name in text, name
    assert "Dockerfile.scoring" in text


def test_the_challenge_dockerfiles_do_not_pull_in_modal() -> None:
    for name in ("Dockerfile.challenge", "Dockerfile.scoring", "install-cli.sh"):
        text = (REPO_ROOT / "eval" / name).read_text(encoding="utf-8").lower()
        assert "modal" not in text, name
