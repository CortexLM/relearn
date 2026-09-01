"""The seed and commitment ports, pinned against cortex's own Rust.

Every value below was produced by compiling `relearn_t2i_task` and
`relearn_challenge_task` from `CortexLM/cortex` unmodified and printing the
result. They are not "what this repository currently computes" — they are what
the control plane computes, so a refactor that changes a preimage fails here
instead of silently making every miner's images incomparable and every holdout
commitment unverifiable.

If one of these ever has to change, the crate changed first, and the pin,
the champion, and this file all move together.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from relearn_eval.commitment import holdout_commitment
from relearn_eval.request import HoldoutItem
from relearn_image_eval.commitment import frozen_prompt_commitment
from relearn_image_eval.seeds import cell_key, derive_generation_seed, seed_cells


@dataclass(frozen=True)
class Prompt:
    id: int
    text: str
    upsampled_json: str = ""


#: `relearn_t2i_task::derive_generation_seed`.
SEEDS = {
    (1, 0, "cortex-t2i-v0"): 5_534_307_901_387_864_795,
    (1, 1, "cortex-t2i-v0"): 8_396_957_163_666_059_367,
    (26, 3, "cortex-t2i-v0"): 2_009_180_236_085_125_894,
    (1000, 7, "cortex-t2i-v0"): 6_694_602_453_166_862_693,
    (7, 0, "salt-a"): 2_697_998_632_308_631_408,
    (7, 0, ""): 1_932_744_928_684_836_595,
    (901, 2, "cortex-t2i-dev-holdout-v0"): 3_331_479_512_056_451_455,
}

#: `relearn_t2i_task::frozen_prompt_commitment`.
COMMITMENTS = {
    "single": (
        [Prompt(900, "a red cube")],
        "b3e682d17b74bd6c108db8c598a6cc2d75aea093323fe63a17227f2a9352efe5",
    ),
    "pair": (
        [Prompt(900, "a red cube on a wooden table"), Prompt(901, "two cats")],
        "5c62f1e9657001aa9dd6d0932b2e3a95daf99a4da3084dd29323e08a2a5b43e2",
    ),
    "reversed_pair": (
        [Prompt(901, "two cats"), Prompt(900, "a red cube on a wooden table")],
        "5c62f1e9657001aa9dd6d0932b2e3a95daf99a4da3084dd29323e08a2a5b43e2",
    ),
    "upsampled": (
        [Prompt(5, "a bird", '{"subject":"a bird"}')],
        "fa873362f7bcf088436ff4f6d3aa695bd5b59cbafb50d02586e78b520ad0b977",
    ),
    # A real bench prompt is Chinese with full-width punctuation and embedded
    # quotes; the commitment hashes UTF-8 bytes, so this is the case that would
    # break first if the port ever stopped matching the crate.
    "unicode": (
        [Prompt(151, '崇祯五年十二月，余住西湖 · "quoted"')],  # noqa: RUF001
        "921135e84a6ab9f1f10fdbb4b06eafe2f42d0369c0cb2e9d270608789afd0039",
    ),
    "empty": ([], "15dc79b2eef64e1d4114da783efc6df22e2bd9ce5a1d5df2693fe788a3e8ef6b"),
    "splice_ab": (
        [Prompt(1, "ab"), Prompt(2, "c")],
        "6c0b5b52cb92e22698828027a5540cf5e87f19f25bfe404dc68ffc719c281eaf",
    ),
    "splice_a_bc": (
        [Prompt(1, "a"), Prompt(2, "bc")],
        "f32111aa3e518acb2e1367e5153933bbc350a040e90049caadc6b8afd9756ecc",
    ),
}

#: `relearn_challenge_task::holdout_commitment`, the language challenge's own.
#: Pinned here as well because the published Relearn LLM image depends on it
#: and nothing else in this repository proves that port against the crate.
LLM_COMMITMENTS = {
    "single_text": (
        [HoldoutItem(id=801, prompt="holdout item one", dataset_id="dev", task="text")],
        "8d8b43b8672dcdc1dbc101fe00766793356d338776c9b8ba0394dfbc94c4b533",
    ),
    "text_pair": (
        [
            HoldoutItem(id=801, prompt="holdout item one", dataset_id="dev", task="text"),
            HoldoutItem(id=802, prompt="holdout item two", dataset_id="dev", task="text"),
        ],
        "910a98289e048fdad9d68d6a44bda3a980562d9349dd014e1e09a310ead9b2a7",
    ),
    "with_vision": (
        [
            HoldoutItem(id=801, prompt="holdout item one", dataset_id="dev", task="text"),
            HoldoutItem(
                id=802,
                prompt="what is in this picture",
                dataset_id="dev",
                task="vqa",
                image_hash="aa" * 32,
            ),
        ],
        "60d513d803265e6d2aed98bf1e98dff86bcdd44dbeddb0067d04a2126a1b5acc",
    ),
    "empty": ([], "12d622ccbee464395e55424677d361a98e289bb0cc6c057dc2bf5bf21c8c092d"),
}


@pytest.mark.parametrize(("args", "expected"), sorted(SEEDS.items()))
def test_generation_seed_matches_the_crate(args: tuple[int, int, str], expected: int) -> None:
    assert derive_generation_seed(*args) == expected


def test_generation_seed_fits_positive_i64() -> None:
    # Both the Diffusers and the vLLM-Omni paths take a signed seed.
    for seed in SEEDS.values():
        assert 0 <= seed <= 2**63 - 1


@pytest.mark.parametrize("name", sorted(COMMITMENTS))
def test_frozen_prompt_commitment_matches_the_crate(name: str) -> None:
    prompts, expected = COMMITMENTS[name]
    assert frozen_prompt_commitment(prompts) == expected


@pytest.mark.parametrize("name", sorted(LLM_COMMITMENTS))
def test_language_holdout_commitment_matches_the_crate(name: str) -> None:
    items, expected = LLM_COMMITMENTS[name]
    assert holdout_commitment(items) == expected


def test_cell_keys_are_the_crate_spelling() -> None:
    assert cell_key(12, 3) == "p12#v3"
    assert cell_key(1, 0) == "p1#v0"
    assert cell_key(1000, 15) == "p1000#v15"
    assert cell_key(12, 3) != cell_key(123, 3)


def test_seed_cells_are_sorted_and_deduplicated() -> None:
    # `RelearnT2iPin::seed_cells` sorts and dedupes first, so cell order does
    # not depend on how the request happened to be serialized.
    once = list(seed_cells([3, 1, 3], 2, "cortex-t2i-v0"))
    again = list(seed_cells([1, 3], 2, "cortex-t2i-v0"))
    assert once == again
    assert len(once) == 4
    assert once[0] == (1, 0, derive_generation_seed(1, 0, "cortex-t2i-v0"))


def test_commitment_separates_reordering_from_splicing() -> None:
    # Order independence and length prefixing are the two properties that make
    # the commitment worth having; both are asserted against the crate above,
    # and this states why the two vectors differ.
    assert COMMITMENTS["pair"][1] == COMMITMENTS["reversed_pair"][1]
    assert COMMITMENTS["splice_ab"][1] != COMMITMENTS["splice_a_bc"][1]
