"""The commitment must agree with the control plane's, bit for bit.

Port of the property tests in `relearn_challenge_task::holdout::tests`. If this
drifts, every live run fails closed on a commitment mismatch — which is the
right failure, but a useless one, so the properties are pinned here.
"""

from __future__ import annotations

from relearn_eval import HoldoutItem
from relearn_eval.commitment import HOLDOUT_DOMAIN, commitments_match, holdout_commitment


def item(item_id: int, prompt: str, dataset_id: str = "dev") -> HoldoutItem:
    return HoldoutItem(id=item_id, prompt=prompt, dataset_id=dataset_id)


def two() -> list[HoldoutItem]:
    return [
        item(900, "a red cube sits on a wooden table near a lamp"),
        item(901, "two cats sleep on a sunlit windowsill together"),
    ]


def test_domain_is_the_control_planes():
    assert HOLDOUT_DOMAIN == b"base-relearn-holdout-v1"


def test_commitment_is_order_independent_and_length_prefixed():
    forward = holdout_commitment(two())
    assert forward == holdout_commitment(list(reversed(two())))
    assert len(forward) == 64
    spliced_ab = holdout_commitment([item(1, "ab", ""), item(2, "c", "")])
    spliced_bc = holdout_commitment([item(1, "a", ""), item(2, "bc", "")])
    assert spliced_ab != spliced_bc


def test_commitment_changes_with_the_body():
    edited = two()
    edited[0] = item(900, edited[0].prompt + ".")
    assert holdout_commitment(two()) != holdout_commitment(edited)


def test_commitment_covers_task_and_image_hash():
    base = [item(5, "caption this scene")]
    vision = [
        HoldoutItem(
            id=5,
            prompt="caption this scene",
            dataset_id="dev",
            task="captioning",
            image_hash="ab" * 32,
        )
    ]
    assert holdout_commitment(base) != holdout_commitment(vision)


def test_image_hash_is_trimmed_before_hashing():
    padded = [
        HoldoutItem(
            id=5, prompt="p", dataset_id="dev", task="captioning", image_hash="  " + "ab" * 32
        )
    ]
    tight = [
        HoldoutItem(id=5, prompt="p", dataset_id="dev", task="captioning", image_hash="ab" * 32)
    ]
    assert holdout_commitment(padded) == holdout_commitment(tight)


def test_commitments_match_ignores_case_and_padding():
    digest = holdout_commitment(two())
    assert commitments_match(digest, f"  {digest.upper()}  ")
    assert not commitments_match(digest, "aa" * 32)


def test_empty_set_still_commits_to_the_domain():
    # Length-prefixed, so the empty set is a defined value rather than the bare
    # domain digest. The request validator refuses it separately.
    assert len(holdout_commitment([])) == 64
    assert holdout_commitment([]) != holdout_commitment(two())
