"""Q-Judger's wire format, and the paper's aggregation.

The behaviour these tests pin down is not stylistic. `N/A` excluded rather than
zeroed is the difference between "this criterion does not apply to this prompt"
and "this model failed this criterion", and prompts differ in which criteria
apply, so getting it wrong would penalise prompts unevenly across the split.
"""

from __future__ import annotations

import pytest

from relearn_common.judge import (
    JudgeError,
    guard_judge_call,
    image_data_url,
    looks_like_artifact_digest,
    model_matches,
)
from relearn_image_eval.judge import aggregate, parse_reply, score_reply
from relearn_image_eval.pillars import (
    AESTHETICS,
    ALIGNMENT,
    ALL_PILLARS,
    CREATIVE_GENERATION,
    QUALITY,
    REAL_WORLD_FIDELITY,
    parse_pillar,
)
from relearn_image_eval.pins import JUDGE_MODEL_ID

FULL_REPLY = """
Let me look at the image. The cube is red {not JSON} and sharp.
```json
{
  "Quality": {
    "Realism": {"Physical Logic": {"score": 1}, "Material Texture": {"score": 2}},
    "Detail": {"Noise": {"score": 1}, "Edge Clarity": {"score": 1}, "Naturalness": {"score": 1}},
    "Resolution": {"Resolution": {"score": 2}}
  },
  "Aesthetics": {
    "Composition": {"Composition": {"score": 2}},
    "Color Harmony": {"Color Harmony": {"score": 1}}
  },
  "Alignment": {
    "Attributes": {"Color": {"score": 2}, "Quantity": {"score": "N/A"}}
  },
  "Real-world Fidelity": {
    "Safety & Compliance": {"Safety & Compliance": {"score": 1}}
  },
  "Creative Generation": {
    "Text Rendering": {"Text Accuracy": {"score": "N/A"}, "Font": {"score": "N/A"}},
    "Imagination": {"Imagination": {"score": 2}}
  }
}
```
"""


def test_parses_a_thinking_preamble_then_the_tree() -> None:
    tree = parse_reply(FULL_REPLY)
    assert set(tree) == {pillar.wire for pillar in ALL_PILLARS}


def test_aggregation_follows_the_paper() -> None:
    score = score_reply(FULL_REPLY)
    # Quality: Realism (60+100)/2 = 80, Detail 60, Resolution 100 -> 80.
    assert score.per_pillar[QUALITY.wire] == pytest.approx(80.0)
    # Aesthetics: Composition 100, Color Harmony 60 -> 80.
    assert score.per_pillar[AESTHETICS.wire] == pytest.approx(80.0)
    # Alignment: Attributes averages only Color (100); Quantity is N/A.
    assert score.per_pillar[ALIGNMENT.wire] == pytest.approx(100.0)
    assert score.per_pillar[REAL_WORLD_FIDELITY.wire] == pytest.approx(60.0)
    # Creative Generation: Text Rendering is entirely N/A and drops out,
    # leaving Imagination = 100.
    assert score.per_pillar[CREATIVE_GENERATION.wire] == pytest.approx(100.0)
    assert score.total == pytest.approx((80.0 + 80.0 + 100.0 + 60.0 + 100.0) / 5.0)


def test_na_is_excluded_not_zeroed() -> None:
    with_na = '{"Quality": {"Detail": {"Noise": {"score": 2}, "Edge Clarity": {"score": "N/A"}}}}'
    zeroed = '{"Quality": {"Detail": {"Noise": {"score": 2}, "Edge Clarity": {"score": 0}}}}'
    excluded = score_reply(with_na)
    punished = score_reply(zeroed)
    assert excluded.total == pytest.approx(100.0)
    assert punished.total == pytest.approx(50.0)
    assert excluded.total > punished.total
    assert (excluded.na_items, excluded.scored_items) == (1, 1)


def test_a_reply_that_is_entirely_na_is_a_failed_run_not_a_zero() -> None:
    with pytest.raises(JudgeError, match="entirely N/A"):
        score_reply('{"Quality": {"Detail": {"Noise": {"score": "N/A"}}}}')


def test_an_unknown_score_vocabulary_fails_loudly() -> None:
    # A judge that started emitting new words must not be coerced into zeros,
    # which would look like every image getting worse at once.
    with pytest.raises(JudgeError, match="unparsable score"):
        parse_reply('{"Quality": {"Detail": {"Noise": {"score": "excellent"}}}}')


def test_a_reply_with_no_json_is_refused() -> None:
    with pytest.raises(JudgeError, match="no JSON object"):
        parse_reply("I could not evaluate this image.")


def test_an_unfenced_reply_uses_the_last_balanced_object() -> None:
    # The thinking trace contains braces of its own, and the score tree is what
    # the model emits last.
    raw = (
        'thinking: the layout {looks} fine, and I considered {"score": 0} briefly.\n'
        '{"Quality": {"Detail": {"Noise": {"score": 2}}}}'
    )
    assert score_reply(raw).total == pytest.approx(100.0)


def test_bare_leaf_scores_are_accepted() -> None:
    score = score_reply('{"Quality": {"Detail": {"Noise": 1, "Naturalness": 2}}}')
    assert score.total == pytest.approx(80.0)


def test_a_single_pillar_reply_needs_the_assumed_pillar() -> None:
    body = '{"Physical Logic": {"score": 1}}'
    with pytest.raises(JudgeError, match="no recognized L1 pillar"):
        parse_reply('{"Realism": {"Physical Logic": {"score": 1}}}')
    tree = parse_reply('{"Realism": {"Physical Logic": {"score": 1}}}', QUALITY)
    assert set(tree) == {QUALITY.wire}
    assert body  # the shape above is what the spot check receives


def test_pillars_round_trip_through_the_judge_spelling() -> None:
    for pillar in ALL_PILLARS:
        assert parse_pillar(pillar.card) is pillar
        assert parse_pillar(pillar.wire) is pillar
    assert parse_pillar("real_world_fidelity") is REAL_WORLD_FIDELITY
    assert parse_pillar("Creative-Generation") is CREATIVE_GENERATION
    assert parse_pillar("Vibes") is None


def test_normalization_puts_one_dead_zone_unit_on_one_paper_point() -> None:
    score = score_reply(FULL_REPLY)
    assert score.normalized_total() == pytest.approx(score.total / 100.0)
    assert score.normalized_pillar(QUALITY) == pytest.approx(0.80)


def test_an_empty_tree_is_refused_rather_than_scored() -> None:
    with pytest.raises(JudgeError):
        aggregate({})


def test_only_q_judger_may_be_called() -> None:
    guard_judge_call(JUDGE_MODEL_ID, (JUDGE_MODEL_ID,), "a candidate image")
    guard_judge_call("qwen/qwen-image-bench", (JUDGE_MODEL_ID,), "case insensitive")
    with pytest.raises(JudgeError, match="not a judge"):
        guard_judge_call("gpt-4o", (JUDGE_MODEL_ID,), "a candidate image")


def test_an_artifact_digest_is_never_a_judge() -> None:
    # The judge is judge-only. Passing a submitted artifact's digest as the
    # model id would be asking the endpoint to serve the thing under test.
    digest = "ab" * 32
    assert looks_like_artifact_digest(digest)
    assert looks_like_artifact_digest(f"sha256:{digest}")
    with pytest.raises(JudgeError, match="not a judge model"):
        guard_judge_call(digest, (JUDGE_MODEL_ID, digest), "payload")


@pytest.mark.parametrize("token", ["safetensors", "GGUF", "nvfp4", "model.ckpt"])
def test_a_weights_payload_is_never_sent_to_a_judge(token: str) -> None:
    with pytest.raises(JudgeError, match="not a judge payload"):
        guard_judge_call(JUDGE_MODEL_ID, (JUDGE_MODEL_ID,), f"here are the {token} bytes")


def test_model_ids_compare_the_way_the_pin_does() -> None:
    pinned = "nvidia/Cosmos3-Super-Text2Image"
    assert model_matches("NVIDIA/cosmos3-super-text2image", pinned)
    # The revision is pinned separately, so a stale card cannot pass as the pin.
    assert model_matches("nvidia/Cosmos3-Super-Text2Image@da579b9", pinned)
    assert not model_matches("nvidia/Cosmos3-Super-Image2Video", pinned)
    assert not model_matches("", pinned)


def test_images_are_sent_inline_rather_than_by_url() -> None:
    # The judge must not have to reach the pod, and a URL it fetched later
    # would not be the bytes that were scored.
    url = image_data_url(b"\x89PNG fake")
    assert url.startswith("data:image/png;base64,")
