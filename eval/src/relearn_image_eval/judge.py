"""Q-Judger: the only judge this challenge accepts.

Port of `relearn_t2i_judge`. Q-Judger (`Qwen/Qwen-Image-Bench`, Apache-2.0,
fine-tuned from Qwen3.6-27B) is handed a bench prompt plus one generated image
and replies with a chain-of-thought preamble followed by a JSON score tree:

    { "<L1 pillar>": { "<L2 group>": { "<L3 item>": { "score": 0|1|2|"N/A" } } } }

The paper's mapping and aggregation are reproduced exactly: raw `0|1|2` map to
`0|60|100`, `N/A` is **excluded rather than zeroed**, level 3 averages into
level 2, level 2 into level 1, and the five pillars average into the total.

Zeroing `N/A` is not a rounding choice, it is a correctness bug: a prompt where
a criterion does not apply would be punished for it, and prompts differ in
which criteria apply, so the punishment would land unevenly across the split.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from relearn_common.env import env, env_first
from relearn_common.errors import ContractError
from relearn_common.jsonscan import extract_json_object
from relearn_common.judge import (
    ChatJudge,
    JudgeError,
    guard_judge_call,
    image_part,
    text_part,
)

from .pillars import ALIGNMENT, ALL_PILLARS, Pillar, parse_pillar
from .pins import (
    ENV_PREFIX,
    JUDGE_ENABLE_THINKING,
    JUDGE_INFERENCE,
    JUDGE_MAX_NEW_TOKENS,
    JUDGE_MODEL_ID,
    MAPPED_EXCEL,
    MAPPED_FAIL,
    MAPPED_MAX,
    MAPPED_PASS,
)

log = logging.getLogger(__name__)

#: What the judge is asked to do. Deliberately terse: the scoring rubric is the
#: model's own, from its card, and restating it here would be a second rubric.
_JUDGE_SYSTEM = (
    "You are Qwen-Image-Bench. Evaluate the generated image against the prompt "
    "across the five level-1 dimensions. Think first, then output the score "
    "tree as a single JSON object where every leaf is 0, 1, 2, or \"N/A\"."
)

#: The faithfulness spot check. A separate, narrower pass than the full tree:
#: it enumerates the concrete, checkable properties (object counts, rendered
#: text, spatial relations) rather than asking for an overall impression.
_SPOT_CHECK_SYSTEM = (
    "You are Qwen-Image-Bench performing a targeted prompt-faithfulness check. "
    "Verify only concrete, checkable claims in the prompt: how many of each "
    "object appear, whether text the prompt asks for is rendered and spelled "
    "correctly, and whether the stated spatial relations hold. Output a single "
    "JSON object of Alignment sub-criteria whose leaves are 0, 1, 2, or "
    "\"N/A\". Do not score style, beauty, or overall quality."
)

#: Level-3 verdicts the judge may return, on the paper scale.
_RAW_TO_MAPPED: dict[str, float | None] = {
    "0": MAPPED_FAIL,
    "1": MAPPED_PASS,
    "2": MAPPED_EXCEL,
    "n/a": None,
    "na": None,
}


@dataclass(frozen=True)
class ImageScore:
    """Aggregated Q-Judger scores for one generated image."""

    per_pillar: dict[str, float]
    total: float
    scored_items: int
    na_items: int

    def normalized_total(self) -> float:
        """Total on the `0..=1` scale the paired test compares.

        Normalizing here makes one `prism_competition` dead-zone unit equal one
        paper point.
        """
        return self.total / MAPPED_MAX

    def normalized_pillar(self, pillar: Pillar) -> float | None:
        value = self.per_pillar.get(pillar.wire)
        return None if value is None else value / MAPPED_MAX


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _raw_score(value: object, path: str) -> float | None:
    """Map one level-3 verdict, or refuse it.

    An unrecognised value is an error rather than a coerced zero: a judge that
    started emitting a new vocabulary must fail loudly, not quietly score every
    image lower.
    """
    if isinstance(value, bool):
        raise JudgeError(f"unparsable score {value!r} at {path}")
    if isinstance(value, int):
        key = str(value)
    elif isinstance(value, float) and value.is_integer():
        key = str(int(value))
    elif isinstance(value, str):
        key = value.strip().lower()
    else:
        raise JudgeError(f"unparsable score {value!r} at {path}")
    if key not in _RAW_TO_MAPPED:
        raise JudgeError(f"unparsable score {value!r} at {path}")
    return _RAW_TO_MAPPED[key]


def _parse_groups(pillar_name: str, value: object) -> dict[str, dict[str, float | None]]:
    if not isinstance(value, Mapping):
        raise JudgeError("no JSON object in judge reply")
    groups: dict[str, dict[str, float | None]] = {}
    for level2, level2_value in value.items():
        if not isinstance(level2_value, Mapping):
            continue
        leaves: dict[str, float | None] = {}
        for level3, level3_value in level2_value.items():
            raw = level3_value
            if isinstance(level3_value, Mapping):
                raw = level3_value.get("score", level3_value.get("Score", level3_value))
            leaves[str(level3)] = _raw_score(raw, f"{pillar_name}.{level2}.{level3}")
        if leaves:
            groups[str(level2)] = leaves
    return groups


def parse_reply(raw: str, assume_pillar: Pillar | None = None) -> dict[str, dict]:
    """Parse a Q-Judger reply into a score tree keyed by pillar wire name.

    Accepts either a full tree keyed by the five pillars, or a single-pillar
    tree when `assume_pillar` names which pillar was requested.
    """
    body = extract_json_object(raw)
    if body is None:
        raise JudgeError("no JSON object in judge reply")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge JSON parse: {exc}") from exc
    if not isinstance(value, Mapping):
        raise JudgeError("no JSON object in judge reply")

    tree: dict[str, dict] = {}
    if any(parse_pillar(str(key)) is not None for key in value):
        for key, sub in value.items():
            pillar = parse_pillar(str(key))
            if pillar is None:
                continue
            groups = _parse_groups(pillar.card, sub)
            if groups:
                tree[pillar.wire] = groups
    elif assume_pillar is not None:
        groups = _parse_groups(assume_pillar.card, value)
        if groups:
            tree[assume_pillar.wire] = groups

    if not tree:
        raise JudgeError("judge reply has no recognized L1 pillar")
    return tree


def aggregate(tree: Mapping[str, dict]) -> ImageScore:
    """Aggregate a score tree the way the paper does.

    Level 3 into level 2 averaging only non-`N/A` items; a level-2 group whose
    items are all `N/A` drops out entirely; level 2 into level 1 averages the
    surviving groups; the total averages the surviving pillars.
    """
    per_pillar: dict[str, float] = {}
    scored_items = 0
    na_items = 0

    for pillar_wire, groups in tree.items():
        group_means: list[float] = []
        for leaves in groups.values():
            usable: list[float] = []
            for mapped in leaves.values():
                if mapped is None:
                    na_items += 1
                else:
                    usable.append(mapped)
                    scored_items += 1
            group_mean = _mean(usable)
            if group_mean is not None:
                group_means.append(group_mean)
        pillar_mean = _mean(group_means)
        if pillar_mean is not None:
            per_pillar[pillar_wire] = pillar_mean

    total = _mean(list(per_pillar.values()))
    if total is None:
        # Every level-3 item was N/A. That is a failed judge run, and treating
        # it as a score of zero would read as "the model is terrible here".
        raise JudgeError("judge reply is entirely N/A")
    return ImageScore(
        per_pillar=per_pillar, total=total, scored_items=scored_items, na_items=na_items
    )


def score_reply(raw: str, assume_pillar: Pillar | None = None) -> ImageScore:
    """Parse and aggregate in one step."""
    return aggregate(parse_reply(raw, assume_pillar))


@dataclass
class QJudger:
    """Q-Judger over an operator-configured OpenAI-compatible endpoint."""

    chat: ChatJudge
    _pillars: tuple[Pillar, ...] = field(default=ALL_PILLARS, repr=False)

    def score_image(self, prompt: str, image: bytes) -> ImageScore:
        """Full five-pillar score for one generated image."""
        reply = self.chat.complete(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": [text_part(f"Prompt:\n{prompt}"), image_part(image)],
                },
            ],
            max_tokens=JUDGE_MAX_NEW_TOKENS,
        )
        return score_reply(reply)

    def spot_check_alignment(self, prompt: str, image: bytes) -> ImageScore:
        """A targeted Alignment-only pass, for the faithfulness evidence."""
        reply = self.chat.complete(
            [
                {"role": "system", "content": _SPOT_CHECK_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        text_part(
                            "Prompt:\n"
                            f"{prompt}\n\n"
                            "Check only counts, rendered text, and spatial relations."
                        ),
                        image_part(image),
                    ],
                },
            ],
            max_tokens=JUDGE_MAX_NEW_TOKENS,
        )
        return score_reply(reply, ALIGNMENT)


def build_judge(judge_model: str) -> QJudger:
    """Build the judge for this run from the operator's environment.

    # Raises
    [`ContractError`] when no endpoint is configured. Without Q-Judger there
    are no numbers, and the run must fail rather than guess at them.
    """
    url = env_first(f"{ENV_PREFIX}JUDGE_API_URL", "RELEARN_T2I_JUDGE_API_URL").rstrip("/")
    if not url:
        raise ContractError(
            f"{ENV_PREFIX}JUDGE_API_URL is unset; the pod has no judge and cannot score"
        )
    model = judge_model.strip() or JUDGE_MODEL_ID
    extra: dict[str, object] = dict(JUDGE_INFERENCE)
    extra["chat_template_kwargs"] = {"enable_thinking": JUDGE_ENABLE_THINKING}
    chat = ChatJudge(
        api_url=url,
        model=model,
        allowed_models=(JUDGE_MODEL_ID,),
        api_key=env_first(f"{ENV_PREFIX}JUDGE_API_KEY", "RELEARN_T2I_JUDGE_API_KEY"),
        timeout_secs=float(env(f"{ENV_PREFIX}JUDGE_TIMEOUT_SECS") or 300.0),
        extra_body=extra,
    )
    # Probe the guard before any image is generated, so a misconfigured judge
    # costs a refusal rather than an hour of GPU time.
    guard_judge_call(model, (JUDGE_MODEL_ID,), "configuration probe")
    return QJudger(chat=chat)
