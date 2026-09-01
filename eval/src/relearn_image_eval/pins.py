"""What the Relearn Image challenge is pinned to, and what it refuses.

Ported from `relearn_t2i_task` in
[`CortexLM/cortex`](https://github.com/CortexLM/cortex). Every constant here is
consensus-relevant: the seed domain and the commitment domain decide which
bytes are hashed, and the model ids decide which submissions are scorable at
all. Where this module and that crate disagree, the crate wins.

Two rules in this file are product decisions, not style:

1. **Flux is rejected.** Its weights are non-commercial, which is incoherent
   for a subnet that pays for redistributable artifacts. A Flux-family base is
   refused at request validation, before any generation happens — never scored
   low, never scored at all.
2. **Q-Judger is the only judge.** A run judged by anything else is not
   comparable with the champion's recorded run, so the judge id is pinned
   rather than configured.
"""

from __future__ import annotations

#: Challenge id as the subnet owner relocked it.
CHALLENGE_ID = "relearn-image"

#: The id this challenge shipped under while it was called "T2I". The cortex
#: crates are still `relearn-t2i-*` and `config/relearn-t2i-pin.toml` still
#: declares `challenge_id = "relearn-t2i"`, so the image accepts a request
#: under either name and echoes back whichever one it was asked with. That
#: keeps a rename on the control plane from needing a new eval image digest.
LEGACY_CHALLENGE_ID = "relearn-t2i"

#: Challenge ids this image will score.
CHALLENGE_IDS = (CHALLENGE_ID, LEGACY_CHALLENGE_ID)

#: Environment prefix for every operator setting this image reads.
ENV_PREFIX = "RELEARN_IMAGE_"

#: Where the control plane stages `request.json`, over stdin. Not a persisted
#: path: the request carries the private holdout for the length of the run.
POD_WORKDIR = "/tmp/relearn_image_eval"  # noqa: S108 - the control plane's staging path

#: Must equal the control plane's metrics schema for this challenge.
SCHEMA_VERSION = 1

#: Artifact id the control plane asks for when it measures the base checkpoint
#: with no miner fine-tune, which is how a live host records a champion.
BASE_CHAMPION_ARTIFACT = "base-relearn-image-champion"

#: Pinned generator seed miners fine-tune. `relearn_t2i_task::BASE_MODEL_ID`.
BASE_MODEL_ID = "nvidia/Cosmos3-Super-Text2Image"

#: License miners inherit from the pinned base.
BASE_MODEL_LICENSE = "OpenMDW-1.1"

#: The only judge. `relearn_t2i_task::JUDGE_MODEL_ID`.
JUDGE_MODEL_ID = "Qwen/Qwen-Image-Bench"

#: Published Qwen-Image-Bench prompt id range.
BENCH_PROMPT_ID_MIN = 1
BENCH_PROMPT_ID_MAX = 1000

#: Base families that may never be the miner seed.
#: `relearn_t2i_task::REJECTED_BASE_SUBSTRINGS`.
REJECTED_BASE_SUBSTRINGS = (
    "flux",
    "black-forest-labs",
    "blackforestlabs",
    "flux.1",
    "flux1",
)

#: Domain tag for per-image generation seeds. `relearn_t2i_task::SEED_DOMAIN`.
SEED_DOMAIN = b"base-relearn-t2i-seed-v1"

#: Domain tag for frozen prompt-set commitments.
#: `relearn_t2i_task::HOLDOUT_DOMAIN`.
HOLDOUT_DOMAIN = b"base-relearn-t2i-holdout-v1"

#: Level-3 verdicts on the paper scale. `relearn_t2i_judge::MAPPED_*`.
MAPPED_FAIL = 0.0
MAPPED_PASS = 60.0
MAPPED_EXCEL = 100.0
MAPPED_MAX = MAPPED_EXCEL

#: Largest share of level-3 items the judge may decline before the run is void.
#: `relearn_t2i_score::MAX_NA_RATE`.
MAX_NA_RATE = 0.25

#: Pinned cells regenerated for the seed-replay check.
#: `relearn_t2i_score::REPLAY_CELLS`.
REPLAY_CELLS = 3

#: Agentic faithfulness spot checks required for a verdict.
#: `relearn_t2i_score::MIN_FAITHFULNESS_CHECKS`.
MIN_FAITHFULNESS_CHECKS = 8

#: Paper-scale Alignment above which a cell counts as "prompt was followed".
#: `relearn_t2i_score::MIN_FAITHFULNESS_AGREEMENT` is the required *agreement
#: rate*; this is the per-cell threshold the two sides are compared at.
FAITHFULNESS_ALIGNMENT_THRESHOLD = 75.0

#: Minimum scored image cells per split. Below this the paired displacement
#: test refuses a verdict, so a request that cannot reach it can never promote
#: anything and is a configuration error rather than a permanent champion-hold.
#: `relearn_t2i_task::MIN_SCORED_CELLS`.
MIN_SCORED_CELLS = 100

#: Q-Judger inference, fixed by the model card. Part of the contract, not a
#: tuning knob: a judge run at a different temperature is not comparable with
#: the champion's recorded run. `relearn_t2i_judge::JudgeInference::default`.
JUDGE_INFERENCE: dict[str, object] = {
    "seed": 42,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
}

#: Generation budget for Q-Judger's thinking trace plus its JSON.
JUDGE_MAX_NEW_TOKENS = 4096

#: Chain-of-thought before the JSON, per the card.
JUDGE_ENABLE_THINKING = True


def base_is_rejected(model_id: str) -> bool:
    """Whether `model_id` names a base family this challenge refuses."""
    lowered = model_id.lower()
    return any(token in lowered for token in REJECTED_BASE_SUBSTRINGS)


def is_bench_prompt_id(prompt_id: int) -> bool:
    """Whether `prompt_id` is inside the published Qwen-Image-Bench range."""
    return BENCH_PROMPT_ID_MIN <= prompt_id <= BENCH_PROMPT_ID_MAX


def normalize_license(value: str) -> str:
    """Compare licenses the way `RelearnT2iPin::attest_artifact_base` does."""
    return value.strip().lower().replace(" ", "-").replace("_", "-")
