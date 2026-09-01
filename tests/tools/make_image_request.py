"""Build a dev Relearn Image harvest request, as the control plane stages one.

Test-side only: it is not installed into the image. CI pipes its output into a
container to exercise the entrypoint end to end — the staging path, the
validation, and the fail-closed behaviour of a pod with no judge and no
generator.

    python tests/tools/make_image_request.py --prompts 25 > request.json

The prompts here are synthetic filler. A real request carries the operator's
private split, which never enters this repo.
"""

from __future__ import annotations

import argparse
import json
import sys

from relearn_image_eval.commitment import frozen_prompt_commitment
from relearn_image_eval.pins import (
    BASE_MODEL_ID,
    BASE_MODEL_LICENSE,
    CHALLENGE_ID,
    JUDGE_MODEL_ID,
    SCHEMA_VERSION,
)
from relearn_image_eval.request import ArtifactManifest, FrozenPrompt, ImageHarvestRequest

DEV_IMAGE_DIGEST = f"sha256:{'ab' * 32}"


def build(
    prompts: int,
    *,
    submission: str,
    artifact: str,
    image_digest: str,
    challenge_id: str = CHALLENGE_ID,
    variations: int = 4,
) -> ImageHarvestRequest:
    # Bench ids 1..=1000. The public split takes the low ids and the holdout
    # the high ones, so the two never overlap.
    public = tuple(
        FrozenPrompt(id=index, text=f"dev public prompt {index}: a red cube on a table")
        for index in range(1, prompts + 1)
    )
    holdout = tuple(
        FrozenPrompt(id=900 + index, text=f"dev holdout prompt {index}: two cats on a rug")
        for index in range(1, prompts + 1)
    )
    return ImageHarvestRequest(
        schema_version=SCHEMA_VERSION,
        challenge_id=challenge_id,
        submission_digest=submission,
        artifact_digest=artifact,
        base_model=BASE_MODEL_ID,
        judge_model=JUDGE_MODEL_ID,
        eval_image_digest=image_digest,
        holdout_commitment=frozen_prompt_commitment(holdout),
        pin_salt="cortex-t2i-v0",
        variations_per_prompt=variations,
        holdout=holdout,
        public=public,
        manifest=ArtifactManifest(base=BASE_MODEL_ID, base_license=BASE_MODEL_LICENSE),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=int, default=25)
    parser.add_argument("--variations", type=int, default=4)
    parser.add_argument("--submission", default="dev-frozen-image-1")
    parser.add_argument("--artifact", default="cd" * 32)
    parser.add_argument("--image-digest", default=DEV_IMAGE_DIGEST)
    parser.add_argument("--challenge-id", default=CHALLENGE_ID)
    args = parser.parse_args(argv)

    request = build(
        args.prompts,
        submission=args.submission,
        artifact=args.artifact,
        image_digest=args.image_digest,
        challenge_id=args.challenge_id,
        variations=args.variations,
    )
    request.validate()
    json.dump(request.to_wire(), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
