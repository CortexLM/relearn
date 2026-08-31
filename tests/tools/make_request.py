"""Build a dev harvest request, the way the control plane stages one.

Test-side only: it is not installed into the image. CI pipes its output into a
container to exercise the entrypoint end to end — the staging path, the
validation, and the fail-closed behaviour of a pod with no judge and no model.

    python tests/tools/make_request.py --size 8 > request.json

The holdout here is synthetic filler. A real request carries the operator's
private split, which never enters this repo.
"""

from __future__ import annotations

import argparse
import json
import sys

from relearn_eval import HarvestRequest, HoldoutItem
from relearn_eval.commitment import holdout_commitment
from relearn_eval.contract import METRICS_SCHEMA_VERSION

DEV_IMAGE_DIGEST = f"sha256:{'ab' * 32}"


def build(size: int, *, submission: str, artifact: str, image_digest: str) -> HarvestRequest:
    holdout = tuple(
        HoldoutItem(
            id=800 + index,
            prompt=f"dev holdout question {index} with enough words for a trigram",
            dataset_id="dev",
            task="text",
        )
        for index in range(1, size + 1)
    )
    return HarvestRequest(
        schema_version=METRICS_SCHEMA_VERSION,
        submission_digest=submission,
        artifact_digest=artifact,
        base_model="Qwen/Qwen3.8-27B",
        teacher_model="glm-5.3",
        eval_image_digest=image_digest,
        holdout_commitment=holdout_commitment(holdout),
        holdout=holdout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--submission", default="dev-frozen-1")
    parser.add_argument("--artifact", default="ab" * 32)
    parser.add_argument("--image-digest", default=DEV_IMAGE_DIGEST)
    args = parser.parse_args(argv)

    request = build(
        args.size,
        submission=args.submission,
        artifact=args.artifact,
        image_digest=args.image_digest,
    )
    request.validate()
    json.dump(request.to_wire(), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
