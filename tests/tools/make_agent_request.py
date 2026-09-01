"""Build a dev Relearn Agent harvest request, as the control plane stages one.

Test-side only: it is not installed into the image. CI pipes its output into a
container to exercise the entrypoint end to end — the staging path, the
validation, and the fail-closed behaviour of a pod with no teacher and no
model.

    python tests/tools/make_agent_request.py --traces 100 > request.json

The episodes here are synthetic filler. A real request carries the operator's
private recordings, which never enter this repo.
"""

from __future__ import annotations

import argparse
import json
import sys

from relearn_agent_eval.commitment import trace_commitment
from relearn_agent_eval.pins import BASE_MODEL_ID, CHALLENGE_ID, SCHEMA_VERSION, TEACHER_MODEL_ID
from relearn_agent_eval.request import AgentHarvestRequest, ToolSchema, Trace, TraceStep

DEV_IMAGE_DIGEST = f"sha256:{'ab' * 32}"

_TOOLS = (
    ToolSchema(
        name="lookup_record",
        description="Fetch one record by id.",
        parameters={"record_id": {"type": "string"}},
    ),
    ToolSchema(
        name="file_report",
        description="File a report about a record.",
        parameters={"record_id": {"type": "string"}, "owner": {"type": "string"}},
    ),
)


def _trace(index: int) -> Trace:
    record = f"REC-{index:04d}"
    owner = f"owner{index % 7}"
    # The second step's `owner` argument exists only in the first step's
    # observation, so a model that ignores observations cannot produce it.
    # That is what the tool-blind control measures.
    return Trace(
        id=4000 + index,
        goal=f"Find who owns record {record} and file a report assigned to them.",
        dataset_id="dev",
        tools=_TOOLS,
        steps=(
            TraceStep(
                index=0,
                tool="lookup_record",
                arguments_json=json.dumps({"record_id": record}, separators=(",", ":")),
                observation=json.dumps({"owner": owner, "state": "open"}),
            ),
            TraceStep(
                index=1,
                tool="file_report",
                arguments_json=json.dumps(
                    {"owner": owner, "record_id": record}, separators=(",", ":"), sort_keys=True
                ),
                observation=json.dumps({"report": f"RPT-{index}"}),
            ),
        ),
        final_answer=f"Record {record} is owned by {owner}; filed RPT-{index} against it.",
    )


def build(
    traces: int, *, submission: str, artifact: str, image_digest: str
) -> AgentHarvestRequest:
    holdout = tuple(_trace(index) for index in range(1, traces + 1))
    return AgentHarvestRequest(
        schema_version=SCHEMA_VERSION,
        challenge_id=CHALLENGE_ID,
        submission_digest=submission,
        artifact_digest=artifact,
        base_model=BASE_MODEL_ID,
        teacher_model=TEACHER_MODEL_ID,
        eval_image_digest=image_digest,
        holdout_commitment=trace_commitment(holdout),
        holdout=holdout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=int, default=100)
    parser.add_argument("--submission", default="dev-frozen-agent-1")
    parser.add_argument("--artifact", default="ef" * 32)
    parser.add_argument("--image-digest", default=DEV_IMAGE_DIGEST)
    args = parser.parse_args(argv)

    request = build(
        args.traces,
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
