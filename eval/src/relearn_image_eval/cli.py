"""`relearn-image-eval` — the Relearn Image entrypoint.

    relearn-image-eval score  --request request.json --out metrics.json
    relearn-image-eval verify --request request.json --metrics metrics.json
    relearn-image-eval verify --request request.json --transcript run.log

`score` is what the control plane runs. It reads the staged request, generates
and judges every cell, writes the one-line sidecar the harvest `cat`s, and only
then prints

    RELEARN_METRICS=<document>
    RELEARN_EVAL_OK

`RELEARN_EVAL_OK` is the last thing the process does, and it is printed only
after the document has passed the image's copy of the control plane's own
acceptance checks. Any failure — an unusable request, a Flux base, no judge, a
pipeline that will not load, a document that would be refused — exits non-zero
with no marker and no sidecar, which the control plane reports as a 503. There
is no path here that prints a document the model did not produce.

stdout carries the document and the markers, nothing else. Diagnostics go to
stderr, and they carry ids, counts, and digests — never a prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from relearn_common.artifact import resolve_artifact, scrub
from relearn_common.clitools import (
    EXIT_ERROR,
    EXIT_REFUSED,
    check_image_identity,
    configure_logging,
    read_text,
    write_sidecar,
)
from relearn_common.errors import ContractError
from relearn_common.transcript import accept
from relearn_common.wire import OK_MARKER, encode_line, marker_line

from .contract import ImageDocument
from .generator import build_generator
from .judge import build_judge
from .pins import BASE_CHAMPION_ARTIFACT, ENV_PREFIX, POD_WORKDIR
from .request import ImageHarvestRequest, read_request
from .scoring import score_request
from .verify import verify_document

log = logging.getLogger("relearn_image_eval")


def _self_accept(document: ImageDocument, request: ImageHarvestRequest) -> None:
    """Prove the control plane would accept this transcript, before printing it."""
    verify_document(document, request)
    transcript = f"{marker_line(document.to_wire())}\n{OK_MARKER}\n"
    accepted = accept(
        transcript,
        ImageDocument.from_wire,
        lambda parsed: verify_document(parsed, request),
    )
    # Compare the encoded forms: publishing rounds the scores, so the decoded
    # document is only equal to the measured one after that rounding.
    if encode_line(accepted.to_wire()) != encode_line(document.to_wire()):
        raise ContractError("document does not survive its own encoding")


def _score(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    request = read_request(args.request)
    request.validate()
    check_image_identity(request.eval_image_digest, prefix=ENV_PREFIX)
    log.info(
        "scoring challenge=%s submission=%s artifact=%s holdout_cells=%d public_cells=%d",
        request.challenge_id,
        request.submission_digest,
        request.artifact_digest,
        len(request.holdout) * request.variations_per_prompt,
        len(request.public) * request.variations_per_prompt,
    )

    judge = build_judge(request.judge_model)
    artifact_dir = resolve_artifact(
        request.artifact_digest,
        workdir,
        prefix=ENV_PREFIX,
        base_champion_id=BASE_CHAMPION_ARTIFACT,
        artifact_uri=request.artifact_uri,
    )
    generator = None
    try:
        generator = build_generator(request.base_model, artifact_dir, request.sampler)
        document = score_request(request, generator, judge)
        _self_accept(document, request)
        write_sidecar(Path(args.out), document.to_wire())
    finally:
        if generator is not None:
            generator.close()
        if artifact_dir is not None and not args.keep_artifact:
            scrub(workdir / "artifact")

    # Markers last, and in this order: a consumer that reads OK without a
    # document, or a document the checks above rejected, must never happen.
    print(marker_line(document.to_wire()))
    print(OK_MARKER)
    sys.stdout.flush()
    return 0


def _verify(args: argparse.Namespace) -> int:
    request = read_request(args.request)
    request.validate()
    if args.transcript:
        document = accept(
            read_text(args.transcript),
            ImageDocument.from_wire,
            lambda parsed: verify_document(parsed, request),
        )
    else:
        document = ImageDocument.from_wire(json.loads(read_text(args.metrics)))
        verify_document(document, request)
    log.info(
        "document accepted: submission=%s artifact=%s holdout_cells=%d",
        document.identity.submission_digest,
        document.identity.artifact_digest,
        len(document.measurement.holdout),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relearn-image-eval",
        description=(
            "Score a Relearn Image submission: generate the frozen prompt split "
            "at the shared seeds and judge it with Q-Judger."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    score = subcommands.add_parser("score", help="score one artifact")
    score.add_argument(
        "--request",
        default="request.json",
        help="staged request, or - for stdin (default: request.json)",
    )
    score.add_argument(
        "--out", default="metrics.json", help="sidecar to write (default: metrics.json)"
    )
    score.add_argument(
        "--workdir", default=POD_WORKDIR, help=f"run workdir (default: {POD_WORKDIR})"
    )
    score.add_argument(
        "--keep-artifact",
        action="store_true",
        help="leave the unpacked artifact in place (local debugging only)",
    )
    score.set_defaults(handler=_score)

    verify = subcommands.add_parser(
        "verify", help="check a document or a pod transcript against a request"
    )
    verify.add_argument("--request", default="request.json")
    group = verify.add_mutually_exclusive_group(required=True)
    group.add_argument("--metrics", help="a metrics.json sidecar")
    group.add_argument("--transcript", help="captured pod stdout")
    verify.set_defaults(handler=_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging(ENV_PREFIX)
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ContractError as exc:
        log.error("refused: %s", exc)
        return EXIT_REFUSED
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("failed: %s: %s", type(exc).__name__, exc)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
