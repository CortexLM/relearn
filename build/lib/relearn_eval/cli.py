"""`relearn-eval` — the image entrypoint.

    relearn-eval score  --request request.json --out metrics.json
    relearn-eval verify --request request.json --metrics metrics.json
    relearn-eval verify --request request.json --transcript run.log

`score` is what the control plane runs. It reads the staged request, scores the
artifact, writes the one-line sidecar the harvest `cat`s, and only then prints

    RELEARN_METRICS=<document>
    RELEARN_EVAL_OK

`RELEARN_EVAL_OK` is the last thing the process does, and it is printed only
after the document has passed the control plane's own acceptance checks
(`verify`, `harvest`). Any failure — an unusable request, no judge, a missing
image, a document that would be refused — exits non-zero with no marker and no
sidecar, which the control plane reports as a 503. There is no path here that
prints a document the model did not produce.

stdout carries the document and the markers, nothing else. Diagnostics go to
stderr, and they carry ids, counts, and digests — never a holdout prompt.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .artifact import resolve_artifact, scrub
from .contract import (
    OK_MARKER,
    POD_WORKDIR,
    ContractError,
    EvalDocument,
    decode_document,
    encode_document,
    marker_line,
)
from .harvest import accept
from .request import HarvestRequest, RequestError, read_request
from .runner import ModelRunner, build_runner
from .scoring import score_request
from .teacher import build_teacher
from .verify import verify_document

log = logging.getLogger("relearn_eval")

#: Refused: a bad request, an unacceptable document, an unreachable judge.
EXIT_REFUSED = 2
#: Something unexpected. Still fail-closed; the log tail is the diagnosis.
EXIT_ERROR = 1


def _configure_logging() -> None:
    level = os.environ.get("RELEARN_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _check_image_identity(request: HarvestRequest) -> None:
    """Refuse a request meant for a different image build.

    The digest cannot be baked in at build time — it does not exist until the
    build is pushed — so this only fires when the operator pins it on the pod.
    """
    declared = os.environ.get("RELEARN_EVAL_IMAGE_DIGEST", "").strip()
    if declared and declared != request.eval_image_digest.strip():
        raise RequestError(
            "request eval_image_digest is not this image "
            f"({request.eval_image_digest} vs {declared})"
        )


def _write_sidecar(path: Path, document: EvalDocument) -> None:
    """Write the document as exactly one line, with no trailing newline.

    The harvest reads it with `printf '<marker>'; cat metrics.json`, so a
    trailing newline would split the marker line and a partial write would hand
    the control plane half a document. Written to a temporary file and renamed
    so the sidecar either exists complete or not at all.
    """
    body = encode_document(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _self_accept(document: EvalDocument, request: HarvestRequest) -> None:
    """Prove the control plane would accept this transcript, before printing it."""
    verify_document(document, request)
    transcript = f"{marker_line(document)}\n{OK_MARKER}\n"
    accepted = accept(transcript, request)
    if accepted != document:
        raise ContractError("document does not survive its own encoding")


def _score(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    request = read_request(args.request)
    request.validate()
    _check_image_identity(request)
    log.info(
        "scoring submission=%s artifact=%s holdout_items=%d vision_families=%s",
        request.submission_digest,
        request.artifact_digest,
        len(request.holdout),
        ",".join(request.vision_families()) or "none",
    )

    teacher = build_teacher(request.teacher_model)
    artifact_dir = resolve_artifact(request.artifact_digest, workdir, request.artifact_uri)
    runner: ModelRunner | None = None
    try:
        runner = build_runner(request.base_model, artifact_dir)
        document = score_request(request, runner, teacher)
        _self_accept(document, request)
        _write_sidecar(Path(args.out), document)
    finally:
        if runner is not None:
            runner.close()
        if artifact_dir is not None and not args.keep_artifact:
            scrub(workdir / "artifact")

    # Markers last, and in this order: a consumer that reads OK without a
    # document, or a document the checks above rejected, must never happen.
    print(marker_line(document))
    print(OK_MARKER)
    sys.stdout.flush()
    return 0


def _verify(args: argparse.Namespace) -> int:
    request = read_request(args.request)
    request.validate()
    if args.transcript:
        document = accept(Path(args.transcript).read_text(encoding="utf-8"), request)
    else:
        document = decode_document(Path(args.metrics).read_text(encoding="utf-8"))
        verify_document(document, request)
    log.info(
        "document accepted: submission=%s artifact=%s holdout=%d",
        document.submission_digest,
        document.artifact_digest,
        len(document.measurement.holdout),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relearn-eval",
        description="Score a Relearn submission on the holdout the request delivers.",
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
    _configure_logging()
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
