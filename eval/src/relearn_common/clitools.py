"""Pieces every eval image's entrypoint needs, and must get identically right.

The sidecar rule in particular is easy to get subtly wrong and impossible to
notice locally: the harvest reconstructs the marker line with `printf
'RELEARN_METRICS='; cat metrics.json`, so a trailing newline splits the marker
line and a partial write hands the control plane half a document. Both would
show up as an unexplained 503 long after the pod is gone.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from .env import env
from .errors import ContractError
from .wire import encode_line

#: Refused: a bad request, an unacceptable document, an unreachable judge.
EXIT_REFUSED = 2

#: Something unexpected. Still fail-closed; the log tail is the diagnosis.
EXIT_ERROR = 1


def configure_logging(prefix: str) -> None:
    """Send diagnostics to stderr, at the operator's level.

    stdout is reserved for the document and the two markers. Nothing else is
    ever printed there, because the harvest reads it.
    """
    level = env(f"{prefix}LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def check_image_identity(requested_digest: str, *, prefix: str) -> None:
    """Refuse a request meant for a different build of this image.

    The digest cannot be baked in at build time — it does not exist until the
    build is pushed — so this only fires when the operator pins it on the pod.
    """
    declared = env(f"{prefix}EVAL_IMAGE_DIGEST")
    if declared and declared != requested_digest.strip():
        raise ContractError(
            f"request eval_image_digest is not this image ({requested_digest} vs {declared})"
        )


def write_sidecar(path: Path, payload: Mapping[str, object]) -> None:
    """Write the document as exactly one line, with no trailing newline.

    Written to a temporary file and renamed, so the sidecar either exists
    complete or does not exist at all.
    """
    body = encode_line(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_text(source: str | Path) -> str:
    """Read a staged file, or stdin when `source` is `-`.

    The control plane delivers the request over stdin so no run input is ever
    interpolated into the remote command.
    """
    if str(source) == "-":
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8")
