"""Relearn LLM live eval image.

The control plane boots this image on a digest pin, stages a
`HarvestRequest` into `/tmp/relearn_eval`, runs

    relearn-eval score --request request.json --out metrics.json

and accepts a score only when the pod prints `RELEARN_METRICS=<document>` and
`RELEARN_EVAL_OK`. The document is bound to the run that asked for it, so a pod
cannot answer with another artifact's numbers or replay an earlier run.

Contract: `docs/RELEARN.md` § Eval image contract in `CortexLM/cortex`, client
in `crates/relearn-lium-harvest`. Local notes: `docs/EVAL-IMAGE.md`.

Nothing in this package can produce a score without the model: there is no
offline harness, no simulated series, and no fallback judge.
"""

from __future__ import annotations

from .contract import (
    BASE_CHAMPION_ARTIFACT,
    BASE_CHAMPION_RUN,
    METRICS_MARKER,
    METRICS_SCHEMA_VERSION,
    OK_MARKER,
    POD_WORKDIR,
    ContractError,
    EvalDocument,
    Measurement,
    ShuffleEvidence,
    decode_document,
    encode_document,
    marker_line,
)
from .harvest import TranscriptError, accept, extract_metrics, has_ok_marker
from .request import HarvestRequest, HoldoutItem, RequestError
from .verify import VerificationError, verify_document

__all__ = [
    "BASE_CHAMPION_ARTIFACT",
    "BASE_CHAMPION_RUN",
    "METRICS_MARKER",
    "METRICS_SCHEMA_VERSION",
    "OK_MARKER",
    "POD_WORKDIR",
    "ContractError",
    "EvalDocument",
    "HarvestRequest",
    "HoldoutItem",
    "Measurement",
    "RequestError",
    "ShuffleEvidence",
    "TranscriptError",
    "VerificationError",
    "accept",
    "decode_document",
    "encode_document",
    "extract_metrics",
    "has_ok_marker",
    "marker_line",
    "verify_document",
]

__version__ = "1.0.0"
