"""Contract machinery shared by the Relearn Image and Relearn Agent eval images.

Three challenges on this subnet ship a live eval image, and the control plane
drives all of them the same way: boot a digest-pinned image, stage a request,
read back one `RELEARN_METRICS=<document>` line followed by `RELEARN_EVAL_OK`.
That transport is what this package holds. Everything a challenge measures —
its request, its document, its graders — lives in the challenge's own package.

`relearn_eval` (the Relearn LLM image, `ghcr.io/cortexlm/relearn-eval`) does
not import this package. It is already published and pinned by digest, so it
keeps its own copy of the transport rather than being refactored underneath a
live pin. The duplication is deliberate; `tests/test_shared_contract.py` asserts
the two copies still agree on every wire constant.
"""

from .errors import ContractError
from .wire import (
    METRICS_MARKER,
    OK_MARKER,
    SCORE_DECIMALS,
    clamp_score,
    decode_line,
    encode_line,
    marker_line,
    mean,
    series_wire,
)

__all__ = [
    "METRICS_MARKER",
    "OK_MARKER",
    "SCORE_DECIMALS",
    "ContractError",
    "clamp_score",
    "decode_line",
    "encode_line",
    "marker_line",
    "mean",
    "series_wire",
]
