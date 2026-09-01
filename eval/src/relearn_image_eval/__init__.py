"""The Relearn Image live eval image (`relearn-image`, formerly `relearn-t2i`).

Generates the frozen Qwen-Image-Bench prompt split with the pinned Cosmos3
checkpoint at seeds the whole subnet shares, judges every image with Q-Judger,
and prints one metrics document bound to the run that asked for it.

Flux is never a base. Q-Judger is the only judge. There is no simulated
fallback: a pod that cannot generate or cannot reach the judge refuses.
"""

from .pins import CHALLENGE_ID, CHALLENGE_IDS, LEGACY_CHALLENGE_ID, POD_WORKDIR

__all__ = ["CHALLENGE_ID", "CHALLENGE_IDS", "LEGACY_CHALLENGE_ID", "POD_WORKDIR"]
