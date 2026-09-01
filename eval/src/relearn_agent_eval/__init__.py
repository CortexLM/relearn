"""The Relearn Agent live eval image (`relearn-agent`).

Replays recorded tool-use episodes against the model under test, grades every
action against what was actually recorded, and replays the same episodes with
the tool results withheld and with observation pixels destroyed. A model that
ignores its tools or its screenshots does not lose ground on those replays,
which is what fails it.

Not the Relearn LLM image under another name: same base checkpoint, different
holdout, different graders, different controls, different image.
"""

from .pins import BASE_MODEL_ID, CHALLENGE_ID, CHALLENGE_IDS, POD_WORKDIR

__all__ = ["BASE_MODEL_ID", "CHALLENGE_ID", "CHALLENGE_IDS", "POD_WORKDIR"]
