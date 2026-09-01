"""The five Qwen-Image-Bench level-1 pillars.

Port of `relearn_t2i_task::L1Dimension`. The wire names are the crate's serde
representation (`rename_all = "snake_case"`), because the control plane keys
`holdout_by_pillar` by them and gates each pillar separately. The card names
are what Q-Judger actually emits as JSON keys.

Pillars are gated one at a time on the control plane, which is the point of
carrying them separately in the document: a large Alignment gain must not be
able to hide a Quality collapse behind a higher total.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pillar:
    """One level-1 pillar."""

    #: Wire name, matching the crate's serde encoding.
    wire: str
    #: Card-spelled name, matching Q-Judger's JSON key.
    card: str


QUALITY = Pillar("quality", "Quality")
AESTHETICS = Pillar("aesthetics", "Aesthetics")
ALIGNMENT = Pillar("alignment", "Alignment")
REAL_WORLD_FIDELITY = Pillar("real_world_fidelity", "Real-world Fidelity")
CREATIVE_GENERATION = Pillar("creative_generation", "Creative Generation")

#: All five pillars, in the paper's order. The order is part of the wire
#: format, so it is not sorted here.
ALL_PILLARS = (
    QUALITY,
    AESTHETICS,
    ALIGNMENT,
    REAL_WORLD_FIDELITY,
    CREATIVE_GENERATION,
)

#: Wire names of every pillar, for verification.
PILLAR_WIRE_NAMES = tuple(pillar.wire for pillar in ALL_PILLARS)


def _normalize(key: str) -> str:
    return "".join(char.lower() for char in key if char.isalnum())


def parse_pillar(key: str) -> Pillar | None:
    """Parse a pillar from a Q-Judger JSON key.

    Tolerant of case, spaces, hyphens, and underscores, because the judge's
    key spelling is the model's and not ours. An unknown key returns `None`
    and the caller drops it: silently folding an unrecognised pillar into a
    known one would move a score for reasons nobody could audit.
    """
    wanted = _normalize(key)
    for pillar in ALL_PILLARS:
        if _normalize(pillar.card) == wanted or _normalize(pillar.wire) == wanted:
            return pillar
    return None
