"""Shared generation seeds, so two artifacts are compared on one trajectory.

Byte-for-byte port of `relearn_t2i_task::derive_generation_seed` and
`cell_key`. Every miner generates the scored split at these seeds, which is
what makes the paired A/B test a comparison of models rather than of sampler
luck. The salt lives in the pin so an operator can rotate the whole lattice
without changing the formula.

`tests/test_image_golden_vectors.py` pins the outputs of this module against
values produced by compiling and running the cortex crate itself. A refactor
that changes the preimage fails there rather than silently making every
miner's images incomparable.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator

from .pins import SEED_DOMAIN


def _u32_le(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFF_FFFF)


def derive_generation_seed(prompt_id: int, variation_index: int, pin_salt: str) -> int:
    """Deterministic generation seed for one `(prompt_id, variation_index)` cell.

    The result is masked into positive `i64` range because the Diffusers and
    vLLM-Omni paths both take a signed seed.
    """
    digest = hashlib.sha256()
    digest.update(SEED_DOMAIN)
    digest.update(b"\xff")
    digest.update(pin_salt.encode("utf-8"))
    digest.update(b"\xff")
    digest.update(_u32_le(int(prompt_id)))
    digest.update(b"\xff")
    digest.update(_u32_le(int(variation_index)))
    return int.from_bytes(digest.digest()[:8], "little") >> 1


def cell_key(prompt_id: int, variation_index: int) -> str:
    """Stable key for one scored image: `p{prompt_id}#v{variation_index}`.

    Both sides of a paired comparison key on this string, so the test only ever
    lines up images generated from the same prompt at the same seed. The key
    carries an id, never prompt text, which is what keeps the private holdout
    out of a document that is logged and stored off the pod.
    """
    return f"p{prompt_id}#v{variation_index}"


def seed_cells(
    prompt_ids: list[int], variations_per_prompt: int, pin_salt: str
) -> Iterator[tuple[int, int, int]]:
    """Every `(prompt_id, variation_index, seed)` cell, in deterministic order.

    Port of `RelearnT2iPin::seed_cells`: ids are sorted and de-duplicated
    first, so the cell order does not depend on how the request happened to be
    serialized.
    """
    for prompt_id in sorted(set(prompt_ids)):
        for variation_index in range(variations_per_prompt):
            yield (
                prompt_id,
                variation_index,
                derive_generation_seed(prompt_id, variation_index, pin_salt),
            )
