"""Image bytes for vision holdout items, and the pixel-shuffle control.

A holdout item names its image by `sha256` — the pixels are not in the request
and never in git. The operator mounts a content-addressed store on the pod
(`RELEARN_IMAGE_STORE`); the image verifies every file against the hash the
item declared. A vision item whose pixels are missing or wrong fails the run:
the document must carry one score per requested item, and a vision item scored
without its image is not a score.

The shuffle seed comes from the image hash alone, so the champion and every
challenger see the *same* destroyed image and the control is a comparison
rather than a coin flip.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .contract import ContractError


class ImageStoreError(ContractError):
    """A vision item's pixels could not be produced."""


def store_root() -> Path | None:
    raw = os.environ.get("RELEARN_IMAGE_STORE", "").strip()
    if not raw:
        return None
    root = Path(raw)
    if not root.is_dir():
        raise ImageStoreError(f"RELEARN_IMAGE_STORE {raw} is not a directory")
    return root


def _candidates(root: Path, image_hash: str) -> list[Path]:
    return [
        root / image_hash,
        *(root / f"{image_hash}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".webp")),
        root / image_hash[:2] / image_hash,
    ]


def load_image(image_hash: str) -> bytes:
    """Read and verify the bytes for one image hash.

    # Raises
    [`ImageStoreError`] when no store is configured, the file is missing, or
    the bytes do not hash to `image_hash`.
    """
    wanted = image_hash.strip().lower()
    root = store_root()
    if root is None:
        raise ImageStoreError(
            "holdout carries vision items but RELEARN_IMAGE_STORE is unset"
        )
    for candidate in _candidates(root, wanted):
        if not candidate.is_file():
            continue
        body = candidate.read_bytes()
        measured = hashlib.sha256(body).hexdigest()
        if measured != wanted:
            raise ImageStoreError(
                f"image store file hashes to {measured}, item declared {wanted}"
            )
        return body
    raise ImageStoreError(f"no image in the store for {wanted}")


def shuffle_pixels(body: bytes, image_hash: str) -> bytes:
    """Destroy the image's spatial content, keeping its pixel histogram.

    Same pixels, permuted: a model that answers as well on the shuffled image
    was not reading the image. The permutation is seeded from `image_hash`, so
    it is identical for every model scored on that item.

    # Raises
    [`ImageStoreError`] when the imaging runtime is missing — a shuffle control
    that silently returned the original image would let a text-only model pass
    the pixel-shuffle gate.
    """
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime extra
        raise ImageStoreError("pillow / numpy are required to shuffle pixels") from exc

    with Image.open(BytesIO(body)) as opened:
        picture = opened.convert("RGB")
        pixels = np.asarray(picture)
    flat = pixels.reshape(-1, pixels.shape[-1])
    seed = int.from_bytes(hashlib.sha256(bytes.fromhex(image_hash)).digest()[:8], "little")
    order = np.random.default_rng(seed).permutation(flat.shape[0])
    shuffled = flat[order].reshape(pixels.shape)
    out = BytesIO()
    Image.fromarray(shuffled).save(out, format="PNG")
    return out.getvalue()
