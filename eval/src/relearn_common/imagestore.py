"""Pixels: fetching them, destroying them on purpose, and comparing them.

Two challenges need image bytes for reasons that look different and are the
same underneath.

The Agent challenge replays traces whose observations include screenshots. The
pixels are not in the request and never in git; an item names its image by
`sha256` and the operator mounts a content-addressed store on the pod. Every
file is verified against the hash the item declared, because an observation
scored without its image is not a score.

Both challenges also need to destroy an image deliberately. A model that
answers as well on shuffled pixels was not looking at the picture, and that is
the only way to tell reading from guessing from the prompt. The permutation is
seeded from the image hash alone, so the champion and every challenger see the
*same* destroyed image and the control is a comparison rather than a coin flip.

The Image challenge additionally needs to ask "is this the same picture as
before" for its seed-replay evidence, which is what [`descriptor`] and
[`cosine_distance`] are for.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .env import env
from .errors import ContractError

#: Side of the square the replay descriptor downsamples to. Small enough that a
#: driver-level difference in a few pixels does not register, large enough that
#: a different composition does.
DESCRIPTOR_SIDE = 32


class ImageError(ContractError):
    """A run needed pixels it could not produce, or could not process them."""


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def store_root(prefix: str) -> Path | None:
    raw = env(f"{prefix}IMAGE_STORE")
    if not raw:
        return None
    root = Path(raw)
    if not root.is_dir():
        raise ImageError(f"{prefix}IMAGE_STORE {raw} is not a directory")
    return root


def _candidates(root: Path, image_hash: str) -> list[Path]:
    return [
        root / image_hash,
        *(root / f"{image_hash}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".webp")),
        root / image_hash[:2] / image_hash,
    ]


def load_image(image_hash: str, *, prefix: str) -> bytes:
    """Read and verify the bytes for one image hash.

    # Raises
    [`ImageError`] when no store is configured, the file is missing, or the
    bytes do not hash to `image_hash`.
    """
    wanted = image_hash.strip().lower()
    root = store_root(prefix)
    if root is None:
        raise ImageError(f"this run needs pixels but {prefix}IMAGE_STORE is unset")
    for candidate in _candidates(root, wanted):
        if not candidate.is_file():
            continue
        body = candidate.read_bytes()
        measured = sha256_bytes(body)
        if measured != wanted:
            raise ImageError(f"image store file hashes to {measured}, item declared {wanted}")
        return body
    raise ImageError(f"no image in the store for {wanted}")


def _imaging():
    """Import the imaging runtime, or refuse.

    A shuffle control that silently returned the original image, or a replay
    descriptor that silently returned a constant, would let a model pass a gate
    it never took. Both are refusals instead.
    """
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime extra
        raise ImageError("pillow / numpy are required to process pixels") from exc
    return BytesIO, np, Image


def shuffle_pixels(body: bytes, seed_hex: str) -> bytes:
    """Destroy an image's spatial content while keeping its pixel histogram.

    Same pixels, permuted. `seed_hex` is the item's own image hash, so the
    permutation is a property of the item rather than of the run.
    """
    bytes_io, np, image_mod = _imaging()
    with image_mod.open(bytes_io(body)) as opened:
        picture = opened.convert("RGB")
        pixels = np.asarray(picture)
    flat = pixels.reshape(-1, pixels.shape[-1])
    seed = int.from_bytes(hashlib.sha256(seed_hex.encode("utf-8")).digest()[:8], "little")
    order = np.random.default_rng(seed).permutation(flat.shape[0])
    shuffled = flat[order].reshape(pixels.shape)
    out = bytes_io()
    image_mod.fromarray(shuffled).save(out, format="PNG")
    return out.getvalue()


def descriptor(body: bytes) -> list[float]:
    """A deterministic, L2-normalized descriptor of an image.

    Not a learned embedding, and deliberately not: the eval image must not
    depend on a second model to decide whether two of its own generations are
    the same picture. This is the image itself, downsampled to
    [`DESCRIPTOR_SIDE`] squared RGB and normalized, which is enough to tell
    "same composition, different last bit" from "different picture" and is
    reproducible on any pod.
    """
    bytes_io, np, image_mod = _imaging()
    with image_mod.open(bytes_io(body)) as opened:
        small = opened.convert("RGB").resize(
            (DESCRIPTOR_SIDE, DESCRIPTOR_SIDE), image_mod.Resampling.BILINEAR
        )
        vector = np.asarray(small, dtype="float64").reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        # An all-black image has no direction; call it its own descriptor so
        # two black images compare as identical rather than as an error.
        return [0.0] * vector.size
    return [float(value) for value in (vector / norm)]


def cosine_distance(left: list[float], right: list[float]) -> float:
    """`1 - cosine` between two descriptors, clamped into `[0, 1]`."""
    if len(left) != len(right):
        raise ImageError("descriptors have different lengths")
    if not left:
        raise ImageError("descriptors are empty")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0 if left_norm == right_norm else 1.0
    return min(1.0, max(0.0, 1.0 - dot / (left_norm * right_norm)))
