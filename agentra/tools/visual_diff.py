"""Perceptual image hashing for screenshot change detection."""

from __future__ import annotations

import io

from PIL import Image


def compute_image_hash(image_data: bytes, hash_size: int = 8) -> str:
    """Compute an average perceptual hash (aHash) for a PNG/JPEG image.

    The image is resized to *hash_size* x *hash_size* grayscale, then each
    pixel is compared against the mean to produce a compact binary fingerprint
    returned as a hex string.
    """
    img = Image.open(io.BytesIO(image_data)).convert("L").resize(
        (hash_size, hash_size), Image.LANCZOS
    )
    pixels = list(img.getdata()) if not hasattr(img, "get_flattened_data") else list(img.get_flattened_data())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for px in pixels:
        bits = (bits << 1) | (1 if px > mean else 0)
    hex_len = (hash_size * hash_size + 3) // 4
    return format(bits, f"0{hex_len}x")


def _hamming_distance(a: str, b: str) -> int:
    """Return the number of differing bits between two hex hash strings."""
    val_a = int(a, 16)
    val_b = int(b, 16)
    return bin(val_a ^ val_b).count("1")


def images_are_similar(
    hash_a: str | None,
    hash_b: str | None,
    threshold: int = 5,
) -> bool:
    """Return *True* when two image hashes are within *threshold* bits.

    Returns *False* when either hash is ``None`` (e.g. no previous screenshot).
    """
    if hash_a is None or hash_b is None:
        return False
    return _hamming_distance(hash_a, hash_b) <= threshold
