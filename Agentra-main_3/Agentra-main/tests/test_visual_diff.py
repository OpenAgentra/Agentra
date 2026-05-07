"""Tests for visual_diff — perceptual image hashing and comparison."""

from __future__ import annotations

import io

from PIL import Image

from agentra.tools.visual_diff import compute_image_hash, images_are_similar


def _make_png(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (64, 64)) -> bytes:
    """Create a minimal solid-colour PNG image."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_gradient_png(start: int = 0, end: int = 255, size: tuple[int, int] = (64, 64)) -> bytes:
    """Create a horizontal gradient PNG — useful for distinct hash values."""
    img = Image.new("L", size)
    for x in range(size[0]):
        val = int(start + (end - start) * x / max(size[0] - 1, 1))
        for y in range(size[1]):
            img.putpixel((x, y), val)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestComputeImageHash:
    def test_returns_hex_string(self) -> None:
        png = _make_png()
        h = compute_image_hash(png)
        assert isinstance(h, str)
        int(h, 16)  # must be valid hex

    def test_identical_images_produce_same_hash(self) -> None:
        png = _make_png((0, 128, 255))
        assert compute_image_hash(png) == compute_image_hash(png)

    def test_different_images_produce_different_hash(self) -> None:
        h1 = compute_image_hash(_make_gradient_png(0, 255))
        h2 = compute_image_hash(_make_gradient_png(255, 0))
        # Opposite gradients — hashes must differ
        assert h1 != h2

    def test_custom_hash_size(self) -> None:
        h = compute_image_hash(_make_png(), hash_size=4)
        # 4x4 = 16 bits → 4 hex chars
        assert len(h) == 4


class TestImagesAreSimilar:
    def test_identical_hashes(self) -> None:
        png = _make_png()
        h = compute_image_hash(png)
        assert images_are_similar(h, h) is True

    def test_none_hash_returns_false(self) -> None:
        h = compute_image_hash(_make_png())
        assert images_are_similar(None, h) is False
        assert images_are_similar(h, None) is False
        assert images_are_similar(None, None) is False

    def test_very_different_images_not_similar(self) -> None:
        h1 = compute_image_hash(_make_gradient_png(0, 255))
        h2 = compute_image_hash(_make_gradient_png(255, 0))
        assert images_are_similar(h1, h2) is False

    def test_similar_colours_within_threshold(self) -> None:
        h1 = compute_image_hash(_make_png((100, 100, 100)))
        h2 = compute_image_hash(_make_png((102, 100, 100)))
        # Nearly identical solid colour → should be similar
        assert images_are_similar(h1, h2) is True

    def test_strict_threshold(self) -> None:
        h1 = compute_image_hash(_make_png((100, 100, 100)))
        h2 = compute_image_hash(_make_png((100, 100, 100)))
        assert images_are_similar(h1, h2, threshold=0) is True
