"""Tests for C3: pixel-level observation drift detection."""

from __future__ import annotations

import io

from PIL import Image

from agentra.tools.computer import ComputerTool
from agentra.tools.visual_diff import (
    compute_structural_hash,
    images_structurally_similar,
)


def _solid_png(color: tuple[int, int, int] = (100, 150, 200), size: tuple[int, int] = (128, 128)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gradient_png(start: int = 0, end: int = 255, size: tuple[int, int] = (128, 128)) -> bytes:
    img = Image.new("L", size)
    for x in range(size[0]):
        val = int(start + (end - start) * x / max(size[0] - 1, 1))
        for y in range(size[1]):
            img.putpixel((x, y), val)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestStructuralHash:
    def test_returns_hex_string(self) -> None:
        h = compute_structural_hash(_solid_png())
        assert isinstance(h, str)
        int(h, 16)

    def test_default_block_size_is_16(self) -> None:
        # 16*16 = 256 bits → 64 hex chars
        h = compute_structural_hash(_solid_png())
        assert len(h) == 64

    def test_custom_block_size(self) -> None:
        h = compute_structural_hash(_solid_png(), block_size=8)
        assert len(h) == 16  # 8*8=64 bits → 16 hex chars

    def test_identical_images_same_hash(self) -> None:
        a = _solid_png((50, 100, 150))
        b = _solid_png((50, 100, 150))
        assert compute_structural_hash(a) == compute_structural_hash(b)

    def test_different_images_different_hash(self) -> None:
        h1 = compute_structural_hash(_gradient_png(0, 255))
        h2 = compute_structural_hash(_gradient_png(255, 0))
        assert h1 != h2


class TestStructurallySimilar:
    def test_identical_hashes_similar(self) -> None:
        h = compute_structural_hash(_solid_png())
        assert images_structurally_similar(h, h) is True

    def test_none_hash_returns_false(self) -> None:
        h = compute_structural_hash(_solid_png())
        assert images_structurally_similar(None, h) is False
        assert images_structurally_similar(h, None) is False

    def test_very_different_not_similar(self) -> None:
        h1 = compute_structural_hash(_gradient_png(0, 255))
        h2 = compute_structural_hash(_gradient_png(255, 0))
        assert images_structurally_similar(h1, h2) is False

    def test_stricter_threshold_than_default(self) -> None:
        h1 = compute_structural_hash(_solid_png())
        h2 = compute_structural_hash(_solid_png())
        # default threshold is 3 bits
        assert images_structurally_similar(h1, h2, threshold=3) is True


class TestDesktopObservationDrifted:
    """ComputerTool._desktop_observation_drifted with new pixel check."""

    def test_no_drift_when_observations_identical(self) -> None:
        obs = {
            "window_handle": 100,
            "window_title": "Notepad",
            "cursor_x": 500,
            "cursor_y": 300,
            "structural_hash": "abc",
        }
        assert ComputerTool._desktop_observation_drifted(obs, obs) is False

    def test_drift_when_pixel_hash_differs(self) -> None:
        prev = {
            "window_handle": 100,
            "window_title": "Notepad",
            "cursor_x": 500,
            "cursor_y": 300,
            "structural_hash": compute_structural_hash(_gradient_png(0, 255)),
        }
        curr = {
            "window_handle": 100,
            "window_title": "Notepad",
            "cursor_x": 500,
            "cursor_y": 300,
            "structural_hash": compute_structural_hash(_gradient_png(255, 0)),
        }
        assert ComputerTool._desktop_observation_drifted(curr, prev) is True

    def test_no_drift_when_hash_missing_falls_back(self) -> None:
        prev = {"window_handle": 100, "window_title": "Notepad", "cursor_x": 500, "cursor_y": 300}
        curr = {"window_handle": 100, "window_title": "Notepad", "cursor_x": 500, "cursor_y": 300}
        assert ComputerTool._desktop_observation_drifted(curr, prev) is False

    def test_drift_via_handle_change(self) -> None:
        prev = {"window_handle": 100, "window_title": "A", "cursor_x": 0, "cursor_y": 0}
        curr = {"window_handle": 200, "window_title": "B", "cursor_x": 0, "cursor_y": 0}
        assert ComputerTool._desktop_observation_drifted(curr, prev) is True

    def test_drift_via_cursor_movement(self) -> None:
        prev = {"window_handle": 100, "window_title": "A", "cursor_x": 0, "cursor_y": 0}
        curr = {"window_handle": 100, "window_title": "A", "cursor_x": 50, "cursor_y": 50}
        assert ComputerTool._desktop_observation_drifted(curr, prev) is True

    def test_none_observations_no_drift(self) -> None:
        assert ComputerTool._desktop_observation_drifted(None, None) is False
        assert ComputerTool._desktop_observation_drifted({}, None) is False
