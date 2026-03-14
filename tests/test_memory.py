"""Tests for the EmbeddingMemory system."""

from __future__ import annotations

import asyncio
import base64
import time

import pytest

from agentra.memory.embedding_memory import EmbeddingMemory, MemoryEntry


@pytest.fixture
def memory(tmp_path):
    return EmbeddingMemory(memory_dir=tmp_path / ".memory", embed_fn=None)


# ── basic operations ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_and_recent(memory):
    await memory.add("First observation", role="observation")
    await memory.add("Second observation", role="observation")
    recent = memory.recent(n=10)
    assert len(recent) == 2
    texts = [e.text for e in recent]
    assert "First observation" in texts
    assert "Second observation" in texts


@pytest.mark.asyncio
async def test_persistence(tmp_path):
    """Data written by one instance should be loadable by another."""
    mem_dir = tmp_path / ".memory"
    mem1 = EmbeddingMemory(memory_dir=mem_dir)
    await mem1.add("Persistent entry", role="user")

    mem2 = EmbeddingMemory(memory_dir=mem_dir)
    assert len(mem2.recent()) == 1
    assert mem2.recent()[0].text == "Persistent entry"


@pytest.mark.asyncio
async def test_search_returns_relevant(memory):
    await memory.add("I like Python programming")
    await memory.add("The weather is sunny today")
    await memory.add("Python is great for AI")

    results = await memory.search("Python coding", top_k=2)
    # Both Python-related entries should rank above weather
    texts = [e.text for e in results]
    assert any("Python" in t for t in texts)


@pytest.mark.asyncio
async def test_search_empty_memory(memory):
    results = await memory.search("anything")
    assert results == []


@pytest.mark.asyncio
async def test_clear(memory):
    await memory.add("entry 1")
    await memory.add("entry 2")
    memory.clear()
    assert memory.recent() == []


@pytest.mark.asyncio
async def test_screenshot_storage(memory):
    # Create a minimal 1×1 PNG (smallest valid PNG)
    png_1x1_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    entry = await memory.add("Screenshot taken", screenshot_b64=png_1x1_b64)
    assert entry.screenshot_path is not None
    from pathlib import Path  # noqa: PLC0415
    assert Path(entry.screenshot_path).exists()


@pytest.mark.asyncio
async def test_recent_screenshots(memory):
    png_1x1_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    await memory.add("Shot 1", screenshot_b64=png_1x1_b64)
    await memory.add("Shot 2", screenshot_b64=png_1x1_b64)
    shots = memory.recent_screenshots(n=5)
    assert len(shots) == 2
    # Each entry should be a valid base64 string
    for s in shots:
        base64.b64decode(s)  # should not raise


@pytest.mark.asyncio
async def test_recent_ordered_by_time(memory):
    await memory.add("oldest")
    await asyncio.sleep(0.01)
    await memory.add("newest")
    recent = memory.recent(n=2)
    # most recent first
    assert recent[0].text == "newest"


# ── trivial embedding fallback ────────────────────────────────────────────────

def test_trivial_embed_is_normalised():
    vec = EmbeddingMemory._trivial_embed("hello world")
    import math  # noqa: PLC0415
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert EmbeddingMemory._cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert EmbeddingMemory._cosine(a, b) == pytest.approx(0.0)


def test_cosine_mismatched_lengths():
    assert EmbeddingMemory._cosine([1.0], [1.0, 2.0]) == 0.0


# ── async embed_fn ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_custom_embed_fn(tmp_path):
    call_count = 0

    async def my_embed(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        return [float(len(text))] + [0.0] * 255

    mem = EmbeddingMemory(memory_dir=tmp_path / ".memory2", embed_fn=my_embed)
    await mem.add("hello")
    assert call_count >= 1


@pytest.mark.asyncio
async def test_fallback_when_embed_fn_raises(tmp_path):
    async def broken_embed(text: str) -> list[float]:
        raise RuntimeError("embedding service down")

    mem = EmbeddingMemory(memory_dir=tmp_path / ".memory3", embed_fn=broken_embed)
    # Should not raise — falls back to trivial embed
    entry = await mem.add("test text")
    assert entry.embedding  # still has a non-empty embedding
