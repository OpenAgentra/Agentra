"""
Embedding-based long-term memory for the agent.

Stores text snippets (observations, tool results, user messages) and their
dense vector embeddings.  Retrieval is cosine-similarity nearest-neighbour
using only NumPy — no external vector DB required by default.

Screenshots are stored separately as PNG files and referenced by ID.
"""

from __future__ import annotations

import base64
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class MemoryEntry:
    """A single item stored in memory."""

    id: str
    text: str
    embedding: list[float]
    timestamp: float
    role: str = "observation"  # "observation" | "user" | "assistant"
    screenshot_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingMemory:
    """
    Lightweight in-process embedding memory with disk persistence.

    The embeddings can be produced by any callable that accepts a string
    and returns ``list[float]``.  Pass in the LLM provider's ``embed``
    method for best results.

    Embeddings and screenshots are stored under *memory_dir*.
    """

    def __init__(
        self,
        memory_dir: Path,
        embed_fn: Optional[Any] = None,
        screenshot_history: int = 10,
    ) -> None:
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._screenshots_dir = self._dir / "screenshots"
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._embed_fn = embed_fn  # async callable: str -> list[float]
        self._screenshot_history = screenshot_history
        self._entries: list[MemoryEntry] = []
        self._load()

    # ── public API ─────────────────────────────────────────────────────────────

    async def add(
        self,
        text: str,
        role: str = "observation",
        screenshot_b64: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Embed *text*, optionally save a screenshot, and persist to disk."""
        embedding = await self._embed(text)
        entry_id = str(uuid.uuid4())
        screenshot_path: Optional[str] = None

        if screenshot_b64:
            screenshot_path = str(self._save_screenshot(entry_id, screenshot_b64))

        entry = MemoryEntry(
            id=entry_id,
            text=text,
            embedding=embedding,
            timestamp=time.time(),
            role=role,
            screenshot_path=screenshot_path,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._save()
        return entry

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Return the *top_k* entries most similar to *query*."""
        if not self._entries:
            return []
        q_emb = await self._embed(query)
        scored = [
            (self._cosine(q_emb, e.embedding), e)
            for e in self._entries
            if e.embedding
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the *n* most recent entries."""
        return sorted(self._entries, key=lambda e: e.timestamp, reverse=True)[:n]

    def recent_screenshots(self, n: Optional[int] = None) -> list[str]:
        """Return base64-encoded PNGs for recent screenshots (up to *n*)."""
        limit = n if n is not None else self._screenshot_history
        entries_with_shots = [
            e for e in sorted(self._entries, key=lambda x: x.timestamp, reverse=True)
            if e.screenshot_path
        ][:limit]
        result = []
        for entry in reversed(entries_with_shots):
            path = Path(entry.screenshot_path)
            if path.exists():
                b64 = base64.b64encode(path.read_bytes()).decode()
                result.append(b64)
        return result

    def clear(self) -> None:
        """Remove all in-memory entries (does not delete files)."""
        self._entries.clear()

    # ── persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        data = [asdict(e) for e in self._entries]
        self._index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._entries = [MemoryEntry(**d) for d in data]
        except Exception:  # noqa: BLE001
            self._entries = []

    # ── helpers ────────────────────────────────────────────────────────────────

    def _save_screenshot(self, entry_id: str, b64: str) -> Path:
        path = self._screenshots_dir / f"{entry_id}.png"
        path.write_bytes(base64.b64decode(b64))
        return path

    async def _embed(self, text: str) -> list[float]:
        if self._embed_fn is None:
            return self._trivial_embed(text)
        try:
            return await self._embed_fn(text)
        except Exception:  # noqa: BLE001
            return self._trivial_embed(text)

    @staticmethod
    def _trivial_embed(text: str) -> list[float]:
        """
        Deterministic, dependency-free fallback embedding.
        Uses character-level bigram frequency as a simple sparse vector
        (sufficient for basic overlap matching when no LLM is available).
        """
        vec = [0.0] * 256
        chars = text.lower()
        for i in range(len(chars) - 1):
            idx = (ord(chars[i]) ^ ord(chars[i + 1])) % 256
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
