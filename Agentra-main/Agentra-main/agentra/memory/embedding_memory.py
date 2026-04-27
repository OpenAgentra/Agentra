"""
Disk-backed embedding memory primitives for Agentra.

Supports both thread-local working memory and project-wide long-term memory.
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

from agentra.memory.providers import CallableEmbeddingProvider, EmbeddingProvider


class MemoryScope:
    """Simple labels for persisted memory records."""

    THREAD = "thread"
    LONG_TERM = "long_term"


@dataclass
class MemoryRecord:
    """A single stored memory item."""

    id: str
    text: str
    embedding: list[float]
    timestamp: float
    role: str = "observation"
    scope: str = MemoryScope.THREAD
    screenshot_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_text: str = ""


MemoryEntry = MemoryRecord


class DiskBackedMemoryStore:
    """
    Lightweight in-process embedding memory with disk persistence.

    Records are embedded using either the raw text or a normalized retrieval
    document that includes useful metadata like summary, tool, URL, and
    extracted text.
    """

    def __init__(
        self,
        memory_dir: Path,
        embed_provider: Optional[EmbeddingProvider] = None,
        embed_fn: Optional[Any] = None,
        screenshot_history: int = 10,
        *,
        default_scope: str = MemoryScope.THREAD,
    ) -> None:
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._screenshots_dir = self._dir / "screenshots"
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._embed_provider = embed_provider
        if self._embed_provider is None and embed_fn is not None:
            self._embed_provider = CallableEmbeddingProvider(embed_fn)
        self._screenshot_history = screenshot_history
        self._default_scope = default_scope
        self._entries: list[MemoryRecord] = []
        self._load()

    async def add(
        self,
        text: str,
        role: str = "observation",
        screenshot_b64: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        *,
        scope: str | None = None,
        retrieval_text: Optional[str] = None,
    ) -> MemoryRecord:
        payload = dict(metadata or {})
        normalized = retrieval_text or self._compose_retrieval_text(text, payload)
        embedding = await self._embed(normalized)
        entry_id = str(uuid.uuid4())
        screenshot_path: Optional[str] = None

        if screenshot_b64:
            screenshot_path = str(self._save_screenshot(entry_id, screenshot_b64))
            payload.setdefault("screenshot_path", screenshot_path)

        entry = MemoryRecord(
            id=entry_id,
            text=text,
            embedding=embedding,
            timestamp=time.time(),
            role=role,
            scope=scope or self._default_scope,
            screenshot_path=screenshot_path,
            metadata=payload,
            retrieval_text=normalized,
        )
        self._entries.append(entry)
        self._save()
        return entry

    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        metadata_filters: Optional[dict[str, Any]] = None,
        scopes: Optional[list[str]] = None,
    ) -> list[MemoryRecord]:
        if not self._entries:
            return []
        q_emb = await self._embed(query)
        scored = []
        allowed_scopes = set(scopes or [])
        for entry in self._entries:
            if allowed_scopes and entry.scope not in allowed_scopes:
                continue
            if metadata_filters and not self._matches_metadata(entry.metadata, metadata_filters):
                continue
            if not entry.embedding:
                continue
            scored.append((self._cosine(q_emb, entry.embedding), entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def recent(self, n: int = 10, *, scopes: Optional[list[str]] = None) -> list[MemoryRecord]:
        allowed_scopes = set(scopes or [])
        entries = [
            entry for entry in self._entries
            if not allowed_scopes or entry.scope in allowed_scopes
        ]
        return sorted(entries, key=lambda entry: entry.timestamp, reverse=True)[:n]

    def recent_screenshots(self, n: Optional[int] = None, *, scopes: Optional[list[str]] = None) -> list[str]:
        limit = n if n is not None else self._screenshot_history
        allowed_scopes = set(scopes or [])
        entries_with_shots = [
            entry
            for entry in sorted(self._entries, key=lambda item: item.timestamp, reverse=True)
            if entry.screenshot_path and (not allowed_scopes or entry.scope in allowed_scopes)
        ][:limit]
        result = []
        for entry in reversed(entries_with_shots):
            path = Path(entry.screenshot_path)
            if path.exists():
                result.append(base64.b64encode(path.read_bytes()).decode())
        return result

    def clear(self) -> None:
        self._entries.clear()

    def records(self) -> list[MemoryRecord]:
        return list(self._entries)

    def _save(self) -> None:
        data = [asdict(entry) for entry in self._entries]
        self._index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._entries = [MemoryRecord(**item) for item in data]
        except Exception:  # noqa: BLE001
            self._entries = []

    def _save_screenshot(self, entry_id: str, b64: str) -> Path:
        path = self._screenshots_dir / f"{entry_id}.png"
        path.write_bytes(base64.b64decode(b64))
        return path

    async def _embed(self, text: str) -> list[float]:
        if self._embed_provider is None:
            return self._trivial_embed(text)
        try:
            return await self._embed_provider.embed(text)
        except Exception:  # noqa: BLE001
            return self._trivial_embed(text)

    @staticmethod
    def _compose_retrieval_text(text: str, metadata: dict[str, Any]) -> str:
        fields = [text]
        for key in ("summary", "tool", "url", "active_url", "active_title", "source_type", "extracted_text"):
            value = metadata.get(key)
            if value:
                fields.append(str(value))
        return "\n".join(part for part in fields if part).strip()

    @staticmethod
    def _matches_metadata(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    @staticmethod
    def _trivial_embed(text: str) -> list[float]:
        vec = [0.0] * 256
        chars = text.lower()
        for index in range(len(chars) - 1):
            bucket = (ord(chars[index]) ^ ord(chars[index + 1])) % 256
            vec[bucket] += 1.0
        norm = math.sqrt(sum(item * item for item in vec)) or 1.0
        return [item / norm for item in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class EmbeddingMemory(DiskBackedMemoryStore):
    """Backwards-compatible embedding memory interface."""

    def __init__(
        self,
        memory_dir: Path,
        embed_provider: Optional[EmbeddingProvider] = None,
        embed_fn: Optional[Any] = None,
        screenshot_history: int = 10,
    ) -> None:
        super().__init__(
            memory_dir,
            embed_provider=embed_provider,
            embed_fn=embed_fn,
            screenshot_history=screenshot_history,
            default_scope=MemoryScope.THREAD,
        )


class ThreadWorkingMemory(EmbeddingMemory):
    """Thread-local short-term memory."""

    async def add(
        self,
        text: str,
        role: str = "observation",
        screenshot_b64: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        *,
        retrieval_text: Optional[str] = None,
    ) -> MemoryRecord:
        return await super().add(
            text,
            role=role,
            screenshot_b64=screenshot_b64,
            metadata=metadata,
            scope=MemoryScope.THREAD,
            retrieval_text=retrieval_text,
        )


class LongTermMemoryStore(EmbeddingMemory):
    """Project-wide searchable memory across runs and threads."""

    async def add(
        self,
        text: str,
        role: str = "observation",
        screenshot_b64: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        *,
        retrieval_text: Optional[str] = None,
    ) -> MemoryRecord:
        return await super().add(
            text,
            role=role,
            screenshot_b64=screenshot_b64,
            metadata=metadata,
            scope=MemoryScope.LONG_TERM,
            retrieval_text=retrieval_text,
        )
