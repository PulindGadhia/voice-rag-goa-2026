"""SQLite-backed deterministic embedding cache."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from threading import RLock
from pathlib import Path
from typing import Sequence

from .base import EmbeddingConfig, EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingCacheStats:
    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @property
    def requests(self) -> int:
        return self.hits + self.misses


class CachedEmbedder:
    """Wrap any provider and cache vectors by model/config/text content."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        config: EmbeddingConfig | None = None,
        cache_path: str | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or getattr(provider, "config", EmbeddingConfig())
        self.cache_path = cache_path or self.config.cache_path
        self.stats = EmbeddingCacheStats()
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        if self.cache_path != ":memory:":
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        # Retrieval is invoked from the application's worker thread. Allow
        # that thread to reuse the long-lived cache connection; calls remain
        # serialized by the lock below.
        self._connection = sqlite3.connect(self.cache_path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA temp_store=MEMORY")
        self._connection.execute("PRAGMA mmap_size=67108864")  # 64 MB mmap
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                input_type TEXT NOT NULL,
                normalized INTEGER NOT NULL,
                text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def _key(self, text: str, input_type: str) -> tuple[str, str]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        payload = json.dumps(
            [self.config.model_name, input_type, self.config.normalize, text_hash],
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), text_hash

    def embed_text(self, text: str, *, input_type: str = "passage") -> list[float]:
        return self.embed_batch([text], input_type=input_type)[0]

    def embed_batch(
        self, texts: Sequence[str], *, input_type: str = "passage"
    ) -> list[list[float]]:
        with self._lock:
            return self._embed_batch(texts, input_type=input_type)

    def _embed_batch(
        self, texts: Sequence[str], *, input_type: str = "passage"
    ) -> list[list[float]]:
        if not texts:
            return []
        assert self._connection is not None
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[str, list[int], str, str]] = []
        missing_by_key: dict[str, tuple[list[int], str, str]] = {}
        for index, text in enumerate(texts):
            key, text_hash = self._key(text, input_type)
            row = self._connection.execute(
                "SELECT vector_json FROM embeddings WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                existing_missing = missing_by_key.get(key)
                if existing_missing is None:
                    positions = [index]
                    missing_by_key[key] = (positions, text_hash, text)
                    missing.append((key, positions, text_hash, text))
                    self.stats.misses += 1
                else:
                    existing_missing[0].append(index)
                    # A duplicate in the same request is served by the one
                    # provider call and is therefore a cache-equivalent hit.
                    self.stats.hits += 1
            else:
                self.stats.hits += 1
                vectors[index] = [float(value) for value in json.loads(row[0])]

        if missing:
            embedded = self.provider.embed_batch(
                [item[3] for item in missing], input_type=input_type
            )
            if len(embedded) != len(missing):
                raise ValueError("embedding provider returned the wrong number of vectors")
            for (key, positions, text_hash, _), vector in zip(missing, embedded):
                vector_list = [float(value) for value in vector]
                for index in positions:
                    vectors[index] = vector_list
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (cache_key, model_name, input_type, normalized, text_hash, vector_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        self.config.model_name,
                        input_type,
                        int(self.config.normalize),
                        text_hash,
                        json.dumps(vector_list),
                    ),
                )
                self.stats.writes += 1
            self._connection.commit()
        logger.info(
            "embedding_cache_batch",
            extra={"cache_hits": self.stats.hits, "cache_misses": self.stats.misses},
        )
        return [vector for vector in vectors if vector is not None]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
