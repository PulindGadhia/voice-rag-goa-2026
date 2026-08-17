"""Embedding contracts and configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Provider-neutral contract used by indexing and retrieval."""

    @property
    def dimension(self) -> int:
        ...

    def embed_text(self, text: str, *, input_type: str = "passage") -> list[float]:
        ...

    def embed_batch(
        self, texts: Sequence[str], *, input_type: str = "passage"
    ) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "intfloat/multilingual-e5-small"
    batch_size: int = 32
    normalize: bool = True
    device: str | None = None
    cache_path: str = ".cache/embeddings.sqlite3"
    cache_enabled: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        device = os.getenv("EMBEDDING_DEVICE") or None
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", cls.model_name),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", str(cls.batch_size))),
            normalize=os.getenv("EMBEDDING_NORMALIZE", "true").lower()
            in {"1", "true", "yes", "on"},
            device=device,
            cache_path=os.getenv("EMBEDDING_CACHE_PATH", cls.cache_path),
            cache_enabled=os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower()
            in {"1", "true", "yes", "on"},
            query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", cls.query_prefix),
            passage_prefix=os.getenv("EMBEDDING_PASSAGE_PREFIX", cls.passage_prefix),
        )

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")


@dataclass
class EmbeddingStats:
    items: int = 0
    batches: int = 0
    elapsed_seconds: float = 0.0

    @property
    def average_seconds_per_item(self) -> float:
        return self.elapsed_seconds / self.items if self.items else 0.0


def timed_batch_embed(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    *,
    batch_size: int,
    input_type: str,
) -> tuple[list[list[float]], EmbeddingStats]:
    """Embed in batches and return measured provider statistics."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    vectors: list[list[float]] = []
    stats = EmbeddingStats(items=len(texts))
    started = perf_counter()
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        vectors.extend(provider.embed_batch(batch, input_type=input_type))
        stats.batches += 1
    stats.elapsed_seconds = perf_counter() - started
    if len(vectors) != len(texts):
        raise ValueError("embedding provider returned the wrong number of vectors")
    return vectors, stats
