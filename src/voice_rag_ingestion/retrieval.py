"""Qdrant-independent vector retrieval interface."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .embeddings.base import EmbeddingProvider
from .qdrant_store import QdrantVectorStore, RetrievedChunk


@dataclass(frozen=True)
class RetrievalTiming:
    query_embedding_seconds: float
    qdrant_search_seconds: float
    total_seconds: float


class VectorRetriever:
    """Compose query embedding and vector store without leaking Qdrant APIs."""

    def __init__(self, embedder: EmbeddingProvider, vector_store: QdrantVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        results, _ = self.retrieve_with_timing(query, top_k=top_k)
        return results

    def retrieve_with_timing(
        self, query: str, *, top_k: int = 5
    ) -> tuple[list[RetrievedChunk], RetrievalTiming]:
        if not query or not query.strip():
            return [], RetrievalTiming(0.0, 0.0, 0.0)
        started = perf_counter()
        embedding_started = perf_counter()
        vector = self.embedder.embed_text(query.strip(), input_type="query")
        embedding_elapsed = perf_counter() - embedding_started
        search_started = perf_counter()
        results = self.vector_store.search(vector, top_k=top_k)
        search_elapsed = perf_counter() - search_started
        return results, RetrievalTiming(
            query_embedding_seconds=embedding_elapsed,
            qdrant_search_seconds=search_elapsed,
            total_seconds=perf_counter() - started,
        )
