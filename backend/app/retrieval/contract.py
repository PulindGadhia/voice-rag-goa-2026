"""Application-facing retrieval contract.

The API and orchestration layers depend on these types only.  They do not know
whether retrieval is backed by Qdrant, BM25, RRF, or a particular reranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResponse:
    query: str
    results: list[RetrievalCandidate]
    top_k: int
    candidate_top_k: int
    total_latency_ms: float
    hybrid_latency_ms: float
    rerank_latency_ms: float
    embedding_latency_ms: float = 0.0
    vector_search_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    rrf_latency_ms: float = 0.0
    model_name: str | None = None
    device: str | None = None
    retrieval_route: str | None = None
    translation_used: bool = False
    translated_query: str | None = None
    detected_language: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.results)


class RetrievalEngine(Protocol):
    async def warmup(self) -> object:
        """Initialize configured retrieval model kernels before readiness."""
        ...

    async def retrieve(
        self, query: str, *, top_k: int, candidate_top_k: int | None = None
    ) -> RetrievalResponse:
        ...

    async def health_check(self) -> dict[str, Any]:
        ...
