"""Provider-neutral interfaces and result types for passage reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..rrf import HybridResult


@dataclass(frozen=True)
class RerankResult:
    """A hybrid candidate augmented with an independent reranker score."""

    chunk_id: str
    document_id: str
    text: str
    vector_score: float | None
    bm25_score: float | None
    rrf_score: float
    metadata: dict[str, Any]
    rerank_score: float
    rerank_rank: int

    @property
    def score(self) -> float:
        """Expose the reranker score through the common retrieval API."""

        return self.rerank_score

    @classmethod
    def from_candidate(
        cls, candidate: HybridResult, *, rerank_score: float, rerank_rank: int
    ) -> "RerankResult":
        return cls(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            text=candidate.text,
            vector_score=candidate.vector_score,
            bm25_score=candidate.bm25_score,
            rrf_score=candidate.rrf_score,
            metadata=dict(candidate.metadata),
            rerank_score=float(rerank_score),
            rerank_rank=rerank_rank,
        )


class Reranker(Protocol):
    """Minimal contract implemented by local or hosted rerankers."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridResult],
        top_k: int,
    ) -> list[RerankResult]:
        """Score query/passage pairs independently and return ranked results."""


@dataclass(frozen=True)
class RerankPhaseTiming:
    """Per-request timing breakdown for reranker profiling."""

    model_load_seconds: float
    tokenizer_seconds: float
    preprocessing_seconds: float
    inference_seconds: float
    postprocessing_seconds: float
    total_seconds: float
    device: str
    pair_count: int
    model_calls: int
    batched: bool


def deduplicate_candidates(candidates: Sequence[HybridResult]) -> list[HybridResult]:
    """Deduplicate by chunk ID while retaining the first candidate occurrence."""

    unique: list[HybridResult] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.chunk_id in seen:
            continue
        seen.add(candidate.chunk_id)
        unique.append(candidate)
    return unique
