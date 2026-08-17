"""Hybrid retrieval followed by configurable candidate reranking."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ..hybrid import HybridRetriever, HybridTiming
from ..rrf import HybridResult
from .base import RerankResult, Reranker
from .config import RerankerConfig


@dataclass(frozen=True)
class RerankingTiming:
    hybrid_seconds: float
    rerank_seconds: float
    total_seconds: float
    hybrid_timing: HybridTiming


class HybridRerankRetriever:
    """Compose the existing hybrid retriever and an independent reranker."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        *,
        config: RerankerConfig | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.config = config or RerankerConfig.from_env()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_top_k: int | None = None,
    ) -> list[RerankResult]:
        results, _ = self.retrieve_with_timing(
            query, top_k=top_k, candidate_top_k=candidate_top_k
        )
        return results

    def retrieve_with_timing(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_top_k: int | None = None,
    ) -> tuple[list[RerankResult], RerankingTiming]:
        final_top_k = top_k or self.config.final_top_k
        candidate_k = candidate_top_k or self.config.candidate_top_k
        if final_top_k <= 0 or candidate_k <= 0:
            raise ValueError("reranking top_k values must be > 0")
        if candidate_k < final_top_k:
            raise ValueError("candidate_top_k must be >= final top_k")
        if not query or not query.strip():
            empty_hybrid_timing = HybridTiming(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return [], RerankingTiming(0.0, 0.0, 0.0, empty_hybrid_timing)

        started = perf_counter()
        candidates, hybrid_timing = self.hybrid_retriever.retrieve_with_timing(
            query, top_k=candidate_k
        )
        rerank_started = perf_counter()
        results = self.reranker.rerank(query, candidates, final_top_k)
        rerank_seconds = perf_counter() - rerank_started
        return results, RerankingTiming(
            hybrid_seconds=hybrid_timing.total_seconds,
            rerank_seconds=rerank_seconds,
            total_seconds=perf_counter() - started,
            hybrid_timing=hybrid_timing,
        )


__all__ = ["HybridRerankRetriever", "RerankingTiming"]
