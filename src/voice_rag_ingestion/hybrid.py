"""Hybrid vector + BM25 retrieval without score-scale mixing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter

from .bm25 import BM25Retriever
from .qdrant_store import RetrievedChunk
from .retrieval import RetrievalTiming, VectorRetriever
from .rrf import HybridResult, RRFFuser
from .tokenization import TokenizerConfig


@dataclass(frozen=True)
class HybridConfig:
    vector_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 10
    rrf_k: int = 60
    chunking_strategy: str = "fixed"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    tokenizer: TokenizerConfig = TokenizerConfig()

    @classmethod
    def from_env(cls) -> "HybridConfig":
        return cls(
            vector_top_k=int(os.getenv("HYBRID_VECTOR_TOP_K", "20")),
            bm25_top_k=int(os.getenv("HYBRID_BM25_TOP_K", "20")),
            final_top_k=int(os.getenv("HYBRID_FINAL_TOP_K", "10")),
            rrf_k=int(os.getenv("HYBRID_RRF_K", "60")),
            chunking_strategy=os.getenv("HYBRID_CHUNKING_STRATEGY", "fixed"),
            bm25_k1=float(os.getenv("BM25_K1", "1.5")),
            bm25_b=float(os.getenv("BM25_B", "0.75")),
            tokenizer=TokenizerConfig(
                lowercase=os.getenv("BM25_LOWERCASE", "true").lower()
                in {"1", "true", "yes", "on"},
                unicode_normalization=os.getenv("BM25_UNICODE_NORMALIZATION", "NFC"),
                min_token_length=int(os.getenv("BM25_MIN_TOKEN_LENGTH", "1")),
            ),
        )

    def __post_init__(self) -> None:
        if self.vector_top_k <= 0 or self.bm25_top_k <= 0 or self.final_top_k <= 0:
            raise ValueError("hybrid top_k values must be > 0")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")


@dataclass(frozen=True)
class HybridTiming:
    vector_seconds: float
    bm25_seconds: float
    rrf_seconds: float
    total_seconds: float
    vector_embedding_seconds: float
    vector_search_seconds: float


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        *,
        config: HybridConfig | None = None,
        fuser: RRFFuser | None = None,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.config = config or HybridConfig()
        self.fuser = fuser or RRFFuser(self.config.rrf_k)

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[HybridResult]:
        results, _ = self.retrieve_with_timing(query, top_k=top_k)
        return results

    def retrieve_with_timing(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[list[HybridResult], HybridTiming]:
        final_top_k = top_k or self.config.final_top_k
        if not query or not query.strip():
            return [], HybridTiming(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        started = perf_counter()
        vector_results, vector_timing = self.vector_retriever.retrieve_with_timing(
            query, top_k=self.config.vector_top_k
        )
        bm25_results, bm25_seconds = self.bm25_retriever.retrieve_with_timing(
            query, top_k=self.config.bm25_top_k
        )
        fusion_started = perf_counter()
        fused = self.fuser.fuse(vector_results, bm25_results, top_k=final_top_k)
        rrf_seconds = perf_counter() - fusion_started
        return fused, HybridTiming(
            vector_seconds=vector_timing.total_seconds,
            bm25_seconds=bm25_seconds,
            rrf_seconds=rrf_seconds,
            total_seconds=perf_counter() - started,
            vector_embedding_seconds=vector_timing.query_embedding_seconds,
            vector_search_seconds=vector_timing.qdrant_search_seconds,
        )
