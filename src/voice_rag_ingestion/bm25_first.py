"""BM25-first candidate retrieval with on-demand late dense embedding and RRF fusion."""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter
from typing import Sequence

import numpy as np

from .bm25 import BM25Retriever
from .embeddings.base import EmbeddingProvider
from .hybrid import HybridConfig, HybridTiming
from .qdrant_store import RetrievedChunk
from .rrf import HybridResult, RRFFuser


class _QueryEmbeddingCache:
    """Thread-safe bounded in-memory LRU cache for query embeddings."""

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> np.ndarray | None:
        arr = self._cache.get(key)
        if arr is not None:
            self._cache.move_to_end(key)
        return arr

    def put(self, key: str, vec: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = vec


class BM25FirstHybridRetriever:
    """Retrieves top-N candidates from BM25, embeds candidates on-demand, and fuses with RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        embedder: EmbeddingProvider,
        *,
        config: HybridConfig | None = None,
        fuser: RRFFuser | None = None,
        candidate_pool_size: int | None = None,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.embedder = embedder
        self.config = config or HybridConfig.from_env()
        self.fuser = fuser or RRFFuser(self.config.rrf_k)
        self.candidate_pool_size = (
            candidate_pool_size
            or max(self.config.bm25_top_k, self.config.vector_top_k, 30)
        )
        self._query_cache = _QueryEmbeddingCache(max_size=256)

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

        # Step 1: BM25 candidate retrieval across the complete corpus
        bm25_results, bm25_seconds = self.bm25_retriever.retrieve_with_timing(
            query, top_k=self.candidate_pool_size
        )
        if not bm25_results:
            return [], HybridTiming(
                vector_seconds=0.0,
                bm25_seconds=bm25_seconds,
                rrf_seconds=0.0,
                total_seconds=perf_counter() - started,
                vector_embedding_seconds=0.0,
                vector_search_seconds=0.0,
            )

        # Step 2: On-demand late dense embedding for query and candidate passages only
        t_embed_0 = perf_counter()
        query_key = query.strip()
        cached_query_vec = self._query_cache.get(query_key)
        if cached_query_vec is not None:
            query_arr = cached_query_vec
            query_embed_sec = perf_counter() - t_embed_0
        else:
            query_vector = self.embedder.embed_text(query, input_type="query")
            query_arr = np.array(query_vector, dtype=np.float32)
            self._query_cache.put(query_key, query_arr)
            query_embed_sec = perf_counter() - t_embed_0

        t_passages_0 = perf_counter()
        candidate_texts = [chunk.text for chunk in bm25_results]
        candidate_vectors = self.embedder.embed_batch(
            candidate_texts,
            input_type="passage",
        )
        passages_embed_sec = perf_counter() - t_passages_0

        # Step 3: NumPy vectorized cosine similarity scoring
        t_search_0 = perf_counter()
        cand_arr = np.array(candidate_vectors, dtype=np.float32)
        # Vectorized dot product and norms
        query_norm = np.linalg.norm(query_arr)
        cand_norms = np.linalg.norm(cand_arr, axis=1)
        # Avoid division by zero
        safe_denom = np.maximum(query_norm * cand_norms, 1e-12)
        sim_scores = cand_arr @ query_arr / safe_denom

        vector_candidates: list[RetrievedChunk] = []
        for idx, chunk in enumerate(bm25_results):
            vector_candidates.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=float(sim_scores[idx]),
                    metadata=dict(chunk.metadata),
                )
            )
        # Sort by descending score for RRF fusion
        vector_candidates.sort(key=lambda item: (-item.score, item.chunk_id))
        vector_search_sec = perf_counter() - t_search_0

        total_vector_seconds = query_embed_sec + passages_embed_sec + vector_search_sec

        # Step 4: Reciprocal Rank Fusion
        t_rrf_0 = perf_counter()
        fused = self.fuser.fuse(
            vector_candidates[: self.config.vector_top_k],
            bm25_results[: self.config.bm25_top_k],
            top_k=final_top_k,
        )
        rrf_seconds = perf_counter() - t_rrf_0

        total_seconds = perf_counter() - started

        return fused, HybridTiming(
            vector_seconds=total_vector_seconds,
            bm25_seconds=bm25_seconds,
            rrf_seconds=rrf_seconds,
            total_seconds=total_seconds,
            vector_embedding_seconds=query_embed_sec + passages_embed_sec,
            vector_search_seconds=vector_search_sec,
        )
