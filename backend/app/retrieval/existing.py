"""Adapter from Pulind's existing hybrid/reranking stack to the app contract."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any

from voice_rag_ingestion.bm25 import BM25Index, BM25Retriever
from voice_rag_ingestion.bm25_first import BM25FirstHybridRetriever
from voice_rag_ingestion.bm25_sqlite import BM25SqliteIndex
from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.embeddings import (
    CachedEmbedder,
    EmbeddingConfig,
    SentenceTransformerEmbedder,
)
from voice_rag_ingestion.hybrid import HybridConfig, HybridRetriever
from voice_rag_ingestion.indexing import index_documents
from voice_rag_ingestion.loader import DatasetLoader
from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig
from voice_rag_ingestion.reranking import (
    CrossEncoderReranker,
    HybridRerankRetriever,
    RerankerConfig,
)
from voice_rag_ingestion.retrieval import VectorRetriever

from .contract import RetrievalCandidate, RetrievalResponse


class ExistingRetrievalEngine:
    """Async facade over the already-built vector/BM25/RRF/reranker pipeline."""

    def __init__(self, retriever: HybridRerankRetriever) -> None:
        self.retriever = retriever

    async def warmup(self) -> object:
        """Warm the already-loaded reranker; do not construct another model."""

        reranker = self.retriever.reranker
        warmup = getattr(reranker, "warmup", None)
        if warmup is None:
            raise RuntimeError("configured reranker does not support warmup")
        try:
            res = await asyncio.to_thread(warmup)
            # Pre-warm hybrid retrieval, embedding, and SQLite BM25 page cache.
            # Using representative Assamese queries exercises the high-frequency
            # term posting pages that would otherwise be cold on the first real
            # request, adding ~200 ms to P95 latency.
            _warmup_queries = [
                "warmup query",
                "কৰ্পোৰেচন কি?",           # short Assamese — exercises common pages
                "কোম্পানী গঠন কৰা হয",       # medium stop-word query
                "ভাৰতৰ ৰাজধানী",            # another short Assamese
            ]
            for _wq in _warmup_queries:
                await asyncio.to_thread(
                    self.retriever.retrieve, _wq, top_k=3, candidate_top_k=5
                )
            return res
        except Exception as exc:
            raise RuntimeError("reranker warmup failed during application startup") from exc

    async def retrieve(
        self, query: str, *, top_k: int = 3, candidate_top_k: int | None = None
    ) -> RetrievalResponse:
        started = perf_counter()
        results, timing = await asyncio.to_thread(
            self.retriever.retrieve_with_timing,
            query,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
        )
        candidates = [
            RetrievalCandidate(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                score=item.rerank_score,
                metadata={
                    **item.metadata,
                    "vector_score": item.vector_score,
                    "bm25_score": item.bm25_score,
                    "rrf_score": item.rrf_score,
                    "rerank_score": item.rerank_score,
                    "rerank_rank": item.rerank_rank,
                },
            )
            for item in results
        ]
        phase = getattr(self.retriever.reranker, "last_timing", None)
        return RetrievalResponse(
            query=query,
            results=candidates,
            top_k=top_k,
            candidate_top_k=candidate_top_k or self.retriever.config.candidate_top_k,
            total_latency_ms=(perf_counter() - started) * 1000.0,
            hybrid_latency_ms=timing.hybrid_seconds * 1000.0,
            rerank_latency_ms=timing.rerank_seconds * 1000.0,
            embedding_latency_ms=timing.hybrid_timing.vector_embedding_seconds * 1000.0,
            vector_search_latency_ms=timing.hybrid_timing.vector_search_seconds * 1000.0,
            bm25_latency_ms=timing.hybrid_timing.bm25_seconds * 1000.0,
            rrf_latency_ms=timing.hybrid_timing.rrf_seconds * 1000.0,
            model_name=self.retriever.config.model_name,
            device=getattr(phase, "device", None),
            retrieval_route="quality",
        )

    async def retrieve_fast(
        self, query: str, *, top_k: int = 3, candidate_top_k: int | None = None
    ) -> RetrievalResponse:
        """Fast path: hybrid retrieval only (BM25 + dense + RRF), no reranker."""
        started = perf_counter()
        candidate_k = candidate_top_k or self.retriever.config.candidate_top_k
        hybrid_results, hybrid_timing = await asyncio.to_thread(
            self.retriever.hybrid_retriever.retrieve_with_timing,
            query,
            top_k=candidate_k,
        )
        # Convert HybridResults to RetrievalCandidates using RRF score
        candidates = [
            RetrievalCandidate(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                score=item.rrf_score,
                metadata={
                    **item.metadata,
                    "vector_score": item.vector_score,
                    "bm25_score": item.bm25_score,
                    "rrf_score": item.rrf_score,
                },
            )
            for item in hybrid_results[:top_k]
        ]
        return RetrievalResponse(
            query=query,
            results=candidates,
            top_k=top_k,
            candidate_top_k=candidate_k,
            total_latency_ms=(perf_counter() - started) * 1000.0,
            hybrid_latency_ms=hybrid_timing.total_seconds * 1000.0,
            rerank_latency_ms=0.0,
            embedding_latency_ms=hybrid_timing.vector_embedding_seconds * 1000.0,
            vector_search_latency_ms=hybrid_timing.vector_search_seconds * 1000.0,
            bm25_latency_ms=hybrid_timing.bm25_seconds * 1000.0,
            rrf_latency_ms=hybrid_timing.rrf_seconds * 1000.0,
            model_name=self.retriever.config.model_name,
            retrieval_route="fast",
        )

    async def retrieve_adaptive(
        self,
        query: str,
        *,
        top_k: int = 3,
        candidate_top_k: int | None = None,
        rrf_threshold: float = 0.025,
        min_agreeing_sources: int = 2,
        bm25_threshold: float = 5.0,
    ) -> RetrievalResponse:
        """Adaptive path: fast when confidence is high, quality when uncertain."""
        started = perf_counter()
        candidate_k = candidate_top_k or self.retriever.config.candidate_top_k

        # Step 1: always run hybrid retrieval (BM25 + dense + RRF)
        hybrid_results, hybrid_timing = await asyncio.to_thread(
            self.retriever.hybrid_retriever.retrieve_with_timing,
            query,
            top_k=candidate_k,
        )

        if not hybrid_results:
            return RetrievalResponse(
                query=query,
                results=[],
                top_k=top_k,
                candidate_top_k=candidate_k,
                total_latency_ms=(perf_counter() - started) * 1000.0,
                hybrid_latency_ms=hybrid_timing.total_seconds * 1000.0,
                rerank_latency_ms=0.0,
                embedding_latency_ms=hybrid_timing.vector_embedding_seconds * 1000.0,
                vector_search_latency_ms=hybrid_timing.vector_search_seconds * 1000.0,
                bm25_latency_ms=hybrid_timing.bm25_seconds * 1000.0,
                rrf_latency_ms=hybrid_timing.rrf_seconds * 1000.0,
                model_name=self.retriever.config.model_name,
                retrieval_route="fast",
            )

        # Step 2: evaluate confidence for fast-path eligibility
        top_result = hybrid_results[0]
        top_rrf = top_result.rrf_score
        top_bm25 = top_result.bm25_score or 0.0

        # Count agreeing sources: results that appear in both BM25 and dense rankings
        agreeing_sources = sum(
            1 for r in hybrid_results[:top_k]
            if r.vector_score is not None and r.bm25_score is not None
        )

        is_high_confidence = (
            top_rrf >= rrf_threshold
            and top_bm25 >= bm25_threshold
            and agreeing_sources >= min_agreeing_sources
        )

        if is_high_confidence:
            # FAST PATH: skip reranker, use RRF scores directly
            candidates = [
                RetrievalCandidate(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    text=item.text,
                    score=item.rrf_score,
                    metadata={
                        **item.metadata,
                        "vector_score": item.vector_score,
                        "bm25_score": item.bm25_score,
                        "rrf_score": item.rrf_score,
                    },
                )
                for item in hybrid_results[:top_k]
            ]
            return RetrievalResponse(
                query=query,
                results=candidates,
                top_k=top_k,
                candidate_top_k=candidate_k,
                total_latency_ms=(perf_counter() - started) * 1000.0,
                hybrid_latency_ms=hybrid_timing.total_seconds * 1000.0,
                rerank_latency_ms=0.0,
                embedding_latency_ms=hybrid_timing.vector_embedding_seconds * 1000.0,
                vector_search_latency_ms=hybrid_timing.vector_search_seconds * 1000.0,
                bm25_latency_ms=hybrid_timing.bm25_seconds * 1000.0,
                rrf_latency_ms=hybrid_timing.rrf_seconds * 1000.0,
                model_name=self.retriever.config.model_name,
                retrieval_route="fast",
            )

        # QUALITY PATH: run cross-encoder reranker on hybrid candidates
        rerank_started = perf_counter()
        final_top_k = top_k
        reranked = await asyncio.to_thread(
            self.retriever.reranker.rerank,
            query,
            hybrid_results,
            final_top_k,
        )
        rerank_seconds = perf_counter() - rerank_started

        candidates = [
            RetrievalCandidate(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                score=item.rerank_score,
                metadata={
                    **item.metadata,
                    "vector_score": item.vector_score,
                    "bm25_score": item.bm25_score,
                    "rrf_score": item.rrf_score,
                    "rerank_score": item.rerank_score,
                    "rerank_rank": item.rerank_rank,
                },
            )
            for item in reranked
        ]
        phase = getattr(self.retriever.reranker, "last_timing", None)
        return RetrievalResponse(
            query=query,
            results=candidates,
            top_k=top_k,
            candidate_top_k=candidate_k,
            total_latency_ms=(perf_counter() - started) * 1000.0,
            hybrid_latency_ms=hybrid_timing.total_seconds * 1000.0,
            rerank_latency_ms=rerank_seconds * 1000.0,
            embedding_latency_ms=hybrid_timing.vector_embedding_seconds * 1000.0,
            vector_search_latency_ms=hybrid_timing.vector_search_seconds * 1000.0,
            bm25_latency_ms=hybrid_timing.bm25_seconds * 1000.0,
            rrf_latency_ms=hybrid_timing.rrf_seconds * 1000.0,
            model_name=self.retriever.config.model_name,
            device=getattr(phase, "device", None),
            retrieval_route="quality",
        )

    async def retrieve_multilingual(
        self,
        query: str,
        *,
        top_k: int = 3,
        candidate_top_k: int | None = None,
        rrf_threshold: float = 0.025,
        min_agreeing_sources: int = 2,
        bm25_threshold: float = 5.0,
        source_language: str = "en",
        translator: Any | None = None,
        translation_cache: Any | None = None,
    ) -> RetrievalResponse:
        """Multilingual retrieval: try original query, fallback to translated English.

        1. Detect language (already done by caller)
        2. Expand query with bilingual terms for BM25
        3. Try adaptive retrieval with expanded query
        4. If confidence is low and language != English, translate and retry
        """
        from dataclasses import replace as dc_replace
        from voice_rag_ingestion.query_expansion import expand_query

        started = perf_counter()

        # Step 1: Expand query with English terms for BM25 recall
        expanded_query = expand_query(query, source_language) if source_language != "en" else query

        # Step 2: Try retrieval with expanded query
        result = await self.retrieve_adaptive(
            expanded_query,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
            rrf_threshold=rrf_threshold,
            min_agreeing_sources=min_agreeing_sources,
            bm25_threshold=bm25_threshold,
        )

        # Step 3: Check confidence — if high, return immediately
        is_confident = (
            result.found
            and len(result.results) > 0
            and result.results[0].score >= 0.015
        )

        if is_confident:
            return dc_replace(
                result,
                detected_language=source_language,
                total_latency_ms=(perf_counter() - started) * 1000.0,
            )

        # Step 4: If low confidence and not English, translate and retry
        if source_language != "en" and translator is not None:
            translated_query: str | None = None

            # Check translation cache first
            if translation_cache is not None:
                translated_query = translation_cache.get(query, source_language, "en")

            if translated_query is None:
                translated_query = await asyncio.to_thread(
                    translator.translate, query, source_language, "en"
                )
                if translation_cache is not None and translated_query != query:
                    translation_cache.put(query, source_language, "en", translated_query)

            # Only retry if translation actually produced something different
            if translated_query and translated_query != query:
                result = await self.retrieve_adaptive(
                    translated_query,
                    top_k=top_k,
                    candidate_top_k=candidate_top_k,
                    rrf_threshold=rrf_threshold,
                    min_agreeing_sources=min_agreeing_sources,
                    bm25_threshold=bm25_threshold,
                )
                return dc_replace(
                    result,
                    translation_used=True,
                    translated_query=translated_query,
                    detected_language=source_language,
                    total_latency_ms=(perf_counter() - started) * 1000.0,
                )

        return dc_replace(
            result,
            detected_language=source_language,
            total_latency_ms=(perf_counter() - started) * 1000.0,
        )

    async def health_check(self) -> dict[str, Any]:
        hybrid = self.retriever.hybrid_retriever
        if isinstance(hybrid, BM25FirstHybridRetriever):
            bm25 = hybrid.bm25_retriever.index
            return {
                "retrieval_engine": "bm25_first_hybrid_rrf_reranker",
                "bm25_chunks": bm25.size,
                "reranker_model": self.retriever.config.model_name,
            }
        store = hybrid.vector_retriever.vector_store
        bm25 = hybrid.bm25_retriever.index
        return {
            "retrieval_engine": "existing_hybrid_rrf_reranker",
            "collection": store.config.collection_name,
            "collection_exists": await asyncio.to_thread(store.collection_exists),
            "bm25_chunks": bm25.size,
            "reranker_model": self.retriever.config.model_name,
        }


def build_existing_retrieval_engine(
    *,
    app_settings: Any,
    auto_index: bool | None = None,
) -> ExistingRetrievalEngine:
    """Build one retrieval stack using Pulind's existing components.

    Supports dual (Qdrant + BM25) and BM25-first (on-demand candidate embedding) strategies.
    """

    embedding_config = EmbeddingConfig.from_env()
    provider = SentenceTransformerEmbedder(embedding_config)
    embedder = (
        CachedEmbedder(provider, config=embedding_config)
        if embedding_config.cache_enabled
        else provider
    )
    store_config = VectorStoreConfig.from_env()
    store_config = store_config.__class__(
        url=app_settings.qdrant_url,
        collection_name=app_settings.qdrant_collection,
        api_key=app_settings.qdrant_api_key,
        recreate_collection=app_settings.app_index_recreate,
        timeout=store_config.timeout,
        upsert_batch_size=store_config.upsert_batch_size,
    )
    vector_store = QdrantVectorStore(store_config)
    bm25_json_path = Path(__import__("os").getenv("BM25_INDEX_PATH", ".cache/bm25_index.json"))
    bm25_db_path = Path(__import__("os").getenv("BM25_SQLITE_PATH", ".cache/bm25_index.db"))
    # Prefer SQLite; fall back to JSON; otherwise empty
    if bm25_db_path.exists():
        bm25_index = BM25SqliteIndex.load(bm25_db_path)
    elif bm25_json_path.exists():
        bm25_index = BM25Index.load(bm25_json_path)
    else:
        bm25_index = BM25Index()

    should_index = app_settings.app_auto_index if auto_index is None else auto_index
    if should_index and not vector_store.collection_exists() and bm25_index.size == 0:
        loader_config = LoaderConfig.from_env(sample_size=app_settings.app_index_sample_size)
        documents, _ = DatasetLoader(loader_config).load_documents()
        index_stats, chunks = index_documents(
            documents,
            embedder=embedder,
            vector_store=vector_store,
            chunking_config=ChunkingConfig.from_env(),
            embedding_batch_size=embedding_config.batch_size,
            recreate_collection=app_settings.app_index_recreate,
        )
        del index_stats
        if bm25_index.size == 0:
            bm25_index.rebuild(chunks)
            bm25_index.save(bm25_path)

    import os
    retrieval_strategy = os.getenv("RETRIEVAL_STRATEGY", "bm25_first").lower()
    if retrieval_strategy == "bm25_first" or (bm25_index.size > 0 and not vector_store.collection_exists()):
        hybrid = BM25FirstHybridRetriever(
            BM25Retriever(bm25_index),
            embedder,
            config=HybridConfig.from_env(),
        )
    else:
        hybrid = HybridRetriever(
            VectorRetriever(embedder, vector_store),
            BM25Retriever(bm25_index),
            config=HybridConfig.from_env(),
        )
    reranker = CrossEncoderReranker(RerankerConfig.from_env())
    return ExistingRetrievalEngine(HybridRerankRetriever(hybrid, reranker))
