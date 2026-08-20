#!/usr/bin/env python3
"""Inspect vector, BM25, and RRF-fused retrieval on a bounded sample."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.bm25 import BM25Config, BM25Index, BM25Retriever
from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.embeddings import CachedEmbedder, EmbeddingConfig, SentenceTransformerEmbedder
from voice_rag_ingestion.evaluation import EvaluationCase, evaluate_rankings
from voice_rag_ingestion.hybrid import HybridConfig, HybridRetriever
from voice_rag_ingestion.indexing import index_documents
from voice_rag_ingestion.loader import DatasetLoader
from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig
from voice_rag_ingestion.retrieval import VectorRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--mock-embeddings", action="store_true")
    parser.add_argument("--bm25-first", action="store_true", help="Use BM25-first on-demand candidate embedding retrieval")
    parser.add_argument("--bm25-index-path", default=os.getenv("BM25_INDEX_PATH", ".cache/bm25_index.json"))
    parser.add_argument("--query", action="append", dest="queries")
    args = parser.parse_args()

    loader_config = LoaderConfig.from_env(sample_size=args.sample_size)
    embedding_config = EmbeddingConfig.from_env()
    if args.mock_embeddings:
        embedding_config = replace(embedding_config, model_name="dev-hash-embedding")
        from voice_rag_ingestion.embeddings.dev import HashEmbeddingProvider

        provider = HashEmbeddingProvider()
    else:
        provider = SentenceTransformerEmbedder(embedding_config)
    embedder = CachedEmbedder(provider, config=embedding_config) if embedding_config.cache_enabled else provider
    hybrid_config = HybridConfig.from_env()
    if args.top_k is not None:
        hybrid_config = replace(
            hybrid_config,
            final_top_k=args.top_k,
            vector_top_k=args.top_k,
            bm25_top_k=args.top_k,
        )
    documents, stats = DatasetLoader(loader_config).load_documents()
    print(f"Loaded {len(documents)} documents from {stats.records_read} records using backend={loader_config.backend} (sample_size={loader_config.sample_size})")
    store = QdrantVectorStore(VectorStoreConfig.from_env())
    index_stats, chunks = index_documents(
        documents,
        embedder=embedder,
        vector_store=store,
        chunking_config=replace(ChunkingConfig.from_env(), strategy=hybrid_config.chunking_strategy),
        embedding_batch_size=embedding_config.batch_size,
        recreate_collection=True,
    )
    bm25_index = BM25Index(
        BM25Config(
            k1=hybrid_config.bm25_k1,
            b=hybrid_config.bm25_b,
            tokenizer=hybrid_config.tokenizer,
        )
    )
    bm25_index.rebuild(chunks)
    bm25_index.save(args.bm25_index_path)
    vector_retriever = VectorRetriever(embedder, store)
    bm25_retriever = BM25Retriever(bm25_index)
    if args.bm25_first:
        from voice_rag_ingestion.bm25_first import BM25FirstHybridRetriever

        hybrid_retriever = BM25FirstHybridRetriever(bm25_retriever, embedder, config=hybrid_config)
    else:
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever, config=hybrid_config)
    queries = args.queries or ["What is a corporation?", "কৰ্পোৰেচন কি?", "कंपनी क्या है?"]

    print(f"Indexed documents: {index_stats.documents}; chunks: {index_stats.chunks}; BM25 size: {bm25_index.size}")
    print(f"BM25 index saved: {args.bm25_index_path}")
    for query in queries:
        vector_results = vector_retriever.retrieve(query, top_k=hybrid_config.vector_top_k)
        bm25_results = bm25_retriever.retrieve(query, top_k=hybrid_config.bm25_top_k)
        hybrid_results, timing = hybrid_retriever.retrieve_with_timing(query, top_k=hybrid_config.final_top_k)
        print(f"\nQUERY\n{query}")
        print("\nVector results:")
        for rank, result in enumerate(vector_results[:hybrid_config.final_top_k], 1):
            print(f"{rank}. {result.chunk_id} / {result.score:.6f} / {result.text[:220]}")
        print("\nBM25 results:")
        for rank, result in enumerate(bm25_results[:hybrid_config.final_top_k], 1):
            print(f"{rank}. {result.chunk_id} / {result.score:.6f} / {result.text[:220]}")
        print("\nHybrid RRF results:")
        for rank, result in enumerate(hybrid_results, 1):
            print(
                f"{rank}. {result.chunk_id} / RRF={result.rrf_score:.6f} / "
                f"vector={result.vector_score} / BM25={result.bm25_score} / {result.text[:220]}"
            )
        print(
            f"timing: vector={timing.vector_seconds:.6f}s BM25={timing.bm25_seconds:.6f}s "
            f"RRF={timing.rrf_seconds:.6f}s total={timing.total_seconds:.6f}s"
        )

    cases_by_query: dict[str, set[str]] = {}
    for chunk in chunks:
        if chunk.source.get("is_selected") in (1, True, "1"):
            query = chunk.metadata.get("query")
            if query:
                cases_by_query.setdefault(query, set()).add(chunk.chunk_id)
    cases = [EvaluationCase(query=query, relevant_chunk_ids=frozenset(ids)) for query, ids in cases_by_query.items()]
    if cases:
        rankings = [hybrid_retriever.retrieve(case.query, top_k=hybrid_config.final_top_k) for case in cases]
        metrics = evaluate_rankings(rankings, cases, top_k=hybrid_config.final_top_k)
        print(f"\nQuality evaluation (selected passages as relevance; cases={metrics.cases}):")
        print(f"Recall@{hybrid_config.final_top_k}: {metrics.recall_at_k:.4f}")
        print(f"MRR@{hybrid_config.final_top_k}: {metrics.mrr_at_k:.4f}")
    else:
        print("\nQuality evaluation: unavailable; sample contained no selected-passage labels.")
    if isinstance(embedder, CachedEmbedder):
        print(f"Cache hits: {embedder.stats.hits}; misses: {embedder.stats.misses}")
        embedder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
