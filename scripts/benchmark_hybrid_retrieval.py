#!/usr/bin/env python3
"""Benchmark vector-only, BM25-only, RRF, and total hybrid retrieval."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.bm25 import BM25Config, BM25Index, BM25Retriever
from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from voice_rag_ingestion.hybrid import HybridConfig, HybridRetriever
from voice_rag_ingestion.indexing import index_documents
from voice_rag_ingestion.loader import DatasetLoader
from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig
from voice_rag_ingestion.retrieval import VectorRetriever


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def report(label: str, values: list[float]) -> None:
    print(
        f"{label}: count={len(values)} average={mean(values):.6f}s "
        f"P50={percentile(values, 50):.6f}s P70={percentile(values, 70):.6f}s "
        f"P95={percentile(values, 95):.6f}s P100={percentile(values, 100):.6f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--with-cache", action="store_true", help="Include embedding cache lookup behavior")
    parser.add_argument("--mock-embeddings", action="store_true")
    args = parser.parse_args()
    loader_config = LoaderConfig.from_env(sample_size=args.sample_size)
    embedding_config = EmbeddingConfig.from_env()
    if not args.with_cache:
        embedding_config = replace(embedding_config, cache_enabled=False)
    if args.mock_embeddings:
        embedding_config = replace(embedding_config, model_name="dev-hash-embedding")
        from voice_rag_ingestion.embeddings.dev import HashEmbeddingProvider

        provider = HashEmbeddingProvider()
    else:
        provider = SentenceTransformerEmbedder(embedding_config)
    hybrid_config = HybridConfig.from_env()
    if args.top_k is not None:
        hybrid_config = replace(
            hybrid_config,
            final_top_k=args.top_k,
            vector_top_k=args.top_k,
            bm25_top_k=args.top_k,
        )
    documents, _ = DatasetLoader(loader_config).load_documents()
    store = QdrantVectorStore(VectorStoreConfig.from_env())
    _, chunks = index_documents(
        documents,
        embedder=provider,
        vector_store=store,
        chunking_config=replace(ChunkingConfig.from_env(), strategy=hybrid_config.chunking_strategy),
        embedding_batch_size=embedding_config.batch_size,
        recreate_collection=True,
    )
    bm25_index = BM25Index(BM25Config(k1=hybrid_config.bm25_k1, b=hybrid_config.bm25_b, tokenizer=hybrid_config.tokenizer))
    bm25_index.rebuild(chunks)
    vector_retriever = VectorRetriever(provider, store)
    bm25_retriever = BM25Retriever(bm25_index)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever, config=hybrid_config)
    queries = ["What is a corporation?", "কৰ্পোৰেচন কি?", "कंपनी क्या है?"]
    vector_times: list[float] = []
    bm25_times: list[float] = []
    rrf_times: list[float] = []
    total_times: list[float] = []
    for _ in range(args.runs):
        for query in queries:
            _, vector_timing = vector_retriever.retrieve_with_timing(query, top_k=hybrid_config.vector_top_k)
            _, bm25_timing = bm25_retriever.retrieve_with_timing(query, top_k=hybrid_config.bm25_top_k)
            _, hybrid_timing = hybrid_retriever.retrieve_with_timing(query, top_k=hybrid_config.final_top_k)
            vector_times.append(vector_timing.total_seconds)
            bm25_times.append(bm25_timing)
            rrf_times.append(hybrid_timing.rrf_seconds)
            total_times.append(hybrid_timing.total_seconds)
    print(f"Benchmark queries: {len(queries)}; runs: {args.runs}; observations: {len(total_times)}")
    report("vector-only retrieval", vector_times)
    report("BM25-only retrieval", bm25_times)
    report("RRF fusion", rrf_times)
    report("total hybrid retrieval", total_times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
