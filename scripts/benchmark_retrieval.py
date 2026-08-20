#!/usr/bin/env python3
"""Benchmark query embedding, Qdrant search, and total vector retrieval."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.embeddings import CachedEmbedder, EmbeddingConfig, SentenceTransformerEmbedder
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


def report(name: str, values: list[float]) -> None:
    print(f"{name}: count={len(values)} average={mean(values):.6f}s "
          f"P50={percentile(values, 50):.6f}s P70={percentile(values, 70):.6f}s "
          f"P95={percentile(values, 95):.6f}s P100={percentile(values, 100):.6f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mock-embeddings", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be > 0")
    loader_config = LoaderConfig.from_env(sample_size=args.sample_size)
    embedding_config = EmbeddingConfig.from_env()
    if args.mock_embeddings:
        embedding_config = replace(embedding_config, model_name="dev-hash-embedding")
        from voice_rag_ingestion.embeddings.dev import HashEmbeddingProvider

        provider = HashEmbeddingProvider()
    else:
        provider = SentenceTransformerEmbedder(embedding_config)
    embedder = CachedEmbedder(provider, config=embedding_config) if embedding_config.cache_enabled else provider
    store = QdrantVectorStore(VectorStoreConfig.from_env())
    documents, _ = DatasetLoader(loader_config).load_documents()
    index_documents(
        documents,
        embedder=embedder,
        vector_store=store,
        chunking_config=replace(ChunkingConfig.from_env(), strategy="fixed"),
        embedding_batch_size=embedding_config.batch_size,
        recreate_collection=True,
    )
    retriever = VectorRetriever(embedder, store)
    queries = ["What is a corporation?", "কৰ্পোৰেচন কি?", "कंपनी क्या है?"]
    embedding_times: list[float] = []
    search_times: list[float] = []
    total_times: list[float] = []
    for _ in range(args.runs):
        for query in queries:
            _, timing = retriever.retrieve_with_timing(query, top_k=args.top_k)
            embedding_times.append(timing.query_embedding_seconds)
            search_times.append(timing.qdrant_search_seconds)
            total_times.append(timing.total_seconds)
    print(f"Benchmark queries: {len(queries)}; runs: {args.runs}; observations: {len(total_times)}")
    report("query embedding", embedding_times)
    report("qdrant search", search_times)
    report("total retrieval", total_times)
    if isinstance(embedder, CachedEmbedder):
        print(f"cache hits: {embedder.stats.hits}; cache misses: {embedder.stats.misses}")
        embedder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
