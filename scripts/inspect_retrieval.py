#!/usr/bin/env python3
"""Bootstrap a small vector index and inspect multilingual top-k results."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.embeddings import CachedEmbedder, EmbeddingConfig, SentenceTransformerEmbedder
from voice_rag_ingestion.indexing import index_documents
from voice_rag_ingestion.loader import DatasetLoader
from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig
from voice_rag_ingestion.retrieval import VectorRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mock-embeddings", action="store_true")
    parser.add_argument("--query", action="append", dest="queries")
    args = parser.parse_args()
    loader_config = LoaderConfig.from_env(sample_size=args.sample_size)
    chunking_config = replace(ChunkingConfig.from_env(), strategy="fixed")
    embedding_config = EmbeddingConfig.from_env()
    if args.mock_embeddings:
        embedding_config = replace(embedding_config, model_name="dev-hash-embedding")
        from voice_rag_ingestion.embeddings.dev import HashEmbeddingProvider

        provider = HashEmbeddingProvider()
    else:
        provider = SentenceTransformerEmbedder(embedding_config)
    embedder = CachedEmbedder(provider, config=embedding_config) if embedding_config.cache_enabled else provider
    store = QdrantVectorStore(VectorStoreConfig.from_env())
    documents, stats = DatasetLoader(loader_config).load_documents()
    print(f"Loaded {len(documents)} documents from {stats.records_read} records using backend={loader_config.backend} (sample_size={loader_config.sample_size})")
    index_documents(
        documents,
        embedder=embedder,
        vector_store=store,
        chunking_config=chunking_config,
        embedding_batch_size=embedding_config.batch_size,
        recreate_collection=True,
    )
    retriever = VectorRetriever(embedder, store)
    queries = args.queries or [
        "What is a corporation?",
        "কৰ্পোৰেচন কি?",
        "कंपनी क्या है?",
    ]
    for query in queries:
        results, timing = retriever.retrieve_with_timing(query, top_k=args.top_k)
        print(f"\nQuery: {query}")
        print(f"query embedding seconds: {timing.query_embedding_seconds:.6f}")
        print(f"qdrant search seconds: {timing.qdrant_search_seconds:.6f}")
        print(f"total retrieval seconds: {timing.total_seconds:.6f}")
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. score={result.score:.6f} strategy={result.metadata.get('chunk_strategy')} "
                f"language={result.metadata.get('language')} "
                f"document_id={result.document_id} chunk_id={result.chunk_id}"
            )
            print(f"   {result.text[:300]}")
    if isinstance(embedder, CachedEmbedder):
        print(f"\nCache hits: {embedder.stats.hits}; misses: {embedder.stats.misses}")
        embedder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
