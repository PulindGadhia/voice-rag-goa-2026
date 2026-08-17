#!/usr/bin/env python3
"""Index a small normalized/chunked sample into Qdrant."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.embeddings import (
    CachedEmbedder,
    EmbeddingConfig,
    SentenceTransformerEmbedder,
)
from voice_rag_ingestion.indexing import index_documents
from voice_rag_ingestion.loader import DatasetLoadError, DatasetLoader
from voice_rag_ingestion.logging_utils import configure_logging
from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--strategy", choices=("fixed", "sentence", "semantic", "metadata"), default=None)
    parser.add_argument("--max-chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-recreate", action="store_true")
    parser.add_argument("--mock-embeddings", action="store_true", help="Use deterministic vectors for offline smoke tests")
    args = parser.parse_args()

    loader_config = replace(LoaderConfig.from_env(), sample_size=args.sample_size)
    chunking_config = ChunkingConfig.from_env()
    chunking_changes = {}
    if args.strategy:
        chunking_changes["strategy"] = args.strategy
    if args.max_chunk_size is not None:
        chunking_changes["max_chunk_size"] = args.max_chunk_size
    if args.overlap is not None:
        chunking_changes["overlap"] = args.overlap
    if chunking_changes:
        chunking_config = replace(chunking_config, **chunking_changes)
    embedding_config = EmbeddingConfig.from_env()
    if args.mock_embeddings:
        embedding_config = replace(embedding_config, model_name="dev-hash-embedding")
    if args.batch_size is not None:
        embedding_config = replace(embedding_config, batch_size=args.batch_size)
    configure_logging(loader_config.log_level)

    try:
        documents, load_stats = DatasetLoader(loader_config).load_documents()
        if args.mock_embeddings:
            from voice_rag_ingestion.embeddings.dev import HashEmbeddingProvider

            provider = HashEmbeddingProvider()
        else:
            provider = SentenceTransformerEmbedder(embedding_config)
        embedder = (
            CachedEmbedder(provider, config=embedding_config)
            if embedding_config.cache_enabled
            else provider
        )
        vector_store = QdrantVectorStore(VectorStoreConfig.from_env())
        index_stats, _ = index_documents(
            documents,
            embedder=embedder,
            vector_store=vector_store,
            chunking_config=chunking_config,
            embedding_batch_size=embedding_config.batch_size,
            recreate_collection=not args.no_recreate,
        )
    except (DatasetLoadError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if "embedder" in locals() and isinstance(embedder, CachedEmbedder):
            embedder.close()

    print("Indexing completed")
    print(f"raw records loaded: {load_stats.records_read}")
    print(f"documents indexed: {index_stats.documents}")
    print(f"chunks indexed: {index_stats.chunks}")
    print(f"vectors upserted: {index_stats.upserted}")
    print(f"embedding dimension: {provider.dimension}")
    print(f"embedding batches: {index_stats.embedding_batches}")
    print(f"embedding seconds: {index_stats.embedding_seconds:.4f}")
    if isinstance(embedder, CachedEmbedder):
        print(f"cache hits: {embedder.stats.hits}")
        print(f"cache misses: {embedder.stats.misses}")
        print(f"cache writes: {embedder.stats.writes}")
    print(f"qdrant collection: {vector_store.config.collection_name}")
    print(f"qdrant url: {vector_store.config.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
