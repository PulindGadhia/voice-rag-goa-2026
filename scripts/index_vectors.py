#!/usr/bin/env python3
"""Streaming ingestion CLI for MSMARCO-XI dataset into Qdrant vector store and BM25 index."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

# Ensure src and project roots are in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

from voice_rag_ingestion.bm25 import BM25Index
from voice_rag_ingestion.bm25_sqlite import BM25SqliteIndex
from voice_rag_ingestion.checkpoint import DEFAULT_CHECKPOINT_PATH
from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.embeddings import (
    CachedEmbedder,
    EmbeddingConfig,
    SentenceTransformerEmbedder,
)
from voice_rag_ingestion.indexing import (
    StreamingIndexStats,
    stream_index_dataset,
    stream_index_from_parquet,
)
from voice_rag_ingestion.loader import DatasetLoadError, DatasetLoader
from voice_rag_ingestion.logging_utils import configure_logging
from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig


def print_progress(stats: StreamingIndexStats) -> None:
    sys.stdout.write(
        f"\r[Ingestion Progress] Records: {stats.records_read:5d} | "
        f"Docs: {stats.documents_processed:5d} (Skip: {stats.documents_skipped:3d}) | "
        f"Chunks: {stats.chunks_created:5d} (Qdrant: {stats.qdrant_chunks_indexed:5d}, BM25: {stats.bm25_chunks_indexed:5d}) | "
        f"Langs: {len(stats.languages_processed):2d} | "
        f"Rate: {stats.throughput_chunks_per_sec:5.1f} c/s | "
        f"Time: {stats.elapsed_seconds:5.1f}s"
    )
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream and index ai4bharat/MSMARCO-XI into Qdrant & BM25 without loading full dataset into RAM."
    )
    parser.add_argument("--split", type=str, default=None, help="Dataset split ('validation' or 'train')")
    parser.add_argument("--sample-size", "--max-records", dest="sample_size", type=int, default=None, help="Number of records to stream (default: all or from HF_SAMPLE_SIZE)")
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated target languages (e.g. 'asm_Beng,hin_Deva,tam_Taml')")
    parser.add_argument("--stream-batch-size", type=int, default=250, help="Document batch size for incremental streaming (default: 250)")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size (default: 64 or 128)")
    parser.add_argument("--backend", choices=("dataset_server", "hf_datasets"), default=None, help="Dataset loading backend (default: dataset_server for <=100, hf_datasets for >100 or full stream)")
    parser.add_argument("--strategy", choices=("fixed", "sentence", "semantic", "metadata"), default=None, help="Chunking strategy (default: metadata)")
    parser.add_argument("--max-chunk-size", type=int, default=None, help="Max chunk word size (default: 256)")
    parser.add_argument("--overlap", type=int, default=None, help="Chunk overlap word count (default: 32)")
    parser.add_argument("--bm25-path", type=str, default=None, help="Path to BM25 index JSON (default: .cache/bm25_index.json)")
    parser.add_argument("--bm25-db", type=str, default=None, help="Path to BM25 SQLite database (default: .cache/bm25_index.db). Auto-migrates from JSON if only JSON exists.")
    parser.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to ingestion checkpoint file (default: .cache/ingestion_checkpoint.json)")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint; start from row 0")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate Qdrant collection, BM25 index, and checkpoint")
    parser.add_argument("--no-recreate", action="store_true", help="Do not recreate Qdrant collection (append/upsert mode)")
    parser.add_argument("--no-bm25", action="store_true", help="Skip updating BM25 index")
    parser.add_argument("--bm25-only", action="store_true", help="Build BM25 index only without generating dense vectors during ingestion")
    parser.add_argument("--mock-embeddings", action="store_true", help="Use deterministic vectors for offline smoke tests")
    # Local Parquet ingestion
    parser.add_argument("--local-parquet-dir", type=str, default=None, help="Path to a directory of local *.parquet files; bypasses HF streaming entirely (use after download_dataset.py)")
    parser.add_argument("--parquet-batch-size", type=int, default=1000, help="Rows per PyArrow read batch for local Parquet ingestion (default: 1000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for local Parquet ingestion (default: 1; reserved for future use)")
    parser.add_argument("--fast-ingest", action="store_true", help="Enable aggressive bulk ingestion: 5000-doc batches, SQLite synchronous=OFF, checkpoint every 10 batches")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Save checkpoint every N batches (default: 1; --fast-ingest default: 10)")
    args = parser.parse_args()

    loader_config = LoaderConfig.from_env(
        sample_size=args.sample_size,
        backend=args.backend,
        split=args.split or LoaderConfig.split,
        streaming=True,
    )

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

    recreate = args.recreate or (not args.no_recreate and bool(os.getenv("APP_INDEX_RECREATE", "false").lower() in {"1", "true"}))
    bm25_json_path = Path(args.bm25_path or os.getenv("BM25_INDEX_PATH", ".cache/bm25_index.json"))
    bm25_db_path = Path(args.bm25_db or os.getenv("BM25_SQLITE_PATH", ".cache/bm25_index.db"))
    checkpoint_path = Path(args.checkpoint_path)

    # Determine BM25 storage mode: prefer SQLite
    use_sqlite = True

    print("=" * 80)
    print("VOICE RAG GOA 2026 — STREAMING INGESTION ENGINE")
    print("=" * 80)
    print(f"Dataset:            {loader_config.dataset_name} (Split: {loader_config.split})")
    print(f"Sample / Limit:     {loader_config.sample_size if loader_config.sample_size is not None else 'ALL (Full Stream)'}")
    if args.local_parquet_dir:
        print(f"Source:             LOCAL PARQUET  ({args.local_parquet_dir})")
    else:
        print(f"Backend:            {loader_config.backend}")
    print(f"Languages Filter:   {args.languages or 'ALL (11 Indic Languages)'}")
    print(f"Chunking Strategy:  {chunking_config.strategy} (max_size={chunking_config.max_chunk_size}, overlap={chunking_config.overlap})")
    print(f"Ingestion Mode:     {'BM25-FIRST (BM25 only, on-demand query vectors)' if args.bm25_only else 'DUAL (Qdrant + BM25)'}")
    if not args.bm25_only:
        print(f"Embedding Model:    {'mock-hash-embedding' if args.mock_embeddings else embedding_config.model_name} (batch_size={embedding_config.batch_size})")
        print(f"Qdrant Collection:  {os.getenv('QDRANT_COLLECTION', 'voice_rag_chunks')} (Recreate: {recreate})")
    print(f"BM25 Storage:       SQLite ({bm25_db_path})")
    print(f"BM25 JSON (legacy): {bm25_json_path}")
    print(f"BM25 Enabled:       {not args.no_bm25}")
    if args.fast_ingest:
        print(f"Fast Ingest:        ENABLED (sync=OFF, 5000-doc batches, checkpoint every {args.checkpoint_interval or 10} batches)")
    print(f"Checkpoint Path:    {checkpoint_path} (Resume: {not args.no_resume and not recreate})")
    print("=" * 80 + "\n")


    bm25_index: BM25Index | BM25SqliteIndex | None = None
    # Path used by stream_index_from_parquet for post-batch persistence
    bm25_path_for_save = bm25_db_path
    if not args.no_bm25:
        if recreate:
            # Fresh start: delete existing SQLite DB if present
            if bm25_db_path.exists():
                bm25_db_path.unlink()
            bm25_index = BM25SqliteIndex(bm25_db_path, fast_ingest=args.fast_ingest)
        elif bm25_db_path.exists():
            # SQLite DB already exists — just open it
            bm25_index = BM25SqliteIndex.load(bm25_db_path, fast_ingest=args.fast_ingest)
            print(f"Loaded existing SQLite BM25 index: {bm25_index.size} chunks")
        elif bm25_json_path.exists():
            # Auto-migrate from JSON to SQLite
            print(f"Migrating JSON BM25 index → SQLite...")
            print(f"  Source: {bm25_json_path}")
            print(f"  Target: {bm25_db_path}")
            t_mig_start = time.time()
            bm25_index = BM25SqliteIndex.migrate_from_json(bm25_json_path, bm25_db_path)
            t_mig_elapsed = time.time() - t_mig_start
            print(f"  Migration complete: {bm25_index.size} chunks in {t_mig_elapsed:.1f}s")
        else:
            # No existing index — create fresh
            bm25_index = BM25SqliteIndex(bm25_db_path, fast_ingest=args.fast_ingest)

    languages_filter = [l.strip() for l in args.languages.split(",")] if args.languages else None

    try:
        if args.local_parquet_dir:
            # ── Local Parquet ingestion path (no HF network calls) ──────────────
            parquet_dir = Path(args.local_parquet_dir)
            if not parquet_dir.is_dir():
                print(f"\nERROR: --local-parquet-dir does not exist: {parquet_dir}", file=sys.stderr)
                return 1

            embedder = None
            vector_store = None
            if not args.bm25_only:
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

            stats = stream_index_from_parquet(
                parquet_dir=parquet_dir,
                dataset_name=loader_config.dataset_name,
                dataset_config=loader_config.dataset_config,
                split=loader_config.split,
                embedder=embedder,
                vector_store=vector_store,
                bm25_index=bm25_index,
                bm25_path=bm25_path_for_save if not args.no_bm25 else None,
                chunking_config=chunking_config,
                stream_batch_size=args.stream_batch_size,
                embedding_batch_size=embedding_config.batch_size,
                recreate_collection=recreate,
                languages=languages_filter,
                checkpoint_path=checkpoint_path,
                resume=not args.no_resume,
                bm25_only=args.bm25_only,
                parquet_batch_size=args.parquet_batch_size,
                sample_size=loader_config.sample_size,
                fast_ingest=args.fast_ingest,
                checkpoint_interval=args.checkpoint_interval,
                workers=args.workers,
            )
        else:
            # ── HF streaming ingestion path ──────────────────────────────────────
            loader = DatasetLoader(loader_config)
            embedder = None
            vector_store = None
            if not args.bm25_only:
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

            stats = stream_index_dataset(
                loader=loader,
                embedder=embedder,
                vector_store=vector_store,
                bm25_index=bm25_index,
                bm25_path=bm25_path_for_save if not args.no_bm25 else None,
                chunking_config=chunking_config,
                stream_batch_size=args.stream_batch_size,
                embedding_batch_size=embedding_config.batch_size,
                recreate_collection=recreate,
                languages=languages_filter,
                checkpoint_path=checkpoint_path,
                resume=not args.no_resume,
                bm25_only=args.bm25_only,
                progress_callback=print_progress,
            )
        print("\n")

        # Save / close BM25 index
        if bm25_index is not None:
            if isinstance(bm25_index, BM25SqliteIndex):
                bm25_index.save()
                print(f"BM25 SQLite index checkpointed: {bm25_db_path} (Total indexed chunks: {bm25_index.size})")
                bm25_index.close()
            else:
                bm25_index.save(bm25_json_path)
                print(f"BM25 JSON index saved to: {bm25_json_path} (Total indexed chunks: {bm25_index.size})")

    except (DatasetLoadError, RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if "embedder" in locals() and isinstance(embedder, CachedEmbedder):
            embedder.close()

    print("=" * 80)
    print("STREAMING INGESTION COMPLETED")
    print("=" * 80)
    print(f"Records Read:             {stats.records_read}")
    print(f"Documents Processed:      {stats.documents_processed}")
    print(f"Documents Skipped:        {stats.documents_skipped}")
    print(f"Chunks Created:           {stats.chunks_created}")
    print(f"Qdrant Chunks Indexed:    {stats.qdrant_chunks_indexed}")
    print(f"BM25 Chunks Indexed:      {stats.bm25_chunks_indexed}")
    print(f"Languages Processed ({len(stats.languages_processed)}): {sorted(stats.languages_processed)}")
    print(f"Failures:                 {stats.failures}")
    print(f"Total Ingestion Time:     {stats.elapsed_seconds:.2f}s (Embedding Time: {stats.embedding_seconds:.2f}s)")
    print(f"Overall Throughput:       {stats.throughput_chunks_per_sec:.2f} chunks/sec")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
