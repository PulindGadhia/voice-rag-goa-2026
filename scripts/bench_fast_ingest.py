#!/usr/bin/env python3
"""Benchmark: Normal vs Fast-Ingest ingestion on 100,000 records.

Usage:
    .venv/bin/python scripts/bench_fast_ingest.py
"""

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voice_rag_ingestion.bm25_sqlite import BM25SqliteIndex
from voice_rag_ingestion.indexing import StreamingIndexStats, stream_index_from_parquet
from voice_rag_ingestion.chunking import ChunkingConfig


PARQUET_DIR = Path(".cache/msmarco_xi/validation")
SAMPLE_SIZE = 10_000


def run_benchmark(label: str, fast_ingest: bool) -> dict:
    """Run a single benchmark pass."""
    # Use temp paths to avoid touching the real index
    db_path = Path(f".cache/_bench_{label}_bm25.db")
    cp_path = Path(f".cache/_bench_{label}_checkpoint.json")

    # Clean up any previous run
    for p in [db_path, cp_path]:
        if p.exists():
            p.unlink()
    for suffix in ["-wal", "-shm"]:
        wal = db_path.with_name(db_path.name + suffix)
        if wal.exists():
            wal.unlink()

    idx = BM25SqliteIndex(db_path, fast_ingest=fast_ingest)

    stream_batch_size = 5000 if fast_ingest else 250
    parquet_batch_size = 10000 if fast_ingest else 1000
    checkpoint_interval = 10 if fast_ingest else 1

    t0 = time.time()
    stats = stream_index_from_parquet(
        parquet_dir=PARQUET_DIR,
        bm25_index=idx,
        bm25_path=db_path,
        chunking_config=ChunkingConfig(strategy="metadata"),
        stream_batch_size=stream_batch_size,
        parquet_batch_size=parquet_batch_size,
        checkpoint_path=cp_path,
        resume=False,
        bm25_only=True,
        sample_size=SAMPLE_SIZE,
        fast_ingest=fast_ingest,
        checkpoint_interval=checkpoint_interval,
    )
    elapsed = time.time() - t0
    idx.close()

    db_size_mb = db_path.stat().st_size / (1024 * 1024)

    result = {
        "label": label,
        "fast_ingest": fast_ingest,
        "elapsed_sec": elapsed,
        "records_read": stats.records_read,
        "bm25_chunks_indexed": stats.bm25_chunks_indexed,
        "records_per_sec": stats.records_read / elapsed if elapsed > 0 else 0,
        "chunks_per_sec": stats.bm25_chunks_indexed / elapsed if elapsed > 0 else 0,
        "db_size_mb": db_size_mb,
    }

    # Clean up
    for p in [db_path, cp_path]:
        if p.exists():
            p.unlink()
    for suffix in ["-wal", "-shm"]:
        wal = db_path.with_name(db_path.name + suffix)
        if wal.exists():
            wal.unlink()

    return result


def main() -> int:
    if not PARQUET_DIR.is_dir():
        print(f"ERROR: Parquet directory not found: {PARQUET_DIR}")
        return 1

    print(f"\nBenchmarking {SAMPLE_SIZE:,} records from {PARQUET_DIR}")
    print("=" * 80)

    # Normal mode
    print(f"\n[1/2] Normal mode (stream_batch=250, parquet_batch=1000, checkpoint every batch)...")
    normal = run_benchmark("normal", fast_ingest=False)
    print(f"  Done: {normal['elapsed_sec']:.1f}s, {normal['records_per_sec']:.0f} rec/s, "
          f"{normal['bm25_chunks_indexed']:,} chunks, DB={normal['db_size_mb']:.1f} MB")

    # Fast-ingest mode
    print(f"\n[2/2] Fast-ingest mode (stream_batch=5000, parquet_batch=10000, checkpoint every 10 batches)...")
    fast = run_benchmark("fast", fast_ingest=True)
    print(f"  Done: {fast['elapsed_sec']:.1f}s, {fast['records_per_sec']:.0f} rec/s, "
          f"{fast['bm25_chunks_indexed']:,} chunks, DB={fast['db_size_mb']:.1f} MB")

    # Summary
    speedup = normal["elapsed_sec"] / fast["elapsed_sec"] if fast["elapsed_sec"] > 0 else float("inf")
    print("\n" + "=" * 80)
    print(f"{'Metric':<35} {'Normal':>15} {'Fast-Ingest':>15} {'Speedup':>10}")
    print("-" * 80)
    print(f"{'Total time (s)':<35} {normal['elapsed_sec']:>15.1f} {fast['elapsed_sec']:>15.1f} {speedup:>9.1f}x")
    print(f"{'Records/sec':<35} {normal['records_per_sec']:>15.0f} {fast['records_per_sec']:>15.0f}")
    print(f"{'BM25 chunks/sec':<35} {normal['chunks_per_sec']:>15.0f} {fast['chunks_per_sec']:>15.0f}")
    print(f"{'BM25 chunks indexed':<35} {normal['bm25_chunks_indexed']:>15,} {fast['bm25_chunks_indexed']:>15,}")
    print(f"{'Records read':<35} {normal['records_read']:>15,} {fast['records_read']:>15,}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
