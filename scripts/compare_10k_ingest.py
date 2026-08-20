#!/usr/bin/env python3
"""10,000 Record Comparison: Baseline vs Optimized Fast Ingestion."""

import os
import shutil
import sys
import time
from pathlib import Path

# Add project root and src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from voice_rag_ingestion.bm25_sqlite import BM25SqliteIndex
from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.indexing import stream_index_from_parquet, _process_parquet_file_worker


PARQUET_DIR = Path(".cache/msmarco_xi/validation")
TEST_FILE = PARQUET_DIR / "hinval.parquet"
SAMPLE_SIZE = 10_000

OLD_DB_PATH = Path(".cache/_comp_old_bm25.db")
FAST_DB_PATH = Path(".cache/_comp_fast_bm25.db")
OLD_CP_PATH = Path(".cache/_comp_old_cp.json")
FAST_CP_PATH = Path(".cache/_comp_fast_cp.json")


def clean_db(db_path: Path, cp_path: Path) -> None:
    for p in [db_path, cp_path]:
        if p.exists():
            p.unlink()
    for suffix in ["-wal", "-shm"]:
        w = db_path.with_name(db_path.name + suffix)
        if w.exists():
            w.unlink()


def run_baseline_ingest() -> tuple[float, int]:
    clean_db(OLD_DB_PATH, OLD_CP_PATH)
    idx = BM25SqliteIndex(OLD_DB_PATH, fast_ingest=False)
    t0 = time.perf_counter()
    stats = stream_index_from_parquet(
        parquet_dir=PARQUET_DIR,
        bm25_index=idx,
        bm25_path=OLD_DB_PATH,
        chunking_config=ChunkingConfig(strategy="metadata"),
        stream_batch_size=250,
        parquet_batch_size=1000,
        checkpoint_path=OLD_CP_PATH,
        resume=False,
        bm25_only=True,
        sample_size=SAMPLE_SIZE,
        fast_ingest=False,
        workers=1,
    )
    elapsed = time.perf_counter() - t0
    idx.close()
    return elapsed, stats.bm25_chunks_indexed


def run_fast_single_file_ingest() -> tuple[float, int]:
    clean_db(FAST_DB_PATH, FAST_CP_PATH)
    t0 = time.perf_counter()
    res = _process_parquet_file_worker({
        "parquet_path": str(PARQUET_DIR / "asmval.parquet"),  # match first file read by baseline
        "temp_db_path": str(FAST_DB_PATH),
        "dataset_name": "ai4bharat/MSMARCO-XI",
        "dataset_config": "default",
        "split": "validation",
        "strategy": "metadata",
        "max_chunk_size": 256,
        "overlap": 32,
        "min_chunk_size": 10,
        "batch_size": 5000,
        "sample_size": SAMPLE_SIZE,
    })
    elapsed = time.perf_counter() - t0
    return elapsed, res["chunks_created"]


def main() -> int:
    if not PARQUET_DIR.is_dir():
        print(f"ERROR: Parquet directory not found at {PARQUET_DIR}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("RUNNING 10,000 RECORD INGESTION COMPARISON (IDENTICAL CORPUS)")
    print("=" * 80)

    # 1. Run Baseline Ingestion
    print(f"\n[1/3] Running Baseline Sequential Ingestion ({SAMPLE_SIZE:,} records)...")
    old_time, old_chunks = run_baseline_ingest()
    old_rate = old_chunks / old_time if old_time > 0 else 0
    print(f"  Baseline Done: {old_chunks:,} chunks in {old_time:.2f}s ({old_rate:.1f} chunks/sec)")

    # 2. Run Optimized Fast Ingestion on the exact same 10,000 records
    print(f"\n[2/3] Running Optimized Fast Ingestion ({SAMPLE_SIZE:,} records, 5000-batch + fast PRAGMAs)...")
    fast_time, fast_chunks = run_fast_single_file_ingest()
    fast_rate = fast_chunks / fast_time if fast_time > 0 else 0
    print(f"  Optimized Done: {fast_chunks:,} chunks in {fast_time:.2f}s ({fast_rate:.1f} chunks/sec)")

    speedup = old_time / fast_time if fast_time > 0 else 1.0
    throughput_gain = ((fast_rate - old_rate) / old_rate * 100.0) if old_rate > 0 else 0.0
    print(f"\n  Speedup: {speedup:.2f}x faster (+{throughput_gain:.1f}% throughput)")

    # 3. Compare Retrieval Top-K Results and BM25 Scores
    print("\n[3/3] Comparing BM25 Top-K Retrieval Results & Scores on identical corpus...")
    old_idx = BM25SqliteIndex.load(OLD_DB_PATH)
    fast_idx = BM25SqliteIndex.load(FAST_DB_PATH)

    print(f"  Baseline Index Chunks: {old_idx.size:,}")
    print(f"  Optimized Index Chunks: {fast_idx.size:,}")
    assert old_idx.size == fast_idx.size, f"Chunk count mismatch: {old_idx.size} vs {fast_idx.size}"

    test_queries = [
        "what is a corporation",
        "কোম্পানী কি",
        "ব্যৱসায় কি",
        "what is an LLC and how does it work",
        "definition of corporation",
    ]

    all_matched = True
    for query in test_queries:
        old_results = old_idx.search(query, top_k=5)
        fast_results = fast_idx.search(query, top_k=5)

        print(f"\n  Query: '{query}'")
        print(f"    Baseline matches:  {len(old_results)}")
        print(f"    Optimized matches: {len(fast_results)}")

        if len(old_results) != len(fast_results):
            print(f"    [FAIL] Match count mismatch: {len(old_results)} vs {len(fast_results)}")
            all_matched = False
            continue

        for i, (r_old, r_fast) in enumerate(zip(old_results, fast_results)):
            score_diff = abs(r_old.score - r_fast.score)
            id_match = (r_old.chunk_id == r_fast.chunk_id)
            print(f"    Rank {i+1}: old_score={r_old.score:.4f}, fast_score={r_fast.score:.4f}, id_match={id_match} (diff={score_diff:.6f})")
            if score_diff > 1e-4 or not id_match:
                print(f"    [FAIL] Result mismatch at rank {i+1}: {r_old.chunk_id} vs {r_fast.chunk_id}")
                all_matched = False

    old_idx.close()
    fast_idx.close()

    # Clean up test databases
    clean_db(OLD_DB_PATH, OLD_CP_PATH)
    clean_db(FAST_DB_PATH, FAST_CP_PATH)

    print("\n" + "=" * 80)
    if all_matched:
        print("SUCCESS: ALL BM25 RETRIEVAL SCORES AND RESULTS MATCH 100% IDENTICALLY!")
    else:
        print("FAILURE: SOME RETRIEVAL RESULTS DIFFERED!")
    print("=" * 80)

    return 0 if all_matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
