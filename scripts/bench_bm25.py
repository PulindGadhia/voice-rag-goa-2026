#!/usr/bin/env python3
"""Benchmark: JSON BM25Index.save() vs SQLite BM25SqliteIndex.add().

Compares ingestion speed, persistence cost, query latency, memory, and index size.
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.bm25 import BM25Index
from voice_rag_ingestion.bm25_sqlite import BM25SqliteIndex
from voice_rag_ingestion.chunking.base import Chunk


def make_chunks(n: int, offset: int = 0) -> list[Chunk]:
    """Generate n synthetic Chunk objects."""
    chunks = []
    for i in range(offset, offset + n):
        chunks.append(
            Chunk(
                chunk_id=f"bench_{i:08d}",
                document_id=f"doc_{i // 10:06d}",
                text=f"This is benchmark passage number {i}. It discusses topic alpha "
                     f"beta gamma delta epsilon about item {i} in the multilingual "
                     f"corpus of documents related to general knowledge and science.",
                language="eng_Latn",
                chunk_index=i % 10,
                chunk_strategy="metadata",
                source={"text_source": "translated", "passage_index": i % 10},
                metadata={"query": f"question {i}", "english_query": f"question {i}"},
            )
        )
    return chunks


def bench_json(n: int, tmp_dir: Path, batch_size: int = 250) -> dict:
    """Benchmark the old JSON BM25Index with full save after each batch."""
    json_path = tmp_dir / "bench_bm25.json"
    if json_path.exists():
        json_path.unlink()

    tracemalloc.start()
    idx = BM25Index()
    t0 = time.perf_counter()

    total_persist_time = 0.0
    for batch_start in range(0, n, batch_size):
        batch = make_chunks(min(batch_size, n - batch_start), offset=batch_start)
        idx.add(batch)
        t_save_0 = time.perf_counter()
        idx.save(json_path)
        total_persist_time += time.perf_counter() - t_save_0

    total_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Query latency
    queries = ["alpha beta gamma", "general knowledge science", "multilingual corpus"]
    t_query_0 = time.perf_counter()
    for q in queries:
        idx.search(q, top_k=10)
    query_time = (time.perf_counter() - t_query_0) / len(queries)

    index_size = json_path.stat().st_size if json_path.exists() else 0

    return {
        "backend": "JSON",
        "chunks": n,
        "total_sec": total_time,
        "persist_sec": total_persist_time,
        "records_per_sec": n / total_time if total_time > 0 else 0,
        "peak_memory_mb": peak / (1024 * 1024),
        "query_latency_ms": query_time * 1000,
        "index_size_mb": index_size / (1024 * 1024),
    }


def bench_sqlite(n: int, tmp_dir: Path, batch_size: int = 250) -> dict:
    """Benchmark the new SQLite BM25SqliteIndex."""
    db_path = tmp_dir / "bench_bm25.db"
    if db_path.exists():
        db_path.unlink()

    tracemalloc.start()
    idx = BM25SqliteIndex(db_path)
    t0 = time.perf_counter()

    total_persist_time = 0.0
    for batch_start in range(0, n, batch_size):
        batch = make_chunks(min(batch_size, n - batch_start), offset=batch_start)
        t_add_0 = time.perf_counter()
        idx.add(batch)
        idx.save()  # WAL checkpoint
        total_persist_time += time.perf_counter() - t_add_0

    total_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Query latency
    queries = ["alpha beta gamma", "general knowledge science", "multilingual corpus"]
    t_query_0 = time.perf_counter()
    for q in queries:
        idx.search(q, top_k=10)
    query_time = (time.perf_counter() - t_query_0) / len(queries)

    idx.close()
    index_size = db_path.stat().st_size if db_path.exists() else 0

    return {
        "backend": "SQLite",
        "chunks": n,
        "total_sec": total_time,
        "persist_sec": total_persist_time,
        "records_per_sec": n / total_time if total_time > 0 else 0,
        "peak_memory_mb": peak / (1024 * 1024),
        "query_latency_ms": query_time * 1000,
        "index_size_mb": index_size / (1024 * 1024),
    }


def bench_sqlite_skip_existing(n: int, tmp_dir: Path, batch_size: int = 250) -> dict:
    """Benchmark SQLite when re-adding existing chunks (skip-existing path)."""
    db_path = tmp_dir / "bench_bm25_skip.db"
    if db_path.exists():
        db_path.unlink()

    # Pre-populate
    idx = BM25SqliteIndex(db_path)
    all_chunks = make_chunks(n)
    idx.add(all_chunks)
    idx.close()

    # Now re-add them (should all be skipped)
    tracemalloc.start()
    idx2 = BM25SqliteIndex.load(db_path)
    t0 = time.perf_counter()
    total_added = 0

    for batch_start in range(0, n, batch_size):
        batch = all_chunks[batch_start : batch_start + batch_size]
        added = idx2.add(batch)
        total_added += added

    total_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    idx2.close()

    return {
        "backend": "SQLite (skip-existing)",
        "chunks": n,
        "total_sec": total_time,
        "persist_sec": total_time,  # all time is checking + skipping
        "records_per_sec": n / total_time if total_time > 0 else 0,
        "peak_memory_mb": peak / (1024 * 1024),
        "newly_added": total_added,
    }


def print_results(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("BM25 PERSISTENCE BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Backend':<25} {'Chunks':>8} {'Total(s)':>10} {'Persist(s)':>12} "
          f"{'Rec/sec':>10} {'Peak MB':>10} {'Query ms':>10} {'Index MB':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['backend']:<25} {r['chunks']:>8} {r['total_sec']:>10.2f} "
              f"{r['persist_sec']:>12.2f} {r['records_per_sec']:>10.1f} "
              f"{r['peak_memory_mb']:>10.1f} {r.get('query_latency_ms', 0):>10.2f} "
              f"{r.get('index_size_mb', 0):>10.1f}")
    print("=" * 90)

    # Speedup
    json_results = [r for r in results if r["backend"] == "JSON"]
    sqlite_results = [r for r in results if r["backend"] == "SQLite"]
    if json_results and sqlite_results:
        for j, s in zip(json_results, sqlite_results):
            if j["chunks"] == s["chunks"]:
                speedup = j["total_sec"] / s["total_sec"] if s["total_sec"] > 0 else float("inf")
                persist_speedup = j["persist_sec"] / s["persist_sec"] if s["persist_sec"] > 0 else float("inf")
                print(f"\n  Speedup at {j['chunks']} chunks:")
                print(f"    Total:    {speedup:.1f}x faster")
                print(f"    Persist:  {persist_speedup:.1f}x faster")
                print(f"    Query:    JSON={j.get('query_latency_ms', 0):.2f}ms vs "
                      f"SQLite={s.get('query_latency_ms', 0):.2f}ms")


def main() -> None:
    import tempfile

    sizes = [1_000, 10_000]
    # Check if user wants 50K (slower)
    if "--large" in sys.argv:
        sizes.append(50_000)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        all_results: list[dict] = []

        for n in sizes:
            print(f"\nBenchmarking {n:,} chunks...")

            print(f"  JSON BM25Index.save()...")
            r_json = bench_json(n, tmp_dir)
            all_results.append(r_json)

            print(f"  SQLite BM25SqliteIndex.add()...")
            r_sqlite = bench_sqlite(n, tmp_dir)
            all_results.append(r_sqlite)

            print(f"  SQLite skip-existing...")
            r_skip = bench_sqlite_skip_existing(n, tmp_dir)
            all_results.append(r_skip)

        print_results(all_results)


if __name__ == "__main__":
    main()
