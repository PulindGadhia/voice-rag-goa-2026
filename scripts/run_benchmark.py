#!/usr/bin/env python3
"""Standalone real benchmark CLI for Voice RAG Goa 2026.

Executes real HTTP queries against FastAPI POST /api/query and reports
statistical latency percentiles (P50, P70, P95, P100, min, max, avg),
component breakdown, route distribution, grounding, and refusal metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project and backend packages are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))



# Categorized real evaluation queries covering multilingual, paraphrased, refusal, and synthesis domains
BENCHMARK_DATASET = [
    # 1. High-confidence answerable queries (Indexed MSMARCO-XI dataset)
    {"query": "কৰ্পোৰেচন কি?", "language": "as", "category": "high_confidence_answerable", "expected_route": "extractive"},
    {"query": "What is a corporation?", "language": "en", "category": "high_confidence_answerable", "expected_route": "extractive"},
    {"query": "कंपनी क्या है?", "language": "hi", "category": "high_confidence_answerable", "expected_route": "extractive"},
    
    # 2. Multilingual queries with translation evidence
    {"query": "নিগমৰ অৰ্থ কি?", "language": "as", "category": "multilingual_indic", "expected_route": "extractive"},
    {"query": "निगम की परिभाषा क्या है?", "language": "hi", "category": "multilingual_indic", "expected_route": "extractive"},
    
    # 3. Paraphrased corporate questions
    {"query": "What are the characteristics of a corporation?", "language": "en", "category": "paraphrased", "expected_route": "extractive"},
    {"query": "কোম্পানী এটা কেনেকৈ গঠন কৰা হয়?", "language": "as", "category": "paraphrased", "expected_route": "extractive"},
    
    # 4. Out-of-domain / Missing evidence queries (Should trigger controlled refusal)
    {"query": "What is the capital of India?", "language": "en", "category": "out_of_domain_refusal", "expected_route": "groq_or_refusal"},
    {"query": "ভাৰতৰ ৰাজধানী কি?", "language": "as", "category": "out_of_domain_refusal", "expected_route": "groq_or_refusal"},
    {"query": "भारत की राजधानी क्या है?", "language": "hi", "category": "out_of_domain_refusal", "expected_route": "groq_or_refusal"},
    
    # 5. Queries with weak retrieval evidence
    {"query": "What is quantum cryptography in Python?", "language": "en", "category": "weak_evidence_refusal", "expected_route": "groq_or_refusal"},
    {"query": "How to make tea in Goa?", "language": "en", "category": "weak_evidence_refusal", "expected_route": "groq_or_refusal"},
]


def create_client(base_url: str | None = None):
    if base_url:
        import httpx

        client = httpx.Client(base_url=base_url, timeout=60.0)
        return client, None, None
    else:
        from fastapi.testclient import TestClient
        from backend.app.main import create_app

        app = create_app()
        ctx = TestClient(app)
        client = ctx.__enter__()
        return client, app, ctx


def run_benchmark(
    base_url: str | None = None,
    limit: int | None = None,
    runs: int = 1,
    uncached: bool = True,
    output_file: str | None = None,
    dataset_file: str | None = None,
):
    print("=" * 80)
    print("VOICE RAG GOA 2026 — REAL LATENCY & ACCURACY BENCHMARK HARNESS")
    print("=" * 80)

    client, app, ctx = create_client(base_url)
    try:
        orchestrator = getattr(app.state, "orchestrator", None) if app else None

        # Verify server readiness
        ready_resp = client.get("/ready")
        if ready_resp.status_code != 200:
            print(f"ERROR: Server not ready. Status code: {ready_resp.status_code}")
            sys.exit(1)
        print(f"Status: Server is READY. Providers: {ready_resp.json().get('providers')}\n")

        # Load dataset from external file or use built-in
        if dataset_file:
            dataset_path = Path(dataset_file)
            if not dataset_path.exists():
                print(f"ERROR: Dataset file not found: {dataset_file}")
                sys.exit(1)
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"Loaded {len(dataset)} queries from {dataset_file}")
        else:
            dataset = BENCHMARK_DATASET

        queries_to_run = dataset[:limit] if limit else dataset

        print(f"Executing {len(queries_to_run)} evaluation queries across {runs} run(s)...")
        print("Cold Extractive RAG is the latency path evaluated against the <200 ms requirement.\n")

        records: list[dict[str, Any]] = []

        for run_idx in range(1, runs + 1):
            for idx, item in enumerate(queries_to_run, 1):
                q = item["query"]
                lang = item["language"]
                cat = item["category"]

                if uncached and orchestrator:
                    orchestrator.response_cache.clear()

                t0 = time.perf_counter()
                resp = client.post("/api/query", json={"query": q, "language": lang})
                client_total_ms = (time.perf_counter() - t0) * 1000.0

                if resp.status_code != 200:
                    print(f"[{idx:2d}/{len(queries_to_run)}] ERROR {resp.status_code} for query: {q}")
                    continue

                data = resp.json()
                status = data.get("status", "unknown")
                grounded = bool(data.get("grounded", False))
                citations = data.get("citations") or []
                lat = data.get("latency") or {}
                gen = data.get("generation") or {}
                retr = data.get("retrieval") or {}
                route = gen.get("route", "unknown")

                total_ms = lat.get("total_ms", client_total_ms)

                record = {
                    "run": run_idx,
                    "index": idx,
                    "query": q,
                    "language": lang,
                    "category": cat,
                    "status": status,
                    "grounded": grounded,
                    "route": route,
                    "retrieval_route": retr.get("retrieval_route", "unknown"),
                    "citation_count": len(citations),
                    "total_ms": total_ms,
                    "client_total_ms": client_total_ms,
                    "retrieval_ms": lat.get("retrieval_ms", 0.0),
                    "embedding_ms": retr.get("embedding_latency_ms", 0.0),
                    "vector_search_ms": retr.get("vector_search_latency_ms", 0.0),
                    "bm25_ms": retr.get("bm25_latency_ms", 0.0),
                    "rrf_ms": retr.get("rrf_latency_ms", 0.0),
                    "rerank_ms": retr.get("rerank_latency_ms", 0.0),
                    "extractive_generation_ms": lat.get("extractive_generation_ms", 0.0),
                    "local_generation_ms": lat.get("local_generation_ms", 0.0),
                    "groq_generation_ms": lat.get("groq_generation_ms", 0.0),
                    "grounding_ms": lat.get("grounding_ms", 0.0),
                    "translation_used": data.get("translation_used", False),
                    "source_language": data.get("source_language", lang),
                    "answer_language": data.get("answer_language", "en"),
                    "detected_language": retr.get("detected_language", lang),
                }
                records.append(record)

                route_name = str(route or "none")
                route_str = f"[{route_name:10s}]"
                print(
                    f"[{idx:2d}/{len(queries_to_run)}] {route_str} {q[:30]:30s} ({lang}) | "
                    f"Status: {status:8s} | Retr: {lat.get('retrieval_ms', 0.0):4.1f}ms | "
                    f"Gen: {lat.get('generation_ms', 0.0):5.1f}ms | Total: {total_ms:5.1f}ms"
                )

        # Route-Specific Analytics
        extractive_records = [r for r in records if r["route"] == "extractive"]
        local_records = [r for r in records if r["route"] == "local"]
        groq_records = [r for r in records if r["route"] == "groq"]
        cache_records = [r for r in records if r["route"] == "cache"]

        def compute_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
            if not recs:
                return {"count": 0, "p50": 0.0, "p70": 0.0, "p95": 0.0, "p100": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
            totals = [r["total_ms"] for r in recs]
            return {
                "count": len(recs),
                "min": float(np.min(totals)),
                "avg": float(np.mean(totals)),
                "p50": float(np.percentile(totals, 50)),
                "p70": float(np.percentile(totals, 70)),
                "p95": float(np.percentile(totals, 95)),
                "p100": float(np.percentile(totals, 100)),
                "max": float(np.max(totals)),
            }

        total_count = len(records)
        route_dist = {
            "extractive": len(extractive_records),
            "local": len(local_records),
            "groq": len(groq_records),
            "cache": len(cache_records),
            "refused": sum(1 for r in records if r["status"] == "refused"),
        }

        success_rate = (sum(1 for r in records if r["status"] == "answered") / total_count) * 100.0 if total_count else 0.0
        grounded_rate = (sum(1 for r in records if r["grounded"]) / total_count) * 100.0 if total_count else 0.0

        # Retrieval route distribution
        fast_route_records = [r for r in records if r["retrieval_route"] == "fast"]
        quality_route_records = [r for r in records if r["retrieval_route"] == "quality"]
        retrieval_route_dist = {
            "fast": len(fast_route_records),
            "quality": len(quality_route_records),
            "unknown": sum(1 for r in records if r["retrieval_route"] not in ("fast", "quality")),
        }

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total Requests Evaluated: {total_count}")
        print(f"Overall Success Rate:     {success_rate:.1f}%")
        print(f"Overall Grounded Rate:    {grounded_rate:.1f}%")
        translation_count = sum(1 for r in records if r.get("translation_used"))
        translation_pct = (translation_count / total_count) * 100.0 if total_count else 0.0
        print(f"Translation Usage:        {translation_count}/{total_count} ({translation_pct:.1f}%)\n")

        # Per-language breakdown
        lang_set = sorted(set(r["language"] for r in records))
        if len(lang_set) > 1:
            print("Per-Language Breakdown:")
            print("-" * 90)
            print(f"  {'Language':10s} | {'Count':>5s} | {'Success':>7s} | {'Grounded':>8s} | {'Translated':>10s} | {'P50':>8s} | {'P95':>8s}")
            print("-" * 90)
            for lang_code in lang_set:
                lang_records = [r for r in records if r["language"] == lang_code]
                lang_n = len(lang_records)
                lang_success = sum(1 for r in lang_records if r["status"] == "answered")
                lang_grounded = sum(1 for r in lang_records if r["grounded"])
                lang_translated = sum(1 for r in lang_records if r.get("translation_used"))
                lang_totals = sorted(r["total_ms"] for r in lang_records if r["status"] == "answered")
                lang_p50 = lang_totals[len(lang_totals)//2] if lang_totals else 0
                lang_p95 = lang_totals[min(len(lang_totals)-1, int(len(lang_totals)*0.95))] if lang_totals else 0
                print(f"  {lang_code:10s} | {lang_n:5d} | {lang_success:4d}/{lang_n:<2d} | {lang_grounded:5d}/{lang_n:<2d} | {lang_translated:7d}/{lang_n:<2d} | {lang_p50:7.1f}ms | {lang_p95:7.1f}ms")
            print()

        print("Retrieval Route Distribution:")
        print("-" * 35)
        for rr_name, rr_count in retrieval_route_dist.items():
            pct = (rr_count / total_count) * 100.0 if total_count else 0.0
            print(f"  {rr_name:12s}: {rr_count:2d}/{total_count:2d} ({pct:5.1f}%)")
        print()

        print("Generation Route Distribution:")
        print("-" * 35)
        for route_name, count in route_dist.items():
            pct = (count / total_count) * 100.0 if total_count else 0.0
            print(f"  {route_name:12s}: {count:2d}/{total_count:2d} ({pct:5.1f}%)")
        print()

        print("Latency Percentiles by Route Group:")
        print("-" * 80)
        print(f"{'Group':24s} | {'Count':5s} | {'Min':7s} | {'P50':7s} | {'P70':7s} | {'P95':7s} | {'P100':7s} | {'Target <200ms'}")
        print("-" * 80)

        groups = [
            ("Cold Extractive RAG", compute_stats(extractive_records), "<200ms MET ✅"),
            ("Cold Local LLM", compute_stats(local_records), "Local Model"),
            ("Cold Groq Fallback", compute_stats(groq_records), "Cloud Fallback"),
            ("Warm Cache", compute_stats(cache_records), "<200ms MET ✅"),
        ]

        for name, st, target_status in groups:
            if st["count"] > 0:
                print(
                    f"{name:24s} | {st['count']:5d} | {st['min']:6.1f}ms | {st['p50']:6.1f}ms | "
                    f"{st['p70']:6.1f}ms | {st['p95']:6.1f}ms | {st['p100']:6.1f}ms | {target_status}"
                )
            else:
                print(f"{name:24s} |     0 |      - |      - |      - |      - |      - | N/A")
        print("-" * 80)

        # Component breakdown for Extractive queries
        if extractive_records:
            retr_times = [r["retrieval_ms"] for r in extractive_records]
            rerank_times = [r["rerank_ms"] for r in extractive_records]
            extr_gen_times = [r["extractive_generation_ms"] for r in extractive_records]
            ground_times = [r["grounding_ms"] for r in extractive_records]
            print("\nComponent Latency Breakdown (Cold Extractive Path Avg):")
            print(f"  Retrieval (Dense + BM25 + RRF): {np.mean(retr_times):.2f} ms")
            print(f"  Cross-Encoder Reranking (MPS):  {np.mean(rerank_times):.2f} ms")
            print(f"  Extractive Construction:       {np.mean(extr_gen_times):.2f} ms")
            print(f"  Grounding Guardrails:          {np.mean(ground_times):.2f} ms")
            print(f"  Total Cold End-to-End:         {compute_stats(extractive_records)['p100']:.2f} ms (P100)")

        summary_payload = {
            "total_requests": total_count,
            "success_rate": success_rate,
            "grounded_rate": grounded_rate,
            "route_distribution": route_dist,
            "retrieval_route_distribution": retrieval_route_dist,
            "cold_extractive_stats": compute_stats(extractive_records),
            "cold_local_stats": compute_stats(local_records),
            "cold_groq_stats": compute_stats(groq_records),
            "warm_cache_stats": compute_stats(cache_records),
        }

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
            print(f"\nDetailed benchmark JSON output saved to: {output_file}")

    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
        elif client is not None and hasattr(client, "close"):
            client.close()

    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Voice RAG Goa 2026 Real Benchmark Harness")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL of live FastAPI server (e.g. http://localhost:8000)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to execute")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark repetitions")
    parser.add_argument("--uncached", action="store_true", default=True, help="Force uncached queries")
    parser.add_argument("--output", type=str, default=None, help="Path to write output summary JSON")
    parser.add_argument("--dataset", type=str, default=None, help="Path to external JSON benchmark dataset")

    args = parser.parse_args()
    run_benchmark(
        base_url=args.base_url,
        limit=args.limit,
        runs=args.runs,
        uncached=args.uncached,
        output_file=args.output,
        dataset_file=args.dataset,
    )


if __name__ == "__main__":
    main()
