#!/usr/bin/env python3
"""Diagnostic script to analyze all benchmark refusals.

Executes queries from scripts/multilingual_queries_200.json against the orchestrator,
captures every refusal, and records detailed diagnostic fields.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project and backend packages are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from fastapi.testclient import TestClient
from backend.app.main import create_app


def main():
    app = create_app()
    with TestClient(app) as client:
        # Verify server readiness
        ready_resp = client.get("/ready")
        assert ready_resp.status_code == 200, "Server not ready"

        dataset_path = _REPO_ROOT / "scripts" / "multilingual_queries_200.json"
        with open(dataset_path, "r", encoding="utf-8") as f:
            queries = json.load(f)

        orchestrator = getattr(app.state, "orchestrator", None)

        refusals: list[dict[str, Any]] = []
        all_results: list[dict[str, Any]] = []

        for idx, item in enumerate(queries, 1):
            q = item["query"]
            lang = item["language"]
            cat = item.get("category", "")

            if orchestrator:
                orchestrator.response_cache.clear()

            t0 = time.perf_counter()
            resp = client.post("/api/query", json={"query": q, "language": lang})
            client_total_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code != 200:
                print(f"Error {resp.status_code} for query: {q}")
                continue

            data = resp.json()
            status = data.get("status", "unknown")
            grounded = bool(data.get("grounded", False))
            citations = data.get("citations") or []
            lat = data.get("latency") or {}
            gen = data.get("generation") or {}
            retr = data.get("retrieval") or {}
            guardrail_status = data.get("guardrail_status", "unknown")
            guardrail_reason = data.get("guardrail_reason")

            record = {
                "index": idx,
                "query": q,
                "language": lang,
                "category": cat,
                "status": status,
                "grounded": grounded,
                "detected_language": retr.get("detected_language"),
                "retrieval_route": retr.get("retrieval_route"),
                "generation_route": gen.get("route"),
                "retrieval_num_results": retr.get("num_results", 0),
                "citation_count": len(citations),
                "top_score": citations[0].get("score") if citations else None,
                "top_vector_score": citations[0].get("metadata", {}).get("vector_score") if citations else None,
                "top_bm25_score": citations[0].get("metadata", {}).get("bm25_score") if citations else None,
                "top_rrf_score": citations[0].get("metadata", {}).get("rrf_score") if citations else None,
                "retrieval_ms": lat.get("retrieval_ms", 0.0),
                "total_ms": lat.get("total_ms", client_total_ms),
                "translation_used": data.get("translation_used", False),
                "source_language": data.get("source_language"),
                "answer_language": data.get("answer_language"),
                "guardrail_status": guardrail_status,
                "guardrail_reason": guardrail_reason,
                "confidence_decision": gen.get("confidence_decision"),
            }

            all_results.append(record)

            if status == "refused" or not grounded:
                # Classify refusal origin
                refusal_origin = "UNKNOWN"
                if guardrail_status == "blocked":
                    refusal_origin = "A. input guardrail"
                elif retr.get("num_results", 0) == 0:
                    refusal_origin = "B. retrieval.found == false (no passages)"
                elif not grounded and status == "answered":
                    refusal_origin = "C. output grounding"
                elif guardrail_status == "low_confidence":
                    refusal_origin = "D. low confidence / model refused"
                else:
                    refusal_origin = "E. other guardrail"

                record["refusal_origin"] = refusal_origin
                refusals.append(record)

        # Save diagnostic output
        diag_output_path = _REPO_ROOT / "scripts" / "diagnostic_refusals.json"
        with open(diag_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_queries": len(queries),
                "total_refusals": len(refusals),
                "refusal_rate": f"{len(refusals)/len(queries)*100:.1f}%",
                "refusals": refusals,
            }, f, indent=2, ensure_ascii=False)

        print(f"\nDiagnostic completed:")
        print(f"Total queries: {len(queries)}")
        print(f"Total refusals/ungrounded: {len(refusals)}")
        print(f"Saved diagnostic report to {diag_output_path}")

        # Summary by language
        print("\nRefusals by Language:")
        lang_counts: dict[str, int] = {}
        for r in refusals:
            lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            print(f"  {lang}: {count}")

        # Summary by refusal origin
        print("\nRefusals by Origin:")
        origin_counts: dict[str, int] = {}
        for r in refusals:
            origin_counts[r["refusal_origin"]] = origin_counts.get(r["refusal_origin"], 0) + 1
        for origin, count in sorted(origin_counts.items(), key=lambda x: -x[1]):
            print(f"  {origin}: {count}")


if __name__ == "__main__":
    main()
