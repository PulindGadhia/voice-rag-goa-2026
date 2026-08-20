# Om backend integration analysis

Audit source: `/Users/pulindgadhia/Downloads/Voice-Rag-Goa-om.zip`.

| Om component | Decision | Integration result |
|---|---|---|
| FastAPI, routes, lifecycle | Adapt | `backend/app/main.py` and `backend/app/api/` |
| Request/response schemas | Adapt | Pydantic schemas under `backend/app/schemas/` |
| Orchestrator/harness | Adapt | `backend/app/orchestrator.py` with dependency injection |
| LLM providers | Adapt | Lazy Groq/Gemini clients under `backend/app/generation/` |
| Structured output/context | Adapt | `backend/app/generation/output.py` and `backend/app/context.py` |
| Grounding/guardrails | Adapt | `backend/app/guardrails/` |
| Metrics/timing | Adapt | `backend/app/observability/` plus retrieval phase timings |
| Sarvam STT | Adapt | Lazy `backend/app/providers/stt/sarvam.py` |
| Dataset loader/chunking | Ignore | Existing ingestion and chunking remain the source of truth |
| Embeddings/Qdrant | Ignore | Existing embedding/cache and Qdrant store remain the source of truth |
| Vector-only retriever/BM25/hybrid/RRF/reranker | Ignore | Existing hybrid/RRF/mMARCO path remains the source of truth |

## Boundary

The application calls only:

```text
RetrievalEngine.retrieve(query) -> RetrievalResponse
```

`ExistingRetrievalEngine` adapts the current `HybridRerankRetriever`, retaining
chunk/document IDs, text, metadata, vector score, BM25 score, RRF score,
reranker score, and phase timings. The adapter runs synchronous model/index
work in a worker thread; the existing SQLite embedding cache was made
thread-safe for this lifecycle.

## Verification

- Existing and new tests: 41 passed.
- Static compilation and `git diff --check`: passed.
- FastAPI OpenAPI paths: `/health`, `/ready`, `/api/query`,
  `/api/voice/query`, `/api/metrics`.
- Fake-provider route smoke test passed for text and voice routes.
- Real bounded retrieval adapter smoke test passed with one development
  record, in-memory Qdrant, mMARCO reranker on MPS, and 3 returned results.
- No full MSMARCO-XI download or indexing was performed.
- A real LLM/STT end-to-end request was not run because no provider API keys
  were supplied; provider failures return structured errors.

