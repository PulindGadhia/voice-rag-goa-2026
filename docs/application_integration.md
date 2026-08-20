# Application-layer integration

Om's backend application layer is integrated under `backend/app`. The
orchestrator, HTTP routes, generation provider, Sarvam STT adapter, guardrails,
schemas, metrics, and lifecycle code live there.

Retrieval remains owned by `src/voice_rag_ingestion`. `backend/app/retrieval`
defines the application contract:

```text
RetrievalEngine.retrieve(query) -> RetrievalResponse
```

`ExistingRetrievalEngine` adapts the existing vector + BM25 + RRF + reranking
pipeline. Om's duplicate `Retriever`, Qdrant wrapper, dataset loader, and
chunking implementations were intentionally not copied.

## Run

Install the application extras with `pip install -e '.[app]'`. Provider SDKs
are optional and can be installed with `pip install -e '.[providers]'`.

From the repository root:

```bash
PYTHONPATH=src:backend uvicorn app.main:app --reload
```

Development defaults use a bounded in-memory index. Set `APP_AUTO_INDEX=false`
when a configured Qdrant collection and BM25 index already exist. API keys are
read only from environment variables; no key is required to import the
retrieval or application contracts.

The generation prompt requests structured JSON (`answer`, `grounded`,
`source_ids`, `confidence`). Responses are validated before API serialization;
malformed output is returned as a structured generation error. Voice requests
are transcribed and then enter the same text-query pipeline.

Run the complete test suite with:

```bash
.venv/bin/python -m pytest -q
```

The current retrieval benchmark scripts remain under `scripts/`; the
application does not replace or duplicate them.
