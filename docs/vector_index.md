# Embeddings and Qdrant vector index

This milestone uses `intfloat/multilingual-e5-small` through the
provider-neutral `EmbeddingProvider` interface.

Why this model:

- The model card reports support for 94 languages and retrieval fine-tuning
  involving Mr. TyDi and MIRACL.
- Its 384-dimensional vectors and roughly 0.1B parameters are practical for
  local development and lower-latency indexing than a 1024-dimensional model
  such as BGE-M3.
- E5 query/passage prefixes are applied only inside the model adapter.

The model is loaded lazily. The first model-backed run downloads its weights
to the Hugging Face cache. No model weights or vectors are stored in this
repository.

## Local Qdrant

For a persistent local Qdrant service:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=msmarco_xi_dev
```

The default `QDRANT_URL=:memory:` keeps development smoke tests isolated and
does not require a running service. `QDRANT_API_KEY` is optional and is read
only from the environment.

## Development commands

```bash
HF_SAMPLE_SIZE=1 HF_LOADER_BACKEND=dataset_server \
  .venv/bin/python scripts/index_vectors.py --sample-size 1

HF_SAMPLE_SIZE=1 HF_LOADER_BACKEND=dataset_server \
  .venv/bin/python scripts/inspect_retrieval.py --sample-size 1 --top-k 5

HF_SAMPLE_SIZE=1 HF_LOADER_BACKEND=dataset_server \
  .venv/bin/python scripts/benchmark_retrieval.py --sample-size 1 --runs 5
```

Use `--mock-embeddings` for an offline Qdrant smoke test. Mock vectors are
stored under a separate cache model key and must not be used as retrieval
quality evidence.
