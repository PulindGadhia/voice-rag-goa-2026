# Multilingual reranking

Milestone 5 adds reranking after the existing vector + BM25 + RRF pipeline:

```text
HybridRetriever -> candidate pool -> CrossEncoderReranker -> final results
```

The measured default is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`: a
15-language, approximately 0.1B-parameter multilingual MiniLM cross-encoder
with Apache-2.0 licensing. Its model card declares Hindi support and documents
direct batched pair scoring. GTE remains selectable with
`RERANKER_MODEL=Alibaba-NLP/gte-multilingual-reranker-base`; it is the larger
306M-parameter, 75-language alternative. The project defaults to
`max_length=512` for bounded latency; this is a runtime truncation setting and
can be changed with `RERANKER_MAX_LENGTH`.

The cross-encoder receives all `(query, passage)` pairs in one model call by
default and returns independent relevance scores. It does not reuse vector
similarity, BM25, or RRF scores. Those scores and the original metadata remain
on every `RerankResult` together with `rerank_score` and `rerank_rank`.

Configuration is available through `.env` variables prefixed with
`RERANKER_`. Development can use `--mock-reranker`; that deterministic lexical
overlap implementation is only a smoke-test substitute, not a quality result.
On Apple Silicon, `RERANKER_DEVICE=cpu` is a stable fallback if the local
PyTorch/MPS stack shows device-specific issues.

The benchmark uses one persistent model instance, excludes model loading from
request latency, warms up once, and reports tokenizer, preprocessing, model
inference, and postprocessing separately. Run the required candidate-size
comparison with:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/benchmark_reranking.py \
  --sample-size 10 --runs 3 \
  --models Alibaba-NLP/gte-multilingual-reranker-base,cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

The optimized path performs one model call for all candidates. ONNX/OpenVINO
was not added: neither runtime is installed in the project environment, and
the measured native MPS path already meets the target with mMARCO at the
recommended candidate size.

Examples:

```bash
.venv/bin/python scripts/inspect_reranking.py --sample-size 3 --mock-embeddings --mock-reranker
.venv/bin/python scripts/benchmark_reranking.py --sample-size 3 --runs 3 --mock-embeddings --mock-reranker
```
