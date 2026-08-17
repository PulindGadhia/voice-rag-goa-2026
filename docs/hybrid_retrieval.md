# BM25 and hybrid retrieval

BM25 is implemented locally in `bm25.py` rather than through an additional
library. This keeps the implementation dependency-light, makes the tokenizer
replaceable, and allows development indexes to be saved as safe JSON.

`UnicodeWordTokenizer` normalizes Unicode with NFC, case-folds by default, and
groups Unicode letters, numbers, and combining marks. Punctuation is a token
boundary; non-Latin characters are not transliterated or removed.

Hybrid retrieval executes vector and BM25 retrieval independently. It never
adds their raw scores. `RRFFuser` merges rankings by chunk ID using
`1 / (rrf_k + rank)` and retains both original scores for later reranking.

Development commands:

```bash
HF_SAMPLE_SIZE=2 HF_LOADER_BACKEND=dataset_server \
  .venv/bin/python scripts/inspect_hybrid_retrieval.py --sample-size 2 --top-k 5

HF_SAMPLE_SIZE=2 HF_LOADER_BACKEND=dataset_server \
  .venv/bin/python scripts/benchmark_hybrid_retrieval.py --sample-size 2 --runs 5
```

The optional quality evaluation treats chunks whose inherited
`source.is_selected` equals `1` as relevant. This is taken directly from the
MSMARCO-XI passage metadata; no labels are invented. Results from a tiny
development sample must not be interpreted as dataset-level quality.
