#!/usr/bin/env python3
"""Profile and benchmark hybrid retrieval followed by multilingual reranking."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.bm25 import BM25Config, BM25Index, BM25Retriever
from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from voice_rag_ingestion.evaluation import EvaluationCase, evaluate_rankings
from voice_rag_ingestion.hybrid import HybridConfig, HybridRetriever
from voice_rag_ingestion.indexing import index_documents
from voice_rag_ingestion.loader import DatasetLoader
from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig
from voice_rag_ingestion.reranking import (
    CrossEncoderReranker,
    HybridRerankRetriever,
    LexicalOverlapReranker,
    RerankerConfig,
)
from voice_rag_ingestion.retrieval import VectorRetriever


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def report(label: str, values: list[float]) -> None:
    print(
        f"{label}: n={len(values)} average={mean(values):.6f}s "
        f"P50={percentile(values, 50):.6f}s P70={percentile(values, 70):.6f}s "
        f"P95={percentile(values, 95):.6f}s P100={percentile(values, 100):.6f}s"
    )


def parse_sizes(value: str) -> list[tuple[int, int]]:
    sizes = []
    for item in value.split(","):
        candidate, final = item.split(":", 1)
        candidate_k, final_k = int(candidate), int(final)
        if candidate_k < final_k or candidate_k <= 0:
            raise ValueError(f"invalid candidate:final pair {item}")
        sizes.append((candidate_k, final_k))
    return sizes


def build_cases(chunks, max_cases: int) -> list[EvaluationCase]:
    grouped: dict[str, set[str]] = {}
    for chunk in chunks:
        if chunk.source.get("is_selected") not in (1, True, "1"):
            continue
        query = chunk.metadata.get("query")
        if query:
            grouped.setdefault(str(query), set()).add(chunk.chunk_id)
    return [
        EvaluationCase(query=query, relevant_chunk_ids=frozenset(ids))
        for query, ids in list(grouped.items())[:max_cases]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--models",
        default=os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
    )
    parser.add_argument("--sizes", default="3:2,5:3,8:3,10:5")
    parser.add_argument("--max-eval-cases", type=int, default=20)
    parser.add_argument("--mock-embeddings", action="store_true")
    parser.add_argument("--mock-reranker", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--compare-unbatched", action="store_true")
    args = parser.parse_args()
    try:
        sizes = parse_sizes(args.sizes)
    except ValueError as exc:
        parser.error(str(exc))
    max_candidate_k = max(candidate_k for candidate_k, _ in sizes)

    loader_config = LoaderConfig.from_env(sample_size=args.sample_size)
    embedding_config = EmbeddingConfig.from_env()
    if args.mock_embeddings:
        embedding_config = replace(embedding_config, model_name="dev-hash-embedding", cache_enabled=False)
        from voice_rag_ingestion.embeddings.dev import HashEmbeddingProvider

        provider = HashEmbeddingProvider()
    else:
        provider = SentenceTransformerEmbedder(replace(embedding_config, cache_enabled=False))
    hybrid_config = replace(
        HybridConfig.from_env(),
        vector_top_k=max_candidate_k,
        bm25_top_k=max_candidate_k,
        final_top_k=max_candidate_k,
    )
    documents, load_stats = DatasetLoader(loader_config).load_documents()
    store = QdrantVectorStore(VectorStoreConfig.from_env())
    index_stats, chunks = index_documents(
        documents,
        embedder=provider,
        vector_store=store,
        chunking_config=replace(ChunkingConfig.from_env(), strategy=hybrid_config.chunking_strategy),
        embedding_batch_size=embedding_config.batch_size,
        recreate_collection=True,
    )
    bm25_index = BM25Index(
        BM25Config(k1=hybrid_config.bm25_k1, b=hybrid_config.bm25_b, tokenizer=hybrid_config.tokenizer)
    )
    bm25_index.rebuild(chunks)
    vector_retriever = VectorRetriever(provider, store)
    bm25_retriever = BM25Retriever(bm25_index)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever, config=hybrid_config)
    queries = ["What is a corporation?", "কৰ্পোৰেচন কি?", "कंपनी क्या है?"]
    cases = build_cases(chunks, args.max_eval_cases)

    print(
        f"Dataset records={load_stats.records_read}; documents={index_stats.documents}; "
        f"chunks={index_stats.chunks}; runs={args.runs}; queries={len(queries)}; "
        f"evaluation_cases={len(cases)}"
    )
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    for model_name in model_names:
        reranker_config = replace(
            RerankerConfig.from_env(),
            model_name=model_name,
            candidate_top_k=max_candidate_k,
            final_top_k=max(final_k for _, final_k in sizes),
            warmup_enabled=not args.no_warmup,
        )
        if args.mock_reranker:
            reranker = LexicalOverlapReranker()
            print("\nModel=lexical development stub")
        else:
            reranker = CrossEncoderReranker(reranker_config)
            print(
                f"\nModel={model_name}; device={reranker.device}; "
                f"model_load={reranker.model_load_seconds:.6f}s; "
                f"batch_all_candidates={reranker_config.batch_all_candidates}; "
                f"max_length={reranker_config.max_length}"
            )
            if reranker_config.warmup_enabled:
                warmup = reranker.warmup()
                print(
                    f"warmup=enabled (excluded from request metrics); warmup_model_calls={warmup.model_calls}"
                )
            else:
                print("warmup=disabled")
        pipeline = HybridRerankRetriever(hybrid_retriever, reranker, config=reranker_config)

        quality_rankings = []
        for candidate_k, final_k in sizes:
            rerank_times: list[float] = []
            total_times: list[float] = []
            tokenizer_times: list[float] = []
            preprocessing_times: list[float] = []
            inference_times: list[float] = []
            postprocessing_times: list[float] = []
            model_calls: list[int] = []
            for _ in range(args.runs):
                for query in queries:
                    results, pipeline_timing = pipeline.retrieve_with_timing(
                        query, candidate_top_k=candidate_k, top_k=final_k
                    )
                    rerank_timing = getattr(reranker, "last_timing", None)
                    if rerank_timing is None:
                        rerank_times.append(pipeline_timing.rerank_seconds)
                    else:
                        rerank_times.append(rerank_timing.total_seconds)
                        tokenizer_times.append(rerank_timing.tokenizer_seconds)
                        preprocessing_times.append(rerank_timing.preprocessing_seconds)
                        inference_times.append(rerank_timing.inference_seconds)
                        postprocessing_times.append(rerank_timing.postprocessing_seconds)
                        model_calls.append(rerank_timing.model_calls)
                    total_times.append(pipeline_timing.total_seconds)
            print(f"\nCandidate top_k={candidate_k} -> final top_k={final_k}")
            report("reranker latency", rerank_times)
            report("total hybrid + reranker", total_times)
            if tokenizer_times:
                report("tokenizer", tokenizer_times)
                report("preprocessing", preprocessing_times)
                report("model inference", inference_times)
                report("postprocessing", postprocessing_times)
                print(f"model_calls_per_request={sorted(set(model_calls))}")

            if cases and candidate_k == 5 and final_k == 3:
                quality_rankings = [
                    pipeline.retrieve(case.query, candidate_top_k=candidate_k, top_k=final_k)
                    for case in cases
                ]

        if quality_rankings:
            metrics = evaluate_rankings(quality_rankings, cases, top_k=3)
            if metrics:
                print(
                    f"quality at 5->3: cases={metrics.cases}; "
                    f"Recall@3={metrics.recall_at_k:.4f}; MRR@3={metrics.mrr_at_k:.4f}"
                )

        if args.compare_unbatched and not args.mock_reranker:
            compare_config = replace(
                reranker_config,
                device=reranker.device,
                batch_all_candidates=False,
                batch_size=1,
            )
            unbatched = CrossEncoderReranker(compare_config, model=reranker.model)
            if compare_config.warmup_enabled:
                unbatched.warmup()
            batched_times: list[float] = []
            unbatched_times: list[float] = []
            for _ in range(args.runs):
                for query in queries:
                    candidates = hybrid_retriever.retrieve(query, top_k=5)
                    reranker.rerank(query, candidates, top_k=3)
                    batched_times.append(reranker.last_timing.total_seconds)
                    unbatched.rerank(query, candidates, top_k=3)
                    unbatched_times.append(unbatched.last_timing.total_seconds)
            print("\nBatched comparison at 5 candidates -> 3 results")
            report("batched reranker", batched_times)
            print(f"batched model_calls={reranker.last_timing.model_calls}")
            report("unbatched reranker", unbatched_times)
            print(f"unbatched model_calls={unbatched.last_timing.model_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
