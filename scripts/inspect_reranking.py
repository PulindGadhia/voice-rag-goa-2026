#!/usr/bin/env python3
"""Inspect hybrid candidates and multilingual cross-encoder reranking."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.bm25 import BM25Config, BM25Index, BM25Retriever
from voice_rag_ingestion.chunking import ChunkingConfig
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.embeddings import CachedEmbedder, EmbeddingConfig, SentenceTransformerEmbedder
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


def show_metrics(label: str, rankings, cases, top_k: int) -> None:
    metrics = evaluate_rankings(rankings, cases, top_k=top_k)
    if metrics is None:
        print(f"{label}: unavailable")
        return
    print(
        f"{label}: cases={metrics.cases} Recall@{top_k}={metrics.recall_at_k:.4f} "
        f"MRR@{top_k}={metrics.mrr_at_k:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--candidate-top-k", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--max-eval-cases", type=int, default=20)
    parser.add_argument("--mock-embeddings", action="store_true")
    parser.add_argument("--mock-reranker", action="store_true")
    parser.add_argument("--bm25-index-path", default=os.getenv("BM25_INDEX_PATH", ".cache/bm25_index.json"))
    parser.add_argument("--query", action="append", dest="queries")
    args = parser.parse_args()
    if args.candidate_top_k < args.final_top_k:
        parser.error("--candidate-top-k must be >= --final-top-k")

    loader_config = replace(LoaderConfig.from_env(), sample_size=args.sample_size)
    embedding_config = EmbeddingConfig.from_env()
    if args.mock_embeddings:
        embedding_config = replace(embedding_config, model_name="dev-hash-embedding")
        from voice_rag_ingestion.embeddings.dev import HashEmbeddingProvider

        provider = HashEmbeddingProvider()
    else:
        provider = SentenceTransformerEmbedder(embedding_config)
    embedder = CachedEmbedder(provider, config=embedding_config) if embedding_config.cache_enabled else provider

    hybrid_config = replace(
        HybridConfig.from_env(),
        vector_top_k=args.candidate_top_k,
        bm25_top_k=args.candidate_top_k,
        final_top_k=args.candidate_top_k,
    )
    documents, load_stats = DatasetLoader(loader_config).load_documents()
    store = QdrantVectorStore(VectorStoreConfig.from_env())
    index_stats, chunks = index_documents(
        documents,
        embedder=embedder,
        vector_store=store,
        chunking_config=replace(ChunkingConfig.from_env(), strategy=hybrid_config.chunking_strategy),
        embedding_batch_size=embedding_config.batch_size,
        recreate_collection=True,
    )
    bm25_index = BM25Index(BM25Config(k1=hybrid_config.bm25_k1, b=hybrid_config.bm25_b, tokenizer=hybrid_config.tokenizer))
    bm25_index.rebuild(chunks)
    bm25_index.save(args.bm25_index_path)
    vector_retriever = VectorRetriever(embedder, store)
    bm25_retriever = BM25Retriever(bm25_index)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever, config=hybrid_config)
    reranker_config = replace(
        RerankerConfig.from_env(),
        candidate_top_k=args.candidate_top_k,
        final_top_k=args.final_top_k,
    )
    reranker = LexicalOverlapReranker() if args.mock_reranker else CrossEncoderReranker(reranker_config)
    pipeline = HybridRerankRetriever(hybrid_retriever, reranker, config=reranker_config)
    queries = args.queries or ["What is a corporation?", "কৰ্পোৰেচন কি?", "कंपनी क्या है?"]

    print(f"Dataset records loaded: {load_stats.records_read}; documents: {index_stats.documents}; chunks: {index_stats.chunks}")
    print(f"Hybrid candidate top_k: {args.candidate_top_k}; reranker final top_k: {args.final_top_k}")
    print(f"Reranker: {'lexical development stub' if args.mock_reranker else reranker_config.model_name}")
    for query in queries:
        candidates = hybrid_retriever.retrieve(query, top_k=args.candidate_top_k)
        results, timing = pipeline.retrieve_with_timing(
            query, candidate_top_k=args.candidate_top_k, top_k=args.final_top_k
        )
        print(f"\nQUERY\n{query}")
        print("Hybrid candidates:")
        for rank, result in enumerate(candidates, 1):
            print(
                f"{rank}. {result.chunk_id} RRF={result.rrf_score:.6f} "
                f"vector={result.vector_score} BM25={result.bm25_score} {result.text[:220]}"
            )
        print("Reranked results:")
        for result in results:
            print(
                f"{result.rerank_rank}. {result.chunk_id} rerank={result.rerank_score:.6f} "
                f"RRF={result.rrf_score:.6f} vector={result.vector_score} "
                f"BM25={result.bm25_score} language={result.metadata.get('language')} "
                f"{result.text[:220]}"
            )
        print(
            f"timing: hybrid={timing.hybrid_seconds:.6f}s rerank={timing.rerank_seconds:.6f}s "
            f"total={timing.total_seconds:.6f}s"
        )

    cases = build_cases(chunks, args.max_eval_cases)
    if cases:
        vector_rankings = [vector_retriever.retrieve(case.query, top_k=args.final_top_k) for case in cases]
        bm25_rankings = [bm25_retriever.retrieve(case.query, top_k=args.final_top_k) for case in cases]
        hybrid_rankings = [hybrid_retriever.retrieve(case.query, top_k=args.final_top_k) for case in cases]
        reranked_rankings = [pipeline.retrieve(case.query, top_k=args.final_top_k, candidate_top_k=args.candidate_top_k) for case in cases]
        print(f"\nQuality evaluation: selected MSMARCO-XI passages only; cases={len(cases)}")
        show_metrics("vector-only", vector_rankings, cases, args.final_top_k)
        show_metrics("BM25-only", bm25_rankings, cases, args.final_top_k)
        show_metrics("hybrid RRF", hybrid_rankings, cases, args.final_top_k)
        show_metrics("hybrid + reranker", reranked_rankings, cases, args.final_top_k)
    else:
        print("\nQuality evaluation: unavailable; sample contained no selected-passage labels.")
    if isinstance(embedder, CachedEmbedder):
        print(f"Embedding cache hits: {embedder.stats.hits}; misses: {embedder.stats.misses}")
        embedder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
