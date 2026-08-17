from __future__ import annotations

from voice_rag_ingestion.bm25 import BM25Index, BM25Retriever
from voice_rag_ingestion.chunking.base import Chunk
from voice_rag_ingestion.hybrid import HybridConfig, HybridRetriever
from voice_rag_ingestion.qdrant_store import RetrievedChunk
from voice_rag_ingestion.reranking import (
    CrossEncoderReranker,
    HybridRerankRetriever,
    LexicalOverlapReranker,
    RerankerConfig,
)
from voice_rag_ingestion.retrieval import RetrievalTiming
from voice_rag_ingestion.rrf import HybridResult


def candidate(chunk_id: str, text: str, *, rrf: float = 0.1) -> HybridResult:
    return HybridResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        rrf_score=rrf,
        vector_score=0.8,
        bm25_score=2.0,
        metadata={"language": "hi_Deva", "source": {"x": 1}},
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        return self.scores[: len(pairs)]


def test_cross_encoder_reranker_implements_interface_and_preserves_scores():
    model = FakeCrossEncoder([0.2, 0.9])
    reranker = CrossEncoderReranker(
        RerankerConfig(model_name="fake", candidate_top_k=2, final_top_k=1),
        model=model,
    )
    results = reranker.rerank("कंपनी क्या है?", [candidate("a", "कंपनी"), candidate("b", "व्यवसाय")], 1)
    assert [result.chunk_id for result in results] == ["b"]
    assert results[0].rerank_score == 0.9
    assert results[0].rerank_rank == 1
    assert results[0].vector_score == 0.8
    assert results[0].bm25_score == 2.0
    assert results[0].rrf_score == 0.1
    assert results[0].metadata == {"language": "hi_Deva", "source": {"x": 1}}
    assert model.calls == [[("कंपनी क्या है?", "कंपनी"), ("कंपनी क्या है?", "व्यवसाय")]]
    assert reranker.last_timing.pair_count == 2
    assert reranker.last_timing.model_calls == 1
    assert reranker.last_timing.batched is True


def test_reranker_handles_empty_query_and_candidates_without_model_call():
    model = FakeCrossEncoder([1.0])
    reranker = CrossEncoderReranker(RerankerConfig(model_name="fake"), model=model)
    assert reranker.rerank(" ", [candidate("a", "text")], 1) == []
    assert reranker.rerank("query", [], 1) == []
    assert model.calls == []


def test_reranker_deduplicates_truncates_and_orders_ties_deterministically():
    model = FakeCrossEncoder([0.5, 0.5])
    reranker = CrossEncoderReranker(
        RerankerConfig(model_name="fake", candidate_top_k=2), model=model
    )
    results = reranker.rerank(
        "query",
        [candidate("b", "b"), candidate("a", "a"), candidate("c", "c"), candidate("b", "duplicate")],
        2,
    )
    assert [result.chunk_id for result in results] == ["a", "b"]
    assert len(model.calls[0]) == 2
    assert [result.rerank_rank for result in results] == [1, 2]


def test_lexical_reranker_supports_multilingual_text_and_top_k():
    reranker = LexicalOverlapReranker()
    results = reranker.rerank(
        "कंपनी",
        [candidate("a", "कंपनी और व्यवसाय"), candidate("b", "नदी और पहाड़")],
        1,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "a"
    assert results[0].rerank_score == 1.0


class StubVectorRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve_with_timing(self, query, *, top_k):
        return self.results[:top_k], RetrievalTiming(0.001, 0.002, 0.003)


def vector_result(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"text {chunk_id}",
        score=score,
        metadata={"language": "en"},
    )


def test_hybrid_to_reranker_integration_uses_larger_candidate_pool():
    chunks = [
        Chunk("a", "doc-a", "company", "en", 0, "fixed", None, {}, {}),
        Chunk("b", "doc-b", "company corporation", "en", 0, "fixed", None, {}, {}),
        Chunk("c", "doc-c", "river water", "en", 0, "fixed", None, {}, {}),
    ]
    bm25_index = BM25Index()
    bm25_index.rebuild(chunks)
    hybrid = HybridRetriever(
        StubVectorRetriever([vector_result("c", 0.9), vector_result("a", 0.8), vector_result("b", 0.7)]),
        BM25Retriever(bm25_index),
        config=HybridConfig(vector_top_k=3, bm25_top_k=3, final_top_k=3),
    )
    pipeline = HybridRerankRetriever(
        hybrid,
        LexicalOverlapReranker(),
        config=RerankerConfig(model_name="fake", candidate_top_k=3, final_top_k=2),
    )
    results, timing = pipeline.retrieve_with_timing("company", candidate_top_k=3, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "a"
    assert timing.hybrid_seconds >= 0
    assert timing.rerank_seconds >= 0
