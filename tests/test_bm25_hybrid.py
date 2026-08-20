from __future__ import annotations

from voice_rag_ingestion.bm25 import BM25Config, BM25Index, BM25Retriever
from voice_rag_ingestion.chunking.base import Chunk
from voice_rag_ingestion.hybrid import HybridConfig, HybridRetriever
from voice_rag_ingestion.qdrant_store import RetrievedChunk
from voice_rag_ingestion.retrieval import RetrievalTiming
from voice_rag_ingestion.rrf import RRFFuser
from voice_rag_ingestion.tokenization import TokenizerConfig, UnicodeWordTokenizer


def make_chunk(chunk_id: str, text: str, *, selected: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        language="hi_Deva",
        chunk_index=0,
        chunk_strategy="fixed",
        query_id="1",
        source={
            "split": "validation",
            "passage_index": 0,
            "source_lang": "eng_Latn",
            "target_lang": "hi_Deva",
            "text_source": "translated",
            "is_selected": selected,
            "dataset_name": "ai4bharat/MSMARCO-XI",
        },
        metadata={"query": "company", "original_metadata": {"x": 1}},
    )


def test_unicode_tokenizer_handles_scripts_punctuation_and_empty_text():
    tokenizer = UnicodeWordTokenizer()
    tokens = tokenizer.tokenize("Hello, नमस्ते! কোম্পানি 世界.")
    assert tokens == ["hello", "नमस्ते", "কোম্পানি", "世界"]
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize(None) == []


def test_bm25_index_retrieval_result_compatibility_and_empty_query():
    index = BM25Index()
    chunks = [
        make_chunk("a", "company corporation business"),
        make_chunk("b", "river mountain water"),
        make_chunk("c", "कंपनी व्यवसाय निगम"),
    ]
    assert index.rebuild(chunks) == 3
    retriever = BM25Retriever(index)
    results = retriever.retrieve("corporation", top_k=2)
    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk_id == "a"
    assert results[0].score > 0
    assert results[0].metadata["passage_index"] == 0
    assert retriever.retrieve("   ") == []


def test_bm25_supports_indic_query_and_deterministic_ties():
    index = BM25Index(BM25Config(tokenizer=TokenizerConfig(lowercase=True)))
    index.rebuild([make_chunk("b", "कंपनी व्यवसाय"), make_chunk("a", "कंपनी व्यापार")])
    results_first = index.search("कंपनी", top_k=2)
    results_second = index.search("कंपनी", top_k=2)
    assert [item.chunk_id for item in results_first] == [item.chunk_id for item in results_second]
    assert all(item.score > 0 for item in results_first)


def test_bm25_save_and_load_preserves_chunks_and_metadata(tmp_path):
    path = tmp_path / "bm25.json"
    original = BM25Index()
    original.rebuild([make_chunk("a", "company corporation")])
    original.save(path)
    restored = BM25Index.load(path)
    result = restored.search("company", top_k=1)[0]
    assert restored.size == 1
    assert result.chunk_id == "a"
    assert result.metadata["document_metadata"]["original_metadata"] == {"x": 1}


def result(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"text {chunk_id}",
        score=score,
        metadata={"chunk_strategy": "fixed", "language": "en"},
    )


def test_rrf_merges_duplicates_preserves_scores_and_uses_configured_k():
    fused = RRFFuser(rrf_k=1).fuse(
        [result("a", 0.9), result("b", 0.8)],
        [result("b", 4.0), result("c", 3.0)],
        top_k=3,
    )
    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    b = fused[0]
    assert b.vector_score == 0.8
    assert b.bm25_score == 4.0
    assert b.rrf_score == 1 / 3 + 1 / 2
    assert len({item.chunk_id for item in fused}) == 3


class StubVectorRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve_with_timing(self, query, *, top_k):
        return self.results[:top_k], RetrievalTiming(0.001, 0.002, 0.003)


def test_hybrid_retrieval_runs_independent_rankers_and_returns_rrf_results():
    chunks = [make_chunk("a", "company corporation"), make_chunk("b", "river water")]
    bm25 = BM25Retriever(BM25Index())
    bm25.index.rebuild(chunks)
    vector = StubVectorRetriever([result("b", 0.9), result("a", 0.8)])
    hybrid = HybridRetriever(
        vector,
        bm25,
        config=HybridConfig(vector_top_k=2, bm25_top_k=2, final_top_k=2, rrf_k=10),
    )
    results, timing = hybrid.retrieve_with_timing("company", top_k=2)
    assert len(results) == 2
    assert results[0].rrf_score >= results[1].rrf_score
    assert timing.vector_seconds == 0.003
    assert timing.bm25_seconds >= 0
    assert timing.rrf_seconds >= 0


def test_hybrid_config_is_parameterized():
    config = HybridConfig.from_env()
    assert config.vector_top_k > 0
    assert config.bm25_top_k > 0
    assert config.final_top_k > 0
    assert config.rrf_k >= 0


def test_incremental_bm25_add_maintains_statistics_and_postings():
    index = BM25Index()
    # Batch 1
    added_1 = index.add([make_chunk("c1", "first test passage"), make_chunk("c2", "second test passage")])
    assert added_1 == 2
    assert index.size == 2
    assert index._document_frequency["test"] == 2
    assert index._document_frequency["first"] == 1
    assert "c1" in index._postings["first"]
    assert "c2" in index._postings["second"]

    # Batch 2
    added_2 = index.add([make_chunk("c3", "third test passage with new terms")])
    assert added_2 == 1
    assert index.size == 3
    assert index._document_frequency["test"] == 3
    assert index._document_frequency["terms"] == 1
    assert index._total_document_length > 0
    assert index._average_document_length == index._total_document_length / 3


def test_bm25_duplicate_id_replacement_and_idempotency():
    index = BM25Index()
    index.add([make_chunk("c1", "alpha beta gamma")])
    assert index.size == 1
    assert index._document_frequency["gamma"] == 1

    # Idempotent addition of same chunk
    added = index.add([make_chunk("c1", "alpha beta delta")])
    assert added == 0  # Replacement, not new chunk
    assert index.size == 1
    assert "gamma" not in index._document_frequency
    assert index._document_frequency["delta"] == 1
    assert index._postings.get("gamma") is None
    assert "c1" in index._postings["delta"]


def test_posting_index_candidate_filtering_accuracy():
    index = BM25Index()
    index.add([
        make_chunk("c1", "artificial intelligence machine learning"),
        make_chunk("c2", "deep learning neural networks"),
        make_chunk("c3", "botany plant flora biology"),
    ])

    results = index.search("intelligence", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"

    # Query with disjoint term shouldn't match botany doc
    results_ml = index.search("neural learning", top_k=5)
    matched_ids = {r.chunk_id for r in results_ml}
    assert matched_ids == {"c1", "c2"}
    assert "c3" not in matched_ids

