from __future__ import annotations

import pytest

from voice_rag_ingestion.chunking import (
    ChunkingConfig,
    DeterministicEmbedding,
    FixedSizeChunker,
    MetadataAwareChunker,
    SemanticChunker,
    SentenceAwareChunker,
    chunk_document,
    validate_chunks,
)
from voice_rag_ingestion.documents import NormalizedDocument


def make_document(text: str) -> NormalizedDocument:
    return NormalizedDocument(
        document_id="msmarco-xi-test-document",
        text=text,
        language="hin_Deva",
        dataset_name="ai4bharat/MSMARCO-XI",
        query_id="42",
        source={
            "split": "validation",
            "passage_index": 3,
            "text_source": "translated",
            "source_lang": "eng_Latn",
            "target_lang": "hin_Deva",
        },
        metadata={"query_type": "DESCRIPTION", "original_metadata": {"x": 1}},
    )


def test_fixed_chunking_and_overlap():
    document = make_document("one two three four five six seven eight nine ten")
    chunks = FixedSizeChunker(ChunkingConfig(max_chunk_size=4, overlap=1)).chunk(document)
    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]
    assert all(len(chunk.text.split()) <= 4 for chunk in chunks)


def test_sentence_chunking_keeps_boundaries_and_splits_only_long_sentences():
    document = make_document("One two. Three four five. Six seven.")
    chunks = SentenceAwareChunker(ChunkingConfig(max_chunk_size=5, overlap=0)).chunk(document)
    assert [chunk.text for chunk in chunks] == ["One two. Three four five.", "Six seven."]
    long_document = make_document("one two three four five six seven eight.")
    long_chunks = SentenceAwareChunker(ChunkingConfig(max_chunk_size=3, overlap=0)).chunk(long_document)
    assert [chunk.text for chunk in long_chunks] == [
        "one two three",
        "four five six",
        "seven eight.",
    ]


class FixedSemanticEmbeddings:
    def embed(self, texts):
        return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]][: len(texts)]


def test_semantic_chunking_detects_similarity_drop_with_injected_provider():
    document = make_document("Topic one. Topic two. Unrelated topic. Unrelated detail.")
    chunker = SemanticChunker(
        ChunkingConfig(max_chunk_size=20, overlap=0, semantic_similarity_threshold=0.8),
        embedding_provider=FixedSemanticEmbeddings(),
    )
    chunks = chunker.chunk(document)
    assert len(chunks) == 2
    assert chunks[0].text == "Topic one. Topic two."
    assert chunks[1].text == "Unrelated topic. Unrelated detail."


def test_deterministic_embedding_is_multilingual_and_deterministic():
    provider = DeterministicEmbedding()
    assert provider.embed(["नमस्ते 世界"]) == provider.embed(["नमस्ते 世界"])
    assert len(provider.embed(["नमस्ते"])[0]) == 32


def test_metadata_chunking_preserves_source_and_original_metadata():
    chunk = MetadataAwareChunker(ChunkingConfig(max_chunk_size=20, overlap=0)).chunk(
        make_document("एक छोटा वाक्य। दूसरा वाक्य।")
    )[0]
    assert chunk.source["language"] == "hin_Deva"
    assert chunk.source["query_id"] == "42"
    assert chunk.source["document_id"] == "msmarco-xi-test-document"
    assert chunk.source["passage_index"] == 3
    assert chunk.source["text_source"] == "translated"
    assert chunk.source["dataset_name"] == "ai4bharat/MSMARCO-XI"
    assert chunk.source["split"] == "validation"
    assert chunk.metadata["original_metadata"] == {"x": 1}


def test_empty_and_short_documents_are_safe():
    assert FixedSizeChunker().chunk(make_document("   ")) == []
    chunks = FixedSizeChunker(ChunkingConfig(max_chunk_size=256, overlap=0)).chunk(make_document("नमस्ते 世界"))
    assert len(chunks) == 1
    assert chunks[0].text == "नमस्ते 世界"


def test_duplicate_chunks_are_removed_and_ids_are_deterministic():
    document = make_document("Repeat this. Repeat this.")
    first = SentenceAwareChunker(ChunkingConfig(max_chunk_size=20, overlap=0)).chunk(document)
    second = SentenceAwareChunker(ChunkingConfig(max_chunk_size=20, overlap=0)).chunk(document)
    assert len(first) == 1
    assert first[0].chunk_id == second[0].chunk_id
    validate_chunks(first, expected_strategy="sentence")


def test_router_supports_all_strategies_and_rejects_unknown():
    document = make_document("One sentence. Another sentence.")
    for strategy in ("fixed", "sentence", "semantic", "metadata"):
        chunks = chunk_document(document, strategy=strategy, config=ChunkingConfig(strategy=strategy, overlap=0))
        assert all(chunk.chunk_strategy == strategy for chunk in chunks)
    with pytest.raises(ValueError, match="unsupported"):
        chunk_document(document, strategy="unknown")


def test_chunking_config_can_be_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_SIZE", "12")
    monkeypatch.setenv("CHUNK_OVERLAP", "3")
    monkeypatch.setenv("CHUNK_MIN_SIZE", "2")
    monkeypatch.setenv("CHUNK_SEMANTIC_THRESHOLD", "0.8")
    monkeypatch.setenv("CHUNK_STRATEGY", "sentence")
    config = ChunkingConfig.from_env()
    assert config.max_chunk_size == 12
    assert config.overlap == 3
    assert config.min_chunk_size == 2
    assert config.semantic_similarity_threshold == 0.8
    assert config.strategy == "sentence"


def test_chunk_validation_rejects_non_contiguous_indexes():
    document = make_document("One sentence. Another sentence.")
    chunks = FixedSizeChunker(ChunkingConfig(max_chunk_size=2, overlap=0)).chunk(document)
    broken = list(chunks)
    broken[1] = type(broken[1])(
        chunk_id=broken[1].chunk_id,
        document_id=broken[1].document_id,
        text=broken[1].text,
        language=broken[1].language,
        chunk_index=4,
        chunk_strategy=broken[1].chunk_strategy,
        query_id=broken[1].query_id,
        source=broken[1].source,
        metadata=broken[1].metadata,
    )
    with pytest.raises(ValueError, match="contiguous"):
        validate_chunks(broken)
