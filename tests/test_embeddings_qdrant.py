from __future__ import annotations

from dataclasses import dataclass

from voice_rag_ingestion.chunking import ChunkingConfig, FixedSizeChunker
from voice_rag_ingestion.documents import NormalizedDocument
from voice_rag_ingestion.embeddings import CachedEmbedder, EmbeddingConfig, SentenceTransformerEmbedder
from voice_rag_ingestion.embeddings.base import timed_batch_embed
from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig, chunk_payload
from voice_rag_ingestion.retrieval import VectorRetriever


class CountingProvider:
    dimension = 3

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def embed_text(self, text, *, input_type="passage"):
        return self.embed_batch([text], input_type=input_type)[0]

    def embed_batch(self, texts, *, input_type="passage"):
        self.calls += 1
        self.batch_sizes.append(len(texts))
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class FakeSentenceModel:
    def __init__(self):
        self.inputs = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.inputs.append((list(texts), kwargs))
        return [[1.0, 2.0, 3.0] for _ in texts]


def make_document(text="One document. Another passage."):
    return NormalizedDocument(
        document_id="doc-1",
        text=text,
        language="hi_Deva",
        dataset_name="ai4bharat/MSMARCO-XI",
        query_id="7",
        source={
            "split": "validation",
            "passage_index": 2,
            "text_source": "translated",
            "source_lang": "eng_Latn",
            "target_lang": "hi_Deva",
        },
        metadata={"query": "प्रश्न", "original_metadata": {"model": "test"}},
    )


def test_sentence_transformer_adapter_batches_and_applies_e5_prefixes():
    model = FakeSentenceModel()
    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(model_name="test", batch_size=2), model=model
    )
    vectors = embedder.embed_batch(["hello", "नमस्ते"], input_type="query")
    assert embedder.dimension == 3
    assert len(vectors) == 2
    assert model.inputs[0][0] == ["query: hello", "query: नमस्ते"]
    assert model.inputs[0][1]["batch_size"] == 2
    assert model.inputs[0][1]["normalize_embeddings"] is True


def test_batch_embedding_respects_batch_size():
    provider = CountingProvider()
    vectors, stats = timed_batch_embed(provider, ["a", "b", "c", "d", "e"], batch_size=2, input_type="passage")
    assert len(vectors) == 5
    assert stats.batches == 3
    assert provider.batch_sizes == [2, 2, 1]


def test_embedding_cache_hits_and_misses(tmp_path):
    provider = CountingProvider()
    config = EmbeddingConfig(model_name="test-cache", cache_path=str(tmp_path / "vectors.sqlite3"))
    embedder = CachedEmbedder(provider, config=config)
    first = embedder.embed_batch(["same", "different"])
    second = embedder.embed_batch(["same", "different", "new"])
    assert first == second[:2]
    assert embedder.stats.hits == 2
    assert embedder.stats.misses == 3
    assert provider.calls == 2
    embedder.close()


def test_embedding_cache_deduplicates_identical_texts_in_one_batch(tmp_path):
    provider = CountingProvider()
    config = EmbeddingConfig(model_name="test-dedupe", cache_path=str(tmp_path / "vectors.sqlite3"))
    embedder = CachedEmbedder(provider, config=config)
    vectors = embedder.embed_batch(["same", "same", "other"])
    assert vectors[0] == vectors[1]
    assert provider.batch_sizes == [2]
    assert embedder.stats.misses == 2
    assert embedder.stats.hits == 1
    embedder.close()


def test_qdrant_collection_upsert_search_and_payload_preservation():
    store = QdrantVectorStore(VectorStoreConfig(collection_name="test", url=":memory:", recreate_collection=True))
    chunks = FixedSizeChunker(ChunkingConfig(max_chunk_size=5, overlap=0)).chunk(
        make_document("one two three four five six seven eight nine ten")
    )
    store.ensure_collection(3)
    assert store.upsert(chunks, [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]) == len(chunks)
    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == chunks[0].chunk_id
    assert results[0].metadata["document_id"] == "doc-1"
    assert results[0].metadata["query_id"] == "7"
    assert results[0].metadata["passage_index"] == 2
    assert results[0].metadata["text_source"] == "translated"
    assert results[0].metadata["dataset_name"] == "ai4bharat/MSMARCO-XI"
    assert results[0].metadata["split"] == "validation"
    assert results[0].metadata["document_metadata"]["original_metadata"] == {"model": "test"}


def test_retriever_result_schema_empty_query_and_top_k():
    provider = CountingProvider()
    store = QdrantVectorStore(VectorStoreConfig(collection_name="test-retriever", url=":memory:"))
    chunks = FixedSizeChunker(ChunkingConfig(max_chunk_size=5, overlap=0)).chunk(
        make_document("one two three four five six seven eight nine ten")
    )
    store.ensure_collection(provider.dimension)
    store.upsert(chunks, [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]])
    retriever = VectorRetriever(provider, store)
    assert retriever.retrieve("   ") == []
    results = retriever.retrieve("find this", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id
    assert results[0].document_id == "doc-1"
    assert results[0].text
    assert isinstance(results[0].score, float)


def test_qdrant_persistent_path_mode_saves_to_disk(tmp_path):
    qdrant_dir = str(tmp_path / "qdrant_storage")
    config = VectorStoreConfig(collection_name="test_persistent", url=qdrant_dir, recreate_collection=True)
    store = QdrantVectorStore(config)
    chunks = FixedSizeChunker(ChunkingConfig(max_chunk_size=5, overlap=0)).chunk(
        make_document("persistent vector content stored on disk")
    )
    store.ensure_collection(3)
    store.upsert(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert store.collection_exists()
    store.close()

    # Re-open same path with new instance (simulating app restart)
    reloaded_store = QdrantVectorStore(VectorStoreConfig(collection_name="test_persistent", url=qdrant_dir, recreate_collection=False))
    assert reloaded_store.collection_exists()
    results = reloaded_store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == chunks[0].chunk_id
    reloaded_store.close()

