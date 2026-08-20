from __future__ import annotations

import pytest

from voice_rag_ingestion.cleaning import clean_text
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.documents import deduplicate_documents, normalize_record
from voice_rag_ingestion.loader import DatasetLoader, DatasetLoadError


def sample_record(**overrides):
    record = {
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "meta": {"model_name": "test-model", "temperature": 0},
        "query": "  नमूना   प्रश्न ",
        "Answer": "उत्तर",
        "query_id": 42,
        "query_type": "DESCRIPTION",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": ["English passage", "Second passage"],
            "Translated_passages": ["  हिन्दी   अनुच्छेद ", "दूसरा अनुच्छेद"],
        },
        "Eng_Query": "sample question",
        "Eng_Answer": "answer",
    }
    record.update(overrides)
    return record


def test_text_cleaning_preserves_multilingual_text():
    assert clean_text("  à\nनमस्ते\t世界  ") == "à नमस्ते 世界"
    assert clean_text(None) is None
    assert clean_text(" \n\t ") is None


def test_document_normalization_creates_one_document_per_passage():
    documents, removed = normalize_record(
        sample_record(),
        dataset_name="ai4bharat/MSMARCO-XI",
        dataset_config="default",
        split="validation",
        row_index=0,
    )
    assert removed == 0
    assert len(documents) == 2
    assert documents[0].text == "हिन्दी अनुच्छेद"
    assert documents[0].query_id == "42"
    assert documents[0].language == "hin_Deva"
    assert documents[0].source["passage_index"] == 0
    assert documents[0].metadata["english_query"] == "sample question"
    assert documents[0].metadata["original_metadata"]["source_lang"] == "eng_Latn"


def test_missing_fields_are_safe_and_english_is_fallback():
    documents, removed = normalize_record(
        {"passages": {"English_passages": [" English only "]}},
        dataset_name="dataset",
        dataset_config="default",
        split="train",
        row_index=3,
    )
    assert removed == 0
    assert len(documents) == 1
    assert documents[0].text == "English only"
    assert documents[0].query_id is None
    assert documents[0].language is None
    assert documents[0].source["text_source"] == "english_fallback"


def test_empty_passages_are_counted_and_removed():
    documents, removed = normalize_record(
        {"passages": {"Translated_passages": [" ", None], "English_passages": [None, ""]}},
        dataset_name="dataset",
        dataset_config="default",
        split="train",
        row_index=0,
    )
    assert documents == []
    assert removed == 2


def test_duplicate_handling_uses_cleaned_text_and_language():
    first, _ = normalize_record(sample_record(), dataset_name="d", dataset_config="c", split="s", row_index=0)
    second, _ = normalize_record(
        sample_record(query_id=43, passages={"Translated_passages": ["हिन्दी अनुच्छेद"], "English_passages": ["x"]}),
        dataset_name="d", dataset_config="c", split="s", row_index=1,
    )
    unique, removed = deduplicate_documents(first + second)
    assert len(unique) == 2
    assert removed == 1


def test_loader_reads_bounded_sample_and_reports_stats():
    rows = [sample_record(query_id=1), sample_record(query_id=2)]

    def fake_loader(**kwargs):
        assert kwargs["name"] == "default"
        assert kwargs["split"] == "validation"
        assert kwargs["streaming"] is True
        return iter(rows)

    loader = DatasetLoader(
        LoaderConfig(sample_size=1, streaming=True, development_mode=True, backend="hf_datasets"),
        dataset_loader=fake_loader,
    )
    documents, stats = loader.load_documents()
    assert stats.records_read == 1
    assert stats.documents_created == 2
    assert len(documents) == 2


def test_development_mode_disallows_non_streaming():
    with pytest.raises(ValueError, match="streaming"):
        LoaderConfig(streaming=False, development_mode=True)


def test_dataset_server_raises_error_for_sample_size_over_100():
    from voice_rag_ingestion.loader import DatasetLoadError

    loader = DatasetLoader(LoaderConfig(sample_size=101, backend="dataset_server"))
    with pytest.raises(DatasetLoadError, match="dataset_server backend supports at most 100 rows"):
        list(loader.raw_records())


def test_dataset_server_raises_error_for_sample_size_none():
    from voice_rag_ingestion.loader import DatasetLoadError

    loader = DatasetLoader(LoaderConfig(sample_size=None, backend="dataset_server"))
    with pytest.raises(DatasetLoadError, match="dataset_server backend supports at most 100 rows"):
        list(loader.raw_records())


def test_hf_datasets_streaming_large_sample_sizes_and_none():
    large_rows = [sample_record(query_id=i) for i in range(1200)]

    def fake_hf_loader(**kwargs):
        assert kwargs["streaming"] is True
        return iter(large_rows)

    # Test sample_size=500
    loader_500 = DatasetLoader(
        LoaderConfig(sample_size=500, backend="hf_datasets"),
        dataset_loader=fake_hf_loader,
    )
    records_500 = list(loader_500.raw_records())
    assert len(records_500) == 500

    # Test sample_size=1000
    loader_1000 = DatasetLoader(
        LoaderConfig(sample_size=1000, backend="hf_datasets"),
        dataset_loader=fake_hf_loader,
    )
    records_1000 = list(loader_1000.raw_records())
    assert len(records_1000) == 1000

    # Test sample_size=None (full stream)
    loader_all = DatasetLoader(
        LoaderConfig(sample_size=None, backend="hf_datasets"),
        dataset_loader=fake_hf_loader,
    )
    records_all = list(loader_all.raw_records())
    assert len(records_all) == 1200


def test_hf_datasets_uses_take_method_when_available():
    class MockDatasetWithTake:
        def __init__(self, data):
            self.data = data
            self.take_called_with = None

        def take(self, n):
            self.take_called_with = n
            return iter(self.data[:n])

    data = [sample_record(query_id=i) for i in range(50)]
    mock_ds = MockDatasetWithTake(data)

    loader = DatasetLoader(
        LoaderConfig(sample_size=20, backend="hf_datasets"),
        dataset_loader=lambda **kwargs: mock_ds,
    )
    records = list(loader.raw_records())
    assert len(records) == 20
    assert mock_ds.take_called_with == 20


def test_loader_config_from_env_backend_resolution(monkeypatch):
    # Auto-selection when HF_LOADER_BACKEND is not set
    monkeypatch.delenv("HF_LOADER_BACKEND", raising=False)
    monkeypatch.delenv("DATASET_BACKEND", raising=False)

    monkeypatch.setenv("HF_SAMPLE_SIZE", "50")
    config_50 = LoaderConfig.from_env()
    assert config_50.backend == "dataset_server"

    monkeypatch.setenv("HF_SAMPLE_SIZE", "500")
    config_500 = LoaderConfig.from_env()
    assert config_500.backend == "hf_datasets"

    monkeypatch.delenv("HF_SAMPLE_SIZE", raising=False)
    # When sample_size is None (full stream)
    monkeypatch.setenv("HF_SAMPLE_SIZE", "")
    config_none = LoaderConfig.from_env()
    assert config_none.backend == "hf_datasets"

    # Explicit backend override takes precedence (HF_LOADER_BACKEND or DATASET_BACKEND)
    monkeypatch.setenv("HF_LOADER_BACKEND", "dataset_server")
    monkeypatch.setenv("HF_SAMPLE_SIZE", "500")
    config_explicit = LoaderConfig.from_env()
    assert config_explicit.backend == "dataset_server"

    monkeypatch.delenv("HF_LOADER_BACKEND", raising=False)
    monkeypatch.setenv("DATASET_BACKEND", "dataset_server")
    config_dataset_backend = LoaderConfig.from_env()
    assert config_dataset_backend.backend == "dataset_server"

    monkeypatch.setenv("HF_LOADER_BACKEND", "hf_datasets")
    monkeypatch.setenv("HF_SAMPLE_SIZE", "50")
    config_explicit_hf = LoaderConfig.from_env()
    assert config_explicit_hf.backend == "hf_datasets"


def test_loader_config_with_resolved_backend(monkeypatch):
    monkeypatch.delenv("HF_LOADER_BACKEND", raising=False)
    monkeypatch.delenv("DATASET_BACKEND", raising=False)
    monkeypatch.delenv("HF_SAMPLE_SIZE", raising=False)

    # Base configuration starting with default sample_size=10
    base = LoaderConfig.from_env()
    assert base.sample_size == 10
    assert base.backend == "dataset_server"

    # with_resolved_backend resolving to hf_datasets when sample_size > 100
    res_500 = base.with_resolved_backend(sample_size=500)
    assert res_500.sample_size == 500
    assert res_500.backend == "hf_datasets"

    # with_resolved_backend resolving to hf_datasets when sample_size is None
    res_none = base.with_resolved_backend(sample_size=None)
    assert res_none.sample_size is None
    assert res_none.backend == "hf_datasets"

    # with_resolved_backend resolving to dataset_server when sample_size <= 100
    res_50 = base.with_resolved_backend(sample_size=50)
    assert res_50.sample_size == 50
    assert res_50.backend == "dataset_server"

    # Direct LoaderConfig.from_env(sample_size=...) invocation (used in scripts)
    cfg_500 = LoaderConfig.from_env(sample_size=500)
    assert cfg_500.sample_size == 500
    assert cfg_500.backend == "hf_datasets"

    cfg_100 = LoaderConfig.from_env(sample_size=100)
    assert cfg_100.sample_size == 100
    assert cfg_100.backend == "dataset_server"

    # Explicit backend override remains protected
    monkeypatch.setenv("HF_LOADER_BACKEND", "dataset_server")
    res_explicit = base.with_resolved_backend(sample_size=500)
    assert res_explicit.sample_size == 500
    assert res_explicit.backend == "dataset_server"

    # Explicit dataset_server with sample_size > 100 must raise DatasetLoadError
    loader = DatasetLoader(res_explicit)
    with pytest.raises(DatasetLoadError, match="dataset_server backend supports at most 100 rows"):
        list(loader.raw_records())


class MockEmbeddingProvider:
    dimension = 4

    def embed_text(self, text, *, input_type="passage"):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, texts, *, input_type="passage"):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_stream_index_dataset_batching_and_progress():
    from voice_rag_ingestion.bm25 import BM25Index
    from voice_rag_ingestion.chunking import ChunkingConfig
    from voice_rag_ingestion.indexing import stream_index_dataset
    from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig

    rows = [
        sample_record(query_id=1, target_lang="hin_Deva"),
        sample_record(query_id=2, target_lang="asm_Beng"),
        sample_record(query_id=3, target_lang="tam_Taml"),
    ]

    def fake_loader(**kwargs):
        return iter(rows)

    loader = DatasetLoader(
        LoaderConfig(sample_size=3, streaming=True, backend="hf_datasets"),
        dataset_loader=fake_loader,
    )

    vector_store = QdrantVectorStore(VectorStoreConfig(collection_name="test_stream", url=":memory:", recreate_collection=True))
    bm25 = BM25Index()
    embedder = MockEmbeddingProvider()

    progress_ticks = []

    def on_progress(stats):
        progress_ticks.append(stats.documents_processed)

    stats = stream_index_dataset(
        loader=loader,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25,
        chunking_config=ChunkingConfig(max_chunk_size=100, overlap=0),
        stream_batch_size=2,  # Mini-batches of 2 documents
        embedding_batch_size=4,
        progress_callback=on_progress,
    )

    assert stats.records_read == 3
    assert stats.documents_processed == 6  # 3 rows * 2 passages
    assert stats.chunks_created >= 6
    assert stats.qdrant_chunks_indexed == stats.chunks_created
    assert stats.bm25_chunks_indexed == stats.chunks_created
    assert stats.languages_processed == {"hin_Deva", "asm_Beng", "tam_Taml"}
    assert stats.failures == 0
    assert len(progress_ticks) >= 3  # Called across multiple batches


def test_stream_index_dataset_idempotency():
    from voice_rag_ingestion.bm25 import BM25Index
    from voice_rag_ingestion.indexing import stream_index_dataset
    from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig

    rows = [sample_record(query_id=10), sample_record(query_id=20)]

    def fake_loader(**kwargs):
        return iter(rows)

    loader = DatasetLoader(LoaderConfig(sample_size=2, streaming=True, backend="hf_datasets"), dataset_loader=fake_loader)
    vector_store = QdrantVectorStore(VectorStoreConfig(collection_name="test_idempotent", url=":memory:"))
    bm25 = BM25Index()
    embedder = MockEmbeddingProvider()

    # First run
    stats1 = stream_index_dataset(loader=loader, embedder=embedder, vector_store=vector_store, bm25_index=bm25)
    initial_chunks = stats1.qdrant_chunks_indexed
    initial_bm25 = bm25.size

    # Second run without recreating collection (safe idempotent upsert)
    stats2 = stream_index_dataset(loader=loader, embedder=embedder, vector_store=vector_store, bm25_index=bm25)

    assert stats2.chunks_created == initial_chunks
    assert bm25.size == initial_bm25  # BM25 deduplicates chunks by chunk_id


def test_stream_index_dataset_language_filtering():
    from voice_rag_ingestion.indexing import stream_index_dataset
    from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig

    rows = [
        sample_record(query_id=1, target_lang="asm_Beng"),
        sample_record(query_id=2, target_lang="hin_Deva"),
        sample_record(query_id=3, target_lang="tam_Taml"),
    ]

    def fake_loader(**kwargs):
        return iter(rows)

    loader = DatasetLoader(LoaderConfig(sample_size=3, streaming=True, backend="hf_datasets"), dataset_loader=fake_loader)
    vector_store = QdrantVectorStore(VectorStoreConfig(collection_name="test_filter", url=":memory:"))
    embedder = MockEmbeddingProvider()

    # Filter only for Assamese
    stats = stream_index_dataset(
        loader=loader,
        embedder=embedder,
        vector_store=vector_store,
        languages=["asm_Beng"],
    )

    assert stats.records_read == 3
    assert stats.documents_processed == 2  # Only the 2 passages from asm_Beng record
    assert stats.documents_skipped == 4   # 4 passages from other languages skipped
    assert stats.languages_processed == {"asm_Beng"}


def test_stream_index_dataset_failure_handling():
    from voice_rag_ingestion.indexing import stream_index_dataset
    from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig

    rows = [
        sample_record(query_id=1),
        {"invalid": "corrupted row missing passages"},
        sample_record(query_id=2),
    ]

    def fake_loader(**kwargs):
        return iter(rows)

    loader = DatasetLoader(LoaderConfig(sample_size=3, streaming=True, backend="hf_datasets"), dataset_loader=fake_loader)
    vector_store = QdrantVectorStore(VectorStoreConfig(collection_name="test_err", url=":memory:"))
    embedder = MockEmbeddingProvider()

    stats = stream_index_dataset(loader=loader, embedder=embedder, vector_store=vector_store)
    assert stats.records_read == 3
    assert stats.failures == 0  # Invalid dict has no passages, gracefully counted as skipped empty
    assert stats.documents_processed == 4  # Valid 2 records processed


def test_checkpoint_save_load_and_reset(tmp_path):
    from voice_rag_ingestion.checkpoint import (
        IngestionCheckpoint,
        load_checkpoint,
        reset_checkpoint,
        save_checkpoint,
    )

    cp_path = tmp_path / "test_cp.json"
    assert load_checkpoint(cp_path) is None

    cp = IngestionCheckpoint(
        dataset_name="ai4bharat/MSMARCO-XI",
        dataset_config="default",
        split="validation",
        last_processed_row_index=49,
        records_read=50,
        documents_processed=500,
        chunks_created=500,
        qdrant_chunks_indexed=500,
        bm25_chunks_indexed=500,
        languages_processed=["hin_Deva", "asm_Beng"],
        failures=0,
        elapsed_seconds=12.5,
    )
    save_checkpoint(cp_path, cp)

    loaded = load_checkpoint(cp_path)
    assert loaded is not None
    assert loaded.last_processed_row_index == 49
    assert loaded.records_read == 50
    assert loaded.qdrant_chunks_indexed == 500
    assert loaded.languages_processed == ["hin_Deva", "asm_Beng"]
    assert loaded.matches("ai4bharat/MSMARCO-XI", "default", "validation")

    reset_checkpoint(cp_path)
    assert not cp_path.exists()
    assert load_checkpoint(cp_path) is None


def test_stream_index_dataset_checkpoint_resumption(tmp_path):
    from voice_rag_ingestion.bm25 import BM25Index
    from voice_rag_ingestion.checkpoint import (
        IngestionCheckpoint,
        load_checkpoint,
        save_checkpoint,
    )
    from voice_rag_ingestion.indexing import stream_index_dataset
    from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig

    cp_path = tmp_path / "resume_cp.json"
    bm25_path = tmp_path / "bm25.json"

    # Pre-seed a checkpoint indicating row 0 and 1 have been processed
    initial_cp = IngestionCheckpoint(
        dataset_name="ai4bharat/MSMARCO-XI",
        dataset_config="default",
        split="validation",
        last_processed_row_index=1,
        records_read=2,
        documents_processed=4,
        documents_skipped=0,
        chunks_created=4,
        qdrant_chunks_indexed=4,
        bm25_chunks_indexed=4,
        languages_processed=["hin_Deva"],
        failures=0,
        elapsed_seconds=1.0,
    )
    save_checkpoint(cp_path, initial_cp)

    rows = [
        sample_record(query_id=101),
        sample_record(query_id=102),
        sample_record(
            query_id=103,
            passages={
                "is_selected": [1, 0],
                "English_passages": ["P103 English", "P103 second"],
                "Translated_passages": ["अनुच्छेद १०३", "दूसरा १०३"],
            },
        ),
        sample_record(
            query_id=104,
            passages={
                "is_selected": [1, 0],
                "English_passages": ["P104 English", "P104 second"],
                "Translated_passages": ["अनुच्छेद १०४", "दूसरा १०४"],
            },
        ),
    ]

    def fake_loader(**kwargs):
        return iter(rows)

    loader = DatasetLoader(
        LoaderConfig(
            dataset_name="ai4bharat/MSMARCO-XI",
            dataset_config="default",
            split="validation",
            sample_size=4,
            streaming=True,
            backend="hf_datasets",
        ),
        dataset_loader=fake_loader,
    )
    vector_store = QdrantVectorStore(VectorStoreConfig(collection_name="test_resume", url=":memory:"))
    bm25_index = BM25Index()
    embedder = MockEmbeddingProvider()

    stats = stream_index_dataset(
        loader=loader,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        bm25_path=bm25_path,
        checkpoint_path=cp_path,
        resume=True,
        recreate_collection=False,
    )

    # 2 initial + 2 newly processed records = 4 records read total
    assert stats.records_read == 4
    # 4 initial + 4 newly created = 8 documents processed total
    assert stats.documents_processed == 8
    assert stats.qdrant_chunks_indexed == 8
    assert stats.bm25_chunks_indexed == 8

    # Verify final checkpoint was updated to last row index 3
    final_cp = load_checkpoint(cp_path)
    assert final_cp is not None
    assert final_cp.last_processed_row_index == 3
    assert final_cp.records_read == 4


def test_bm25_precomputed_term_frequencies_and_search():
    from voice_rag_ingestion.bm25 import BM25Index
    from voice_rag_ingestion.chunking.base import Chunk

    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            text="A corporation is a legal entity created under law.",
            chunk_index=0,
            language="eng_Latn",
            chunk_strategy="metadata",
            source={"text_source": "translated"},
            metadata={},
        ),
        Chunk(
            chunk_id="c2",
            document_id="d2",
            text="Financial ratios help assess corporate performance.",
            chunk_index=0,
            language="eng_Latn",
            chunk_strategy="metadata",
            source={"text_source": "translated"},
            metadata={},
        ),
    ]

    index = BM25Index()
    index.add(chunks)

    # Verify precomputed term frequencies and doc lengths
    assert index._term_frequencies["c1"]["corporation"] == 1
    assert index._term_frequencies["c1"]["a"] == 2
    assert index._doc_lengths["c1"] == 9
    assert index._term_frequencies["c2"]["financial"] == 1
    assert index._doc_lengths["c2"] == 6

    # Verify search
    results = index.search("corporation entity", top_k=5)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0.0


def test_qdrant_on_disk_payload_configuration():
    from voice_rag_ingestion.qdrant_store import QdrantVectorStore, VectorStoreConfig

    cfg = VectorStoreConfig(
        collection_name="test_on_disk",
        url=":memory:",
        on_disk_payload=True,
    )
    store = QdrantVectorStore(cfg)
    store.ensure_collection(vector_size=32, recreate=True)
    assert store.collection_exists()


def test_bm25_first_hybrid_retriever():
    from voice_rag_ingestion.bm25 import BM25Index, BM25Retriever
    from voice_rag_ingestion.bm25_first import BM25FirstHybridRetriever
    from voice_rag_ingestion.chunking.base import Chunk
    from voice_rag_ingestion.hybrid import HybridConfig

    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            text="A corporation is a legal entity created under company law.",
            chunk_index=0,
            language="eng_Latn",
            chunk_strategy="metadata",
            source={"text_source": "translated"},
            metadata={"domain": "business"},
        ),
        Chunk(
            chunk_id="c2",
            document_id="d2",
            text="Photosynthesis is the biological process used by plants.",
            chunk_index=0,
            language="eng_Latn",
            chunk_strategy="metadata",
            source={"text_source": "translated"},
            metadata={"domain": "biology"},
        ),
        Chunk(
            chunk_id="c3",
            document_id="d3",
            text="Financial market analysis assesses stocks and investments.",
            chunk_index=0,
            language="eng_Latn",
            chunk_strategy="metadata",
            source={"text_source": "translated"},
            metadata={"domain": "finance"},
        ),
    ]

    index = BM25Index()
    index.add(chunks)
    bm25_retriever = BM25Retriever(index)
    embedder = MockEmbeddingProvider()

    retriever = BM25FirstHybridRetriever(
        bm25_retriever,
        embedder,
        config=HybridConfig(vector_top_k=2, bm25_top_k=2, final_top_k=2),
    )

    results, timing = retriever.retrieve_with_timing("corporation company law", top_k=2)

    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].vector_score is not None
    assert results[0].bm25_score is not None
    assert results[0].rrf_score is not None
    assert timing.total_seconds > 0.0
    assert timing.bm25_seconds > 0.0
    assert timing.vector_seconds > 0.0


def test_bm25_only_streaming_ingestion(tmp_path):
    from voice_rag_ingestion.bm25 import BM25Index
    from voice_rag_ingestion.checkpoint import load_checkpoint
    from voice_rag_ingestion.indexing import stream_index_dataset

    bm25_path = tmp_path / "bm25_index.json"
    cp_path = tmp_path / "checkpoint.json"

    rows = [
        {
            "query_id": "q1",
            "query": "what is law",
            "query_language": "eng_Latn",
            "passages": {
                "English_passages": ["Law is a set of rules."],
                "is_selected": [1],
                "url": ["https://example.com/1"],
            },
            "Answer": "Rules.",
        },
        {
            "query_id": "q2",
            "query": "what is science",
            "query_language": "eng_Latn",
            "passages": {
                "English_passages": ["Science explores nature."],
                "is_selected": [1],
                "url": ["https://example.com/2"],
            },
            "Answer": "Nature.",
        },
    ]

    def fake_loader(**kwargs):
        return iter(rows)

    loader = DatasetLoader(
        LoaderConfig(
            dataset_name="ai4bharat/MSMARCO-XI",
            dataset_config="default",
            split="validation",
            sample_size=2,
            streaming=True,
            backend="hf_datasets",
        ),
        dataset_loader=fake_loader,
    )
    bm25_index = BM25Index()

    stats = stream_index_dataset(
        loader=loader,
        bm25_index=bm25_index,
        bm25_path=bm25_path,
        checkpoint_path=cp_path,
        bm25_only=True,
        recreate_collection=True,
    )

    assert stats.records_read == 2
    assert stats.documents_processed == 2
    assert stats.chunks_created == 2
    assert stats.bm25_chunks_indexed == 2
    assert stats.qdrant_chunks_indexed == 0  # skipped dense vector upsert
    assert bm25_index.size == 2
    assert bm25_path.exists()

    cp = load_checkpoint(cp_path)
    assert cp is not None
    assert cp.last_processed_row_index == 1
    assert cp.bm25_chunks_indexed == 2


# ---------------------------------------------------------------------------
# Local Parquet ingestion tests
# ---------------------------------------------------------------------------


def _write_parquet(path, rows: list[dict]) -> None:
    """Write a list of dicts as a Parquet file using pyarrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path))


def _make_msmarco_rows(n: int, lang: str = "hin_Deva", start_id: int = 0) -> list[dict]:
    """Generate synthetic MSMARCO-XI-compatible rows."""
    rows = []
    for i in range(n):
        qid = start_id + i
        rows.append({
            "query_id": qid,
            "source_lang": "eng_Latn",
            "target_lang": lang,
            "query": f"सवाल {qid}",
            "Eng_Query": f"question {qid}",
            "Answer": f"उत्तर {qid}",
            "Eng_Answer": f"answer {qid}",
            "query_type": "DESCRIPTION",
            "passages": {
                "is_selected": [1],
                "English_passages": [f"English passage {qid}"],
                "Translated_passages": [f"हिन्दी अनुच्छेद {qid}"],
            },
            "meta": {"model_name": "test-model", "temperature": 0},
        })
    return rows



def test_stream_index_from_parquet_bm25_only(tmp_path):
    from voice_rag_ingestion.bm25 import BM25Index
    from voice_rag_ingestion.checkpoint import load_checkpoint
    from voice_rag_ingestion.indexing import stream_index_from_parquet

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    _write_parquet(parquet_dir / "file_a.parquet", _make_msmarco_rows(10, start_id=0))
    _write_parquet(parquet_dir / "file_b.parquet", _make_msmarco_rows(10, start_id=10))

    bm25_path = tmp_path / "bm25.json"
    cp_path = tmp_path / "checkpoint.json"
    bm25_index = BM25Index()

    stats = stream_index_from_parquet(
        parquet_dir=parquet_dir,
        bm25_index=bm25_index,
        bm25_path=bm25_path,
        checkpoint_path=cp_path,
        bm25_only=True,
        stream_batch_size=5,
        parquet_batch_size=50,
    )

    assert stats.records_read == 20
    assert stats.documents_processed == 20   # 1 passage per row
    assert stats.bm25_chunks_indexed == 20
    assert stats.qdrant_chunks_indexed == 0  # bm25_only=True
    assert bm25_index.size == 20
    assert bm25_path.exists()

    cp = load_checkpoint(cp_path)
    assert cp is not None
    assert cp.records_read == 20
    assert sorted(cp.parquet_completed_files) == ["file_a.parquet", "file_b.parquet"]
    assert cp.parquet_current_file is None


def test_stream_index_from_parquet_sample_size_cutoff(tmp_path):
    from voice_rag_ingestion.bm25 import BM25Index
    from voice_rag_ingestion.indexing import stream_index_from_parquet

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    _write_parquet(parquet_dir / "file_a.parquet", _make_msmarco_rows(50, start_id=0))
    _write_parquet(parquet_dir / "file_b.parquet", _make_msmarco_rows(50, start_id=50))

    bm25_index = BM25Index()

    stats = stream_index_from_parquet(
        parquet_dir=parquet_dir,
        bm25_index=bm25_index,
        bm25_only=True,
        sample_size=30,
        parquet_batch_size=100,
    )

    assert stats.records_read == 30
    assert stats.documents_processed == 30
    assert bm25_index.size == 30


def test_stream_index_from_parquet_checkpoint_resume(tmp_path):
    from voice_rag_ingestion.bm25 import BM25Index
    from voice_rag_ingestion.checkpoint import (
        IngestionCheckpoint,
        load_checkpoint,
        save_checkpoint,
    )
    from voice_rag_ingestion.indexing import stream_index_from_parquet

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    _write_parquet(parquet_dir / "file_a.parquet", _make_msmarco_rows(10, start_id=0))
    _write_parquet(parquet_dir / "file_b.parquet", _make_msmarco_rows(10, start_id=10))

    cp_path = tmp_path / "checkpoint.json"

    # Pre-seed checkpoint: file_a is already complete
    pre_cp = IngestionCheckpoint(
        dataset_name="ai4bharat/MSMARCO-XI",
        dataset_config="default",
        split="validation",
        records_read=10,
        documents_processed=10,
        chunks_created=10,
        bm25_chunks_indexed=10,
        parquet_completed_files=["file_a.parquet"],
    )
    save_checkpoint(cp_path, pre_cp)

    bm25_index = BM25Index()
    stats = stream_index_from_parquet(
        parquet_dir=parquet_dir,
        dataset_name="ai4bharat/MSMARCO-XI",
        dataset_config="default",
        split="validation",
        bm25_index=bm25_index,
        checkpoint_path=cp_path,
        resume=True,
        bm25_only=True,
        stream_batch_size=5,
        parquet_batch_size=50,
    )

    # file_a skipped; only file_b (10 rows) processed; counts carry over from checkpoint
    assert stats.records_read == 20        # 10 carried + 10 new
    assert stats.documents_processed == 20
    assert bm25_index.size == 10           # only file_b added to this fresh index

    final_cp = load_checkpoint(cp_path)
    assert final_cp is not None
    assert sorted(final_cp.parquet_completed_files) == ["file_a.parquet", "file_b.parquet"]


def test_checkpoint_backward_compat_without_parquet_fields(tmp_path):
    """Checkpoints written before parquet fields were added still load cleanly."""
    import json
    from voice_rag_ingestion.checkpoint import load_checkpoint

    old_format = {
        "dataset_name": "ai4bharat/MSMARCO-XI",
        "dataset_config": "default",
        "split": "validation",
        "last_processed_row_index": 99,
        "records_read": 100,
        "documents_processed": 100,
        "documents_skipped": 0,
        "chunks_created": 100,
        "qdrant_chunks_indexed": 100,
        "bm25_chunks_indexed": 100,
        "languages_processed": ["hin_Deva"],
        "failures": 0,
        "elapsed_seconds": 5.0,
        "timestamp": "2026-01-01T00:00:00+00:00",
        # parquet_completed_files / parquet_current_file intentionally absent
    }
    cp_path = tmp_path / "old_checkpoint.json"
    cp_path.write_text(json.dumps(old_format), encoding="utf-8")

    cp = load_checkpoint(cp_path)
    assert cp is not None
    assert cp.parquet_completed_files == []
    assert cp.parquet_current_file is None
    assert cp.records_read == 100
