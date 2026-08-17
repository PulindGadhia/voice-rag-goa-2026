from __future__ import annotations

import pytest

from voice_rag_ingestion.cleaning import clean_text
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.documents import deduplicate_documents, normalize_record
from voice_rag_ingestion.loader import DatasetLoader


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
