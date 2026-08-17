"""MSMARCO-XI record normalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .cleaning import clean_text, stable_text_key


@dataclass(frozen=True)
class NormalizedDocument:
    """Stable internal representation consumed by later RAG stages."""

    document_id: str
    text: str
    language: str | None
    dataset_name: str
    query_id: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _stable_id(*parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:20]


def normalize_record(
    record: Mapping[str, Any],
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    row_index: int,
) -> tuple[list[NormalizedDocument], int]:
    """Convert one raw example into one document per usable passage.

    Returns ``(documents, removed_empty_count)``. Missing or malformed nested
    fields are treated as empty rather than raising. Translation is preferred;
    English is retained as a fallback when no translated passage is available.
    """

    row = _as_mapping(record)
    passages = _as_mapping(row.get("passages"))
    translated = _as_list(passages.get("Translated_passages"))
    english = _as_list(passages.get("English_passages"))
    selected = _as_list(passages.get("is_selected"))
    count = max(len(translated), len(english))
    query_id = row.get("query_id")
    query_id_text = str(query_id) if query_id is not None else None
    language = clean_text(row.get("target_lang")) or clean_text(row.get("language"))
    query = clean_text(row.get("query"))
    english_query = clean_text(row.get("Eng_Query"))
    answer = clean_text(row.get("Answer"))
    english_answer = clean_text(row.get("Eng_Answer"))
    meta = dict(_as_mapping(row.get("meta")))
    documents: list[NormalizedDocument] = []
    removed_empty = 0

    for passage_index in range(count):
        translated_text = clean_text(translated[passage_index]) if passage_index < len(translated) else None
        english_text = clean_text(english[passage_index]) if passage_index < len(english) else None
        text = translated_text or english_text
        if text is None:
            removed_empty += 1
            continue
        selected_value = selected[passage_index] if passage_index < len(selected) else None
        source = {
            "dataset_config": dataset_config,
            "split": split,
            "row_index": row_index,
            "passage_index": passage_index,
            "text_source": "translated" if translated_text else "english_fallback",
            "source_lang": clean_text(row.get("source_lang")),
            "target_lang": clean_text(row.get("target_lang")),
            "is_selected": selected_value,
            "english_text": english_text,
            "translated_text": translated_text,
        }
        metadata = {
            "query": query,
            "english_query": english_query,
            "answer": answer,
            "english_answer": english_answer,
            "query_type": clean_text(row.get("query_type")),
            "translation_meta": meta,
            "original_metadata": {
                key: value
                for key, value in row.items()
                if key not in {"passages", "query", "Answer", "Eng_Query", "Eng_Answer"}
            },
        }
        document_id = "msmarco-xi-" + _stable_id(
            dataset_name, dataset_config, split, query_id_text, passage_index, text
        )
        documents.append(
            NormalizedDocument(
                document_id=document_id,
                text=text,
                language=language,
                dataset_name=dataset_name,
                query_id=query_id_text,
                source=source,
                metadata=metadata,
            )
        )
    return documents, removed_empty


def deduplicate_documents(
    documents: Iterable[NormalizedDocument],
) -> tuple[list[NormalizedDocument], int]:
    """Drop repeated normalized text within the loaded sample."""

    seen: set[tuple[str | None, str]] = set()
    unique: list[NormalizedDocument] = []
    removed = 0
    for document in documents:
        key = (document.language, stable_text_key(document.text))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(document)
    return unique, removed
