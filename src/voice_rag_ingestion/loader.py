"""Streaming Hugging Face dataset loader and normalization pipeline."""

from __future__ import annotations

import logging
import json
from urllib.parse import urlencode
from urllib.request import urlopen
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .config import LoaderConfig
from .documents import NormalizedDocument, deduplicate_documents, normalize_record

logger = logging.getLogger(__name__)


class DatasetLoadError(RuntimeError):
    """Raised when Hugging Face cannot load the requested dataset selection."""


@dataclass
class LoadStats:
    records_read: int = 0
    documents_created: int = 0
    empty_passages_removed: int = 0
    duplicate_documents_removed: int = 0
    malformed_records: int = 0

    @property
    def documents_removed(self) -> int:
        return self.empty_passages_removed + self.duplicate_documents_removed


class DatasetLoader:
    """Load a bounded sample and normalize it without full development download."""

    def __init__(
        self,
        config: LoaderConfig | None = None,
        *,
        dataset_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or LoaderConfig.from_env()
        self._dataset_loader = dataset_loader
        self.last_raw_fields: set[str] = set()

    def _load_dataset(self) -> Any:
        factory = self._dataset_loader
        if factory is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise DatasetLoadError(
                    "The 'datasets' package is required. Install project dependencies first."
                ) from exc
            factory = load_dataset
        kwargs: dict[str, Any] = {
            "path": self.config.dataset_name,
            "name": self.config.dataset_config,
            "split": self.config.split,
            "streaming": self.config.streaming,
        }
        if self.config.revision:
            kwargs["revision"] = self.config.revision
        if self.config.cache_dir:
            kwargs["cache_dir"] = self.config.cache_dir
        if self.config.trust_remote_code:
            kwargs["trust_remote_code"] = True
        try:
            return factory(**kwargs)
        except Exception as exc:  # datasets exposes several exception types
            raise DatasetLoadError(
                f"Unable to load dataset={self.config.dataset_name!r}, "
                f"config={self.config.dataset_config!r}, split={self.config.split!r}: {exc}"
            ) from exc

    def raw_records(self) -> Iterator[Mapping[str, Any]]:
        if self.config.backend == "dataset_server":
            yield from self._raw_records_from_dataset_server()
            return
        dataset = self._load_dataset()
        records: Iterable[Any] = dataset
        if self.config.sample_size is not None:
            if hasattr(dataset, "take"):
                records = dataset.take(self.config.sample_size)
            else:
                records = (row for index, row in enumerate(dataset) if index < self.config.sample_size)
        for row in records:
            if isinstance(row, Mapping):
                self.last_raw_fields.update(row.keys())
                yield row
            else:
                logger.warning("Skipping non-mapping dataset row", extra={"dataset": self.config.dataset_name})

    def _raw_records_from_dataset_server(self) -> Iterator[Mapping[str, Any]]:
        """Read only the server's bounded first-row window; no Parquet download."""

        if self.config.sample_size == 0:
            return
        if self.config.sample_size is not None and self.config.sample_size > 100:
            raise DatasetLoadError(
                "dataset_server backend supports at most 100 rows per bounded request; "
                "use backend='hf_datasets' for larger streaming samples"
            )
        length = min(self.config.sample_size or 100, 100)
        query = urlencode(
            {
                "dataset": self.config.dataset_name,
                "config": self.config.dataset_config,
                "split": self.config.split,
            }
        )
        url = f"{self.config.dataset_server_url}/first-rows?{query}"
        try:
            with urlopen(url, timeout=60) as response:
                payload = json.load(response)
        except Exception as exc:
            raise DatasetLoadError(
                f"Unable to query dataset-server for {self.config.dataset_name!r}, "
                f"config={self.config.dataset_config!r}, split={self.config.split!r}: {exc}"
            ) from exc
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            raise DatasetLoadError("dataset-server returned an invalid rows payload")
        for item in rows[:length]:
            row = item.get("row") if isinstance(item, Mapping) else None
            if isinstance(row, Mapping):
                self.last_raw_fields.update(row.keys())
                yield row

    def available_fields(self) -> list[str]:
        return sorted(self.last_raw_fields)

    def load_documents(self) -> tuple[list[NormalizedDocument], LoadStats]:
        stats = LoadStats()
        candidates: list[NormalizedDocument] = []
        logger.info(
            "dataset_load_started",
            extra={
                "dataset": self.config.dataset_name,
                "config": self.config.dataset_config,
                "split": self.config.split,
                "sample_size": self.config.sample_size,
            },
        )
        for row_index, record in enumerate(self.raw_records()):
            stats.records_read += 1
            try:
                documents, removed_empty = normalize_record(
                    record,
                    dataset_name=self.config.dataset_name,
                    dataset_config=self.config.dataset_config,
                    split=self.config.split,
                    row_index=row_index,
                )
            except Exception:
                stats.malformed_records += 1
                logger.exception("record_normalization_failed", extra={"dataset": self.config.dataset_name})
                continue
            candidates.extend(documents)
            stats.empty_passages_removed += removed_empty
        documents, stats.duplicate_documents_removed = deduplicate_documents(candidates)
        stats.documents_created = len(documents)
        logger.info(
            "dataset_load_completed",
            extra={
                "dataset": self.config.dataset_name,
                "config": self.config.dataset_config,
                "split": self.config.split,
                "records": stats.records_read,
                "documents": stats.documents_created,
            },
        )
        return documents, stats

    def available_configurations(self) -> list[str]:
        if self.config.backend == "dataset_server":
            query = urlencode({"dataset": self.config.dataset_name})
            try:
                with urlopen(
                    f"{self.config.dataset_server_url}/info?{query}", timeout=60
                ) as response:
                    payload = json.load(response)
                return sorted(payload.get("dataset_info", {}).keys())
            except Exception as exc:
                raise DatasetLoadError(
                    f"Unable to discover configurations for {self.config.dataset_name!r}: {exc}"
                ) from exc
        try:
            from datasets import get_dataset_config_names
        except ImportError as exc:
            raise DatasetLoadError(
                "The 'datasets' package is required to discover configurations."
            ) from exc
        try:
            return list(get_dataset_config_names(self.config.dataset_name))
        except Exception as exc:
            raise DatasetLoadError(
                f"Unable to discover configurations for {self.config.dataset_name!r}: {exc}"
            ) from exc
