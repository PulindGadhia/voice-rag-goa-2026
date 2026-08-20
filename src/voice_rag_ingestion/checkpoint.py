"""Ingestion checkpointing and progress recovery for large dataset streaming."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path(".cache/ingestion_checkpoint.json")


@dataclass
class IngestionCheckpoint:
    """Structured state of streaming ingestion progress."""

    dataset_name: str
    dataset_config: str
    split: str
    last_processed_row_index: int = -1
    records_read: int = 0
    documents_processed: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    qdrant_chunks_indexed: int = 0
    bm25_chunks_indexed: int = 0
    languages_processed: list[str] = field(default_factory=list)
    failures: int = 0
    elapsed_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Parquet-specific resume fields (optional; absent in HF-streaming checkpoints)
    parquet_completed_files: list[str] = field(default_factory=list)
    parquet_current_file: str | None = None

    def matches(self, dataset_name: str, dataset_config: str, split: str) -> bool:
        return (
            self.dataset_name == dataset_name
            and self.dataset_config == dataset_config
            and self.split == split
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestionCheckpoint:
        return cls(
            dataset_name=str(data.get("dataset_name", "")),
            dataset_config=str(data.get("dataset_config", "default")),
            split=str(data.get("split", "validation")),
            last_processed_row_index=int(data.get("last_processed_row_index", -1)),
            records_read=int(data.get("records_read", 0)),
            documents_processed=int(data.get("documents_processed", 0)),
            documents_skipped=int(data.get("documents_skipped", 0)),
            chunks_created=int(data.get("chunks_created", 0)),
            qdrant_chunks_indexed=int(data.get("qdrant_chunks_indexed", 0)),
            bm25_chunks_indexed=int(data.get("bm25_chunks_indexed", 0)),
            languages_processed=list(data.get("languages_processed", [])),
            failures=int(data.get("failures", 0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            parquet_completed_files=list(data.get("parquet_completed_files", [])),
            parquet_current_file=data.get("parquet_current_file"),
        )


def load_checkpoint(path: Path | str = DEFAULT_CHECKPOINT_PATH) -> IngestionCheckpoint | None:
    """Load checkpoint from disk if it exists and is valid JSON."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        checkpoint = IngestionCheckpoint.from_dict(data)
        logger.info(
            "ingestion_checkpoint_loaded",
            extra={
                "path": str(file_path),
                "last_processed_row": checkpoint.last_processed_row_index,
                "chunks_indexed": checkpoint.qdrant_chunks_indexed,
            },
        )
        return checkpoint
    except Exception as exc:
        logger.warning(
            "ingestion_checkpoint_corrupt_or_unreadable",
            extra={"path": str(file_path), "error": str(exc)},
        )
        return None


def save_checkpoint(
    path: Path | str = DEFAULT_CHECKPOINT_PATH,
    checkpoint: IngestionCheckpoint | None = None,
) -> None:
    """Atomically write checkpoint to disk."""
    if checkpoint is None:
        return
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.timestamp = datetime.now(timezone.utc).isoformat()
    temp_path = file_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
    temp_path.replace(file_path)


def reset_checkpoint(path: Path | str = DEFAULT_CHECKPOINT_PATH) -> None:
    """Remove checkpoint file if it exists."""
    file_path = Path(path)
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info("ingestion_checkpoint_reset", extra={"path": str(file_path)})
        except OSError:
            pass
