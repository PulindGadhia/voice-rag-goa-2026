"""Configuration for the ingestion layer."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _int_env(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0, got {parsed}")
    return parsed


@dataclass(frozen=True)
class LoaderConfig:
    """Runtime settings; no credentials are stored here."""

    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_config: str = "default"
    split: str = "validation"
    sample_size: int | None = 10
    streaming: bool = True
    development_mode: bool = True
    revision: str | None = "main"
    cache_dir: str | None = None
    trust_remote_code: bool = False
    backend: str = "dataset_server"
    dataset_server_url: str = "https://datasets-server.huggingface.co"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name cannot be empty")
        if not self.dataset_config.strip():
            raise ValueError("dataset_config cannot be empty")
        if not self.split.strip():
            raise ValueError("split cannot be empty")
        if self.sample_size is not None and self.sample_size < 0:
            raise ValueError("sample_size must be >= 0 or None")
        if self.development_mode and not self.streaming:
            raise ValueError("development_mode requires streaming=True")
        if self.backend not in {"dataset_server", "hf_datasets"}:
            raise ValueError("backend must be 'dataset_server' or 'hf_datasets'")

    @classmethod
    def from_env(cls) -> "LoaderConfig":
        sample_size = _int_env("HF_SAMPLE_SIZE", 10)
        return cls(
            dataset_name=os.getenv("HF_DATASET_NAME", cls.dataset_name),
            dataset_config=os.getenv("HF_DATASET_CONFIG", cls.dataset_config),
            split=os.getenv("HF_DATASET_SPLIT", cls.split),
            sample_size=sample_size,
            streaming=_bool_env("HF_STREAMING", cls.streaming),
            development_mode=_bool_env(
                "INGESTION_DEVELOPMENT_MODE", cls.development_mode
            ),
            revision=os.getenv("HF_REVISION", cls.revision) or None,
            cache_dir=os.getenv("HF_CACHE_DIR") or None,
            trust_remote_code=_bool_env(
                "HF_TRUST_REMOTE_CODE", cls.trust_remote_code
            ),
            backend=os.getenv("HF_LOADER_BACKEND", cls.backend),
            dataset_server_url=os.getenv(
                "HF_DATASET_SERVER_URL", cls.dataset_server_url
            ).rstrip("/"),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
        )
