"""Configuration for the ingestion layer."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any


_ENV_LOADED = False


def _load_local_env() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            pass
        _ENV_LOADED = True


def _bool_env(name: str, default: bool) -> bool:
    _load_local_env()
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
    if value is None:
        return default
    val_str = value.strip().lower()
    if not val_str or val_str in {"none", "all", "null"}:
        return None
    parsed = int(val_str)
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

    def with_resolved_backend(
        self,
        *,
        sample_size: int | None | object = ...,
        backend: str | None = None,
        **overrides: Any,
    ) -> "LoaderConfig":
        """Return a new LoaderConfig with sample_size and backend resolved deterministically."""
        target_sample_size = (
            self.sample_size if sample_size is ... else sample_size
        )
        env_backend = os.getenv("HF_LOADER_BACKEND") or os.getenv("DATASET_BACKEND")
        if backend is not None:
            resolved_backend = backend.strip().lower()
        elif env_backend:
            resolved_backend = env_backend.strip().lower()
        else:
            if target_sample_size is None or target_sample_size > 100:
                resolved_backend = "hf_datasets"
            else:
                resolved_backend = "dataset_server"

        return replace(
            self,
            sample_size=target_sample_size,
            backend=resolved_backend,
            **overrides,
        )

    def resolve_backend(self) -> "LoaderConfig":
        """Re-resolve backend based on current sample_size and environment."""
        return self.with_resolved_backend()

    @classmethod
    def from_env(
        cls,
        *,
        sample_size: int | None | object = ...,
        backend: str | None = None,
        **overrides: Any,
    ) -> "LoaderConfig":
        _load_local_env()
        env_sample_size = _int_env("HF_SAMPLE_SIZE", 10)
        target_sample_size = (
            env_sample_size if sample_size is ... else sample_size
        )
        env_backend = os.getenv("HF_LOADER_BACKEND") or os.getenv("DATASET_BACKEND")
        if backend is not None:
            resolved_backend = backend.strip().lower()
        elif env_backend:
            resolved_backend = env_backend.strip().lower()
        else:
            if target_sample_size is None or target_sample_size > 100:
                resolved_backend = "hf_datasets"
            else:
                resolved_backend = cls.backend

        base = cls(
            dataset_name=os.getenv("HF_DATASET_NAME", cls.dataset_name),
            dataset_config=os.getenv("HF_DATASET_CONFIG", cls.dataset_config),
            split=os.getenv("HF_DATASET_SPLIT", cls.split),
            sample_size=target_sample_size,
            streaming=_bool_env("HF_STREAMING", cls.streaming),
            development_mode=_bool_env(
                "INGESTION_DEVELOPMENT_MODE", cls.development_mode
            ),
            revision=os.getenv("HF_REVISION", cls.revision) or None,
            cache_dir=os.getenv("HF_CACHE_DIR") or None,
            trust_remote_code=_bool_env(
                "HF_TRUST_REMOTE_CODE", cls.trust_remote_code
            ),
            backend=resolved_backend,
            dataset_server_url=os.getenv(
                "HF_DATASET_SERVER_URL", cls.dataset_server_url
            ).rstrip("/"),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
        )
        if overrides:
            return base.with_resolved_backend(**overrides)
        return base
