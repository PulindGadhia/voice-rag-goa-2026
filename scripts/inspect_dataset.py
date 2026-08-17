#!/usr/bin/env python3
"""Inspect a bounded MSMARCO-XI sample and its normalized representation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.loader import DatasetLoadError, DatasetLoader
from voice_rag_ingestion.logging_utils import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    env_config = LoaderConfig.from_env()
    config = LoaderConfig(
        dataset_name=env_config.dataset_name,
        dataset_config=args.config or env_config.dataset_config,
        split=args.split or env_config.split,
        sample_size=args.sample_size if args.sample_size is not None else env_config.sample_size,
        streaming=True,
        development_mode=True,
        revision=env_config.revision,
        cache_dir=env_config.cache_dir,
        trust_remote_code=env_config.trust_remote_code,
        backend=env_config.backend,
        dataset_server_url=env_config.dataset_server_url,
        log_level=env_config.log_level,
    )
    configure_logging(config.log_level)
    loader = DatasetLoader(config)
    try:
        available = loader.available_configurations()
        documents, stats = loader.load_documents()
    except DatasetLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Dataset configuration")
    print(json.dumps({
        "dataset_name": config.dataset_name,
        "selected_config": config.dataset_config,
        "available_configs": available,
        "split": config.split,
        "sample_size": config.sample_size,
        "streaming": config.streaming,
        "development_mode": config.development_mode,
        "backend": config.backend,
    }, ensure_ascii=False, indent=2))
    print("\nRaw/normalized record statistics")
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))
    print(f"Raw fields: {loader.available_fields()}")
    print(f"\nNormalized fields: {list(documents[0].to_dict()) if documents else []}")
    print(f"Languages: {dict(Counter(document.language for document in documents))}")
    print("\nNormalized samples")
    for document in documents[:3]:
        print(json.dumps(document.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
