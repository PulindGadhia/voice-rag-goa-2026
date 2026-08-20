#!/usr/bin/env python3
"""Run all chunking strategies on a bounded live ingestion sample."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_rag_ingestion.chunking import ChunkingConfig, chunk_document
from voice_rag_ingestion.config import LoaderConfig
from voice_rag_ingestion.loader import DatasetLoadError, DatasetLoader
from voice_rag_ingestion.logging_utils import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--max-chunk-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=8)
    args = parser.parse_args()
    config = LoaderConfig.from_env(sample_size=args.sample_size)
    configure_logging(config.log_level)
    try:
        documents, stats = DatasetLoader(config).load_documents()
    except DatasetLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Documents processed: {len(documents)} (raw records: {stats.records_read})")
    for strategy in ("fixed", "sentence", "semantic", "metadata"):
        chunk_config = ChunkingConfig(
            max_chunk_size=args.max_chunk_size,
            overlap=args.overlap,
            strategy=strategy,
        )
        chunks = [chunk for document in documents for chunk in chunk_document(document, config=chunk_config)]
        lengths = [len(chunk.text.split()) for chunk in chunks]
        per_document = Counter(chunk.document_id for chunk in chunks)
        print(f"\n[{strategy}]")
        print(f"chunks generated: {len(chunks)}")
        print(f"average chunk length: {sum(lengths) / len(lengths) if lengths else 0:.2f} words")
        print(f"minimum chunk length: {min(lengths) if lengths else 0} words")
        print(f"maximum chunk length: {max(lengths) if lengths else 0} words")
        print(f"chunks per document: {dict(per_document)}")
        for chunk in chunks[:2]:
            print(f"sample {chunk.chunk_id}: {chunk.text[:240]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
