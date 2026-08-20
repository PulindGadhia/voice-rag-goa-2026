"""Reusable normalized-document to vector-index pipeline with streaming and checkpointing support."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from .bm25 import BM25Index
from .bm25_sqlite import BM25SqliteIndex
from .checkpoint import (
    DEFAULT_CHECKPOINT_PATH,
    IngestionCheckpoint,
    load_checkpoint,
    reset_checkpoint,
    save_checkpoint,
)
from .chunking import ChunkingConfig, chunk_document
from .chunking.base import Chunk
from .documents import NormalizedDocument, deduplicate_documents, normalize_record
from .embeddings.base import EmbeddingProvider, timed_batch_embed
from .loader import DatasetLoader
from .qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    embedding_batches: int = 0
    embedding_seconds: float = 0.0
    upserted: int = 0


@dataclass
class StreamingIndexStats:
    records_read: int = 0
    documents_processed: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    qdrant_chunks_indexed: int = 0
    bm25_chunks_indexed: int = 0
    languages_processed: set[str] = field(default_factory=set)
    failures: int = 0
    elapsed_seconds: float = 0.0
    embedding_seconds: float = 0.0

    @property
    def throughput_chunks_per_sec(self) -> float:
        total = self.qdrant_chunks_indexed or self.bm25_chunks_indexed or self.chunks_created
        return (
            (total / self.elapsed_seconds)
            if self.elapsed_seconds > 0
            else 0.0
        )


def index_documents(
    documents: Sequence[NormalizedDocument],
    *,
    embedder: EmbeddingProvider,
    vector_store: QdrantVectorStore,
    chunking_config: ChunkingConfig,
    embedding_batch_size: int,
    recreate_collection: bool | None = None,
) -> tuple[IndexStats, list[Chunk]]:
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, config=chunking_config)
    ]
    vector_store.ensure_collection(embedder.dimension, recreate=recreate_collection)
    vectors, embedding_stats = timed_batch_embed(
        embedder,
        [chunk.text for chunk in chunks],
        batch_size=embedding_batch_size,
        input_type="passage",
    )
    upserted = vector_store.upsert(chunks, vectors)
    return (
        IndexStats(
            documents=len(documents),
            chunks=len(chunks),
            vectors=len(vectors),
            embedding_batches=embedding_stats.batches,
            embedding_seconds=embedding_stats.elapsed_seconds,
            upserted=upserted,
        ),
        chunks,
    )


def stream_index_dataset(
    loader: DatasetLoader,
    *,
    embedder: EmbeddingProvider | None = None,
    vector_store: QdrantVectorStore | None = None,
    bm25_index: BM25Index | None = None,
    bm25_path: Path | str | None = None,
    chunking_config: ChunkingConfig | None = None,
    stream_batch_size: int = 250,
    embedding_batch_size: int = 64,
    recreate_collection: bool = False,
    languages: Sequence[str] | set[str] | None = None,
    checkpoint_path: Path | str | None = None,
    resume: bool = True,
    bm25_only: bool = False,
    progress_callback: Callable[[StreamingIndexStats], None] | None = None,
) -> StreamingIndexStats:
    """Stream dataset records, normalize, chunk, embed (optional), and index incrementally in memory-bounded batches with checkpointing."""

    config = chunking_config or ChunkingConfig.from_env()
    stats = StreamingIndexStats()
    start_time = perf_counter()
    allowed_langs = {lang.strip() for lang in languages} if languages else None

    # Handle checkpoint resetting on recreate or resuming
    resume_after_row = -1
    if recreate_collection:
        if checkpoint_path:
            reset_checkpoint(checkpoint_path)
    elif resume and checkpoint_path:
        cp = load_checkpoint(checkpoint_path)
        if cp is not None and cp.matches(
            loader.config.dataset_name,
            loader.config.dataset_config,
            loader.config.split,
        ):
            resume_after_row = cp.last_processed_row_index
            stats.records_read = cp.records_read
            stats.documents_processed = cp.documents_processed
            stats.documents_skipped = cp.documents_skipped
            stats.chunks_created = cp.chunks_created
            stats.qdrant_chunks_indexed = cp.qdrant_chunks_indexed
            stats.bm25_chunks_indexed = cp.bm25_chunks_indexed
            stats.languages_processed = set(cp.languages_processed)
            stats.failures = cp.failures
            stats.elapsed_seconds = cp.elapsed_seconds
            logger.info(
                "resuming_stream_ingestion",
                extra={
                    "resume_after_row": resume_after_row,
                    "records_read": stats.records_read,
                    "bm25_indexed": stats.bm25_chunks_indexed,
                    "qdrant_indexed": stats.qdrant_chunks_indexed,
                },
            )

    # Ensure Qdrant collection is ready if dense indexing is enabled
    if not bm25_only and vector_store is not None and embedder is not None:
        vector_store.ensure_collection(embedder.dimension, recreate=recreate_collection)

    batch_docs: list[NormalizedDocument] = []
    last_row_index = resume_after_row

    def process_batch(documents: list[NormalizedDocument], current_row_index: int) -> None:
        if not documents:
            return
        unique_docs, removed_dups = deduplicate_documents(documents)
        stats.documents_skipped += removed_dups

        chunks: list[Chunk] = []
        for doc in unique_docs:
            try:
                doc_chunks = chunk_document(doc, config=config)
                chunks.extend(doc_chunks)
            except Exception:
                stats.failures += 1
                logger.exception("chunking_failed", extra={"document_id": doc.document_id})

        stats.chunks_created += len(chunks)
        if not chunks:
            return

        # Incremental Embedding & Vector Upsert (skipped in bm25_only mode)
        if not bm25_only and vector_store is not None and embedder is not None:
            t_embed_0 = perf_counter()
            vectors, _ = timed_batch_embed(
                embedder,
                [c.text for c in chunks],
                batch_size=embedding_batch_size,
                input_type="passage",
            )
            stats.embedding_seconds += (perf_counter() - t_embed_0)

            upserted = vector_store.upsert(chunks, vectors)
            stats.qdrant_chunks_indexed += upserted

        # Incremental BM25 Inverted Index update
        if bm25_index is not None:
            added_bm25 = bm25_index.add(chunks)
            stats.bm25_chunks_indexed += added_bm25
            if bm25_path:
                bm25_index.save(bm25_path)

        # Update and persist checkpoint after successful batch completion
        stats.elapsed_seconds = perf_counter() - start_time
        if checkpoint_path:
            cp = IngestionCheckpoint(
                dataset_name=loader.config.dataset_name,
                dataset_config=loader.config.dataset_config,
                split=loader.config.split,
                last_processed_row_index=current_row_index,
                records_read=stats.records_read,
                documents_processed=stats.documents_processed,
                documents_skipped=stats.documents_skipped,
                chunks_created=stats.chunks_created,
                qdrant_chunks_indexed=stats.qdrant_chunks_indexed,
                bm25_chunks_indexed=stats.bm25_chunks_indexed,
                languages_processed=sorted(stats.languages_processed),
                failures=stats.failures,
                elapsed_seconds=stats.elapsed_seconds,
            )
            save_checkpoint(checkpoint_path, cp)

    try:
        for row_index, record in enumerate(loader.raw_records()):
            if row_index <= resume_after_row:
                continue

            last_row_index = row_index
            stats.records_read += 1
            try:
                docs, empty_removed = normalize_record(
                    record,
                    dataset_name=loader.config.dataset_name,
                    dataset_config=loader.config.dataset_config,
                    split=loader.config.split,
                    row_index=row_index,
                )
                stats.documents_skipped += empty_removed
            except Exception:
                stats.failures += 1
                logger.exception("record_normalization_failed")
                continue

            for doc in docs:
                if allowed_langs and doc.language and doc.language not in allowed_langs:
                    stats.documents_skipped += 1
                    continue
                if doc.language:
                    stats.languages_processed.add(doc.language)
                batch_docs.append(doc)
                stats.documents_processed += 1

                if len(batch_docs) >= stream_batch_size:
                    process_batch(batch_docs, last_row_index)
                    batch_docs.clear()
                    if progress_callback:
                        progress_callback(stats)

        # Process any remaining documents in trailing batch
        if batch_docs:
            process_batch(batch_docs, last_row_index)
            batch_docs.clear()

    finally:
        stats.elapsed_seconds = perf_counter() - start_time
        if progress_callback:
            progress_callback(stats)

    return stats


PARQUET_LANG_NAMES: dict[str, str] = {
    "asm": "Assamese (অসমীয়া)",
    "ben": "Bengali (বাংলা)",
    "guj": "Gujarati (ગુજરાતી)",
    "hin": "Hindi (हिन्दी)",
    "kan": "Kannada (ಕನ್ನಡ)",
    "mal": "Malayalam (മലയാളം)",
    "mar": "Marathi (मराठी)",
    "nep": "Nepali (नेपाली)",
    "ori": "Odia (ଓଡ଼ିଆ)",
    "pan": "Punjabi (ਪੰਜਾਬੀ)",
    "san": "Sanskrit (संस्कृत)",
    "tam": "Tamil (தமிழ்)",
    "tel": "Telugu (తెలుగు)",
    "urd": "Urdu (اردو)",
}


def _get_lang_display_name(stem: str) -> str:
    prefix = stem[:3].lower()
    return PARQUET_LANG_NAMES.get(prefix, stem.capitalize())


def _format_eta(seconds: float) -> str:
    import math

    if seconds <= 0 or not math.isfinite(seconds):
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def _process_parquet_file_worker(worker_input: dict[str, Any]) -> dict[str, Any]:
    """Process a single parquet file into an isolated temporary SQLite BM25 database.

    Runs in a separate worker process safely under multiprocessing (spawn or fork).
    """
    import json
    import sqlite3
    from collections import Counter
    from pathlib import Path
    from time import perf_counter
    import pyarrow.parquet as pq
    from .chunking import ChunkingConfig, chunk_document
    from .documents import deduplicate_documents, normalize_record
    from .tokenization import TokenizerConfig, UnicodeWordTokenizer

    parquet_path = Path(worker_input["parquet_path"])
    temp_db_path = Path(worker_input["temp_db_path"])
    dataset_name = worker_input.get("dataset_name", "ai4bharat/MSMARCO-XI")
    dataset_config = worker_input.get("dataset_config", "default")
    split = worker_input.get("split", "validation")
    batch_size = worker_input.get("batch_size", 5000)
    sample_size = worker_input.get("sample_size")
    chunking_config = ChunkingConfig(
        strategy=worker_input.get("strategy", "metadata"),
        max_chunk_size=worker_input.get("max_chunk_size", 256),
        overlap=worker_input.get("overlap", 32),
        min_chunk_size=worker_input.get("min_chunk_size", 10),
    )
    allowed_langs = worker_input.get("allowed_langs")
    progress_queue = worker_input.get("progress_queue")

    stem = parquet_path.stem
    lang_name = _get_lang_display_name(stem)

    temp_db_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_db_path.exists():
        temp_db_path.unlink()
    for suffix in ["-wal", "-shm"]:
        w_f = temp_db_path.with_name(temp_db_path.name + suffix)
        if w_f.exists():
            w_f.unlink()

    conn = sqlite3.connect(str(temp_db_path))
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")
    conn.execute("PRAGMA locking_mode = EXCLUSIVE")

    conn.execute("""
        CREATE TABLE chunks (
            chunk_id        TEXT PRIMARY KEY,
            document_id     TEXT NOT NULL,
            text            TEXT NOT NULL,
            language        TEXT,
            chunk_index     INTEGER,
            chunk_strategy  TEXT,
            parent_chunk_id TEXT,
            query_id        TEXT,
            source          TEXT,
            metadata        TEXT,
            doc_length      INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE term_frequencies (
            chunk_id TEXT NOT NULL,
            term     TEXT NOT NULL,
            count    INTEGER NOT NULL,
            PRIMARY KEY (chunk_id, term)
        ) WITHOUT ROWID
    """)
    conn.execute("""
        CREATE TABLE postings (
            term     TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            PRIMARY KEY (term, chunk_id)
        ) WITHOUT ROWID
    """)
    conn.execute("""
        CREATE TABLE stats (
            id                       INTEGER PRIMARY KEY DEFAULT 1,
            schema_version           INTEGER NOT NULL DEFAULT 1,
            total_chunks             INTEGER NOT NULL DEFAULT 0,
            total_document_length    INTEGER NOT NULL DEFAULT 0,
            k1                       REAL    NOT NULL DEFAULT 1.5,
            b                        REAL    NOT NULL DEFAULT 0.75,
            tokenizer_lowercase      INTEGER NOT NULL DEFAULT 1,
            tokenizer_normalization  TEXT    NOT NULL DEFAULT 'NFC',
            tokenizer_min_length     INTEGER NOT NULL DEFAULT 1
        )
    """)

    tokenizer = UnicodeWordTokenizer(TokenizerConfig())
    tokenize = tokenizer.tokenize

    t_start = perf_counter()
    records_read = 0
    documents_processed = 0
    documents_skipped = 0
    chunks_created = 0
    total_doc_length = 0
    languages_processed: set[str] = set()
    failures = 0

    pf = pq.ParquetFile(str(parquet_path))
    total_file_rows = pf.metadata.num_rows

    try:
        sample_size_reached = False
        for arrow_batch in pf.iter_batches(batch_size=batch_size):
            if sample_size_reached:
                break
            records = arrow_batch.to_pylist()
            batch_docs = []
            for row_dict in records:
                if sample_size is not None and records_read >= sample_size:
                    sample_size_reached = True
                    break
                row_index = records_read
                records_read += 1
                try:
                    docs, empty_removed = normalize_record(
                        row_dict,
                        dataset_name=dataset_name,
                        dataset_config=dataset_config,
                        split=split,
                        row_index=row_index,
                    )
                    documents_skipped += empty_removed
                except Exception:
                    failures += 1
                    continue

                for doc in docs:
                    if allowed_langs and doc.language and doc.language not in allowed_langs:
                        documents_skipped += 1
                        continue
                    if doc.language:
                        languages_processed.add(doc.language)
                    batch_docs.append(doc)
                    documents_processed += 1

            if not batch_docs:
                continue

            unique_docs, removed_dups = deduplicate_documents(batch_docs)
            documents_skipped += removed_dups

            chunks = []
            for doc in unique_docs:
                try:
                    chunks.extend(chunk_document(doc, config=chunking_config))
                except Exception:
                    failures += 1

            if not chunks:
                continue

            chunk_rows = []
            tf_rows = []
            postings_rows = []
            batch_doc_len = 0

            for chunk in chunks:
                if not chunk.chunk_id or not chunk.text or not chunk.text.strip():
                    continue
                tokens = tokenize(chunk.text)
                counts = Counter(tokens)
                doc_len = len(tokens)
                batch_doc_len += doc_len

                chunk_rows.append((
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.text,
                    chunk.language,
                    chunk.chunk_index,
                    chunk.chunk_strategy,
                    chunk.parent_chunk_id,
                    chunk.query_id,
                    json.dumps(dict(chunk.source), ensure_ascii=False) if chunk.source else "{}",
                    json.dumps(dict(chunk.metadata), ensure_ascii=False) if chunk.metadata else "{}",
                    doc_len,
                ))
                for term, count in counts.items():
                    tf_rows.append((chunk.chunk_id, term, count))
                    postings_rows.append((term, chunk.chunk_id))

            cur = conn.cursor()
            cur.execute("BEGIN TRANSACTION")
            cur.executemany("INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", chunk_rows)
            cur.executemany("INSERT OR IGNORE INTO term_frequencies VALUES (?, ?, ?)", tf_rows)
            cur.executemany("INSERT OR IGNORE INTO postings VALUES (?, ?)", postings_rows)
            cur.execute("COMMIT")

            chunks_created += len(chunk_rows)
            total_doc_length += batch_doc_len

            elapsed = perf_counter() - t_start
            speed = chunks_created / elapsed if elapsed > 0 else 0
            remaining_records = max(0, total_file_rows - records_read)
            est_remaining_sec = (remaining_records / (records_read / elapsed)) if records_read > 0 and elapsed > 0 else 0

            if progress_queue is not None:
                try:
                    progress_queue.put({
                        "event": "progress",
                        "file_name": parquet_path.name,
                        "lang_name": lang_name,
                        "records_read": records_read,
                        "total_file_rows": total_file_rows,
                        "chunks_created": chunks_created,
                        "chunks_per_sec": speed,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": est_remaining_sec,
                    })
                except Exception:
                    pass

        # Save stats table in worker database
        conn.execute(
            "INSERT INTO stats (id, schema_version, total_chunks, total_document_length, "
            "k1, b, tokenizer_lowercase, tokenizer_normalization, tokenizer_min_length) "
            "VALUES (1, 1, ?, ?, 1.5, 0.75, 1, 'NFC', 1)",
            (chunks_created, total_doc_length),
        )
        conn.commit()
        conn.close()

        elapsed = perf_counter() - t_start
        speed = chunks_created / elapsed if elapsed > 0 else 0

        res = {
            "file_name": parquet_path.name,
            "lang_name": lang_name,
            "records_read": records_read,
            "documents_processed": documents_processed,
            "documents_skipped": documents_skipped,
            "chunks_created": chunks_created,
            "total_doc_length": total_doc_length,
            "languages_processed": list(languages_processed),
            "failures": failures,
            "elapsed_seconds": elapsed,
            "chunks_per_sec": speed,
            "db_path": str(temp_db_path),
            "success": True,
            "error": None,
        }

        if progress_queue is not None:
            try:
                progress_queue.put({
                    "event": "file_complete",
                    **res,
                })
            except Exception:
                pass

        return res

    except Exception as exc:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        err_res = {
            "file_name": parquet_path.name,
            "lang_name": lang_name,
            "records_read": records_read,
            "documents_processed": documents_processed,
            "documents_skipped": documents_skipped,
            "chunks_created": chunks_created,
            "total_doc_length": total_doc_length,
            "languages_processed": list(languages_processed),
            "failures": failures + 1,
            "elapsed_seconds": perf_counter() - t_start,
            "chunks_per_sec": 0.0,
            "db_path": str(temp_db_path),
            "success": False,
            "error": str(exc),
        }
        if progress_queue is not None:
            try:
                progress_queue.put({
                    "event": "file_error",
                    **err_res,
                })
            except Exception:
                pass
        return err_res


def _parallel_stream_index_from_parquet(
    parquet_dir: Path,
    parquet_files: list[Path],
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    bm25_index: BM25Index | BM25SqliteIndex | None,
    bm25_path: Path | str | None,
    chunking_config: ChunkingConfig,
    workers: int,
    recreate_collection: bool,
    languages: Sequence[str] | set[str] | None,
    checkpoint_path: Path | str | None,
    resume: bool,
    sample_size: int | None,
    parquet_batch_size: int,
    progress_callback: Callable[[StreamingIndexStats], None] | None,
    fast_ingest: bool,
) -> StreamingIndexStats:
    import math
    import multiprocessing as mp
    import os
    import sys
    import threading
    from time import perf_counter

    start_time = perf_counter()
    stats = StreamingIndexStats()
    allowed_langs = {lang.strip() for lang in languages} if languages else None

    # Checkpoint handling
    completed_files: set[str] = set()
    if recreate_collection and checkpoint_path:
        reset_checkpoint(checkpoint_path)
    elif resume and checkpoint_path:
        cp = load_checkpoint(checkpoint_path)
        if cp is not None and cp.matches(dataset_name, dataset_config, split):
            completed_files = set(cp.parquet_completed_files)
            stats.records_read = cp.records_read
            stats.documents_processed = cp.documents_processed
            stats.documents_skipped = cp.documents_skipped
            stats.chunks_created = cp.chunks_created
            stats.bm25_chunks_indexed = cp.bm25_chunks_indexed
            stats.languages_processed = set(cp.languages_processed)
            stats.failures = cp.failures

    pending_files = [f for f in parquet_files if f.name not in completed_files]
    if not pending_files:
        stats.elapsed_seconds = perf_counter() - start_time
        return stats

    temp_base_dir = Path(".cache/bm25_workers")
    temp_base_dir.mkdir(parents=True, exist_ok=True)

    num_workers = min(len(pending_files), workers) if workers > 1 else min(len(pending_files), os.cpu_count() or 4, 6)

    target_bm25_db = Path(bm25_path or (bm25_index.db_path if isinstance(bm25_index, BM25SqliteIndex) else ".cache/bm25_index.db"))

    # Prepare worker tasks
    tasks = []
    per_file_sample = math.ceil(sample_size / len(pending_files)) if sample_size else None

    for pf in pending_files:
        stem = pf.stem
        temp_db_path = temp_base_dir / f"{stem}.db"
        tasks.append({
            "parquet_path": str(pf),
            "temp_db_path": str(temp_db_path),
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "strategy": chunking_config.strategy,
            "max_chunk_size": chunking_config.max_chunk_size,
            "overlap": chunking_config.overlap,
            "min_chunk_size": chunking_config.min_chunk_size,
            "batch_size": 5000 if fast_ingest else parquet_batch_size,
            "sample_size": per_file_sample,
            "allowed_langs": list(allowed_langs) if allowed_langs else None,
        })

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress_queue = manager.Queue()

    for task in tasks:
        task["progress_queue"] = progress_queue

    # Background progress printer
    stop_event = threading.Event()
    last_print_time: dict[str, float] = {}

    def _monitor():
        while not stop_event.is_set():
            try:
                msg = progress_queue.get(timeout=0.2)
            except Exception:
                continue
            if msg is None:
                break
            if msg.get("event") == "progress":
                lang = msg["lang_name"]
                now = perf_counter()
                if now - last_print_time.get(lang, 0) >= 2.0:
                    last_print_time[lang] = now
                    eta_str = _format_eta(msg["eta_seconds"])
                    print(
                        f"\n------------------------------------------------------------\n"
                        f"Language:            {lang}\n"
                        f"Processed documents: {msg['records_read']:,}/{msg['total_file_rows']:,}\n"
                        f"Created chunks:      {msg['chunks_created']:,}\n"
                        f"Chunks/sec:          {msg['chunks_per_sec']:.1f} chunks/sec\n"
                        f"Estimated remaining: {eta_str}\n"
                        f"------------------------------------------------------------"
                    )
                    sys.stdout.flush()
            elif msg.get("event") == "file_complete":
                lang = msg["lang_name"]
                print(
                    f"\n[Completed {lang}] Chunks: {msg['chunks_created']:,} in {msg['elapsed_seconds']:.1f}s ({msg['chunks_per_sec']:.1f} chunks/sec)"
                )
                sys.stdout.flush()

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    logger.info(
        "parallel_ingestion_started",
        extra={
            "num_workers": num_workers,
            "num_files": len(pending_files),
            "target_db": str(target_bm25_db),
        },
    )

    worker_results = []
    try:
        with ctx.Pool(processes=num_workers) as pool:
            worker_results = pool.map(_process_parquet_file_worker, tasks)
    finally:
        stop_event.set()
        try:
            progress_queue.put(None)
        except Exception:
            pass
        monitor_thread.join(timeout=2.0)
        try:
            manager.shutdown()
        except Exception:
            pass

    # Collect results and merge SQLite DBs
    source_dbs = []
    new_completed = set(completed_files)
    for r in worker_results:
        stats.records_read += r.get("records_read", 0)
        stats.documents_processed += r.get("documents_processed", 0)
        stats.documents_skipped += r.get("documents_skipped", 0)
        stats.chunks_created += r.get("chunks_created", 0)
        stats.failures += r.get("failures", 0)
        for lang_code in r.get("languages_processed", []):
            stats.languages_processed.add(lang_code)
        if r.get("success"):
            source_dbs.append(r["db_path"])
            if not sample_size:
                new_completed.add(r["file_name"])

    print(f"\nMerging {len(source_dbs)} language databases into {target_bm25_db}...")
    t_merge_0 = perf_counter()
    BM25SqliteIndex.merge_language_databases(
        target_bm25_db,
        source_dbs,
        cleanup_sources=True,
    )
    t_merge_sec = perf_counter() - t_merge_0
    print(f"Merge completed in {t_merge_sec:.2f}s.")

    # Reopen to verify size
    merged_idx = BM25SqliteIndex.load(target_bm25_db)
    stats.bm25_chunks_indexed = merged_idx.size
    merged_idx.close()

    stats.elapsed_seconds = perf_counter() - start_time

    # Save final checkpoint
    if checkpoint_path:
        cp = IngestionCheckpoint(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            last_processed_row_index=max(stats.records_read - 1, -1),
            records_read=stats.records_read,
            documents_processed=stats.documents_processed,
            documents_skipped=stats.documents_skipped,
            chunks_created=stats.chunks_created,
            qdrant_chunks_indexed=0,
            bm25_chunks_indexed=stats.bm25_chunks_indexed,
            languages_processed=sorted(stats.languages_processed),
            failures=stats.failures,
            elapsed_seconds=stats.elapsed_seconds,
            parquet_completed_files=sorted(new_completed),
            parquet_current_file=None,
        )
        save_checkpoint(checkpoint_path, cp)

    # Print summary & speed comparison
    baseline_speed = 350.0  # historical baseline chunks/sec for single-threaded unoptimized ingest
    actual_speed = stats.throughput_chunks_per_sec
    speedup_pct = ((actual_speed - baseline_speed) / baseline_speed * 100.0) if baseline_speed > 0 else 0.0

    print("\n" + "=" * 80)
    print("PARALLEL BM25 INGESTION COMPLETED")
    print("=" * 80)
    print(f"Total Records Processed:  {stats.records_read:,}")
    print(f"Total Chunks Created:     {stats.chunks_created:,}")
    print(f"Total Chunks Indexed:     {stats.bm25_chunks_indexed:,}")
    print(f"Languages Processed ({len(stats.languages_processed)}): {sorted(stats.languages_processed)}")
    print(f"Total Ingestion Time:     {stats.elapsed_seconds:.2f}s (Merge: {t_merge_sec:.2f}s)")
    print(f"Overall Throughput:       {stats.throughput_chunks_per_sec:.1f} chunks/sec")
    if speedup_pct > 0:
        print(f"Speed Improvement:        +{speedup_pct:.1f}% vs baseline")
    print("=" * 80 + "\n")

    if progress_callback:
        progress_callback(stats)

    return stats


def stream_index_from_parquet(
    parquet_dir: Path | str,
    *,
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    dataset_config: str = "default",
    split: str = "validation",
    embedder: EmbeddingProvider | None = None,
    vector_store: QdrantVectorStore | None = None,
    bm25_index: BM25Index | None = None,
    bm25_path: Path | str | None = None,
    chunking_config: ChunkingConfig | None = None,
    stream_batch_size: int = 250,
    embedding_batch_size: int = 64,
    recreate_collection: bool = False,
    languages: Sequence[str] | set[str] | None = None,
    checkpoint_path: Path | str | None = None,
    resume: bool = True,
    bm25_only: bool = False,
    parquet_batch_size: int = 1000,
    sample_size: int | None = None,
    progress_callback: Callable[[StreamingIndexStats], None] | None = None,
    fast_ingest: bool = False,
    checkpoint_interval: int | None = None,
    workers: int = 1,
) -> StreamingIndexStats:
    """Stream local Parquet files incrementally through normalization → chunking → BM25/Qdrant.

    Reads each Parquet file using PyArrow row-group batching (never loads an entire file into RAM).
    Completed files are recorded in the checkpoint so interrupted runs resume at the next unfinished
    file rather than restarting the full corpus.  Within a partially-processed file, the last few
    batches are re-applied on resume — BM25 deduplication and Qdrant upserts make this safe.

    Args:
        parquet_dir: Directory containing ``*.parquet`` files to process.
        dataset_name: Logical dataset name written into chunks and checkpoints.
        dataset_config: Dataset config name (for checkpoint matching).
        split: Dataset split (for checkpoint matching and logging).
        embedder: Optional dense embedding provider (required unless ``bm25_only=True``).
        vector_store: Optional Qdrant store (required unless ``bm25_only=True``).
        bm25_index: Optional BM25 index to build incrementally.
        bm25_path: Path to persist the BM25 index after each batch.
        chunking_config: Chunking strategy; defaults to ``ChunkingConfig.from_env()``.
        stream_batch_size: Number of normalized documents per indexing batch.
        embedding_batch_size: Number of chunks per embedding API call.
        recreate_collection: Drop and recreate Qdrant collection and reset checkpoint.
        languages: Optional allowlist of language codes to index.
        checkpoint_path: Path to the JSON checkpoint file.
        resume: Whether to resume from a previous checkpoint.
        bm25_only: Skip dense embedding and Qdrant; build BM25 index only.
        parquet_batch_size: Rows per PyArrow read batch (controls peak RAM per file).
        sample_size: Stop after this many records (``None`` = full corpus).
        progress_callback: Callable invoked after each indexing batch with current stats.
        fast_ingest: Use aggressive batching and reduced checkpoint frequency for max throughput.
        checkpoint_interval: Save checkpoint every N processed batches (default: 1; fast_ingest default: 10).
        workers: Number of worker processes (default: 1; >1 enables parallel language file ingestion).

    Returns:
        :class:`StreamingIndexStats` with cumulative ingestion totals.
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for local Parquet ingestion. "
            "Install with: pip install pyarrow"
        ) from exc

    parquet_dir = Path(parquet_dir)
    if not parquet_dir.is_dir():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    # Fast-ingest overrides: larger batches, less frequent checkpoints
    if fast_ingest:
        stream_batch_size = max(stream_batch_size, 5000)
        parquet_batch_size = max(parquet_batch_size, 5000)
        if checkpoint_interval is None:
            checkpoint_interval = 10
        logger.info(
            "fast_ingest_enabled",
            extra={
                "stream_batch_size": stream_batch_size,
                "parquet_batch_size": parquet_batch_size,
                "checkpoint_interval": checkpoint_interval,
            },
        )
    if checkpoint_interval is None:
        checkpoint_interval = 1

    config = chunking_config or ChunkingConfig.from_env()
    stats = StreamingIndexStats()
    start_time = perf_counter()
    allowed_langs = {lang.strip() for lang in languages} if languages else None

    # Sort files deterministically so checkpoint resume is stable
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning("no_parquet_files_found", extra={"dir": str(parquet_dir)})
        return stats

    # Multiprocessing route when bm25_only and (workers > 1 or (fast_ingest and workers != 1))
    effective_workers = workers
    if fast_ingest and effective_workers <= 1:
        import os
        effective_workers = min(len(parquet_files), os.cpu_count() or 4, 6)

    if bm25_only and effective_workers > 1 and len(parquet_files) > 1:
        return _parallel_stream_index_from_parquet(
            parquet_dir=parquet_dir,
            parquet_files=parquet_files,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            bm25_index=bm25_index,
            bm25_path=bm25_path,
            chunking_config=config,
            workers=effective_workers,
            recreate_collection=recreate_collection,
            languages=languages,
            checkpoint_path=checkpoint_path,
            resume=resume,
            sample_size=sample_size,
            parquet_batch_size=parquet_batch_size,
            progress_callback=progress_callback,
            fast_ingest=fast_ingest,
        )

    logger.info(
        "parquet_ingestion_started",
        extra={
            "parquet_dir": str(parquet_dir),
            "num_files": len(parquet_files),
            "files": [f.name for f in parquet_files],
            "bm25_only": bm25_only,
            "sample_size": sample_size,
        },
    )

    # --- Checkpoint / recreation handling ---
    completed_files: set[str] = set()
    if recreate_collection and checkpoint_path:
        reset_checkpoint(checkpoint_path)
    elif resume and checkpoint_path:
        cp = load_checkpoint(checkpoint_path)
        if cp is not None and cp.matches(dataset_name, dataset_config, split):
            completed_files = set(cp.parquet_completed_files)
            stats.records_read = cp.records_read
            stats.documents_processed = cp.documents_processed
            stats.documents_skipped = cp.documents_skipped
            stats.chunks_created = cp.chunks_created
            stats.qdrant_chunks_indexed = cp.qdrant_chunks_indexed
            stats.bm25_chunks_indexed = cp.bm25_chunks_indexed
            stats.languages_processed = set(cp.languages_processed)
            stats.failures = cp.failures
            stats.elapsed_seconds = cp.elapsed_seconds
            logger.info(
                "resuming_parquet_ingestion",
                extra={
                    "completed_files": sorted(completed_files),
                    "records_read": stats.records_read,
                    "bm25_indexed": stats.bm25_chunks_indexed,
                },
            )

    # Ensure Qdrant collection is ready
    if not bm25_only and vector_store is not None and embedder is not None:
        vector_store.ensure_collection(embedder.dimension, recreate=recreate_collection)

    batch_docs: list[NormalizedDocument] = []
    batch_count = 0

    # --- Inner helpers ---

    def _save_checkpoint(current_file: str | None = None) -> None:
        if not checkpoint_path:
            return
        cp = IngestionCheckpoint(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            last_processed_row_index=max(stats.records_read - 1, -1),
            records_read=stats.records_read,
            documents_processed=stats.documents_processed,
            documents_skipped=stats.documents_skipped,
            chunks_created=stats.chunks_created,
            qdrant_chunks_indexed=stats.qdrant_chunks_indexed,
            bm25_chunks_indexed=stats.bm25_chunks_indexed,
            languages_processed=sorted(stats.languages_processed),
            failures=stats.failures,
            elapsed_seconds=perf_counter() - start_time,
            parquet_completed_files=sorted(completed_files),
            parquet_current_file=current_file,
        )
        save_checkpoint(checkpoint_path, cp)

    def _process_batch(documents: list[NormalizedDocument]) -> None:
        if not documents:
            return
        unique_docs, removed_dups = deduplicate_documents(documents)
        stats.documents_skipped += removed_dups

        chunks: list[Chunk] = []
        for doc in unique_docs:
            try:
                doc_chunks = chunk_document(doc, config=config)
                chunks.extend(doc_chunks)
            except Exception:
                stats.failures += 1
                logger.exception("chunking_failed", extra={"document_id": doc.document_id})

        stats.chunks_created += len(chunks)
        if not chunks:
            return

        # Dense embedding + Qdrant upsert (skipped in bm25_only mode)
        if not bm25_only and vector_store is not None and embedder is not None:
            t_embed_0 = perf_counter()
            vectors, _ = timed_batch_embed(
                embedder,
                [c.text for c in chunks],
                batch_size=embedding_batch_size,
                input_type="passage",
            )
            stats.embedding_seconds += perf_counter() - t_embed_0
            upserted = vector_store.upsert(chunks, vectors)
            stats.qdrant_chunks_indexed += upserted

        # BM25 incremental update
        if bm25_index is not None:
            added_bm25 = bm25_index.add(chunks)
            stats.bm25_chunks_indexed += added_bm25
            if bm25_path and not fast_ingest:
                bm25_index.save(bm25_path)

    # --- Main ingestion loop ---
    try:
        sample_size_reached = False
        for parquet_file in parquet_files:
            if sample_size_reached:
                break

            file_name = parquet_file.name
            if file_name in completed_files:
                logger.info("parquet_file_skipped_completed", extra={"file": file_name})
                continue

            logger.info(
                "parquet_file_started",
                extra={"file": file_name, "path": str(parquet_file)},
            )

            try:
                pf = pq.ParquetFile(parquet_file)
                for arrow_batch in pf.iter_batches(batch_size=parquet_batch_size):
                    if sample_size_reached:
                        break
                    for row_dict in arrow_batch.to_pylist():
                        if sample_size is not None and stats.records_read >= sample_size:
                            sample_size_reached = True
                            break

                        row_index = stats.records_read
                        stats.records_read += 1

                        try:
                            docs, empty_removed = normalize_record(
                                row_dict,
                                dataset_name=dataset_name,
                                dataset_config=dataset_config,
                                split=split,
                                row_index=row_index,
                            )
                            stats.documents_skipped += empty_removed
                        except Exception:
                            stats.failures += 1
                            logger.exception("record_normalization_failed")
                            continue

                        for doc in docs:
                            if allowed_langs and doc.language and doc.language not in allowed_langs:
                                stats.documents_skipped += 1
                                continue
                            if doc.language:
                                stats.languages_processed.add(doc.language)
                            batch_docs.append(doc)
                            stats.documents_processed += 1

                        if len(batch_docs) >= stream_batch_size:
                            _process_batch(batch_docs)
                            batch_docs.clear()
                            batch_count += 1
                            stats.elapsed_seconds = perf_counter() - start_time
                            if batch_count % checkpoint_interval == 0:
                                _save_checkpoint(current_file=file_name)
                            if progress_callback:
                                progress_callback(stats)

            except Exception:
                logger.exception("parquet_file_read_failed", extra={"file": file_name})
                stats.failures += 1
                continue

            # Flush trailing batch from this file
            if batch_docs:
                _process_batch(batch_docs)
                batch_docs.clear()

            # Mark file as fully completed only when sample_size hasn't cut us off mid-file
            if not sample_size_reached:
                completed_files.add(file_name)
                logger.info(
                    "parquet_file_completed",
                    extra={"file": file_name, "total_records": stats.records_read},
                )

            stats.elapsed_seconds = perf_counter() - start_time
            _save_checkpoint(
                current_file=None if not sample_size_reached else file_name
            )
            if progress_callback:
                progress_callback(stats)

        # Flush any residual trailing batch (e.g. after sample_size cut-off)
        if batch_docs:
            _process_batch(batch_docs)
            batch_docs.clear()

    finally:
        stats.elapsed_seconds = perf_counter() - start_time
        _save_checkpoint()
        if progress_callback:
            progress_callback(stats)

    logger.info(
        "parquet_ingestion_completed",
        extra={
            "records_read": stats.records_read,
            "bm25_indexed": stats.bm25_chunks_indexed,
            "qdrant_indexed": stats.qdrant_chunks_indexed,
            "files_completed": sorted(completed_files),
        },
    )

    return stats
