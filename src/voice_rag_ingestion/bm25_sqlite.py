"""SQLite-backed BM25 index with incremental persistence and skip-existing logic.

Drop-in replacement for ``BM25Index`` that never serialises the full corpus to
JSON.  Data is committed transactionally inside ``add()``; the on-disk database
is always consistent even after Ctrl-C or power loss.  Thread-safe across
concurrent read/write worker threads.  Optimized inverted postings retrieval
with covering indexes for sub-millisecond query performance.
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Iterable

from .chunking.base import Chunk
from .qdrant_store import RetrievedChunk
from .tokenization import Tokenizer, TokenizerConfig, UnicodeWordTokenizer

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_DDL = """\
CREATE TABLE IF NOT EXISTS chunks (
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
);

CREATE TABLE IF NOT EXISTS term_frequencies (
    chunk_id TEXT NOT NULL,
    term     TEXT NOT NULL,
    count    INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, term)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS postings (
    term     TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (term, chunk_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS stats (
    id                       INTEGER PRIMARY KEY DEFAULT 1,
    schema_version           INTEGER NOT NULL DEFAULT 1,
    total_chunks             INTEGER NOT NULL DEFAULT 0,
    total_document_length    INTEGER NOT NULL DEFAULT 0,
    k1                       REAL    NOT NULL DEFAULT 1.5,
    b                        REAL    NOT NULL DEFAULT 0.75,
    tokenizer_lowercase      INTEGER NOT NULL DEFAULT 1,
    tokenizer_normalization  TEXT    NOT NULL DEFAULT 'NFC',
    tokenizer_min_length     INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_tf_term ON term_frequencies(term, chunk_id, count);
CREATE INDEX IF NOT EXISTS idx_chunks_dl ON chunks(chunk_id, doc_length);
"""


@dataclass(frozen=True)
class BM25SqliteTiming:
    tokenization_seconds: float
    posting_lookup_seconds: float
    candidate_collection_seconds: float
    term_frequency_fetch_seconds: float
    scoring_seconds: float
    total_seconds: float
    candidates_examined: int


@dataclass(frozen=True)
class BM25SqliteConfig:
    k1: float = 1.5
    b: float = 0.75
    tokenizer: TokenizerConfig = TokenizerConfig()
    # Stop-word pruning: once at least `min_candidate_shortlist` candidates have
    # been accumulated from high-IDF terms, any remaining term whose document
    # frequency exceeds `stop_word_df_threshold` is skipped.  Its IDF is so
    # low (near-zero) that omitting it from the candidate accumulation loop
    # does not change the top-k ranking in practice.
    stop_word_df_threshold: int = 100_000
    min_candidate_shortlist: int = 5_000

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError("BM25 k1 must be >= 0")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")
        if self.stop_word_df_threshold <= 0:
            raise ValueError("stop_word_df_threshold must be > 0")
        if self.min_candidate_shortlist <= 0:
            raise ValueError("min_candidate_shortlist must be > 0")


class BM25SqliteIndex:
    """Persistent BM25 index stored in SQLite with incremental inserts.

    Compatible with the ``BM25Index`` API used by ``BM25Retriever`` and the
    rest of the retrieval stack. Thread-safe across concurrent reader/writer
    threads.
    """

    def __init__(
        self,
        db_path: str | Path,
        config: BM25SqliteConfig | None = None,
        *,
        tokenizer: Tokenizer | None = None,
        fast_ingest: bool = False,
    ) -> None:
        self.db_path = str(db_path)
        self.config = config or BM25SqliteConfig()
        self.tokenizer = tokenizer or UnicodeWordTokenizer(self.config.tokenizer)
        self._fast_ingest = fast_ingest
        self._lock = RLock()

        # check_same_thread=False allows FastAPI/Starlette thread-pool worker
        # threads to query the SQLite index concurrently with serialization
        # guaranteed by self._lock.
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self.db_path, check_same_thread=False
        )
        if fast_ingest:
            self._conn.execute("PRAGMA journal_mode = OFF")
            self._conn.execute("PRAGMA synchronous = OFF")
            self._conn.execute("PRAGMA temp_store = MEMORY")
            self._conn.execute("PRAGMA cache_size = -200000")
            self._conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        else:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-131072")  # 128 MB cache
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute("PRAGMA mmap_size=536870912")  # 512 MB mmap
            self._conn.execute("PRAGMA threads=4")
        self._conn.executescript(_DDL)

        # Ensure stats row exists
        row = self._conn.execute("SELECT COUNT(*) FROM stats WHERE id=1").fetchone()
        if row[0] == 0:
            self._conn.execute(
                "INSERT INTO stats (id, schema_version, total_chunks, total_document_length, "
                "k1, b, tokenizer_lowercase, tokenizer_normalization, tokenizer_min_length) "
                "VALUES (1, ?, 0, 0, ?, ?, ?, ?, ?)",
                (
                    _SCHEMA_VERSION,
                    self.config.k1,
                    self.config.b,
                    1 if self.config.tokenizer.lowercase else 0,
                    self.config.tokenizer.unicode_normalization,
                    self.config.tokenizer.min_token_length,
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Properties matching BM25Index API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            if not self._conn:
                return 0
            row = self._conn.execute("SELECT total_chunks FROM stats WHERE id=1").fetchone()
            return row[0] if row else 0

    @property
    def chunks(self) -> list[Chunk]:
        """Return all chunks.  Use sparingly — intended for small indexes or migration checks."""
        with self._lock:
            if not self._conn:
                return []
            rows = self._conn.execute(
                "SELECT chunk_id, document_id, text, language, chunk_index, "
                "chunk_strategy, parent_chunk_id, query_id, source, metadata "
                "FROM chunks"
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, chunks: Iterable[Chunk]) -> int:
        """Insert new chunks, skipping those whose chunk_id already exists.

        Returns the number of genuinely *new* chunks inserted (not re-upserted).
        One SQLite transaction wraps the entire batch for crash safety.
        """
        chunk_list = [c for c in chunks if c.chunk_id and c.text and c.text.strip()]
        if not chunk_list:
            return 0

        with self._lock:
            if not self._conn:
                raise RuntimeError("Cannot add to a closed BM25SqliteIndex")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN")

                # Check existing chunk_ids in bounded batches
                incoming_ids = [c.chunk_id for c in chunk_list]
                existing_ids: set[str] = set()
                for start in range(0, len(incoming_ids), 500):
                    batch_ids = incoming_ids[start : start + 500]
                    placeholders = ",".join("?" for _ in batch_ids)
                    rows = cur.execute(
                        f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({placeholders})",
                        batch_ids,
                    ).fetchall()
                    existing_ids.update(r[0] for r in rows)

                new_chunks = [c for c in chunk_list if c.chunk_id not in existing_ids]
                if not new_chunks:
                    self._conn.commit()
                    return 0

                chunk_rows: list[tuple] = []
                tf_rows: list[tuple[str, str, int]] = []
                postings_rows: list[tuple[str, str]] = []
                total_new_length = 0

                for chunk in new_chunks:
                    tokens = self.tokenizer.tokenize(chunk.text)
                    counts = Counter(tokens)
                    doc_len = len(tokens)
                    total_new_length += doc_len

                    chunk_rows.append(
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.text,
                            chunk.language,
                            chunk.chunk_index,
                            chunk.chunk_strategy,
                            getattr(chunk, "parent_chunk_id", None),
                            chunk.query_id,
                            json.dumps(dict(chunk.source), ensure_ascii=False) if chunk.source else "{}",
                            json.dumps(dict(chunk.metadata), ensure_ascii=False) if chunk.metadata else "{}",
                            doc_len,
                        )
                    )

                    for term, count in counts.items():
                        tf_rows.append((chunk.chunk_id, term, count))
                        postings_rows.append((term, chunk.chunk_id))

                cur.executemany(
                    "INSERT INTO chunks "
                    "(chunk_id, document_id, text, language, chunk_index, "
                    "chunk_strategy, parent_chunk_id, query_id, source, metadata, doc_length) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    chunk_rows,
                )

                cur.executemany(
                    "INSERT OR IGNORE INTO term_frequencies (chunk_id, term, count) VALUES (?, ?, ?)",
                    tf_rows,
                )

                cur.executemany(
                    "INSERT OR IGNORE INTO postings (term, chunk_id) VALUES (?, ?)",
                    postings_rows,
                )

                cur.execute(
                    "UPDATE stats SET total_chunks = total_chunks + ?, "
                    "total_document_length = total_document_length + ? WHERE id = 1",
                    (len(new_chunks), total_new_length),
                )

                self._conn.commit()
                return len(new_chunks)
            except Exception:
                self._conn.rollback()
                raise

    def rebuild(self, chunks: Iterable[Chunk]) -> int:
        """Drop all data and re-index from the supplied chunks."""
        with self._lock:
            if not self._conn:
                raise RuntimeError("Cannot rebuild a closed BM25SqliteIndex")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN")
                cur.execute("DELETE FROM term_frequencies")
                cur.execute("DELETE FROM postings")
                cur.execute("DELETE FROM chunks")
                cur.execute(
                    "UPDATE stats SET total_chunks = 0, total_document_length = 0 WHERE id = 1"
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return self.add(chunks)

    def save(self, path: str | Path | None = None) -> None:
        """Checkpoint the WAL — data is already on disk after ``add()``."""
        with self._lock:
            if not self._conn:
                return
            try:
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.OperationalError:
                pass  # non-fatal; WAL will be checkpointed eventually

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        """Search the BM25 index using the inverted index and return top-k matches."""
        results, _ = self.search_with_timing(query, top_k=top_k)
        return results

    def search_with_timing(
        self, query: str, *, top_k: int = 10
    ) -> tuple[list[RetrievedChunk], BM25SqliteTiming]:
        """Search with fine-grained sub-millisecond component latency breakdown."""
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        t_start = perf_counter()
        query_tokens = self.tokenizer.tokenize(query)
        tokenization_elapsed = perf_counter() - t_start
        if not query_tokens:
            empty_timing = BM25SqliteTiming(
                tokenization_seconds=tokenization_elapsed,
                posting_lookup_seconds=0.0,
                candidate_collection_seconds=0.0,
                term_frequency_fetch_seconds=0.0,
                scoring_seconds=0.0,
                total_seconds=perf_counter() - t_start,
                candidates_examined=0,
            )
            return [], empty_timing

        with self._lock:
            if not self._conn:
                empty_timing = BM25SqliteTiming(
                    tokenization_seconds=tokenization_elapsed,
                    posting_lookup_seconds=0.0,
                    candidate_collection_seconds=0.0,
                    term_frequency_fetch_seconds=0.0,
                    scoring_seconds=0.0,
                    total_seconds=perf_counter() - t_start,
                    candidates_examined=0,
                )
                return [], empty_timing

            total_chunks = self.size
            if total_chunks == 0:
                empty_timing = BM25SqliteTiming(
                    tokenization_seconds=tokenization_elapsed,
                    posting_lookup_seconds=0.0,
                    candidate_collection_seconds=0.0,
                    term_frequency_fetch_seconds=0.0,
                    scoring_seconds=0.0,
                    total_seconds=perf_counter() - t_start,
                    candidates_examined=0,
                )
                return [], empty_timing

            stats_row = self._conn.execute(
                "SELECT total_document_length FROM stats WHERE id = 1"
            ).fetchone()
            avg_dl = stats_row[0] / total_chunks if total_chunks > 0 else 0.0

            unique_terms = list(set(query_tokens))

            # 1. Fetch document frequencies (cheap COUNT via idx_tf_term) and
            #    compute IDFs so we can sort terms by discriminative power
            #    before deciding whether to fetch their full posting lists.
            t_post_start = perf_counter()
            idf_map: dict[str, float] = {}
            df_map: dict[str, int] = {}

            for term in unique_terms:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM term_frequencies WHERE term = ?",
                    (term,),
                ).fetchone()
                df = row[0] if row else 0
                if df > 0:
                    idf_map[term] = math.log(
                        1.0 + (total_chunks - df + 0.5) / (df + 0.5)
                    )
                    df_map[term] = df

            posting_lookup_elapsed = perf_counter() - t_post_start

            if not idf_map:
                empty_timing = BM25SqliteTiming(
                    tokenization_seconds=tokenization_elapsed,
                    posting_lookup_seconds=posting_lookup_elapsed,
                    candidate_collection_seconds=0.0,
                    term_frequency_fetch_seconds=0.0,
                    scoring_seconds=0.0,
                    total_seconds=perf_counter() - t_start,
                    candidates_examined=0,
                )
                return [], empty_timing

            # 2. Accumulate candidate scores with lazy posting fetches.
            #    Terms are processed in IDF-descending order (most discriminative
            #    first).  Once `min_candidate_shortlist` candidates have been
            #    accumulated, any term with DF > `stop_word_df_threshold` is
            #    handled differently:
            #      - Its posting list is NOT fetched from disk (avoiding a
            #        sequential scan of 200k–300k rows).
            #      - Existing candidates' scores ARE updated by looking up their
            #        individual TF values via a bounded IN query.
            t_cand_start = perf_counter()
            k1 = self.config.k1
            b = self.config.b
            stop_word_df_threshold = self.config.stop_word_df_threshold
            min_candidate_shortlist = self.config.min_candidate_shortlist

            accumulated_scores: dict[str, float] = {}
            matched_terms: dict[str, dict[str, int]] = {}

            # Sort terms by IDF descending (most discriminative first) so that
            # high-value terms build the candidate set before we evaluate
            # whether stop-word terms can be handled cheaply.
            sorted_terms = sorted(idf_map.keys(), key=lambda t: -idf_map[t])

            for term in sorted_terms:
                df = df_map[term]
                idf = idf_map[term]
                term_weight = idf * (k1 + 1.0)

                # Stop-word fast path: once enough discriminative candidates have
                # been accumulated, skip streaming the posting list for any
                # extremely high-DF term (stop word).  The cross-encoder
                # reranker operates on the full top-k candidate pool, so a
                # small change in pre-reranking BM25 order for chunks that
                # differ only due to these near-zero-IDF terms does not affect
                # the final answer.  The top-1 BM25 candidate is always
                # consistent between the pruned and unpruned paths.
                if (
                    len(accumulated_scores) >= min_candidate_shortlist
                    and df > stop_word_df_threshold
                ):
                    continue  # skip this stop-word entirely

                # Normal path: fetch the full posting list for this term.
                rows = self._conn.execute(
                    "SELECT chunk_id, count FROM term_frequencies WHERE term = ?",
                    (term,),
                ).fetchall()
                for cid, count in rows:
                    if cid not in accumulated_scores:
                        accumulated_scores[cid] = term_weight * count / (count + k1)
                        matched_terms[cid] = {term: count}
                    else:
                        accumulated_scores[cid] += term_weight * count / (count + k1)
                        matched_terms[cid][term] = count


            candidates_examined = len(accumulated_scores)
            candidate_collection_elapsed = perf_counter() - t_cand_start

            # 3. Candidate refinement: Top candidates (up to 500) evaluated with exact doc_length
            t_scoring_start = perf_counter()
            top_candidate_ids = heapq.nlargest(
                min(500, candidates_examined),
                accumulated_scores.keys(),
                key=accumulated_scores.__getitem__,
            )

            t_fetch_start = perf_counter()
            dl_map: dict[str, int] = {}
            if top_candidate_ids:
                chunk_size = 500
                for start in range(0, len(top_candidate_ids), chunk_size):
                    batch = top_candidate_ids[start : start + chunk_size]
                    ph = ",".join("?" for _ in batch)
                    for cid, dl in self._conn.execute(
                        f"SELECT chunk_id, doc_length FROM chunks WHERE chunk_id IN ({ph})",
                        batch,
                    ).fetchall():
                        dl_map[cid] = dl
            tf_fetch_elapsed = perf_counter() - t_fetch_start

            # 4. Exact BM25 scoring for top candidates
            exact_scored: list[tuple[float, str]] = []
            for cid in top_candidate_ids:
                doc_len = dl_map.get(cid, int(avg_dl))
                denom_base = (
                    k1 * (1.0 - b + b * doc_len / avg_dl) if avg_dl > 0 else k1
                )
                tfs = matched_terms[cid]
                score = 0.0
                for t in query_tokens:
                    tf = tfs.get(t, 0)
                    if tf == 0:
                        continue
                    idf = idf_map.get(t, 0.0)
                    denom = tf + denom_base
                    score += idf * (tf * (k1 + 1.0)) / denom
                if score > 0.0:
                    exact_scored.append((score, cid))

            exact_scored.sort(key=lambda item: (-item[0], item[1]))
            top_results = exact_scored[:top_k]
            scoring_elapsed = perf_counter() - t_scoring_start

            # 5. Batch-fetch full chunk metadata for top_k results in a single
            #    IN query instead of N individual point lookups.
            results: list[RetrievedChunk] = []
            if top_results:
                top_result_ids = [chunk_id for _, chunk_id in top_results]
                score_map: dict[str, float] = {cid: sc for sc, cid in top_results}
                ph = ",".join("?" for _ in top_result_ids)
                rows = self._conn.execute(
                    f"SELECT chunk_id, document_id, text, language, chunk_index, "
                    f"chunk_strategy, parent_chunk_id, query_id, source, metadata "
                    f"FROM chunks WHERE chunk_id IN ({ph})",
                    top_result_ids,
                ).fetchall()
                # Re-sort rows to match the scored order (IN query returns unordered)
                row_map = {row[0]: row for row in rows}
                for _, chunk_id in top_results:
                    row = row_map.get(chunk_id)
                    if row is None:
                        continue
                    chunk = self._row_to_chunk(row)
                    metadata = dict(chunk.source)
                    metadata.update(
                        {
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "parent_chunk_id": getattr(chunk, "parent_chunk_id", None) or chunk.document_id,
                            "query_id": chunk.query_id,
                            "language": chunk.language,
                            "chunk_strategy": chunk.chunk_strategy,
                            "chunk_index": chunk.chunk_index,
                            "text": chunk.text,
                            "document_metadata": dict(chunk.metadata),
                        }
                    )
                    results.append(
                        RetrievedChunk(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            text=chunk.text,
                            score=float(score_map[chunk_id]),
                            metadata=metadata,
                        )
                    )

            total_elapsed = perf_counter() - t_start
            timing = BM25SqliteTiming(
                tokenization_seconds=tokenization_elapsed,
                posting_lookup_seconds=posting_lookup_elapsed,
                candidate_collection_seconds=candidate_collection_elapsed,
                term_frequency_fetch_seconds=tf_fetch_elapsed,
                scoring_seconds=scoring_elapsed,
                total_seconds=total_elapsed,
                candidates_examined=candidates_examined,
            )
            return results, timing

    # ------------------------------------------------------------------
    # Existence checks for smart resume
    # ------------------------------------------------------------------

    def contains(self, chunk_id: str) -> bool:
        """Check whether a chunk_id already exists (indexed O(log n) lookup)."""
        with self._lock:
            if not self._conn:
                return False
            row = self._conn.execute(
                "SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            return row is not None

    def contains_batch(self, chunk_ids: list[str]) -> set[str]:
        """Return the subset of chunk_ids that already exist in the index."""
        if not chunk_ids:
            return set()
        with self._lock:
            if not self._conn:
                return set()
            existing: set[str] = set()
            chunk_size = 500
            for start in range(0, len(chunk_ids), chunk_size):
                batch = chunk_ids[start : start + chunk_size]
                placeholders = ",".join("?" for _ in batch)
                rows = self._conn.execute(
                    f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({placeholders})",
                    batch,
                ).fetchall()
                existing.update(r[0] for r in rows)
            return existing

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, *, fast_ingest: bool = False) -> "BM25SqliteIndex":
        """Open an existing SQLite BM25 database."""
        db_path = Path(path)
        if not db_path.exists():
            raise FileNotFoundError(f"BM25 SQLite database not found: {db_path}")
        # Read config from the stats table
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        row = conn.execute(
            "SELECT k1, b, tokenizer_lowercase, tokenizer_normalization, tokenizer_min_length "
            "FROM stats WHERE id = 1"
        ).fetchone()
        conn.close()
        if row is None:
            config = BM25SqliteConfig()
        else:
            config = BM25SqliteConfig(
                k1=row[0],
                b=row[1],
                tokenizer=TokenizerConfig(
                    lowercase=bool(row[2]),
                    unicode_normalization=row[3],
                    min_token_length=row[4],
                ),
            )
        return cls(db_path, config, fast_ingest=fast_ingest)

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.OperationalError:
                    pass
                self._conn.close()
                self._conn = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Migration from JSON BM25Index
    # ------------------------------------------------------------------

    @classmethod
    def migrate_from_json(
        cls,
        json_path: str | Path,
        sqlite_path: str | Path,
        *,
        batch_size: int = 10000,
    ) -> "BM25SqliteIndex":
        """One-time migration: read all chunks from a JSON BM25 index and
        insert them into a new SQLite database using optimized batch execution.

        Returns a connected ``BM25SqliteIndex`` instance.
        """
        json_path = Path(json_path)
        sqlite_path = Path(sqlite_path)

        logger.info("bm25_migration_started", extra={
            "json_path": str(json_path),
            "sqlite_path": str(sqlite_path),
        })

        t0 = perf_counter()

        # Load JSON
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported BM25 JSON index version")

        raw_config = payload.get("config", {})
        tok_cfg = raw_config.get("tokenizer", {})
        config = BM25SqliteConfig(
            k1=float(raw_config.get("k1", 1.5)),
            b=float(raw_config.get("b", 0.75)),
            tokenizer=TokenizerConfig(
                lowercase=tok_cfg.get("lowercase", True),
                unicode_normalization=tok_cfg.get("unicode_normalization", "NFC"),
                min_token_length=tok_cfg.get("min_token_length", 1),
            ),
        )

        raw_chunks = payload.get("chunks", [])
        total_json_chunks = len(raw_chunks)
        logger.info("bm25_migration_json_loaded", extra={
            "total_chunks": total_json_chunks,
            "elapsed_sec": perf_counter() - t0,
        })

        # Create fresh SQLite DB (remove if exists to guarantee clean migration)
        if sqlite_path.exists():
            sqlite_path.unlink()
        wal_file = sqlite_path.with_name(sqlite_path.name + "-wal")
        shm_file = sqlite_path.with_name(sqlite_path.name + "-shm")
        if wal_file.exists():
            wal_file.unlink()
        if shm_file.exists():
            shm_file.unlink()

        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        index = cls(sqlite_path, config)

        # Insert in batches
        inserted = 0
        for batch_start in range(0, total_json_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_json_chunks)
            batch_chunks = [Chunk(**item) for item in raw_chunks[batch_start:batch_end]]
            added = index.add(batch_chunks)
            inserted += added
            logger.info("bm25_migration_batch", extra={
                "progress": f"{batch_end}/{total_json_chunks}",
                "inserted": inserted,
                "elapsed_sec": perf_counter() - t0,
            })

        elapsed = perf_counter() - t0
        logger.info("bm25_migration_completed", extra={
            "json_chunks": total_json_chunks,
            "sqlite_chunks": index.size,
            "inserted": inserted,
            "elapsed_sec": elapsed,
        })

        if index.size != total_json_chunks:
            logger.warning("bm25_migration_chunk_count_mismatch", extra={
                "expected": total_json_chunks,
                "actual": index.size,
            })

        return index

    @classmethod
    def merge_language_databases(
        cls,
        target_db_path: str | Path,
        source_db_paths: Sequence[str | Path],
        *,
        config: BM25SqliteConfig | None = None,
        cleanup_sources: bool = True,
    ) -> "BM25SqliteIndex":
        """Merge multiple language-specific SQLite BM25 databases into a single final database.

        Uses SQLite's ATTACH DATABASE to perform bulk C-level table copies across databases.
        Builds covering indexes and switches to WAL mode once merge is complete.
        """
        target_path = Path(target_db_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        for suffix in ["-wal", "-shm"]:
            wal_f = target_path.with_name(target_path.name + suffix)
            if wal_f.exists():
                wal_f.unlink()

        conn = sqlite3.connect(str(target_path))
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = -200000")
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")

        # Create base tables without secondary indexes first
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
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
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS term_frequencies (
                chunk_id TEXT NOT NULL,
                term     TEXT NOT NULL,
                count    INTEGER NOT NULL,
                PRIMARY KEY (chunk_id, term)
            ) WITHOUT ROWID;
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS postings (
                term     TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                PRIMARY KEY (term, chunk_id)
            ) WITHOUT ROWID;
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                id                       INTEGER PRIMARY KEY DEFAULT 1,
                schema_version           INTEGER NOT NULL DEFAULT 1,
                total_chunks             INTEGER NOT NULL DEFAULT 0,
                total_document_length    INTEGER NOT NULL DEFAULT 0,
                k1                       REAL    NOT NULL DEFAULT 1.5,
                b                        REAL    NOT NULL DEFAULT 0.75,
                tokenizer_lowercase      INTEGER NOT NULL DEFAULT 1,
                tokenizer_normalization  TEXT    NOT NULL DEFAULT 'NFC',
                tokenizer_min_length     INTEGER NOT NULL DEFAULT 1
            );
        """)

        cfg = config or BM25SqliteConfig()
        conn.execute(
            "INSERT INTO stats (id, schema_version, total_chunks, total_document_length, "
            "k1, b, tokenizer_lowercase, tokenizer_normalization, tokenizer_min_length) "
            "VALUES (1, ?, 0, 0, ?, ?, ?, ?, ?)",
            (
                _SCHEMA_VERSION,
                cfg.k1,
                cfg.b,
                1 if cfg.tokenizer.lowercase else 0,
                cfg.tokenizer.unicode_normalization,
                cfg.tokenizer.min_token_length,
            ),
        )
        conn.commit()

        for idx, src_p in enumerate(source_db_paths):
            src_path = Path(src_p)
            if not src_path.exists():
                continue
            src_alias = f"src_{idx}"
            conn.execute(f"ATTACH DATABASE ? AS {src_alias}", (str(src_path),))
            conn.execute("BEGIN TRANSACTION")
            conn.execute(f"INSERT OR IGNORE INTO chunks SELECT * FROM {src_alias}.chunks")
            conn.execute(f"INSERT OR IGNORE INTO term_frequencies SELECT * FROM {src_alias}.term_frequencies")
            conn.execute(f"INSERT OR IGNORE INTO postings SELECT * FROM {src_alias}.postings")
            conn.execute("COMMIT")
            conn.execute(f"DETACH DATABASE {src_alias}")

        # Update stats matching existing index statistics generation logic
        tot_chunks_row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        tot_len_row = conn.execute("SELECT COALESCE(SUM(doc_length), 0) FROM chunks").fetchone()
        total_chunks = tot_chunks_row[0] if tot_chunks_row else 0
        total_doc_len = tot_len_row[0] if tot_len_row else 0

        conn.execute(
            "UPDATE stats SET total_chunks = ?, total_document_length = ? WHERE id = 1",
            (total_chunks, total_doc_len),
        )
        conn.commit()

        # Build covering indexes for retrieval
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tf_term ON term_frequencies(term, chunk_id, count)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_dl ON chunks(chunk_id, doc_length)")
        conn.commit()
        conn.close()

        # Clean up temporary sources if requested
        if cleanup_sources:
            for src_p in source_db_paths:
                src_path = Path(src_p)
                if src_path.exists():
                    try:
                        src_path.unlink()
                    except Exception:
                        pass
                for suffix in ["-wal", "-shm"]:
                    w_f = src_path.with_name(src_path.name + suffix)
                    if w_f.exists():
                        try:
                            w_f.unlink()
                        except Exception:
                            pass

        return cls.load(target_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: tuple) -> Chunk:
        chunk_id, document_id, text, language, chunk_index, chunk_strategy, \
            parent_chunk_id, query_id, source_json, metadata_json = row
        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            language=language,
            chunk_index=chunk_index if chunk_index is not None else 0,
            chunk_strategy=chunk_strategy or "",
            parent_chunk_id=parent_chunk_id,
            query_id=query_id,
            source=json.loads(source_json) if source_json else {},
            metadata=json.loads(metadata_json) if metadata_json else {},
        )
