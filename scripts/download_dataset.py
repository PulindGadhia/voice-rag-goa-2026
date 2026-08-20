#!/usr/bin/env python3
"""
Download ai4bharat/MSMARCO-XI Parquet files from Hugging Face Hub to a local cache.

Run this ONCE before full-corpus ingestion to decouple dataset acquisition from
dataset processing and avoid repeated HF CDN downloads during indexing:

    python scripts/download_dataset.py --split validation

After the download completes, run the Parquet-based indexer:

    python scripts/index_vectors.py \\
        --local-parquet-dir .cache/msmarco_xi/validation \\
        --bm25-only \\
        --recreate

Re-running download_dataset.py skips files that already exist locally (--no-skip-existing
forces a fresh download of every file).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def discover_parquet_files(
    repo_id: str,
    split: str,
    config: str,
    *,
    token: str | None = None,
    revision: str = "main",
) -> list[str]:
    """Return HF-repo-relative paths of all Parquet files for the given split.

    Handles multiple common HF dataset directory structures:
      - ``{split}/*.parquet``               (e.g. validation/asmval.parquet)
      - ``data/{split}*.parquet``           (e.g. data/validation-00001.parquet)
      - ``data/{config}/{split}*.parquet``  (e.g. data/default/validation-00001.parquet)
      - ``{config}/{split}*.parquet``       (e.g. default/validation-00001.parquet)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "ERROR: huggingface_hub is required. Install with: pip install huggingface_hub",
            file=sys.stderr,
        )
        return []

    api = HfApi(token=token)
    try:
        all_files = list(
            api.list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
            )
        )
    except Exception as exc:
        print(
            f"ERROR: Failed to list repository files for {repo_id!r}: {exc}",
            file=sys.stderr,
        )
        return []

    prefixes = (
        f"{split}/",
        f"data/{split}",
        f"data/{config}/{split}",
        f"{config}/{split}",
    )

    candidates: list[str] = []
    seen: set[str] = set()
    for f in all_files:
        if not f.endswith(".parquet"):
            continue
        if any(f.startswith(p) for p in prefixes):
            if f not in seen:
                seen.add(f)
                candidates.append(f)

    return sorted(candidates)


def download_one_file(
    repo_id: str,
    hf_path: str,
    local_dir: Path,
    *,
    token: str | None = None,
    revision: str = "main",
    skip_existing: bool = True,
) -> tuple[str, bool, str]:
    """Download *hf_path* from *repo_id* into *local_dir* preserving directory structure.

    Returns ``(hf_path, success, message)``.
    """
    dest = local_dir / hf_path
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return hf_path, True, f"SKIPPED  ({_sizeof_fmt(dest.stat().st_size)}, already cached)"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return hf_path, False, "ERROR: huggingface_hub not installed"

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # hf_hub_download with local_dir preserves the path structure:
        # local_dir/hf_path (e.g. .cache/msmarco_xi/validation/asmval.parquet)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            repo_type="dataset",
            local_dir=str(local_dir),
            revision=revision,
            token=token,
            local_dir_use_symlinks=False,
        )
        # Ensure the file ended up where we expected (hf_hub_download may rename)
        actual = Path(local_path)
        if actual != dest and actual.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            actual.rename(dest)
        size = dest.stat().st_size if dest.exists() else 0
        return hf_path, True, f"OK       ({_sizeof_fmt(size)})"
    except Exception as exc:
        return hf_path, False, f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ai4bharat/MSMARCO-XI Parquet files from Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo-id",
        default="ai4bharat/MSMARCO-XI",
        help="HF dataset repository ID (default: ai4bharat/MSMARCO-XI)",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="Dataset config/subset name (default: default)",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to download (default: validation)",
    )
    parser.add_argument(
        "--output-dir",
        default=".cache/msmarco_xi",
        help="Root output directory; files land at OUTPUT_DIR/{split}/*.parquet (default: .cache/msmarco_xi)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel download threads (default: 2; use 1 for low-bandwidth connections)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (falls back to HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset Git revision / branch (default: main)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if the file already exists locally",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Discover and print Parquet files without downloading anything",
    )
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    output_dir = Path(args.output_dir).resolve()
    skip_existing = not args.no_skip_existing

    print("=" * 72)
    print("VOICE RAG GOA 2026 — DATASET DOWNLOADER")
    print("=" * 72)
    print(f"Repository:     {args.repo_id}")
    print(f"Config:         {args.config}")
    print(f"Split:          {args.split}")
    print(f"Output root:    {output_dir}")
    print(f"Workers:        {args.workers}")
    print(f"Skip existing:  {skip_existing}")
    print("=" * 72 + "\n")

    print(f"Discovering Parquet files in {args.repo_id}...", flush=True)
    parquet_files = discover_parquet_files(
        args.repo_id,
        args.split,
        args.config,
        token=token,
        revision=args.revision,
    )

    if not parquet_files:
        print(
            f"\nERROR: No Parquet files found for split={args.split!r} in {args.repo_id!r}",
            file=sys.stderr,
        )
        print(
            "Hint: Check the repo structure with:\n"
            "  python -c \"from huggingface_hub import HfApi; "
            f"print('\\n'.join(HfApi().list_repo_files('{args.repo_id}', repo_type='dataset')))\"",
            file=sys.stderr,
        )
        return 1

    print(f"\nFound {len(parquet_files)} Parquet file(s):")
    for f in parquet_files:
        dest = output_dir / f
        cached = " [cached]" if dest.exists() and dest.stat().st_size > 0 else ""
        print(f"  {f}{cached}")

    if args.list_only:
        print("\n[--list-only: no downloads performed]")
        return 0

    dest_split_dir = output_dir / args.split
    print(f"\nDownloading {len(parquet_files)} file(s) → {dest_split_dir}/ ...\n")
    t0 = time.perf_counter()

    success_count = 0
    fail_count = 0

    if args.workers <= 1:
        for hf_path in parquet_files:
            fname, ok, msg = download_one_file(
                args.repo_id,
                hf_path,
                output_dir,
                token=token,
                revision=args.revision,
                skip_existing=skip_existing,
            )
            icon = "✓" if ok else "✗"
            print(f"  {icon} {Path(fname).name:<35} {msg}")
            if ok:
                success_count += 1
            else:
                fail_count += 1
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for hf_path in parquet_files:
                fut = executor.submit(
                    download_one_file,
                    args.repo_id,
                    hf_path,
                    output_dir,
                    token=token,
                    revision=args.revision,
                    skip_existing=skip_existing,
                )
                futures[fut] = hf_path
            for fut in as_completed(futures):
                fname, ok, msg = fut.result()
                icon = "✓" if ok else "✗"
                print(f"  {icon} {Path(fname).name:<35} {msg}")
                if ok:
                    success_count += 1
                else:
                    fail_count += 1

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 72}")
    print(f"Download complete: {success_count} succeeded, {fail_count} failed  ({elapsed:.1f}s)")

    # Summary of what's available locally
    available = sorted(dest_split_dir.glob("*.parquet")) if dest_split_dir.exists() else []
    if available:
        total_bytes = sum(p.stat().st_size for p in available)
        print(f"\nLocal Parquet files in {dest_split_dir}:")
        for p in available:
            print(f"  {p.name}  ({_sizeof_fmt(p.stat().st_size)})")
        print(f"  ─── Total: {len(available)} file(s), {_sizeof_fmt(total_bytes)}")
        print(f"\nNext step — 1 000-record smoke test:")
        print(
            f"  python scripts/index_vectors.py \\\n"
            f"    --local-parquet-dir {dest_split_dir} \\\n"
            f"    --bm25-only --recreate --sample-size 1000"
        )
        print(f"\nNext step — full BM25 ingestion:")
        print(
            f"  python scripts/index_vectors.py \\\n"
            f"    --local-parquet-dir {dest_split_dir} \\\n"
            f"    --bm25-only --recreate"
        )

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
