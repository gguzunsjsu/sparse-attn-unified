#!/usr/bin/env python3
"""Download model + cache raw text on the login node (internet required).

Tokenization runs offline on GPU compute nodes (see train_llama1b_h100.slurm).

Usage (login node):
  source scripts/activate_env.sh
  hf auth login
  bash scripts/prefetch_offline_assets.sh
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# hf-xet can fail on large model blobs ("Unable to parse string as hex hash value").
# Must be set before any huggingface_hub import (prefetch_offline_assets.sh exports this too).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np


def infer_row_width(total_size: int, seq_length: int) -> int:
    """Infer tokens-per-row from flat memmap size."""
    preferred = seq_length + 1
    if total_size % preferred == 0:
        return preferred
    for stored in (4096, 2048, 8192, 1024, 512):
        width = stored + 1
        if total_size % width == 0:
            return width
    raise ValueError(f"Cannot infer row width from file size {total_size}")


def validate_data_bin(
    bin_path: Path,
    *,
    seq_length: int | None = None,
    num_samples: int = 8,
    vocab_size: int = 128256,
) -> tuple[int, int]:
    """
    Sample rows from a tokenized .bin and fail fast if data looks corrupt.

    Returns (min_token_id, max_token_id) over sampled rows.
    """
    flat = np.memmap(bin_path, dtype=np.int32, mode="r")
    if flat.size == 0:
        raise RuntimeError(f"Empty data bin: {bin_path}")

    row_width = infer_row_width(flat.size, seq_length or 4096)
    stored_seq = row_width - 1
    num_rows = flat.size // row_width
    rows = flat.reshape(num_rows, row_width)

    rng = np.random.default_rng(0)
    idxs = rng.choice(num_rows, size=min(num_samples, num_rows), replace=False)
    sample = np.asarray(rows[idxs], dtype=np.int64)
    mn = int(sample.min())
    mx = int(sample.max())
    zero_frac = float((sample == 0).mean())

    if mx == 0:
        raise RuntimeError(
            f"Corrupt data bin (all token id 0): {bin_path}\n"
            f"  stored_seq={stored_seq}, num_rows={num_rows}\n"
            "Delete the .bin (and .progress if any) and re-tokenize."
        )
    if zero_frac > 0.9:
        raise RuntimeError(
            f"Corrupt data bin (>90% token id 0, zero_frac={zero_frac:.3f}): {bin_path}\n"
            "Delete the .bin and re-tokenize."
        )
    if mx >= vocab_size or mn < 0:
        raise RuntimeError(
            f"Invalid token ids in {bin_path}: range=[{mn}, {mx}], vocab_size={vocab_size}"
        )

    print(
        f"Data bin OK: {bin_path} stored_seq={stored_seq} rows={num_rows} "
        f"sample_range=[{mn}, {mx}] zero_frac={zero_frac:.3f}",
        flush=True,
    )
    return mn, mx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefetch HuggingFace assets for offline HPC training")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--seq-length", type=int, default=4096)
    p.add_argument(
        "--num-sequences",
        type=int,
        default=170_000,
        help="Tokenized sequences (5000 steps x grad_accum 16 = 80k minimum)",
    )
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--skip-model", action="store_true", help="Skip model download (already cached)")
    p.add_argument(
        "--force-model-redownload",
        action="store_true",
        help="Delete partial model cache and download again (use after hf-xet failures)",
    )
    p.add_argument("--skip-data", action="store_true", help="Skip tokenization (default on login node)")
    p.add_argument(
        "--cache-raw",
        action="store_true",
        help="Download parquet shards and build JSONL (login node; use --parquet-only instead)",
    )
    p.add_argument(
        "--parquet-only",
        action="store_true",
        help="Login node: download parquet shards only (needs internet)",
    )
    p.add_argument(
        "--build-jsonl-only",
        action="store_true",
        help="Build JSONL from local parquet shards (offline; used by SLURM)",
    )
    p.add_argument("--skip-raw-cache", action="store_true", help="Skip raw dataset caching")
    p.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Only tokenize from cached raw JSONL (used by SLURM before training)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Read only from local caches (no HuggingFace Hub / dataset downloads)",
    )
    p.add_argument(
        "--force-raw-recache",
        action="store_true",
        help="Delete partial raw JSONL/parquet cache and rebuild from parquet shards",
    )
    p.add_argument(
        "--resume-data",
        action="store_true",
        help="Resume tokenization from .progress sidecar (auto-detected if present)",
    )
    p.add_argument(
        "--data-bin",
        type=Path,
        default=None,
        help="Explicit tokenized .bin path (tokenize output or --validate-only target)",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing tokenized .bin and exit",
    )
    return p.parse_args()


def _token_paths() -> list[Path]:
    home = Path.home()
    hf_home = Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface"))
    return [
        hf_home / "token",
        home / ".cache" / "huggingface" / "token",
        home / ".huggingface" / "token",
    ]


def verify_hf_auth(model_id: str) -> None:
    """Fail fast with actionable instructions if not authenticated for gated models."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError

    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not has_token:
        has_token = any(p.is_file() and p.stat().st_size > 0 for p in _token_paths())

    if not has_token:
        print("ERROR: Not logged in to HuggingFace.", file=sys.stderr)
        print(_auth_instructions(), file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    try:
        user = api.whoami()
        print(f"HuggingFace authenticated as: {user.get('name', user.get('fullname', 'unknown'))}")
    except Exception as exc:
        print(f"ERROR: HuggingFace auth check failed: {exc}", file=sys.stderr)
        print(_auth_instructions(), file=sys.stderr)
        sys.exit(1)

    try:
        api.model_info(model_id)
        print(f"Model access OK: {model_id}")
    except GatedRepoError:
        print(f"ERROR: No access to gated model {model_id}.", file=sys.stderr)
        print(_gated_model_instructions(model_id), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"WARN: Could not verify model access ({exc}); continuing...")


def _auth_instructions() -> str:
    return """
Llama 3.2 1B requires HuggingFace authentication.

1. Accept the license (browser, while on VPN if needed):
   https://huggingface.co/meta-llama/Llama-3.2-1B

2. Create a READ token:
   https://huggingface.co/settings/tokens

3. Login on the login node (pick ONE):

  hf auth login

4. Verify:
   hf auth whoami
   bash scripts/prefetch_offline_assets.sh
"""


def _gated_model_instructions(model_id: str) -> str:
    return f"""
Your HF account is logged in but does NOT have access to {model_id}.

1. Open https://huggingface.co/{model_id}
2. Click "Agree and access repository" (Meta Llama license)
3. Wait ~1 minute, then re-run prefetch.
"""


def download_model(model_id: str, local_dir: Path, *, force: bool = False) -> None:
    from huggingface_hub import snapshot_download

    if force and local_dir.exists():
        import shutil

        print(f"Removing incomplete model cache: {local_dir}")
        shutil.rmtree(local_dir)

    print(f"Downloading model {model_id} -> {local_dir}")
    print("(Using plain HTTP; HF_HUB_DISABLE_XET=1 avoids hf-xet hash errors on large weights.)")
    local_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=token,
            max_workers=4,
        )
    except RuntimeError as exc:
        if "hex hash" in str(exc).lower():
            raise RuntimeError(
                f"{exc}\n\n"
                "Partial download may be corrupted. Re-run with:\n"
                f"  python scripts/prefetch_offline_assets.py --force-model-redownload"
            ) from exc
        raise

    from transformers import AutoTokenizer

    print("Verifying local model files (offline)...")
    AutoTokenizer.from_pretrained(local_dir, local_files_only=True, trust_remote_code=True)
    required = ["config.json", "tokenizer.json"]
    missing = [name for name in required if not (local_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Model cache incomplete; missing: {missing}")
    print(f"Model OK: {local_dir}")


def raw_dataset_path(project_root: Path) -> Path:
    return project_root / "cache" / "datasets" / "fineweb_raw.jsonl"


def _raw_progress_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".progress")


def parquet_shard_dir(project_root: Path) -> Path:
    return project_root / "cache" / "datasets" / "fineweb_parquet"


def _min_raw_docs(num_sequences: int) -> int:
    # Truncated docs usually yield 2+ training sequences after buffering.
    return num_sequences // 2 + 5_000


def _count_jsonl_docs(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _list_parquet_shards(dataset_name: str, num_shards: int) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    all_files = sorted(
        f for f in api.list_repo_files(dataset_name, repo_type="dataset") if f.endswith(".parquet")
    )
    if not all_files:
        raise RuntimeError(f"No parquet files found in dataset repo {dataset_name}")
    return all_files[:num_shards]


def download_parquet_shards(
    dataset_name: str,
    shard_dir: Path,
    *,
    num_shards: int = 6,
) -> list[Path]:
    from huggingface_hub import hf_hub_download

    shard_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(shard_dir.rglob("*.parquet"))
    min_docs_shard = 6  # reuse cache if we already have enough shards
    if len(cached) >= min_docs_shard:
        print(f"Using {len(cached)} cached parquet shards in {shard_dir}")
        return cached

    selected = _list_parquet_shards(dataset_name, num_shards)
    print(f"Downloading {len(selected)} parquet shards -> {shard_dir}")
    paths: list[Path] = []
    for relpath in selected:
        local = hf_hub_download(
            repo_id=dataset_name,
            filename=relpath,
            repo_type="dataset",
            local_dir=str(shard_dir),
        )
        paths.append(Path(local))
        print(f"  {relpath}", flush=True)
    return paths


def build_jsonl_from_parquet(
    out_path: Path,
    num_sequences: int,
    seq_length: int,
    project_root: Path,
    *,
    force: bool = False,
) -> None:
    import pyarrow.parquet as pq

    min_docs = _min_raw_docs(num_sequences)
    max_doc_chars = (seq_length + 1) * 12
    shard_dir = parquet_shard_dir(project_root)
    parquet_files = sorted(shard_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet shards in {shard_dir}. "
            "Run on login node: bash scripts/prefetch_offline_assets.sh"
        )

    if force:
        if out_path.exists():
            out_path.unlink()
        _raw_progress_path(out_path).unlink(missing_ok=True)

    docs_written = _count_jsonl_docs(out_path)
    if docs_written >= min_docs:
        print(f"Raw JSONL already built: {out_path} ({docs_written} docs)")
        _raw_progress_path(out_path).unlink(missing_ok=True)
        return

    if docs_written > 0:
        print(f"Removing incomplete JSONL ({docs_written}/{min_docs} docs) and rebuilding...")
        out_path.unlink()
        docs_written = 0

    print(
        f"Building JSONL from {len(parquet_files)} local parquet shards "
        f"-> {out_path} (target {min_docs} docs)"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for pq_path in parquet_files:
            parquet = pq.ParquetFile(pq_path)
            for batch in parquet.iter_batches(batch_size=512, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    if not text:
                        continue
                    if len(text) > max_doc_chars:
                        text = text[:max_doc_chars]
                    out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    docs_written += 1
                    if docs_written % 5000 == 0:
                        out_f.flush()
                        print(f"  {docs_written}/{min_docs} docs", flush=True)
                    if docs_written >= min_docs:
                        break
                if docs_written >= min_docs:
                    break
            if docs_written >= min_docs:
                break

    if docs_written < min_docs:
        raise RuntimeError(
            f"Only cached {docs_written}/{min_docs} docs from {len(parquet_files)} parquet shards. "
            "Download more shards on the login node or lower --num-sequences."
        )
    print(f"Raw JSONL OK: {out_path} ({docs_written} docs, {out_path.stat().st_size / 1e9:.2f} GB)")


def cache_raw_dataset(
    out_path: Path,
    dataset_name: str,
    dataset_config: str,
    num_sequences: int,
    seq_length: int,
    project_root: Path,
    *,
    force: bool = False,
    parquet_only: bool = False,
) -> None:
    download_parquet_shards(dataset_name, parquet_shard_dir(project_root), num_shards=6)
    if parquet_only:
        print("Parquet shards ready. JSONL build will run offline in the SLURM job.")
        return
    build_jsonl_from_parquet(out_path, num_sequences, seq_length, project_root, force=force)


def _progress_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".progress")


def _read_progress(progress_path: Path) -> tuple[int, int]:
    if not progress_path.is_file():
        return 0, 0
    written = 0
    row = 0
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("written="):
            written = int(line.split("=", 1)[1])
        elif line.startswith("row="):
            row = int(line.split("=", 1)[1])
    return written, row


def _write_progress(progress_path: Path, written: int, row: int) -> None:
    progress_path.write_text(f"written={written}\nrow={row}\n", encoding="utf-8")


def tokenize_dataset(
    model_dir: Path,
    out_path: Path,
    dataset_name: str,
    dataset_config: str,
    seq_length: int,
    num_sequences: int,
    *,
    local_raw_path: Path | None = None,
    resume: bool = False,
) -> None:
    import array

    from datasets import load_dataset
    from transformers import AutoTokenizer

    progress_path = _progress_path(out_path)
    start_written, start_row = (0, 0)
    if resume and progress_path.is_file():
        start_written, start_row = _read_progress(progress_path)
        if start_written >= num_sequences:
            print(f"Dataset already complete: {out_path} ({start_written} sequences)")
            return
        if start_written > 0:
            print(f"Resuming tokenization from sequence {start_written} (dataset row {start_row})")

    print(f"Tokenizing {num_sequences} sequences (len={seq_length}) -> {out_path}")
    if local_raw_path is not None:
        print(f"Reading offline raw text: {local_raw_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_raw_path is not None:
        if not local_raw_path.is_file():
            raise FileNotFoundError(f"Offline raw dataset not found: {local_raw_path}")
        ds = load_dataset(
            "json",
            data_files=str(local_raw_path),
            split="train",
            streaming=True,
        )
    else:
        ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_bytes = (seq_length + 1) * np.dtype(np.int32).itemsize
    if start_written == 0:
        if out_path.exists():
            out_path.unlink()
        out_mode = "wb"
    else:
        expected_size = start_written * row_bytes
        actual_size = out_path.stat().st_size if out_path.exists() else 0
        if actual_size != expected_size:
            raise RuntimeError(
                f"Cannot resume: {out_path} is {actual_size} bytes, expected {expected_size}. "
                "Remove the .bin and .progress files and restart."
            )
        out_mode = "ab"

    # Login nodes often cap virtual memory (ulimit -v). Do NOT np.memmap the full
    # output up front — a 1.4GB map triggers SIGKILL even if RSS stays small.
    chunk_buf = array.array("i", [0] * (seq_length + 1))
    max_chars = (seq_length + 1) * 12
    max_buffer_tokens = (seq_length + 1) * 8

    buffer: list[int] = []
    written = start_written
    row_idx = start_row - 1
    with open(out_path, out_mode) as out_f:
        for row_idx, row in enumerate(ds):
            if row_idx < start_row:
                continue
            text = row["text"]
            if len(text) > max_chars:
                text = text[:max_chars]
            ids = tokenizer(text, add_special_tokens=True)["input_ids"]
            buffer.extend(ids)
            if len(buffer) > max_buffer_tokens:
                buffer = buffer[-max_buffer_tokens:]
            while len(buffer) >= seq_length + 1 and written < num_sequences:
                for i, tok in enumerate(buffer[: seq_length + 1]):
                    chunk_buf[i] = tok
                del buffer[: seq_length + 1]
                out_f.write(chunk_buf.tobytes())
                written += 1
                if written % 1000 == 0:
                    out_f.flush()
                    _write_progress(progress_path, written, row_idx + 1)
                    print(f"  {written}/{num_sequences} sequences", flush=True)
            if written >= num_sequences:
                break

    if written < num_sequences:
        _write_progress(progress_path, written, row_idx + 1)
        raise RuntimeError(
            f"Only tokenized {written}/{num_sequences} sequences. "
            "Increase streaming time or lower --num-sequences."
        )
    progress_path.unlink(missing_ok=True)
    print(f"Dataset OK: {out_path} ({written} sequences, {written * row_bytes / 1e9:.2f} GB)")
    validate_data_bin(out_path, seq_length=seq_length)


def write_manifest(
    project_root: Path,
    model_dir: Path,
    data_path: Path,
    raw_path: Path,
    args: argparse.Namespace,
) -> None:
    manifest = project_root / "cache" / "offline_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        f"model_dir={model_dir}\n"
        f"raw_dataset={raw_path}\n"
        f"data_path={data_path}\n"
        f"seq_length={args.seq_length}\n"
        f"num_sequences={args.num_sequences}\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest}")


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    model_dir = project_root / "cache" / "models" / "Llama-3.2-1B"
    data_path = args.data_bin or (project_root / "cache" / "data" / f"train_{args.seq_length}.bin")
    raw_path = raw_dataset_path(project_root)

    offline = (
        args.offline
        or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        or os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
    )

    if args.validate_only:
        if not data_path.is_file():
            print(f"ERROR: Data bin not found: {data_path}", file=sys.stderr)
            sys.exit(1)
        validate_data_bin(data_path, seq_length=args.seq_length)
        return

    if args.tokenize_only:
        if not model_dir.is_dir():
            print(f"ERROR: Model not found: {model_dir}", file=sys.stderr)
            sys.exit(1)
        if not raw_path.is_file():
            print(f"ERROR: Raw dataset not found: {raw_path}", file=sys.stderr)
            print("Run on login node: bash scripts/prefetch_offline_assets.sh", file=sys.stderr)
            sys.exit(1)
        progress_path = _progress_path(data_path)
        resume_data = args.resume_data or (
            progress_path.is_file() and _read_progress(progress_path)[0] < args.num_sequences
        )
        tokenize_dataset(
            model_dir,
            data_path,
            args.dataset,
            args.dataset_config,
            args.seq_length,
            args.num_sequences,
            local_raw_path=raw_path,
            resume=resume_data,
        )
        write_manifest(project_root, model_dir, data_path, raw_path, args)
        print("\nTokenization complete.")
        return

    if args.build_jsonl_only:
        build_jsonl_from_parquet(
            raw_path,
            args.num_sequences,
            args.seq_length,
            project_root,
            force=args.force_raw_recache,
        )
        write_manifest(project_root, model_dir, data_path, raw_path, args)
        print("\nJSONL build complete.")
        return

    if not args.skip_model:
        verify_hf_auth(args.model)
        download_model(args.model, model_dir, force=args.force_model_redownload)
        gc.collect()
    elif model_dir.is_dir():
        print(f"Skipping model download; using {model_dir}")
    else:
        print(f"ERROR: --skip-model but {model_dir} not found", file=sys.stderr)
        sys.exit(1)

    if (args.cache_raw or args.parquet_only) and not args.skip_raw_cache:
        cache_raw_dataset(
            raw_path,
            args.dataset,
            args.dataset_config,
            args.num_sequences,
            args.seq_length,
            project_root,
            force=args.force_raw_recache,
            parquet_only=args.parquet_only,
        )

    progress_path = _progress_path(data_path)
    resume_data = args.resume_data or (
        not args.skip_data and progress_path.is_file() and _read_progress(progress_path)[0] < args.num_sequences
    )

    if not args.skip_data:
        tokenize_dataset(
            model_dir,
            data_path,
            args.dataset,
            args.dataset_config,
            args.seq_length,
            args.num_sequences,
            local_raw_path=raw_path if offline and raw_path.is_file() else None,
            resume=resume_data,
        )

    write_manifest(project_root, model_dir, data_path, raw_path, args)
    if args.skip_data:
        print("\nPrefetch complete. JSONL build + tokenization run offline in the SLURM job.")
    else:
        print("\nPrefetch complete. GPU jobs can run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
