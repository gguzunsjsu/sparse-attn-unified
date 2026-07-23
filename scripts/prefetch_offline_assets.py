#!/usr/bin/env python3
"""Download model + tokenized training data on the login node (internet required).

GPU compute nodes on SJSU HPC have no outbound network. Run this once on the
login node before submitting SLURM jobs.

Usage (login node):
  source scripts/activate_env.sh
  hf auth login
  python scripts/prefetch_offline_assets.py
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# hf-xet can fail on large model blobs ("Unable to parse string as hex hash value").
# Must be set before any huggingface_hub import (prefetch_offline_assets.sh exports this too).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np


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
    p.add_argument("--skip-data", action="store_true", help="Skip dataset tokenization")
    p.add_argument(
        "--resume-data",
        action="store_true",
        help="Resume tokenization from .progress sidecar (auto-detected if present)",
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
    resume: bool = False,
) -> None:
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
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_bytes = (seq_length + 1) * np.dtype(np.int32).itemsize
    if start_written == 0 and out_path.exists():
        out_path.unlink()
    memmap = np.memmap(
        out_path,
        dtype=np.int32,
        mode="w+" if start_written == 0 else "r+",
        shape=(num_sequences, seq_length + 1),
    )

    # FineWeb pages can be multi-MB; tokenizing whole documents OOMs login nodes.
    max_chars = (seq_length + 1) * 12
    max_buffer_tokens = (seq_length + 1) * 8

    buffer: list[int] = []
    written = start_written
    row_idx = start_row - 1
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
            chunk = buffer[: seq_length + 1]
            buffer = buffer[seq_length:]
            memmap[written] = np.asarray(chunk, dtype=np.int32)
            written += 1
            if written % 1000 == 0:
                memmap.flush()
                _write_progress(progress_path, written, row_idx + 1)
                print(f"  {written}/{num_sequences} sequences", flush=True)
        if written >= num_sequences:
            break

    memmap.flush()
    if written < num_sequences:
        _write_progress(progress_path, written, row_idx + 1)
        raise RuntimeError(
            f"Only tokenized {written}/{num_sequences} sequences. "
            "Increase streaming time or lower --num-sequences."
        )
    progress_path.unlink(missing_ok=True)
    print(f"Dataset OK: {out_path} ({written} sequences, {written * row_bytes / 1e9:.2f} GB)")


def write_manifest(project_root: Path, model_dir: Path, data_path: Path, args: argparse.Namespace) -> None:
    manifest = project_root / "cache" / "offline_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        f"model_dir={model_dir}\n"
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
    data_path = project_root / "cache" / "data" / f"train_{args.seq_length}.bin"

    if not args.skip_model:
        verify_hf_auth(args.model)
        download_model(args.model, model_dir, force=args.force_model_redownload)
        gc.collect()
    elif model_dir.is_dir():
        print(f"Skipping model download; using {model_dir}")
    else:
        print(f"ERROR: --skip-model but {model_dir} not found", file=sys.stderr)
        sys.exit(1)

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
            resume=resume_data,
        )

    write_manifest(project_root, model_dir, data_path, args)
    print("\nPrefetch complete. GPU jobs can run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
