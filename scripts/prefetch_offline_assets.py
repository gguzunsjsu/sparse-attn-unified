#!/usr/bin/env python3
"""Download model + tokenized training data on the login node (internet required).

GPU compute nodes on SJSU HPC have no outbound network. Run this once on the
login node before submitting SLURM jobs.

Usage (login node):
  source scripts/activate_env.sh
  python scripts/prefetch_offline_assets.py

Optional:
  python scripts/prefetch_offline_assets.py --num-sequences 170000 --seq-length 4096
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
        help="Tokenized sequences to cache (10000 steps x grad_accum 16 = 160k minimum)",
    )
    p.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
        help="HF dataset name for streaming tokenization",
    )
    p.add_argument("--dataset-config", default="sample-10BT")
    return p.parse_args()


def download_model(model_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading model {model_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Verifying local model load (offline)...")
    AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True, trust_remote_code=True)
    AutoTokenizer.from_pretrained(local_dir, local_files_only=True, trust_remote_code=True)
    print(f"Model OK: {local_dir}")


def tokenize_dataset(
    model_dir: Path,
    out_path: Path,
    dataset_name: str,
    dataset_config: str,
    seq_length: int,
    num_sequences: int,
) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Tokenizing {num_sequences} sequences (len={seq_length}) -> {out_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Shape [N, seq_length+1] — training script splits into input/labels
    memmap = np.memmap(
        out_path,
        dtype=np.int32,
        mode="w+",
        shape=(num_sequences, seq_length + 1),
    )

    buffer: list[int] = []
    written = 0
    for row in ds:
        ids = tokenizer(row["text"], add_special_tokens=True)["input_ids"]
        buffer.extend(ids)
        while len(buffer) >= seq_length + 1 and written < num_sequences:
            chunk = buffer[: seq_length + 1]
            buffer = buffer[seq_length:]
            memmap[written] = np.asarray(chunk, dtype=np.int32)
            written += 1
            if written % 1000 == 0:
                print(f"  {written}/{num_sequences} sequences", flush=True)
        if written >= num_sequences:
            break

    memmap.flush()
    if written < num_sequences:
        raise RuntimeError(
            f"Only tokenized {written}/{num_sequences} sequences. "
            "Increase streaming time or lower --num-sequences."
        )
    print(f"Dataset OK: {out_path} ({written} sequences)")


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

    if not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache" / "huggingface" / "token").exists():
        print("NOTE: Llama 3.2 1B is gated. Run `huggingface-cli login` on the login node first.")

    download_model(args.model, model_dir)
    tokenize_dataset(
        model_dir,
        data_path,
        args.dataset,
        args.dataset_config,
        args.seq_length,
        args.num_sequences,
    )
    write_manifest(project_root, model_dir, data_path, args)
    print("\nPrefetch complete. GPU jobs can run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
