#!/usr/bin/env python3
"""Quick offline diagnostic for SSA training (run on HPC login or GPU node)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sparse_attn.config import HPCConfig
from sparse_attn.models import llama_ssa
from sparse_attn.models.llama_ssa import LlamaSSAModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--data-bin", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=None)
    args = p.parse_args()

    root = args.project_root
    data_bin = args.data_bin or root / "cache/data/train_4096.bin"
    model_dir = args.model_dir or root / "cache/models/Llama-3.2-1B"

    print(f"sparse_attn: {llama_ssa.__file__}")
    print(f"data_bin: {data_bin} exists={data_bin.is_file()}")
    print(f"model_dir: {model_dir} exists={model_dir.is_dir()}")

    flat = np.memmap(data_bin, dtype=np.int32, mode="r")
    row_width = args.seq_length + 1
    for stored in (4096, 2048, 8192, 1024, 512):
        w = stored + 1
        if flat.size % w == 0:
            row_width = w
            break
    rows = flat.reshape(-1, row_width)
    sample = np.asarray(rows[0, : args.seq_length + 1], dtype=np.int64)
    print(
        f"row_width={row_width} num_rows={rows.shape[0]} "
        f"sample_range=[{sample.min()}, {sample.max()}] zero_frac={(sample == 0).mean():.3f}"
    )

    cfg = HPCConfig(seq_length=args.seq_length, per_device_batch_size=1)
    cfg.ssa.checkpoint_fa = True
    cfg.socket.train_l = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = LlamaSSAModel.from_pretrained_base(
        str(model_dir),
        cfg,
        device=device,
        local_files_only=True,
    ).to(dtype=dtype)

    ids = torch.from_numpy(sample[:-1]).unsqueeze(0).to(device)
    labels = torch.from_numpy(sample[1:]).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        base = model(ids, labels, training=False, inference_mode="full")
    print(f"baseline lm={base['lm_loss'].float().item():.4f}")

    model.train()
    out = model(ids, labels, training=True, global_step=0)
    print(
        f"train lm={out['lm_loss'].detach().float().item():.4f} "
        f"align={out['align_loss'].detach().float().item():.4f}"
    )


if __name__ == "__main__":
    main()
