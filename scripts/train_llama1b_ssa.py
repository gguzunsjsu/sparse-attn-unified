#!/usr/bin/env python3
"""Train Llama-1B with SSA + SOCKET/SAAP on a single H100."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from sparse_attn.config import HPCConfig, SSAConfig
from sparse_attn.models.llama_ssa import LlamaSSAModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSA Llama-1B training")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--sparse-backend", choices=["socket", "saap"], default="socket")
    p.add_argument("--seq-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument(
        "--local-model-dir",
        type=str,
        default=None,
        help="Local model directory (required on offline GPU nodes)",
    )
    p.add_argument(
        "--local-data-bin",
        type=str,
        default=None,
        help="Memmap tokenized data (.bin) from prefetch_offline_assets.py",
    )
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Run 5 steps on random data")
    p.add_argument(
        "--checkpoint-fa",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Gradient-checkpoint the FA stream (saves VRAM, slower)",
    )
    p.add_argument("--socket-train-l", type=int, default=None, help="SOCKET LSH tables during training")
    return p.parse_args()


def setup_cuda_perf() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


class RandomTokenDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_length: int):
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def __iter__(self):
        while True:
            ids = torch.randint(0, self.vocab_size, (self.seq_length + 1,))
            yield {"input_ids": ids[:-1], "labels": ids[1:]}


class LocalMemmapDataset(IterableDataset):
    """Read pre-tokenized sequences from prefetch_offline_assets.py (offline-safe)."""

    def __init__(self, bin_path: Path, seq_length: int):
        import numpy as np

        self.seq_length = seq_length
        flat = np.memmap(bin_path, dtype=np.int32, mode="r")
        row_width = self._infer_row_width(flat.size, seq_length)
        if flat.size % row_width != 0:
            raise ValueError(f"Invalid bin file shape for seq_length={seq_length}: {bin_path}")
        self.stored_seq = row_width - 1
        self.num_rows = flat.size // row_width
        self.data = flat.reshape(self.num_rows, row_width)

    @staticmethod
    def _infer_row_width(total_size: int, seq_length: int) -> int:
        preferred = seq_length + 1
        if total_size % preferred == 0:
            return preferred
        for stored in (4096, 2048, 8192, 1024, 512):
            width = stored + 1
            if total_size % width == 0:
                return width
        raise ValueError(f"Cannot infer row width from file size {total_size}")

    def __iter__(self):
        import numpy as np

        while True:
            perm = np.random.permutation(self.num_rows)
            for idx in perm:
                row = self.data[idx]
                if self.stored_seq > self.seq_length:
                    row = row[: self.seq_length + 1]
                input_ids = torch.from_numpy(np.asarray(row[:-1], dtype=np.int64))
                yield {"input_ids": input_ids, "labels": input_ids.clone()}


class StreamingTextDataset(IterableDataset):
    def __init__(self, cfg: HPCConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset(
            self.cfg.dataset_name,
            self.cfg.dataset_config,
            split="train",
            streaming=True,
        )
        buffer: list[int] = []
        for row in ds:
            ids = self.tokenizer(row["text"], add_special_tokens=True)["input_ids"]
            buffer.extend(ids)
            while len(buffer) >= self.cfg.seq_length + 1:
                chunk = buffer[: self.cfg.seq_length + 1]
                buffer = buffer[self.cfg.seq_length :]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t[:-1], "labels": t[1:]}


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch]).long()
    labels = torch.stack([b["labels"] for b in batch]).long()
    return {"input_ids": input_ids, "labels": labels}


def resolve_offline_paths(project_root: Path, args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    """Default local paths from prefetch script."""
    model_dir = Path(args.local_model_dir) if args.local_model_dir else project_root / "cache/models/Llama-3.2-1B"
    data_bin = Path(args.local_data_bin) if args.local_data_bin else project_root / f"cache/data/train_{args.seq_length}.bin"
    model_path = model_dir if model_dir.is_dir() else None
    data_path = data_bin if data_bin.is_file() else None
    return model_path, data_path


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    local_model, local_data = resolve_offline_paths(project_root, args)
    cfg = HPCConfig(
        seq_length=args.seq_length,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        base_model=args.base_model,
    )
    cfg.ssa = SSAConfig(sparse_backend=args.sparse_backend)
    if args.checkpoint_fa is not None:
        cfg.ssa.checkpoint_fa = args.checkpoint_fa
    if args.socket_train_l is not None:
        cfg.socket.train_l = args.socket_train_l
    if args.output_dir:
        cfg.output_dir = args.output_dir

    setup_cuda_perf()

    if args.smoke_test:
        # Safe defaults for dual-stream SSA smoke test on 1× H100
        if args.seq_length == 4096:
            cfg.seq_length = 512
        if args.batch_size == 2:
            cfg.per_device_batch_size = 1
        # Lighter SOCKET settings for smoke test
        cfg.socket.train_l = min(cfg.socket.train_l, 8)
        cfg.socket.heavy_const = 0.15

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if cfg.bf16 and device == "cuda" else torch.float32

    if args.from_scratch or args.smoke_test:
        model = LlamaSSAModel(cfg, training=True).to(device=device, dtype=dtype)
    else:
        model_source = str(local_model) if local_model else cfg.base_model
        local_files_only = offline or local_model is not None
        if offline and local_model is None:
            raise FileNotFoundError(
                f"Offline mode but model not found at {project_root / 'cache/models/Llama-3.2-1B'}. "
                "Run on login node: bash scripts/prefetch_offline_assets.sh"
            )
        model = LlamaSSAModel.from_pretrained_base(
            model_source,
            cfg,
            device=device,
            local_files_only=local_files_only,
        )
        model = model.to(dtype=dtype)

    print(f"Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(
        f"Backend: {cfg.ssa.sparse_backend} | seq={cfg.seq_length} | batch={cfg.per_device_batch_size} "
        f"| grad_accum={cfg.gradient_accumulation_steps} | checkpoint_fa={cfg.ssa.checkpoint_fa} "
        f"| socket_L={cfg.socket.train_l}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    if args.smoke_test:
        model.train()
        for step in range(5):
            ids = torch.randint(100, 5000, (cfg.per_device_batch_size, cfg.seq_length), device=device)
            labels = ids.clone()
            out = model(ids, labels, training=True, global_step=step)
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()
            print(
                f"step {step}: loss={out['loss'].item():.4f} "
                f"lm={out['lm_loss'].item():.4f} align={out['align_loss'].item():.4f}"
            )
        print("Smoke test passed.")
        return

    from transformers import AutoTokenizer

    if local_data is not None:
        print(f"Using offline data: {local_data}")
        dataset = LocalMemmapDataset(local_data, cfg.seq_length)
        if dataset.stored_seq != cfg.seq_length:
            print(f"  Slicing stored seq={dataset.stored_seq} -> train seq={cfg.seq_length}")
    elif offline:
        raise FileNotFoundError(
            f"Offline mode but data not found at {project_root / f'cache/data/train_{cfg.seq_length}.bin'}. "
            "Submit the SLURM job (tokenizes offline) or run: "
            "python scripts/prefetch_offline_assets.py --tokenize-only --offline --skip-model --skip-raw-cache"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_model) if local_model else cfg.base_model,
            trust_remote_code=True,
            local_files_only=local_model is not None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = StreamingTextDataset(cfg, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        collate_fn=collate,
        num_workers=0,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    accum = 0
    pbar = tqdm(total=cfg.max_steps, desc="train")

    while step < cfg.max_steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels, training=True, global_step=step)
            (out["loss"] / cfg.gradient_accumulation_steps).backward()
            accum += 1

            if accum >= cfg.gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                accum = 0
                step += 1
                pbar.update(1)

                if step % cfg.logging_steps == 0:
                    pbar.set_postfix(
                        loss=f"{out['loss'].item():.3f}",
                        lm=f"{out['lm_loss'].item():.3f}",
                        align=f"{out['align_loss'].item():.3f}",
                    )

                if step % cfg.save_steps == 0:
                    ckpt = out_dir / f"checkpoint-{step}"
                    ckpt.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), ckpt / "model.pt")
                    print(f"Saved {ckpt}")

                if step >= cfg.max_steps:
                    break

    pbar.close()
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    print(f"Training complete. Final checkpoint: {out_dir / 'model_final.pt'}")


if __name__ == "__main__":
    main()
