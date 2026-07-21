#!/usr/bin/env python3
"""Train Llama-1B with SSA + SOCKET/SAAP on a single H100."""

from __future__ import annotations

import argparse
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
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Run 5 steps on random data")
    return p.parse_args()


class RandomTokenDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_length: int):
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def __iter__(self):
        while True:
            ids = torch.randint(0, self.vocab_size, (self.seq_length + 1,))
            yield {"input_ids": ids[:-1], "labels": ids[1:]}


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
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    args = parse_args()
    cfg = HPCConfig(
        seq_length=args.seq_length,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        base_model=args.base_model,
    )
    cfg.ssa = SSAConfig(sparse_backend=args.sparse_backend)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if cfg.bf16 and device == "cuda" else torch.float32

    if args.from_scratch or args.smoke_test:
        model = LlamaSSAModel(cfg, training=True).to(device=device, dtype=dtype)
    else:
        model = LlamaSSAModel.from_pretrained_base(cfg.base_model, cfg, device=device)
        model = model.to(dtype=dtype)

    print(f"Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"Backend: {cfg.ssa.sparse_backend} | seq={cfg.seq_length} | batch={cfg.per_device_batch_size}")

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

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
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
