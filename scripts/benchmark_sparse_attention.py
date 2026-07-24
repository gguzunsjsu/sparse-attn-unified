#!/usr/bin/env python3
"""Benchmark SSA Llama throughput: dense FA vs SOCKET sparse vs dual-stream training."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from sparse_attn.config import HPCConfig, SSAConfig
from sparse_attn.models.llama_ssa import LlamaSSAModel


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class BenchResult:
    name: str
    seq_length: int
    batch_size: int
    ms_per_iter: float
    tokens_per_sec: float
    lm_loss: float | None
    peak_mem_gb: float | None


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _reset_peak_mem(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _peak_mem_gb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated() / 1e9


def load_model(
    *,
    model_dir: Path,
    checkpoint: Path | None,
    device: str,
    dtype: torch.dtype,
    sparse_backend: str,
    checkpoint_fa: bool,
    socket_train_l: int,
    match_saap_sparsity_to_socket: bool = False,
) -> LlamaSSAModel:
    cfg = HPCConfig(seq_length=2048, per_device_batch_size=1)
    cfg.ssa = SSAConfig(sparse_backend=sparse_backend, checkpoint_fa=checkpoint_fa)
    cfg.socket.train_l = socket_train_l
    if match_saap_sparsity_to_socket and sparse_backend == "saap":
        cfg.saap.sink_size = cfg.socket.sink_size
        cfg.saap.window_size = cfg.socket.window_size
        cfg.saap.heavy_const = cfg.socket.heavy_const

    model = LlamaSSAModel.from_pretrained_base(
        str(model_dir),
        cfg,
        device=device,
        local_files_only=True,
    ).to(dtype=dtype)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        _log(f"Loaded checkpoint: {checkpoint}")
    else:
        _log("Using base Llama weights (no SSA fine-tune checkpoint)")

    return model


def _sample_batch(
    data_bin: Path | None,
    *,
    seq_length: int,
    batch_size: int,
    device: str,
    batch_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data_bin is not None and data_bin.is_file():
        import numpy as np

        flat = np.memmap(data_bin, dtype=np.int32, mode="r")
        row_width = seq_length + 1
        for stored in (4096, 2048, 8192, 1024, 512):
            w = stored + 1
            if flat.size % w == 0:
                row_width = w
                break
        rows = flat.reshape(-1, row_width)
        stored_seq = row_width - 1
        chunks: list[torch.Tensor] = []
        for b in range(batch_size):
            need = seq_length + 1
            row_idx = (b + batch_offset) % rows.shape[0]
            if seq_length <= stored_seq:
                row = np.asarray(rows[row_idx, : need], dtype=np.int64)
            else:
                # Concatenate across rows when the .bin was tokenized at shorter seq_length.
                flat_ids: list[int] = []
                walk = row_idx
                while len(flat_ids) < need:
                    flat_ids.extend(int(x) for x in rows[walk % rows.shape[0]].tolist())
                    walk += 1
                row = np.asarray(flat_ids[:need], dtype=np.int64)
            chunks.append(torch.from_numpy(row.copy()))
        batch = torch.stack(chunks, dim=0).long().to(device)
        return batch[:, :-1], batch[:, 1:]

    vocab = 128256
    ids = torch.randint(100, vocab - 1, (batch_size, seq_length), device=device)
    labels = ids.clone()
    return ids, labels


def _run_once(
    model: LlamaSSAModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    training: bool,
    inference_mode: str,
    backward: bool,
) -> torch.Tensor:
    model.train(mode=training)
    ctx = nullcontext() if training else torch.inference_mode()
    with ctx:
        out = model(
            input_ids,
            labels,
            training=training,
            inference_mode=inference_mode,
            global_step=0,
        )
        loss = out["loss"]
        if backward:
            loss.backward()
    return out["lm_loss"].detach()


def _bench(
    name: str,
    model: LlamaSSAModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    training: bool,
    inference_mode: str,
    backward: bool,
    warmup: int,
    iters: int,
    device: str,
) -> tuple[float, float | None]:
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _run_once(
            model,
            input_ids,
            labels,
            training=training,
            inference_mode=inference_mode,
            backward=backward,
        )
    _sync(device)

    _reset_peak_mem(device)
    lm_for_loss: float | None = None
    t0 = time.perf_counter()
    for i in range(iters):
        model.zero_grad(set_to_none=True)
        lm = _run_once(
            model,
            input_ids,
            labels,
            training=training,
            inference_mode=inference_mode,
            backward=backward,
        )
        if i == 0:
            lm_for_loss = float(lm.float().item())
    _sync(device)
    elapsed = time.perf_counter() - t0
    ms = (elapsed / iters) * 1000.0
    return ms, lm_for_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark sparse vs full attention throughput")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None, help="model_final.pt or checkpoint-*/model.pt")
    p.add_argument("--data-bin", type=Path, default=None, help="Optional tokenized .bin for realistic input")
    p.add_argument("--seq-length", type=int, nargs="+", default=[2048])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--sparse-backend", choices=["socket", "saap"], default="socket")
    p.add_argument("--checkpoint-fa", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--socket-train-l", type=int, default=8)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["infer_full", "infer_sparse", "train_forward", "train_step"],
        choices=["infer_full", "infer_sparse", "train_forward", "train_step"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root
    model_dir = args.model_dir or root / "cache/models/Llama-3.2-1B"
    data_bin = args.data_bin or root / "cache/data/train_2048.bin"

    if not model_dir.is_dir():
        _log(f"ERROR: model not found: {model_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        _log("WARN: CUDA not available — timings are not meaningful for H100 comparison.")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        _log(f"GPU: {props.name} | torch {torch.__version__} | dtype={dtype}")

    model = load_model(
        model_dir=model_dir,
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
        sparse_backend=args.sparse_backend,
        checkpoint_fa=args.checkpoint_fa,
        socket_train_l=args.socket_train_l,
    )

    mode_specs = {
        "infer_full": dict(training=False, inference_mode="full", backward=False),
        "infer_sparse": dict(training=False, inference_mode="sparse", backward=False),
        "train_forward": dict(training=True, inference_mode="sparse", backward=False),
        "train_step": dict(training=True, inference_mode="sparse", backward=True),
    }

    results: list[BenchResult] = []

    for seq_len in args.seq_length:
        _log("")
        _log(f"=== seq_length={seq_len} batch_size={args.batch_size} ===")
        # Force RoPE rebuild when seq length changes (avoids stale shorter cache).
        model._cos = torch.empty(0, device=device)
        model._sin = torch.empty(0, device=device)
        input_ids, labels = _sample_batch(
            data_bin if data_bin.is_file() else None,
            seq_length=seq_len,
            batch_size=args.batch_size,
            device=device,
        )
        actual_seq = int(input_ids.shape[1])
        if actual_seq != seq_len:
            _log(
                f"WARN: requested seq_length={seq_len} but input has {actual_seq} tokens "
                f"(data bin row width may be shorter; concatenated rows if possible)."
            )
        tokens = actual_seq * args.batch_size

        for mode in args.modes:
            spec = mode_specs[mode]
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            _reset_peak_mem(device)

            ms, lm_loss = _bench(
                mode,
                model,
                input_ids,
                labels,
                training=spec["training"],
                inference_mode=spec["inference_mode"],
                backward=spec["backward"],
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            peak = _peak_mem_gb(device)
            tps = tokens / (ms / 1000.0)
            results.append(
                BenchResult(
                    name=mode,
                    seq_length=actual_seq,
                    batch_size=args.batch_size,
                    ms_per_iter=ms,
                    tokens_per_sec=tps,
                    lm_loss=lm_loss,
                    peak_mem_gb=peak,
                )
            )
            lm_str = f" lm={lm_loss:.4f}" if lm_loss is not None else ""
            mem_str = f" peak_mem={peak:.2f}GB" if peak is not None else ""
            _log(f"{mode:16s}  {ms:8.2f} ms/iter  {tps:8.1f} tok/s{lm_str}{mem_str}")

        # Speedups vs dense inference on this seq length
        by_name = {r.name: r for r in results if r.seq_length == actual_seq}
        if "infer_full" in by_name and "infer_sparse" in by_name:
            full_ms = by_name["infer_full"].ms_per_iter
            sparse_ms = by_name["infer_sparse"].ms_per_iter
            ratio = full_ms / sparse_ms if sparse_ms > 0 else float("nan")
            _log(
                f"infer_full / infer_sparse speedup: {ratio:.2f}x "
                f"({'sparse faster' if ratio > 1 else 'dense faster'})"
            )
        if "infer_full" in by_name and "train_step" in by_name:
            train_ms = by_name["train_step"].ms_per_iter
            full_ms = by_name["infer_full"].ms_per_iter
            _log(f"train_step / infer_full slowdown: {train_ms / full_ms:.2f}x")

    _log("")
    _log("Notes:")
    _log("- infer_* uses a single attention path (FA or SOCKET); train_* runs dual-stream SSA.")
    _log("- SOCKET here is the PyTorch training masker in this repo, not amarka8/SOCKET CUDA kernels.")
    _log("- SOCKET LSH tables are fixed at model build (train_l); inference_mode=sparse still uses train_l.")


if __name__ == "__main__":
    main()
