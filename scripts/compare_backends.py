#!/usr/bin/env python3
"""Side-by-side dense FA vs SOCKET vs SAAP: LM loss and throughput (same checkpoint & data)."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BENCH_PATH = Path(__file__).resolve().parent / "benchmark_sparse_attention.py"
_spec = importlib.util.spec_from_file_location("benchmark_sparse_attention", _BENCH_PATH)
_bench = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None and _spec.name is not None
sys.modules[_spec.name] = _bench
_spec.loader.exec_module(_bench)

_log = _bench._log
_peak_mem_gb = _bench._peak_mem_gb
_reset_peak_mem = _bench._reset_peak_mem
_bench_fn = _bench._bench
_sample_batch = _bench._sample_batch
_load_model = _bench.load_model
_run_once = _bench._run_once


@dataclass
class CompareRow:
    label: str
    ms_per_iter: float
    tokens_per_sec: float
    lm_loss: float
    peak_mem_gb: float | None
    lm_loss_eval: float | None = None


def _free(model: torch.nn.Module | None, device: str) -> None:
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def _mean_lm_loss(
    model: torch.nn.Module,
    data_bin: Path | None,
    *,
    seq_length: int,
    batch_size: int,
    device: str,
    n_batches: int,
    training: bool,
    inference_mode: str,
) -> float:
    model.eval() if not training else model.train()
    total = 0.0
    for bi in range(n_batches):
        input_ids, labels = _sample_batch(
            data_bin,
            seq_length=seq_length,
            batch_size=batch_size,
            device=device,
            batch_offset=bi * batch_size,
        )
        if training:
            model.train()
            lm = _run_once(
                model,
                input_ids,
                labels,
                training=True,
                inference_mode=inference_mode,
                backward=False,
            )
        else:
            with torch.inference_mode():
                lm = _run_once(
                    model,
                    input_ids,
                    labels,
                    training=False,
                    inference_mode=inference_mode,
                    backward=False,
                )
        total += float(lm.float().item())
    return total / n_batches


def _run_path(
    label: str,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    training: bool,
    inference_mode: str,
    backward: bool,
    warmup: int,
    iters: int,
    device: str,
    tokens: int,
    data_bin: Path | None,
    seq_length: int,
    batch_size: int,
    loss_batches: int,
) -> CompareRow:
    model._cos = torch.empty(0, device=device)
    model._sin = torch.empty(0, device=device)
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    _reset_peak_mem(device)

    lm_eval: float | None = None
    if loss_batches > 1:
        lm_eval = _mean_lm_loss(
            model,
            data_bin,
            seq_length=seq_length,
            batch_size=batch_size,
            device=device,
            n_batches=loss_batches,
            training=training,
            inference_mode=inference_mode,
        )

    ms, lm = _bench_fn(
        label,
        model,
        input_ids,
        labels,
        training=training,
        inference_mode=inference_mode,
        backward=backward,
        warmup=warmup,
        iters=iters,
        device=device,
    )
    peak = _peak_mem_gb(device)
    assert lm is not None
    tps = tokens / (ms / 1000.0)
    return CompareRow(
        label=label,
        ms_per_iter=ms,
        tokens_per_sec=tps,
        lm_loss=lm,
        peak_mem_gb=peak,
        lm_loss_eval=lm_eval,
    )


def _print_table(rows: list[CompareRow], dense_lm: float) -> None:
    _log("")
    _log(f"{'path':<22} {'ms/iter':>10} {'tok/s':>12} {'lm_loss':>10} {'Δ vs dense':>12} {'peak_GB':>9}")
    _log("-" * 80)
    for r in rows:
        lm_show = r.lm_loss_eval if r.lm_loss_eval is not None else r.lm_loss
        delta_show = lm_show - dense_lm
        mem = f"{r.peak_mem_gb:.2f}" if r.peak_mem_gb is not None else "—"
        _log(
            f"{r.label:<22} {r.ms_per_iter:10.2f} {r.tokens_per_sec:12,.0f} "
            f"{lm_show:10.4f} {delta_show:+12.4f} {mem:>9}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare dense FA vs SOCKET vs SAAP (loss + throughput on same weights/data)"
    )
    p.add_argument("--project-root", type=Path, default=_ROOT)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--data-bin", type=Path, default=None)
    p.add_argument("--seq-length", type=int, nargs="+", default=[2048])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--loss-batches", type=int, default=1, help=">1 averages LM loss over extra batches (timed on first batch)")
    p.add_argument("--checkpoint-fa", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--socket-train-l", type=int, default=8)
    p.add_argument(
        "--match-sparsity",
        action="store_true",
        help="Use SOCKET sink/window/heavy_const for SAAP (fairer K; default configs differ)",
    )
    p.add_argument(
        "--include-train",
        action="store_true",
        help="Also benchmark dual-stream train_forward and train_step per sparse backend",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root
    model_dir = args.model_dir or root / "cache/models/Llama-3.2-1B"
    data_bin = args.data_bin or root / "cache/data/train_2048.bin"
    data_path = data_bin if data_bin.is_file() else None

    if not model_dir.is_dir():
        _log(f"ERROR: model not found: {model_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        _log(f"GPU: {props.name} | torch {torch.__version__} | dtype={dtype}")
    else:
        _log("WARN: CUDA not available — timings are not meaningful for H100 comparison.")

    if args.checkpoint:
        _log(f"Checkpoint: {args.checkpoint}")
    if not args.match_sparsity:
        _log(
            "NOTE: default SOCKET (~269 keys @2048) vs SAAP (~80 keys) differ; "
            "pass --match-sparsity for comparable sparsity budgets."
        )

    for seq_len in args.seq_length:
        _log("")
        _log(f"=== seq_length={seq_len} batch_size={args.batch_size} ===")
        input_ids, labels = _sample_batch(
            data_path,
            seq_length=seq_len,
            batch_size=args.batch_size,
            device=device,
        )
        actual_seq = int(input_ids.shape[1])
        tokens = actual_seq * args.batch_size

        infer_rows: list[CompareRow] = []

        model = _load_model(
            model_dir=model_dir,
            checkpoint=args.checkpoint,
            device=device,
            dtype=dtype,
            sparse_backend="socket",
            checkpoint_fa=args.checkpoint_fa,
            socket_train_l=args.socket_train_l,
        )
        infer_rows.append(
            _run_path(
                "dense (FA)",
                model,
                input_ids,
                labels,
                training=False,
                inference_mode="full",
                backward=False,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
                tokens=tokens,
                data_bin=data_path,
                seq_length=actual_seq,
                batch_size=args.batch_size,
                loss_batches=args.loss_batches,
            )
        )
        dense_lm = infer_rows[0].lm_loss_eval if infer_rows[0].lm_loss_eval is not None else infer_rows[0].lm_loss_eval or infer_rows[0].lm_loss
        _free(model, device)

        for backend, label in (("socket", "socket (sparse)"), ("saap", "saap (sparse)")):
            model = _load_model(
                model_dir=model_dir,
                checkpoint=args.checkpoint,
                device=device,
                dtype=dtype,
                sparse_backend=backend,
                checkpoint_fa=args.checkpoint_fa,
                socket_train_l=args.socket_train_l,
                match_saap_sparsity_to_socket=args.match_sparsity,
            )
            infer_rows.append(
                _run_path(
                    label,
                    model,
                    input_ids,
                    labels,
                    training=False,
                    inference_mode="sparse",
                    backward=False,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                    tokens=tokens,
                    data_bin=data_path,
                    seq_length=actual_seq,
                    batch_size=args.batch_size,
                    loss_batches=args.loss_batches,
                )
            )
            _free(model, device)

        _log("")
        _log("Inference (eval forward, single attention path):")
        _print_table(infer_rows, dense_lm)

        dense_ms = infer_rows[0].ms_per_iter
        for r in infer_rows[1:]:
            ratio = dense_ms / r.ms_per_iter if r.ms_per_iter > 0 else float("nan")
            _log(f"  dense / {r.label}: {ratio:.2f}x ({'sparse faster' if ratio > 1 else 'dense faster'})")

        if args.include_train:
            _log("")
            _log("Training (dual-stream SSA on configured sparse backend):")
            train_rows: list[CompareRow] = []
            for backend, label in (("socket", "socket train_step"), ("saap", "saap train_step")):
                model = _load_model(
                    model_dir=model_dir,
                    checkpoint=args.checkpoint,
                    device=device,
                    dtype=dtype,
                    sparse_backend=backend,
                    checkpoint_fa=args.checkpoint_fa,
                    socket_train_l=args.socket_train_l,
                    match_saap_sparsity_to_socket=args.match_sparsity,
                )
                train_rows.append(
                    _run_path(
                        label,
                        model,
                        input_ids,
                        labels,
                        training=True,
                        inference_mode="sparse",
                        backward=True,
                        warmup=max(1, args.warmup - 1),
                        iters=args.iters,
                        device=device,
                        tokens=tokens,
                        data_bin=data_path,
                        seq_length=actual_seq,
                        batch_size=args.batch_size,
                        loss_batches=1,
                    )
                )
                _free(model, device)
            _print_table(train_rows, dense_lm)

    _log("")
    _log("Notes:")
    _log("- SOCKET-trained checkpoints load shared weights into SAAP; router/centroids stay at init.")
    _log("- Throughput uses PyTorch SOCKET/SAAP retrieval in this repo, not amarka8/SOCKET CUDA kernels.")
    _log("- Layer microbench: python scripts/benchmark_attention.py --seq-length 2048 --backend both --compare-dense")


if __name__ == "__main__":
    main()
