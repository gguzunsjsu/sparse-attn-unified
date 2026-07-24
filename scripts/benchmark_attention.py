#!/usr/bin/env python3
"""Attention-layer microbenchmark: retrieval vs sparse-attn kernel vs dense FA."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from sparse_attn.backends.full import FullAttentionBackend
from sparse_attn.backends.saap.attention import SaapBackend
from sparse_attn.backends.socket.attention import SocketBackend
from sparse_attn.config import SaapConfig, SocketConfig
from sparse_attn.kernels.py_gather import sparse_attention


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _bench(fn, warmup: int, iters: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    p = argparse.ArgumentParser(description="Single-layer attention retrieval/kernel benchmark")
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--backend", choices=["socket", "saap", "both"], default="both")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--compare-dense", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    b, h, t, d = args.batch_size, args.heads, args.seq_length, args.head_dim
    q = torch.randn(b, h, t, d, device=device, dtype=dtype)
    k = torch.randn(b, h, t, d, device=device, dtype=dtype)
    v = torch.randn(b, h, t, d, device=device, dtype=dtype)
    tokens = b * t

    print(f"device={device} dtype={dtype} shape=[{b},{h},{t},{d}]", flush=True)

    if args.compare_dense:
        fa = FullAttentionBackend()
        ms = _bench(lambda: fa.forward(q, k, v), args.warmup, args.iters, device)
        print(f"dense_fa          {ms:8.2f} ms   {tokens / (ms / 1000):,.0f} tok/s", flush=True)

    if args.backend in ("socket", "both"):
        print("socket (bucket retrieval)...", flush=True)
        scfg = SocketConfig(train_l=8, top_m_buckets=8, use_bucket_retrieval=True)
        sb = SocketBackend(d, scfg, training=False).to(device=device, dtype=dtype)
        mask_holder: list = []

        def build():
            mask_holder.clear()
            mask_holder.append(sb.build_mask(q, k, v))

        def attn():
            if not mask_holder:
                build()
            sb.forward(q, k, v, mask_holder[0])

        ms_build = _bench(build, args.warmup, args.iters, device)
        ms_attn = _bench(attn, args.warmup, args.iters, device)
        stats = sb.last_retrieval_stats
        print(
            f"socket_retrieval  {ms_build:8.2f} ms   stats={stats} "
            f"(bucket top_m={scfg.top_m_buckets})"
        )
        print(f"socket_attn       {ms_attn:8.2f} ms   {tokens / (ms_attn / 1000):,.0f} tok/s")

        scfg_dense = SocketConfig(train_l=8, use_bucket_retrieval=False)
        sb_d = SocketBackend(d, scfg_dense, training=False).to(device=device, dtype=dtype)
        ms_dense_build = _bench(lambda: sb_d.build_mask(q, k, v), args.warmup, args.iters, device)
        print(f"socket_dense_mask {ms_dense_build:8.2f} ms  (legacy QxT scores)", flush=True)

    if args.backend in ("saap", "both"):
        print("saap (cluster retrieval)...", flush=True)
        acfg = SaapConfig(num_clusters=64, top_m_clusters=2, use_cluster_retrieval=True)
        ab = SaapBackend(d, acfg).to(device=device, dtype=dtype)
        ab.train()
        mask_holder: list = []

        def build_saap():
            mask_holder.clear()
            mask_holder.append(ab.build_mask(q, k, v))

        def attn_saap():
            if not mask_holder:
                build_saap()
            ab.forward(q, k, v, mask_holder[0])

        ms_build = _bench(build_saap, args.warmup, args.iters, device)
        ms_attn = _bench(attn_saap, args.warmup, args.iters, device)
        print(f"saap_retrieval    {ms_build:8.2f} ms   stats={ab.last_retrieval_stats}")
        print(f"saap_attn         {ms_attn:8.2f} ms   {tokens / (ms_attn / 1000):,.0f} tok/s")


if __name__ == "__main__":
    main()
