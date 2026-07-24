"""Sparse attention utilities shared by SOCKET and SAAP backends."""

from __future__ import annotations

import torch


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding. x: [B, H, T, D]."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 500000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


from sparse_attn.kernels.py_gather import sparse_attention  # noqa: E402


def always_keep_indices(
    seq_len: int,
    sink_size: int,
    window_size: int,
    device: torch.device,
) -> torch.LongTensor:
    """Sink + local window token indices."""
    sink = torch.arange(min(sink_size, seq_len), device=device)
    if window_size > 0 and seq_len > sink_size:
        window = torch.arange(max(0, seq_len - window_size), seq_len, device=device)
        keep = torch.unique(torch.cat([sink, window]))
    else:
        keep = sink
    return keep
