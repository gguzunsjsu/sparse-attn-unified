"""PyTorch gather-based sparse attention (debug / fallback kernel)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.LongTensor,
    *,
    query_chunk: int = 512,
) -> torch.Tensor:
    """
    Gather selected keys/values and compute causal softmax attention.

    q: [B, H, Q, D]
    k, v: [B, H, T, D]
    indices: [B, H, Q, K_sel] key positions per query
    """
    _, _, q_len, _ = q.shape
    if q_len <= query_chunk:
        return _sparse_attention_block(q, k, v, indices)

    chunks = []
    for q_start in range(0, q_len, query_chunk):
        q_end = min(q_start + query_chunk, q_len)
        chunks.append(
            _sparse_attention_block(
                q[:, :, q_start:q_end, :],
                k,
                v,
                indices[:, :, q_start:q_end, :],
            )
        )
    return torch.cat(chunks, dim=2)


def _sparse_attention_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.LongTensor,
) -> torch.Tensor:
    b, h, q_len, d = q.shape

    b_idx = torch.arange(b, device=q.device)[:, None, None, None]
    h_idx = torch.arange(h, device=q.device)[None, :, None, None]
    k_g = k[b_idx, h_idx, indices, :]
    v_g = v[b_idx, h_idx, indices, :]

    scale = 1.0 / math.sqrt(d)
    scores = torch.einsum("bhqd,bhqkd->bhqk", q, k_g) * scale

    key_pos = indices
    query_pos = torch.arange(q_len, device=q.device).view(1, 1, q_len, 1)
    causal = key_pos > query_pos
    scores = scores.masked_fill(causal, float("-inf"))

    probs = F.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhqkd->bhqd", probs, v_g)
