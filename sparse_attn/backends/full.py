"""Full (dense) attention backend for the SSA FA stream."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sparse_attn.backends.base import SparseAttentionBackend, SparseMask


class FullAttentionBackend(torch.nn.Module, SparseAttentionBackend):
    """Standard scaled dot-product attention with optional FlashAttention."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    @property
    def name(self) -> str:
        return "full"

    def build_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
    ) -> SparseMask:
        raise NotImplementedError("Full attention does not use sparse masks.")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: SparseMask | None = None,
    ) -> torch.Tensor:
        # q, k, v: [B, H, T, D]
        if self._use_flash:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        t = q.size(-2)
        causal_mask = torch.triu(
            torch.ones(t, k.size(-2), device=q.device, dtype=torch.bool),
            diagonal=k.size(-2) - t + 1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            probs = F.dropout(probs, p=self.dropout)
        return torch.matmul(probs, v)
