"""SOCKET soft-LSH collision scoring and top-k selection (PyTorch)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sparse_attn.backends.base import SparseAttentionBackend, SparseMask
from sparse_attn.backends.utils import always_keep_indices
from sparse_attn.config import SocketConfig
from sparse_attn.kernels.py_gather import sparse_attention
from sparse_attn.retrieval.socket_lsh import (
    socket_collision_scores_dense,
    socket_select_topk_mask,
)


class SocketMasker(torch.nn.Module):
    """
    Soft collision kernel: queries distribute mass over R=2^P buckets per table;
    keys are hard-assigned; scores aggregate across L tables.
    """

    def __init__(self, head_dim: int, cfg: SocketConfig, training: bool = True):
        super().__init__()
        self.cfg = cfg
        self.num_tables = cfg.train_l if training else cfg.bucket_l
        self.num_buckets = 2**cfg.bucket_k
        gen = torch.Generator().manual_seed(42)
        proj = torch.randn(self.num_tables, head_dim, cfg.bucket_k, generator=gen)
        proj = proj / proj.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("hash_proj", proj)

        codes = torch.arange(self.num_buckets)
        powers = 2 ** torch.arange(cfg.bucket_k)
        bits = ((codes.unsqueeze(-1) // powers) % 2).float()
        signs = bits * 2 - 1
        self.register_buffer("bucket_signs", signs)

    def _key_bucket_codes(self, k: torch.Tensor) -> torch.LongTensor:
        dots = torch.einsum("bhtd,lde->bhtle", k, self.hash_proj)
        bits = (dots > 0).long()
        powers = 2 ** torch.arange(self.cfg.bucket_k, device=k.device)
        return (bits * powers).sum(-1)

    def _query_soft_scores(self, q: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum("bhqd,lde->bhqle", q, self.hash_proj)
        logits = torch.einsum("bhqle,re->bhqlr", proj, self.bucket_signs) / self.cfg.tau
        return F.softmax(logits, dim=-1)

    def key_codes_and_query_soft(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._key_bucket_codes(k), self._query_soft_scores(q)

    def score_keys_dense(self, q: torch.Tensor, k: torch.Tensor, *, query_chunk: int = 256) -> torch.Tensor:
        """Legacy dense [B,H,Q,T] scores (parity / use_bucket_retrieval=False)."""
        codes, soft_q = self.key_codes_and_query_soft(q, k)
        return socket_collision_scores_dense(codes, soft_q, query_chunk=query_chunk)


class SocketBackend(SparseAttentionBackend, torch.nn.Module):
    """SOCKET sparse attention for SSA SA stream."""

    def __init__(self, head_dim: int, cfg: SocketConfig, training: bool = True):
        super().__init__()
        self.cfg = cfg
        self.masker = SocketMasker(head_dim, cfg, training=training)
        self.last_retrieval_stats: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "socket"

    def _budget(self, seq_len: int) -> int:
        heavy = max(1, int(round(self.cfg.heavy_const * seq_len)))
        keep = self.cfg.sink_size + self.cfg.window_size + heavy
        return min(keep, seq_len)

    def build_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
    ) -> SparseMask:
        del v
        b, h, q_len, _ = q.shape
        t = k.size(2)
        device = q.device
        budget = self._budget(t)
        always = always_keep_indices(t, self.cfg.sink_size, self.cfg.window_size, device)

        codes, soft_q = self.masker.key_codes_and_query_soft(q, k)

        if self.cfg.use_bucket_retrieval:
            indices, scores, stats = socket_select_topk_mask(
                codes,
                soft_q,
                budget=budget,
                top_m_buckets=self.cfg.top_m_buckets,
                always_idx=always,
                causal=causal,
                query_chunk=self.cfg.retrieval_query_chunk,
            )
            self.last_retrieval_stats = stats
            return SparseMask(indices=indices, scores=scores)

        scores = self.masker.score_keys_dense(q, k)
        if causal:
            pos_q = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
            pos_k = torch.arange(t, device=device).view(1, 1, 1, t)
            scores = scores.masked_fill(pos_k > pos_q, float("-inf"))

        topk_scores, topk_idx = torch.topk(scores, k=min(budget, t), dim=-1)
        always_idx = always.view(1, 1, 1, -1).expand(b, h, q_len, -1)
        combined = torch.cat([always_idx, topk_idx], dim=-1)
        combined = combined[..., : min(budget, combined.size(-1))]
        self.last_retrieval_stats = {"mean_selected_keys": float(combined.size(-1)), "dense_scores": 1.0}
        return SparseMask(indices=combined, scores=topk_scores)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: SparseMask,
    ) -> torch.Tensor:
        return sparse_attention(q, k, v, mask.indices)
