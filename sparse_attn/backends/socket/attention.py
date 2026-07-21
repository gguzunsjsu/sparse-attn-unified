"""SOCKET soft-LSH collision scoring and top-k selection (PyTorch)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sparse_attn.backends.base import SparseAttentionBackend, SparseMask
from sparse_attn.backends.utils import always_keep_indices, sparse_attention
from sparse_attn.config import SocketConfig


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
        # Projections [B, H, Q, L, P] -> bucket logits [B, H, Q, L, R]
        proj = torch.einsum("bhqd,lde->bhqle", q, self.hash_proj)
        logits = torch.einsum("bhqle,re->bhqlr", proj, self.bucket_signs) / self.cfg.tau
        return F.softmax(logits, dim=-1)

    def score_keys(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Soft collision scores [B, H, Q, T]."""
        codes = self._key_bucket_codes(k)
        soft_q = self._query_soft_scores(q)
        b, h, t, l_tables = codes.shape
        _, _, q_len, _, _ = soft_q.shape

        scores = torch.zeros(b, h, q_len, t, device=q.device, dtype=q.dtype)
        for table_idx in range(l_tables):
            bucket = codes[..., table_idx]
            table_scores = torch.gather(
                soft_q[:, :, :, table_idx, :],
                dim=-1,
                index=bucket.unsqueeze(2).expand(b, h, q_len, t),
            )
            scores = scores + table_scores
        return scores / l_tables


class SocketBackend(SparseAttentionBackend, torch.nn.Module):
    """SOCKET sparse attention for SSA SA stream."""

    def __init__(self, head_dim: int, cfg: SocketConfig, training: bool = True):
        super().__init__()
        self.cfg = cfg
        self.masker = SocketMasker(head_dim, cfg, training=training)

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

        scores = self.masker.score_keys(q, k)
        if causal:
            pos_q = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
            pos_k = torch.arange(t, device=device).view(1, 1, 1, t)
            scores = scores.masked_fill(pos_k > pos_q, float("-inf"))

        budget = self._budget(t)
        topk_scores, topk_idx = torch.topk(scores, k=min(budget, t), dim=-1)

        always = always_keep_indices(t, self.cfg.sink_size, self.cfg.window_size, device)
        always_idx = always.view(1, 1, 1, -1).expand(b, h, q_len, -1)
        combined = torch.cat([always_idx, topk_idx], dim=-1)
        combined = combined[..., : min(budget, combined.size(-1))]
        return SparseMask(indices=combined, scores=topk_scores)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: SparseMask,
    ) -> torch.Tensor:
        return sparse_attention(q, k, v, mask.indices)
