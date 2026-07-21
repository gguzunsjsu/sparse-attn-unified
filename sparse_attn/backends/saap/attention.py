"""Soft-SAAP: differentiable asymmetric partitioning for SSA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_attn.backends.base import SparseAttentionBackend, SparseMask
from sparse_attn.backends.utils import always_keep_indices, sparse_attention
from sparse_attn.config import SaapConfig


class SoftSaapRouter(nn.Module):
    """Query-side MLP with Gumbel-Softmax cluster selection."""

    def __init__(self, head_dim: int, cfg: SaapConfig):
        super().__init__()
        self.cfg = cfg
        hidden = min(cfg.router_hidden, head_dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(head_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, cfg.num_clusters),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B, H, Q, D] -> cluster weights [B, H, Q, C]
        logits = self.mlp(q)
        if self.training and self.cfg.use_soft_routing:
            return F.gumbel_softmax(logits, tau=self.cfg.gumbel_tau, hard=False, dim=-1)
        top = logits.topk(self.cfg.top_m_clusters, dim=-1).indices
        weights = torch.zeros_like(logits).scatter_(-1, top, 1.0)
        return weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)


class SaapCentroids(nn.Module):
    """Key-side k-means centroids with nearest-centroid assignment."""

    def __init__(self, num_clusters: int, head_dim: int):
        super().__init__()
        self.num_clusters = num_clusters
        centroids = torch.randn(num_clusters, head_dim)
        centroids = centroids / centroids.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("centroids", centroids)

    @torch.no_grad()
    def refresh(self, keys: torch.Tensor, steps: int = 10) -> None:
        """Lloyd k-means on flattened key vectors."""
        flat = keys.reshape(-1, keys.size(-1))
        n = flat.size(0)
        perm = torch.randperm(n, device=flat.device)[: min(n, 50_000)]
        sample = flat[perm]
        for _ in range(steps):
            dist = torch.cdist(sample, self.centroids)
            assign = dist.argmin(dim=-1)
            for c in range(self.num_clusters):
                mask = assign == c
                if mask.any():
                    self.centroids[c] = sample[mask].mean(dim=0)
        self.centroids.copy_(F.normalize(self.centroids, dim=-1))

    def assign_keys(self, k: torch.Tensor) -> torch.LongTensor:
        # k: [B, H, T, D] -> cluster id [B, H, T]
        b, h, t, d = k.shape
        flat = k.reshape(b * h * t, d)
        dist = torch.cdist(flat, self.centroids)
        assign = dist.argmin(dim=-1)
        return assign.view(b, h, t)


class SaapBackend(SparseAttentionBackend, nn.Module):
    """Soft-SAAP sparse attention for SSA SA stream."""

    def __init__(self, head_dim: int, cfg: SaapConfig):
        super().__init__()
        self.cfg = cfg
        self.router = SoftSaapRouter(head_dim, cfg)
        self.centroids = SaapCentroids(cfg.num_clusters, head_dim)
        self._key_clusters: torch.LongTensor | None = None

    @property
    def name(self) -> str:
        return "saap"

    def maybe_refresh_centroids(self, k: torch.Tensor, step: int) -> None:
        if step > 0 and step % self.cfg.refresh_interval == 0:
            self.centroids.refresh(k.detach())

    def _compute_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        causal: bool = True,
    ) -> torch.Tensor:
        key_clusters = self.centroids.assign_keys(k)
        query_weights = self.router(q)
        key_onehot = F.one_hot(key_clusters, self.cfg.num_clusters).float()
        scores = torch.einsum("bhqc,bhtc->bhqt", query_weights, key_onehot)
        if causal:
            q_len, t = q.size(2), k.size(2)
            device = q.device
            pos_q = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
            pos_k = torch.arange(t, device=device).view(1, 1, 1, t)
            scores = scores.masked_fill(pos_k > pos_q, float("-inf"))
        return scores

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

        scores = self._compute_scores(q, k, causal=causal)
        self._key_clusters = self.centroids.assign_keys(k)

        always = always_keep_indices(t, sink_size=16, window_size=16, device=device)
        always_idx = always.view(1, 1, 1, -1).expand(b, h, q_len, -1)

        heavy = max(32, int(0.15 * t))
        _, topk_idx = torch.topk(scores, k=min(heavy, t), dim=-1)
        combined = torch.cat([always_idx, topk_idx], dim=-1)
        combined = combined[..., : min(heavy + always.numel(), combined.size(-1))]
        return SparseMask(indices=combined, scores=scores)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: SparseMask,
    ) -> torch.Tensor:
        if self.training and self.cfg.use_soft_routing:
            scores = mask.scores if mask.scores is not None else self._compute_scores(q, k)
            probs = F.softmax(scores, dim=-1)
            return torch.einsum("bhqt,bhtd->bhqd", probs, v)
        return sparse_attention(q, k, v, mask.indices)
