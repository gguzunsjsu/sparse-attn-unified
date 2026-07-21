"""SSA dual-stream attention module."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from sparse_attn.backends.base import SparseAttentionBackend
from sparse_attn.backends.full import FullAttentionBackend
from sparse_attn.backends.saap.attention import SaapBackend
from sparse_attn.backends.socket.attention import SocketBackend
from sparse_attn.config import SSAConfig, SaapConfig, SocketConfig
from sparse_attn.ssa.alignment import alignment_distance


def build_sparse_backend(
    name: str,
    head_dim: int,
    socket_cfg: SocketConfig,
    saap_cfg: SaapConfig,
    training: bool,
) -> SparseAttentionBackend:
    if name == "socket":
        return SocketBackend(head_dim, socket_cfg, training=training)
    if name == "saap":
        return SaapBackend(head_dim, saap_cfg)
    raise ValueError(f"Unknown sparse backend: {name}")


class SSADualStreamAttention(nn.Module):
    """
    SSA layer: computes FA + SA every forward pass, applies alignment loss,
    and randomly routes the primary output through FA or SA (p=0.5).
    """

    def __init__(
        self,
        head_dim: int,
        cfg: SSAConfig,
        socket_cfg: SocketConfig,
        saap_cfg: SaapConfig,
        dropout: float = 0.0,
        training: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.fa = FullAttentionBackend(dropout=dropout)
        self.sa = build_sparse_backend(
            cfg.sparse_backend, head_dim, socket_cfg, saap_cfg, training
        )
        self._last_align_loss: torch.Tensor | None = None

    @property
    def alignment_loss(self) -> torch.Tensor:
        if self._last_align_loss is None:
            return torch.tensor(0.0)
        return self._last_align_loss

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        training: bool = True,
        inference_mode: str = "sparse",
        global_step: int = 0,
    ) -> torch.Tensor:
        if not training:
            if inference_mode == "full":
                return self.fa.forward(q, k, v)
            mask = self.sa.build_mask(q, k, v, causal=True)
            return self.sa.forward(q, k, v, mask)

        if isinstance(self.sa, SaapBackend):
            self.sa.maybe_refresh_centroids(k, global_step)

        def fa_fn(q_, k_, v_):
            return self.fa.forward(q_, k_, v_)

        def sa_fn(q_, k_, v_):
            mask = self.sa.build_mask(q_, k_, v_, causal=True)
            return self.sa.forward(q_, k_, v_, mask)

        if self.cfg.checkpoint_fa:
            o_fa = checkpoint(fa_fn, q, k, v, use_reentrant=False)
        else:
            o_fa = fa_fn(q, k, v)
        o_sa = sa_fn(q, k, v)

        self._last_align_loss = alignment_distance(
            o_fa,
            o_sa,
            metric=self.cfg.align_metric,
            stopgrad_sa=self.cfg.align_stopgrad_sa,
        )

        if torch.rand((), device=q.device) < self.cfg.p_sparse:
            return o_sa
        return o_fa
