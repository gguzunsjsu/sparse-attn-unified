"""Sparse attention backend protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class SparseMask:
    """Per-query key indices for sparse attention."""

    indices: torch.LongTensor
    scores: torch.Tensor | None = None


class SparseAttentionBackend(ABC):
    """Pluggable sparse routing engine for the SSA SA stream."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def build_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
    ) -> SparseMask: ...

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: SparseMask,
    ) -> torch.Tensor: ...
