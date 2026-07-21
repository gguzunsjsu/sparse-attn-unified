"""SSA alignment loss and dual-stream routing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sparse_attn.config import SSAConfig


def alignment_distance(
    o_fa: torch.Tensor,
    o_sa: torch.Tensor,
    metric: str = "mse",
    stopgrad_sa: bool = True,
) -> torch.Tensor:
    target_sa = o_sa.detach() if stopgrad_sa else o_sa
    if metric == "cosine":
        fa = F.normalize(o_fa, dim=-1)
        sa = F.normalize(target_sa, dim=-1)
        return 1.0 - (fa * sa).sum(dim=-1).mean()
    return F.mse_loss(o_fa, target_sa)


class SSAAlignmentTracker:
    """Accumulates per-layer alignment losses during forward."""

    def __init__(self):
        self.losses: list[torch.Tensor] = []

    def reset(self) -> None:
        self.losses.clear()

    def add(self, loss: torch.Tensor) -> None:
        self.losses.append(loss)

    def total(self, weight: float = 1.0) -> torch.Tensor:
        if not self.losses:
            return torch.tensor(0.0)
        return weight * torch.stack(self.losses).sum()
