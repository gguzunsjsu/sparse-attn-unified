"""SSA training schedule: when to run FA/SA streams and alignment weight."""

from __future__ import annotations

from sparse_attn.config import SSAConfig


def align_weight(cfg: SSAConfig, global_step: int) -> float:
    if not cfg.schedule_enabled:
        return cfg.align_weight
    if global_step < cfg.fa_warmup_steps:
        return cfg.align_weight * 0.25
    if global_step < cfg.fa_warmup_steps + cfg.align_ramp_steps:
        frac = (global_step - cfg.fa_warmup_steps) / max(cfg.align_ramp_steps, 1)
        return cfg.align_weight * (0.25 + 0.75 * frac)
    return cfg.align_weight


def p_sparse(cfg: SSAConfig, global_step: int) -> float:
    if not cfg.schedule_enabled:
        return cfg.p_sparse
    if cfg.sparse_only_after_step is not None and global_step >= cfg.sparse_only_after_step:
        return 1.0
    if global_step < cfg.fa_warmup_steps:
        return 0.0
    return cfg.p_sparse


def run_fa_stream(cfg: SSAConfig, global_step: int, layer_idx: int) -> bool:
    if not cfg.schedule_enabled:
        return True
    if cfg.sparse_only_after_step is not None and global_step >= cfg.sparse_only_after_step:
        return False
    if cfg.fa_every_n_layers > 1 and (layer_idx % cfg.fa_every_n_layers) != 0:
        return False
    return True


def run_sa_stream(cfg: SSAConfig, global_step: int) -> bool:
    if not cfg.schedule_enabled:
        return True
    return True


def checkpoint_fa(cfg: SSAConfig, global_step: int, layer_idx: int) -> bool:
    if not cfg.checkpoint_fa:
        return False
    if not run_fa_stream(cfg, global_step, layer_idx):
        return False
    return True
