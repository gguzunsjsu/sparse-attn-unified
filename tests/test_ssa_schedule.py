"""Tests for SSA training schedule helpers."""

from sparse_attn.config import SSAConfig
from sparse_attn.ssa.schedule import align_weight, p_sparse, run_fa_stream


def test_schedule_disabled_matches_defaults():
    cfg = SSAConfig(schedule_enabled=False, p_sparse=0.5, align_weight=1.0)
    assert p_sparse(cfg, 1000) == 0.5
    assert align_weight(cfg, 1000) == 1.0
    assert run_fa_stream(cfg, 1000, 3)


def test_sparse_only_phase():
    cfg = SSAConfig(schedule_enabled=True, sparse_only_after_step=100, p_sparse=0.5)
    assert p_sparse(cfg, 200) == 1.0
    assert not run_fa_stream(cfg, 200, 0)
