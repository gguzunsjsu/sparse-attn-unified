"""Unified SSA + SOCKET + Soft-SAAP sparse attention library."""

from sparse_attn.config import HPCConfig, Llama1BConfig, SSAConfig, SaapConfig, SocketConfig
from sparse_attn.layers.ssa_attention import SSADualStreamAttention

__all__ = [
    "HPCConfig",
    "Llama1BConfig",
    "SSAConfig",
    "SaapConfig",
    "SocketConfig",
    "SSADualStreamAttention",
]

__version__ = "0.1.0"
