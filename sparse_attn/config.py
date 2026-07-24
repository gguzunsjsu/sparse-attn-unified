"""Configuration dataclasses for SSA, backends, and SJSU HPC Llama-1B runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SSAConfig:
    """Sparse Sparse Attention dual-stream training settings."""

    p_sparse: float = 0.5
    align_weight: float = 1.0
    align_metric: Literal["mse", "cosine"] = "mse"
    sparse_backend: Literal["socket", "saap"] = "socket"
    align_stopgrad_sa: bool = True
    checkpoint_fa: bool = True
    # Optional training schedule (see sparse_attn.ssa.schedule)
    schedule_enabled: bool = False
    fa_warmup_steps: int = 200
    align_ramp_steps: int = 500
    sparse_only_after_step: int | None = None
    fa_every_n_layers: int = 1


@dataclass
class SocketConfig:
    """SOCKET soft-LSH knobs (inference defaults from paper; train_L reduced for VRAM)."""

    bucket_k: int = 8
    bucket_l: int = 60
    train_l: int = 16
    tau: float = 0.4
    heavy_const: float = 0.10
    sink_size: int = 32
    window_size: int = 32
    top_m_buckets: int = 8
    use_bucket_retrieval: bool = True
    retrieval_query_chunk: int = 128


@dataclass
class SaapConfig:
    """Soft-SAAP: differentiable cluster routing for SSA co-training."""

    num_clusters: int = 64
    top_m_clusters: int = 2
    refresh_interval: int = 500
    router_hidden: int = 512
    gumbel_tau: float = 1.0
    use_soft_routing: bool = True
    use_cluster_retrieval: bool = True
    retrieval_query_chunk: int = 128
    sink_size: int = 16
    window_size: int = 16
    heavy_const: float = 0.15


@dataclass
class Llama1BConfig:
    """Llama 3.2 1B architecture (matches Meta + SSA-1B baselines)."""

    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True


@dataclass
class HPCConfig:
    """Single H100 on SJSU CoE HPC — conservative defaults for 1B SSA training."""

    seq_length: int = 4096
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    max_steps: int = 10_000
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    save_steps: int = 1000
    logging_steps: int = 10
    output_dir: str = "./outputs/llama1b-ssa"
    base_model: str = "meta-llama/Llama-3.2-1B"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    max_train_samples: int | None = 100_000
    ssa: SSAConfig = field(default_factory=SSAConfig)
    socket: SocketConfig = field(default_factory=SocketConfig)
    saap: SaapConfig = field(default_factory=SaapConfig)
