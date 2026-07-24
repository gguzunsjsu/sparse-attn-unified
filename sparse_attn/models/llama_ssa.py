"""Llama 1B transformer with SSA dual-stream attention layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_attn.backends.utils import apply_rope, build_rope_cache
from sparse_attn.config import HPCConfig, Llama1BConfig, SSAConfig, SaapConfig, SocketConfig
from sparse_attn.layers.ssa_attention import SSADualStreamAttention
from sparse_attn.ssa.alignment import SSAAlignmentTracker
from sparse_attn.ssa.schedule import align_weight


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


class LlamaSSABlock(nn.Module):
    def __init__(
        self,
        cfg: Llama1BConfig,
        ssa_cfg: SSAConfig,
        socket_cfg: SocketConfig,
        saap_cfg: SaapConfig,
        layer_idx: int,
        training: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.ssa_attn = SSADualStreamAttention(
            self.head_dim,
            ssa_cfg,
            socket_cfg,
            saap_cfg,
            training=training,
        )

        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def _reshape_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        training: bool = True,
        inference_mode: str = "sparse",
        global_step: int = 0,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        residual = hidden
        x = self.input_layernorm(hidden)

        q = self._reshape_heads(self.q_proj(x), self.num_heads)
        k = self._reshape_heads(self.k_proj(x), self.num_kv_heads)
        v = self._reshape_heads(self.v_proj(x), self.num_kv_heads)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_out = self.ssa_attn(
            q,
            k,
            v,
            training=training,
            inference_mode=inference_mode,
            global_step=global_step,
            layer_idx=layer_idx,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(hidden.size(0), hidden.size(1), -1)
        hidden = residual + self.o_proj(attn_out)

        residual = hidden
        x = self.post_attention_layernorm(hidden)
        hidden = residual + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return hidden


class LlamaSSAModel(nn.Module):
    """Llama 3.2 1B with SSA attention for H100 single-GPU training."""

    def __init__(self, hpc_cfg: HPCConfig | None = None, training: bool = True):
        super().__init__()
        hpc = hpc_cfg or HPCConfig()
        self.cfg = Llama1BConfig()
        self.hpc_cfg = hpc
        self.ssa_cfg = hpc.ssa
        self.align_tracker = SSAAlignmentTracker()

        self.embed_tokens = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        self.layers = nn.ModuleList(
            [
                LlamaSSABlock(
                    self.cfg,
                    hpc.ssa,
                    hpc.socket,
                    hpc.saap,
                    i,
                    training=training,
                )
                for i in range(self.cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(self.cfg.hidden_size, self.cfg.rms_norm_eps)
        self.lm_head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _update_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        cached = int(self._cos.size(2)) if self._cos.numel() else 0
        if cached >= seq_len:
            return
        # Constants — must not be inference tensors or grad checkpointing fails after eval().
        with torch.inference_mode(False):
            with torch.no_grad():
                cos, sin = build_rope_cache(
                    seq_len,
                    self.cfg.hidden_size // self.cfg.num_attention_heads,
                    device,
                    dtype,
                    self.cfg.rope_theta,
                )
        self._cos = cos
        self._sin = sin

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        *,
        training: bool = True,
        inference_mode: str = "sparse",
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        b, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        self._update_rope(seq_len, device, dtype)

        hidden = self.embed_tokens(input_ids)
        self.align_tracker.reset()

        for i, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                self._cos[:, :, :seq_len, :],
                self._sin[:, :, :seq_len, :],
                training=training,
                inference_mode=inference_mode,
                global_step=global_step,
                layer_idx=i,
            )
            self.align_tracker.add(layer.ssa_attn.alignment_loss)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            align_loss = self.align_tracker.total(align_weight(self.ssa_cfg, global_step))
            if align_loss.device != lm_loss.device:
                align_loss = align_loss.to(lm_loss.device)
            out["loss"] = lm_loss + align_loss
            out["lm_loss"] = lm_loss
            out["align_loss"] = align_loss
        return out

    @classmethod
    def from_pretrained_base(
        cls,
        model_name: str,
        hpc_cfg: HPCConfig,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> "LlamaSSAModel":
        """Initialize SSA model weights from HuggingFace Llama 3.2 1B or a local directory."""
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if hpc_cfg.bf16 else torch.float32,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = cls(hpc_cfg, training=True).to(device)
        src = base.state_dict()
        dst = model.state_dict()
        mapping = {}

        for i in range(model.cfg.num_hidden_layers):
            prefix = f"model.layers.{i}"
            ours = f"layers.{i}"
            for stem in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ):
                src_key = f"{prefix}.{stem}"
                dst_key = f"{ours}.{stem.replace('self_attn.', '')}"
                if src_key in src and dst_key in dst:
                    mapping[dst_key] = src_key

        mapping["embed_tokens.weight"] = "model.embed_tokens.weight"
        mapping["norm.weight"] = "model.norm.weight"
        mapping["lm_head.weight"] = "lm_head.weight"

        loaded = 0
        for dst_key, src_key in mapping.items():
            if dst_key in dst and src_key in src and dst[dst_key].shape == src[src_key].shape:
                dst[dst_key].copy_(src[src_key].to(dst[dst_key].dtype))
                loaded += 1

        model.load_state_dict(dst)
        print(f"Loaded {loaded}/{len(mapping)} tensors from {model_name}")
        return model

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
