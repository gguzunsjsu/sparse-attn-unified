"""Unit tests for sparse-attn-unified."""

import math

import pytest
import torch

from sparse_attn.backends.full import FullAttentionBackend
from sparse_attn.backends.saap.attention import SaapBackend
from sparse_attn.backends.socket.attention import SocketBackend, SocketMasker
from sparse_attn.config import HPCConfig, SaapConfig, SocketConfig, SSAConfig
from sparse_attn.layers.ssa_attention import SSADualStreamAttention
from sparse_attn.models.llama_ssa import LlamaSSAModel


@pytest.fixture
def attn_tensors():
    torch.manual_seed(0)
    b, h, t, d = 1, 4, 32, 16
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    return q, k, v


def test_socket_soft_scores_sum_to_one(attn_tensors):
    q, k, _ = attn_tensors
    masker = SocketMasker(head_dim=16, cfg=SocketConfig(bucket_k=4, train_l=4, bucket_l=4))
    soft = masker._query_soft_scores(q)
    assert soft.shape[-1] == 2**4
    sums = soft.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_socket_forward_shape(attn_tensors):
    q, k, v = attn_tensors
    backend = SocketBackend(16, SocketConfig(bucket_k=4, train_l=4, heavy_const=0.5))
    mask = backend.build_mask(q, k, v)
    out = backend.forward(q, k, v, mask)
    assert out.shape == q.shape


def test_saap_router_grad(attn_tensors):
    q, k, v = attn_tensors
    q = q.detach().requires_grad_(True)
    backend = SaapBackend(16, SaapConfig(num_clusters=8, top_m_clusters=2))
    mask = backend.build_mask(q, k, v)
    out = backend.forward(q, k, v, mask)
    loss = out.sum()
    loss.backward()
    assert backend.router.mlp[0].weight.grad is not None


def test_ssa_dual_stream_alignment(attn_tensors):
    q, k, v = attn_tensors
    layer = SSADualStreamAttention(
        16,
        SSAConfig(sparse_backend="socket", checkpoint_fa=False),
        SocketConfig(bucket_k=4, train_l=4, heavy_const=0.5),
        SaapConfig(),
        training=True,
    )
    layer.train()
    out = layer(q, k, v, training=True)
    assert out.shape == q.shape
    assert layer.alignment_loss.item() >= 0


def test_llama_ssa_smoke_forward():
    cfg = HPCConfig(seq_length=64, per_device_batch_size=1)
    cfg.ssa.sparse_backend = "socket"
    cfg.ssa.checkpoint_fa = False
    cfg.socket.train_l = 4
    cfg.socket.bucket_k = 4
    model = LlamaSSAModel(cfg, training=True)
    model.train()
    ids = torch.randint(0, 1000, (1, 64))
    labels = ids.clone()
    out = model(ids, labels, training=True)
    assert "loss" in out
    assert math.isfinite(out["loss"].item())


def test_full_attention_causal(attn_tensors):
    q, k, v = attn_tensors
    fa = FullAttentionBackend()
    out = fa.forward(q, k, v)
    assert out.shape == q.shape
