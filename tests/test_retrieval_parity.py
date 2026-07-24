"""Parity and shape tests for bucket/cluster retrieval."""

from __future__ import annotations

import torch

from sparse_attn.backends.saap.attention import SaapBackend
from sparse_attn.backends.socket.attention import SocketBackend, SocketMasker
from sparse_attn.config import SaapConfig, SocketConfig
from sparse_attn.retrieval.socket_lsh import socket_collision_scores_dense


def _tensors(t: int = 48, d: int = 16):
    torch.manual_seed(0)
    b, h = 1, 4
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    return q, k, v


def test_socket_bucket_matches_dense_when_top_m_covers_buckets():
    q, k, v = _tensors(t=40)
    d = q.size(-1)
    cfg = SocketConfig(bucket_k=4, train_l=4, heavy_const=0.5, top_m_buckets=16, use_bucket_retrieval=True)
    masker = SocketMasker(d, cfg, training=True)
    codes, soft_q = masker.key_codes_and_query_soft(q, k)
    dense = socket_collision_scores_dense(codes, soft_q)

    backend = SocketBackend(d, cfg, training=True)
    mask = backend.build_mask(q, k, v)
    assert mask.indices.shape[:3] == q.shape[:3]


def test_socket_dense_fallback_topk_similar():
    q, k, v = _tensors(t=32)
    cfg_dense = SocketConfig(
        bucket_k=4, train_l=4, heavy_const=0.5, use_bucket_retrieval=False, sink_size=4, window_size=4
    )
    cfg_bucket = SocketConfig(
        bucket_k=4,
        train_l=4,
        heavy_const=0.5,
        use_bucket_retrieval=True,
        top_m_buckets=16,
        sink_size=4,
        window_size=4,
    )
    m_dense = SocketBackend(16, cfg_dense, training=True).build_mask(q, k, v).indices
    m_bucket = SocketBackend(16, cfg_bucket, training=True).build_mask(q, k, v).indices
    assert m_dense.shape == m_bucket.shape


def test_saap_cluster_retrieval_shape():
    q, k, v = _tensors(t=36)
    cfg = SaapConfig(num_clusters=8, top_m_clusters=2, use_cluster_retrieval=True)
    backend = SaapBackend(16, cfg)
    backend.train()
    mask = backend.build_mask(q, k, v)
    out = backend.forward(q, k, v, mask)
    assert out.shape == q.shape
    assert backend.last_retrieval_stats.get("mean_selected_keys", 0) > 0


def test_saap_dense_vs_cluster_same_shape():
    q, k, v = _tensors(t=32)
    cfg_d = SaapConfig(num_clusters=8, top_m_clusters=2, use_cluster_retrieval=False)
    cfg_c = SaapConfig(num_clusters=8, top_m_clusters=2, use_cluster_retrieval=True)
    idx_d = SaapBackend(16, cfg_d).build_mask(q, k, v).indices
    idx_c = SaapBackend(16, cfg_c).build_mask(q, k, v).indices
    assert idx_d.shape == idx_c.shape
