"""Cluster-based SAAP candidate retrieval (no dense [Q,T] score tensor)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _top_keys_from_cluster_match(
    key_clusters: torch.LongTensor,
    cluster_ids: torch.LongTensor,
    *,
    q_start: int,
    q_end: int,
    cap: int,
    causal: bool,
) -> torch.Tensor:
    """Keys in selected clusters [B,H,Qc,cap] (vectorized)."""
    _b, _h, qc = cluster_ids.shape
    t = key_clusters.size(2)
    device = key_clusters.device
    match = key_clusters.unsqueeze(2) == cluster_ids.unsqueeze(-1)
    if causal:
        q_pos = torch.arange(q_start, q_end, device=device).view(1, 1, qc, 1)
        pos_k = torch.arange(t, device=device).view(1, 1, 1, t)
        match = match & (pos_k <= q_pos)
    t_idx = torch.arange(t, device=device, dtype=torch.float32).view(1, 1, 1, t)
    ranked = torch.where(match, t_idx, torch.tensor(-1.0, device=device))
    k = min(cap, t)
    vals, _ = torch.topk(ranked, k=k, dim=-1)
    return vals.to(torch.long)


def saap_scores_for_keys(
    query_weights: torch.Tensor,
    key_clusters: torch.LongTensor,
    key_indices: torch.LongTensor,
) -> torch.Tensor:
    """
    query_weights: [B,H,Q,C]
    key_clusters: [B,H,T]
    key_indices: [B,H,Q,K]
    """
    b, h, q_len, k_sel = key_indices.shape
    b_idx = torch.arange(b, device=key_indices.device)[:, None, None, None]
    h_idx = torch.arange(h, device=key_indices.device)[None, :, None, None]
    clusters = key_clusters[b_idx, h_idx, key_indices.clamp(min=0)]
    onehot = F.one_hot(clusters, query_weights.size(-1)).float()
    return torch.einsum("bhqc,bhqkc->bhqk", query_weights, onehot)


def saap_candidate_mask_and_scores(
    query_weights: torch.Tensor,
    key_clusters: torch.LongTensor,
    *,
    budget: int,
    top_m_clusters: int,
    always_idx: torch.LongTensor,
    causal: bool = True,
    query_chunk: int = 256,
    max_candidates: int | None = None,
) -> tuple[torch.LongTensor, torch.Tensor, dict[str, float]]:
    """Build SAAP indices via cluster union + routing scores on candidates."""
    b, h, q_len, num_clusters = query_weights.shape
    device = query_weights.device
    m = min(top_m_clusters, num_clusters)
    n_always = int(always_idx.numel())
    max_total = max_candidates or (budget + n_always + 32)
    per_cluster_cap = min(24, max(4, (max_total - n_always) // max(m + 1, 1)))

    always = always_idx.view(1, 1, 1, -1).expand(b, h, q_len, -1)
    all_indices: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    total_valid = 0.0
    total_slots = 0.0
    last_cand_width = 0.0

    for q_start in range(0, q_len, query_chunk):
        q_end = min(q_start + query_chunk, q_len)
        qc = q_end - q_start
        qw = query_weights[:, :, q_start:q_end, :]
        top_c = qw.topk(m, dim=-1).indices

        parts: list[torch.Tensor] = [always[:, :, q_start:q_end, :]]
        for mi in range(m):
            parts.append(
                _top_keys_from_cluster_match(
                    key_clusters,
                    top_c[:, :, :, mi],
                    q_start=q_start,
                    q_end=q_end,
                    cap=per_cluster_cap,
                    causal=causal,
                )
            )

        cand = torch.cat(parts, dim=-1)
        if cand.size(-1) > max_total:
            cand = cand[..., :max_total]
        last_cand_width = float(cand.size(-1))
        valid = cand >= 0
        safe = cand.clamp(min=0)
        scores = saap_scores_for_keys(qw, key_clusters, safe)
        scores = scores.masked_fill(~valid, float("-inf"))
        if causal:
            q_abs = torch.arange(q_start, q_end, device=device).view(1, 1, qc, 1)
            scores = scores.masked_fill(cand > q_abs, float("-inf"))

        k = min(budget, cand.size(-1))
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        chosen = torch.gather(cand, -1, top_idx)
        all_indices.append(chosen)
        all_scores.append(top_scores)
        total_valid += (chosen >= 0).sum().item()
        total_slots += chosen.numel()

    indices = torch.cat(all_indices, dim=2)
    scores = torch.cat(all_scores, dim=2)
    stats = {
        "mean_selected_keys": float(indices.size(-1)),
        "mean_valid_candidates": total_valid / max(total_slots, 1) * indices.size(-1),
        "top_m_clusters": float(m),
        "candidate_width": last_cand_width,
    }
    return indices, scores, stats
