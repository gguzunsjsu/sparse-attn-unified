"""Cluster-based SAAP candidate retrieval (no dense [Q,T] score tensor)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pack_cluster_keys(
    key_clusters: torch.LongTensor,
    cluster_ids: torch.LongTensor,
    *,
    cap: int,
    causal: bool,
    q_start: int,
    q_end: int,
) -> torch.LongTensor:
    """Keys belonging to selected clusters per query position."""
    b, h, qc, m = cluster_ids.shape
    t = key_clusters.size(2)
    device = key_clusters.device
    out = torch.full((b, h, qc, cap), -1, device=device, dtype=torch.long)
    for bi in range(b):
        for hi in range(h):
            for qi in range(qc):
                q_abs = q_start + qi
                picked: list[int] = []
                for mi in range(m):
                    cid = int(cluster_ids[bi, hi, qi, mi].item())
                    keys = (key_clusters[bi, hi, : q_abs + 1] == cid).nonzero(as_tuple=True)[0]
                    for tk in keys.tolist():
                        if causal and tk > q_abs:
                            continue
                        if tk not in picked:
                            picked.append(tk)
                        if len(picked) >= cap:
                            break
                    if len(picked) >= cap:
                        break
                if picked:
                    out[bi, hi, qi, : len(picked)] = torch.tensor(picked, device=device)
    return out


def _merge_unique_indices(
    base: torch.LongTensor,
    extra: torch.LongTensor,
    *,
    max_keys: int,
) -> torch.LongTensor:
    b, h, q_len, _ = base.shape
    device = base.device
    out = torch.full((b, h, q_len, max_keys), -1, device=device, dtype=torch.long)
    for bi in range(b):
        for hi in range(h):
            for qi in range(q_len):
                row = torch.cat([base[bi, hi, qi], extra[bi, hi, qi]])
                row = row[row >= 0]
                if row.numel() == 0:
                    continue
                row = torch.unique(row, sorted=True)
                out[bi, hi, qi, : min(row.numel(), max_keys)] = row[:max_keys]
    return out


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
    clusters = key_clusters[b_idx, h_idx, key_indices]
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
    query_chunk: int = 128,
    max_candidates: int | None = None,
) -> tuple[torch.LongTensor, torch.Tensor, dict[str, float]]:
    """Build SAAP indices via cluster union + routing scores on candidates."""
    b, h, q_len, num_clusters = query_weights.shape
    device = query_weights.device
    cap = max_candidates or max(budget * 4, budget + 32)
    m = min(top_m_clusters, num_clusters)

    always = always_idx.view(1, 1, 1, -1).expand(b, h, q_len, -1)
    all_indices: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    total_valid = 0.0
    total_slots = 0.0

    for q_start in range(0, q_len, query_chunk):
        q_end = min(q_start + query_chunk, q_len)
        qc = q_end - q_start
        qw = query_weights[:, :, q_start:q_end, :]
        top_c = qw.topk(m, dim=-1).indices

        cand = always[:, :, q_start:q_end, :].clone()
        extra = _pack_cluster_keys(
            key_clusters,
            top_c,
            cap=cap,
            causal=causal,
            q_start=q_start,
            q_end=q_end,
        )
        cand = _merge_unique_indices(cand, extra, max_keys=cap)

        scores = saap_scores_for_keys(qw, key_clusters, cand)
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
    }
    return indices, scores, stats
