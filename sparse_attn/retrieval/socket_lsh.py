"""Bucket-based SOCKET collision retrieval (no dense [Q,T] score tensor)."""

from __future__ import annotations

import torch


def socket_collision_scores_dense(
    codes: torch.Tensor,
    soft_q: torch.Tensor,
    *,
    query_chunk: int = 256,
) -> torch.Tensor:
    """
    Reference dense soft-collision scores [B, H, Q, T].

    codes: [B, H, T, L]   soft_q: [B, H, Q, L, R]
    """
    b, h, _t, l_tables = codes.shape
    _, _, q_len, _, _ = soft_q.shape
    device = soft_q.device
    dtype = soft_q.dtype

    if q_len * codes.size(2) <= 512 * 512:
        return _dense_block(codes, soft_q, b, h, q_len, codes.size(2), l_tables, device, dtype)

    chunks = []
    for q_start in range(0, q_len, query_chunk):
        q_end = min(q_start + query_chunk, q_len)
        block = _dense_block(
            codes,
            soft_q[:, :, q_start:q_end, :, :],
            b,
            h,
            q_end - q_start,
            codes.size(2),
            l_tables,
            device,
            dtype,
        )
        chunks.append(block)
    return torch.cat(chunks, dim=2)


def _dense_block(
    codes: torch.Tensor,
    soft_q: torch.Tensor,
    b: int,
    h: int,
    q_len: int,
    t: int,
    l_tables: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scores = torch.zeros(b, h, q_len, t, device=device, dtype=dtype)
    for table_idx in range(l_tables):
        bucket = codes[..., table_idx]
        table_scores = torch.gather(
            soft_q[:, :, :, table_idx, :],
            dim=-1,
            index=bucket.unsqueeze(2).expand(b, h, q_len, t),
        )
        scores = scores + table_scores
    return scores / l_tables


def socket_collision_scores_for_keys(
    codes: torch.Tensor,
    soft_q: torch.Tensor,
    key_indices: torch.LongTensor,
) -> torch.Tensor:
    """
    Soft-collision scores for candidate keys only.

    codes: [B, H, T, L]
    soft_q: [B, H, Q, L, R]
    key_indices: [B, H, Q, K]
    returns: [B, H, Q, K]
    """
    b, h, q_len, k_sel = key_indices.shape
    l_tables = codes.size(-1)

    b_idx = torch.arange(b, device=codes.device)[:, None, None, None]
    h_idx = torch.arange(h, device=codes.device)[None, :, None, None]
    key_codes = codes[b_idx, h_idx, key_indices.clamp(min=0), :]

    scores = torch.zeros(b, h, q_len, k_sel, device=soft_q.device, dtype=soft_q.dtype)
    for table_idx in range(l_tables):
        bucket = key_codes[..., table_idx]
        table_scores = torch.gather(
            soft_q[:, :, :, table_idx, :],
            dim=-1,
            index=bucket,
        )
        scores = scores + table_scores
    return scores / l_tables


def _top_keys_from_bucket_match(
    code_l: torch.Tensor,
    bucket: torch.Tensor,
    *,
    q_start: int,
    q_end: int,
    cap: int,
    causal: bool,
) -> torch.Tensor:
    """Up to ``cap`` key indices per query from bucket equality [B,H,Qc,cap]."""
    _b, _h, qc = bucket.shape
    t = code_l.size(-1)
    device = code_l.device
    match = code_l.unsqueeze(2) == bucket.unsqueeze(-1)
    if causal:
        q_pos = torch.arange(q_start, q_end, device=device).view(1, 1, qc, 1)
        pos_k = torch.arange(t, device=device).view(1, 1, 1, t)
        match = match & (pos_k <= q_pos)
    t_idx = torch.arange(t, device=device, dtype=torch.float32).view(1, 1, 1, t)
    ranked = torch.where(match, t_idx, torch.tensor(-1.0, device=device))
    k = min(cap, t)
    vals, _ = torch.topk(ranked, k=k, dim=-1)
    return vals.to(torch.long)


def socket_select_topk_mask(
    codes: torch.Tensor,
    soft_q: torch.Tensor,
    *,
    budget: int,
    top_m_buckets: int,
    always_idx: torch.LongTensor,
    causal: bool = True,
    query_chunk: int = 256,
    max_candidates: int | None = None,
) -> tuple[torch.LongTensor, torch.Tensor, dict[str, float]]:
    """
    Build SOCKET sparse indices via bucket union + collision scoring on candidates.

    Returns (indices [B,H,Q,K], scores [B,H,Q,K], stats).
    """
    b, h, t, l_tables = codes.shape
    _, _, q_len, _, num_buckets = soft_q.shape
    device = codes.device
    top_m = min(top_m_buckets, num_buckets)
    per_bucket_cap = max_candidates or max(budget, 64)

    always = always_idx.view(1, 1, 1, -1).expand(b, h, q_len, -1)
    all_indices: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    total_valid = 0.0
    total_slots = 0.0

    for q_start in range(0, q_len, query_chunk):
        q_end = min(q_start + query_chunk, q_len)
        soft_chunk = soft_q[:, :, q_start:q_end, :, :]
        top_buckets = soft_chunk.topk(top_m, dim=-1).indices

        parts: list[torch.Tensor] = [always[:, :, q_start:q_end, :]]
        for table_idx in range(l_tables):
            code_l = codes[:, :, :, table_idx]
            for m in range(top_m):
                bucket = top_buckets[:, :, :, table_idx, m]
                parts.append(
                    _top_keys_from_bucket_match(
                        code_l,
                        bucket,
                        q_start=q_start,
                        q_end=q_end,
                        cap=per_bucket_cap,
                        causal=causal,
                    )
                )

        cand_lists = torch.cat(parts, dim=-1)
        valid = cand_lists >= 0
        safe_idx = cand_lists.clamp(min=0)
        cand_scores = socket_collision_scores_for_keys(codes, soft_chunk, safe_idx)
        cand_scores = cand_scores.masked_fill(~valid, float("-inf"))
        if causal:
            qc = q_end - q_start
            q_abs = torch.arange(q_start, q_end, device=device).view(1, 1, qc, 1)
            cand_scores = cand_scores.masked_fill(cand_lists > q_abs, float("-inf"))

        k = min(budget, cand_lists.size(-1))
        top_scores, top_idx = torch.topk(cand_scores, k=k, dim=-1)
        chosen = torch.gather(cand_lists, -1, top_idx)
        all_indices.append(chosen)
        all_scores.append(top_scores)
        total_valid += (chosen >= 0).sum().item()
        total_slots += chosen.numel()

    indices = torch.cat(all_indices, dim=2)
    scores = torch.cat(all_scores, dim=2)
    stats = {
        "mean_selected_keys": float(indices.size(-1)),
        "mean_valid_candidates": total_valid / max(total_slots, 1) * indices.size(-1),
        "top_m_buckets": float(top_m),
        "candidate_width": float(cand_lists.size(-1)) if all_indices else 0.0,
    }
    return indices, scores, stats
