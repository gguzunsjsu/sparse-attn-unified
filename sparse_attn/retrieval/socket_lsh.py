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
    key_codes = codes[b_idx, h_idx, key_indices, :]

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


def _pack_matches(masked_t_idx: torch.Tensor, cap: int) -> torch.Tensor:
    """Pack matches [B,H,Q,T] (-1 padded) into [B,H,Q,cap]."""
    b, h, qc, _t = masked_t_idx.shape
    device = masked_t_idx.device
    out = torch.full((b, h, qc, cap), -1, device=device, dtype=torch.long)
    for bi in range(b):
        for hi in range(h):
            for qi in range(qc):
                vals = masked_t_idx[bi, hi, qi]
                vals = vals[vals >= 0]
                if vals.numel() == 0:
                    continue
                vals = torch.unique(vals, sorted=True)
                out[bi, hi, qi, : min(vals.numel(), cap)] = vals[:cap]
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


def socket_select_topk_mask(
    codes: torch.Tensor,
    soft_q: torch.Tensor,
    *,
    budget: int,
    top_m_buckets: int,
    always_idx: torch.LongTensor,
    causal: bool = True,
    query_chunk: int = 128,
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
    cap = max_candidates or max(budget * 4, budget + 32)

    always = always_idx.view(1, 1, 1, -1).expand(b, h, q_len, -1)
    all_indices: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    total_valid = 0.0
    total_slots = 0.0

    for q_start in range(0, q_len, query_chunk):
        q_end = min(q_start + query_chunk, q_len)
        qc = q_end - q_start
        soft_chunk = soft_q[:, :, q_start:q_end, :, :]
        top_buckets = soft_chunk.topk(top_m, dim=-1).indices

        cand_lists = always[:, :, q_start:q_end, :].clone()
        pos_k = torch.arange(t, device=device)

        for table_idx in range(l_tables):
            code_l = codes[:, :, :, table_idx]
            for m in range(top_m):
                bucket = top_buckets[:, :, :, table_idx, m]
                match = code_l.unsqueeze(2) == bucket.unsqueeze(-1)
                if causal:
                    q_pos = torch.arange(q_start, q_end, device=device).view(1, 1, qc, 1)
                    match = match & (pos_k.view(1, 1, 1, t) <= q_pos)
                t_idx = torch.arange(t, device=device).view(1, 1, 1, t).expand(b, h, qc, t)
                masked = torch.where(match, t_idx, torch.full_like(t_idx, -1))
                extra = _pack_matches(masked, cap)
                cand_lists = _merge_unique_indices(cand_lists, extra, max_keys=cap)

        cand_scores = socket_collision_scores_for_keys(codes, soft_chunk, cand_lists)
        if causal:
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
    }
    return indices, scores, stats
