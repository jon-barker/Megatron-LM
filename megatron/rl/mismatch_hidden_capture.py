# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Capture lm_final_hidden digests for W&B logprob mismatch diagnostics."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import mpu
from megatron.rl.determinism_probe import _ACTIVE_SCOPE, _tensor_hash, _token_rows

# Hidden vectors are keyed by prefix_hash: a content hash of all tokens up to and
# including the input that produces lm_final_hidden. This is invariant to prompt
# length, left-padding, and decode-vs-prefill batching, so it aligns the training
# and inference forward passes for the same generated token.
_TRAIN_VECTORS: dict[str, torch.Tensor] = {}
_INF_VECTORS: dict[str, torch.Tensor] = {}
_PAIR_STATS: dict[str, dict[str, Any]] = {}

# Inference top-k captured in-process at the engine, keyed by
# (routing_dump_id, gen_offset). gen_offset is the index of the generated token
# within the generation; in the engine it is the cumulative length of
# request.generated_top_n_logprobs, which is reliable across decode steps.
_INF_TOPK: dict[tuple[str, int], dict[str, list]] = {}


def mismatch_hidden_capture_enabled() -> bool:
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
    except Exception:
        return False
    if not getattr(args, "rl_logprob_mismatch_capture_hidden", True):
        return False
    return int(getattr(args, "rl_logprob_mismatch_num_examples", 0) or 0) > 0


def clear_mismatch_hidden_capture() -> None:
    _TRAIN_VECTORS.clear()
    _INF_VECTORS.clear()
    _PAIR_STATS.clear()
    _INF_TOPK.clear()


def record_inference_topk(
    routing_dump_id: str | None,
    gen_offset: int,
    tokens: list[str],
    logprobs: list[float],
) -> None:
    """Record inference top-k for a generated token, keyed by (dump_id, gen_offset).

    Called in-process from the engine so it does not depend on the inference
    response transport (e.g. the NeMoGym bridge) carrying top_logprobs back.
    """
    if routing_dump_id is None:
        return
    _INF_TOPK[(str(routing_dump_id), int(gen_offset))] = {
        "tokens": [str(t) for t in tokens],
        "logprobs": [float(x) for x in logprobs],
    }


def gather_inference_topk() -> dict[tuple[str, int], dict[str, list]]:
    """Merge locally captured inference top-k across data-parallel ranks."""
    local = dict(_INF_TOPK)
    if not dist.is_initialized():
        return local
    dp_group = mpu.get_data_parallel_group()
    gathered: list[dict[tuple[str, int], dict[str, list]] | None] = [
        None
    ] * dist.get_world_size(dp_group)
    dist.all_gather_object(gathered, local, group=dp_group)
    merged: dict[tuple[str, int], dict[str, list]] = {}
    for rank_topk in gathered:
        if rank_topk:
            merged.update(rank_topk)
    return merged


def _content_key(meta: dict[str, Any]) -> str | None:
    prefix_hash = meta.get("prefix_hash")
    if prefix_hash is None:
        return None
    return str(prefix_hash)


def _pair_stats(train: torch.Tensor, inference: torch.Tensor) -> dict[str, Any]:
    train_vec = train.detach().float().flatten().cpu()
    inf_vec = inference.detach().float().flatten().cpu()
    if train_vec.numel() != inf_vec.numel() or train_vec.numel() == 0:
        return {
            "hidden_hash_match": False,
            "hidden_rel_l2": float("nan"),
            "hidden_cosine": float("nan"),
            "train_hidden_hash": _tensor_hash(train_vec) if train_vec.numel() else "",
            "inference_hidden_hash": _tensor_hash(inf_vec) if inf_vec.numel() else "",
            "train_hidden_l2_norm": float("nan"),
            "inference_hidden_l2_norm": float("nan"),
        }

    diff = train_vec - inf_vec
    train_norm = train_vec.norm().clamp_min(1e-12)
    return {
        "hidden_hash_match": bool(_tensor_hash(train_vec) == _tensor_hash(inf_vec)),
        "hidden_rel_l2": float(diff.norm().item() / train_norm.item()),
        "hidden_cosine": float(
            F.cosine_similarity(train_vec.unsqueeze(0), inf_vec.unsqueeze(0)).item()
        ),
        "train_hidden_hash": _tensor_hash(train_vec),
        "inference_hidden_hash": _tensor_hash(inf_vec),
        "train_hidden_l2_norm": float(train_vec.norm().item()),
        "inference_hidden_l2_norm": float(inf_vec.norm().item()),
    }


def _store_phase_vector(phase: str, key: str, vector: torch.Tensor) -> None:
    vec = vector.detach().float().cpu()
    if phase == "training_old_logprobs":
        _TRAIN_VECTORS[key] = vec
        if key in _INF_VECTORS:
            _PAIR_STATS[key] = _pair_stats(vec, _INF_VECTORS[key])
    elif phase == "inference":
        _INF_VECTORS[key] = vec
        if key in _TRAIN_VECTORS:
            _PAIR_STATS[key] = _pair_stats(_TRAIN_VECTORS[key], vec)


def maybe_capture_lm_final_hidden(hidden_states: torch.Tensor) -> None:
    """Record per-token hidden digests keyed by prefix_hash."""
    if not mismatch_hidden_capture_enabled():
        return

    scope = _ACTIVE_SCOPE.get()
    if scope is None:
        return

    rows = _token_rows(hidden_states, scope)
    if rows is None:
        return

    token_metadata, row_tensor = rows
    for meta, row in zip(token_metadata, row_tensor):
        if meta.get("is_padding"):
            continue
        if "is_generated_target" in meta and not meta["is_generated_target"]:
            continue
        key = _content_key(meta)
        if key is None:
            continue
        _store_phase_vector(scope.phase, key, row)


def gather_hidden_pair_stats() -> dict[str, dict[str, Any]]:
    """Merge locally captured hidden pair stats across data-parallel ranks."""
    local_stats = dict(_PAIR_STATS)
    if not dist.is_initialized():
        return local_stats

    dp_group = mpu.get_data_parallel_group()
    gathered: list[dict[str, dict[str, Any]] | None] = [None] * dist.get_world_size(dp_group)
    dist.all_gather_object(gathered, local_stats, group=dp_group)
    merged: dict[str, dict[str, Any]] = {}
    for rank_stats in gathered:
        if rank_stats:
            merged.update(rank_stats)
    return merged


def lookup_hidden_pair_stats(
    prefix_hash: str | None,
    *,
    pair_stats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if prefix_hash is None:
        return None
    stats = pair_stats if pair_stats is not None else _PAIR_STATS
    return stats.get(str(prefix_hash))


def _rank_in_tokens(token: str, tokens: list[str]) -> int | None:
    try:
        return tokens.index(token) + 1
    except ValueError:
        return None


def _top2_margin(logprobs: list[float]) -> float | None:
    if len(logprobs) < 2:
        return None
    return float(logprobs[0] - logprobs[1])


def enrich_candidate_inference_topk(
    candidate: dict[str, Any],
    inference_topk: dict[tuple[str, int], dict[str, list]] | None = None,
) -> dict[str, Any]:
    """Fill inference top-k columns from in-process captured engine top-k.

    Overrides the transport-derived inference top-k whenever a captured entry
    exists for (routing_dump_id, gen_offset), and refreshes the dependent
    overlap/rank columns against the candidate's train top-k.
    """
    topk = inference_topk if inference_topk is not None else _INF_TOPK
    if not topk:
        return candidate

    routing_dump_id = candidate.get("routing_dump_id")
    if routing_dump_id is None:
        return candidate

    gen_offsets = candidate.get("gen_offset") or []
    sampled_token_strs = candidate.get("sampled_token_str") or []
    num_tokens = int(candidate.get("num_tokens") or len(gen_offsets))

    for row_idx in range(num_tokens):
        gen_offset = gen_offsets[row_idx] if row_idx < len(gen_offsets) else None
        if gen_offset is None:
            continue
        entry = topk.get((str(routing_dump_id), int(gen_offset)))
        if not entry:
            continue
        inf_tokens = entry.get("tokens") or []
        inf_lps = entry.get("logprobs") or []
        if not inf_tokens:
            continue

        candidate["inference_topk_tokens"][row_idx] = inf_tokens
        candidate["inference_topk_logprobs"][row_idx] = [float(x) for x in inf_lps]
        candidate["inference_topk_available"][row_idx] = True
        candidate["inference_top1_token"][row_idx] = inf_tokens[0]
        candidate["inference_top1_logprob"][row_idx] = (
            float(inf_lps[0]) if inf_lps else float("nan")
        )
        candidate["inference_top2_token"][row_idx] = (
            inf_tokens[1] if len(inf_tokens) > 1 else ""
        )
        candidate["inference_top2_logprob"][row_idx] = (
            float(inf_lps[1]) if len(inf_lps) > 1 else float("nan")
        )
        candidate["inference_top2_margin"][row_idx] = _top2_margin(inf_lps)

        train_tokens = candidate.get("train_topk_tokens", [None] * num_tokens)[row_idx] or []
        if train_tokens:
            overlap = len(set(train_tokens) & set(inf_tokens))
            denom = min(len(train_tokens), len(inf_tokens))
            candidate["topk_token_overlap"][row_idx] = overlap
            candidate["topk_token_overlap_frac"][row_idx] = (
                float(overlap / denom) if denom > 0 else None
            )

        sampled_token = (
            sampled_token_strs[row_idx] if row_idx < len(sampled_token_strs) else None
        )
        if sampled_token is not None:
            candidate["sampled_token_inference_rank"][row_idx] = _rank_in_tokens(
                sampled_token, inf_tokens
            )
    return candidate


def collect_candidate_prefix_keys(selected: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for candidate in selected:
        for prefix_hash in candidate.get("prefix_hash") or []:
            if prefix_hash is not None:
                keys.add(str(prefix_hash))
    return keys


def gather_selected_hidden_vectors(
    needed_keys: set[str],
) -> dict[str, dict[str, torch.Tensor]]:
    """Gather train/inference hidden vectors for the given prefix_hash keys.

    Train and inference vectors for the same token can be captured on different
    data-parallel ranks (the inference replica that served a request need not be
    the rank that owns that rollout in the training data split), so we gather and
    merge across the DP group. Vectors are exchanged in fp16 to bound transfer.
    """
    local: dict[str, tuple] = {}
    for key in needed_keys:
        train_vec = _TRAIN_VECTORS.get(key)
        inf_vec = _INF_VECTORS.get(key)
        if train_vec is None and inf_vec is None:
            continue
        local[key] = (
            train_vec.half() if train_vec is not None else None,
            inf_vec.half() if inf_vec is not None else None,
        )

    if not dist.is_initialized():
        gathered_list = [local]
    else:
        dp_group = mpu.get_data_parallel_group()
        gathered_list = [None] * dist.get_world_size(dp_group)
        dist.all_gather_object(gathered_list, local, group=dp_group)

    merged: dict[str, dict[str, torch.Tensor]] = {}
    for rank_dict in gathered_list:
        if not rank_dict:
            continue
        for key, (train_vec, inf_vec) in rank_dict.items():
            entry = merged.setdefault(key, {"train": None, "inf": None})
            if train_vec is not None:
                entry["train"] = train_vec.float()
            if inf_vec is not None:
                entry["inf"] = inf_vec.float()
    return merged


def enrich_candidate_hidden_stats(
    candidate: dict[str, Any],
    pair_stats: dict[str, dict[str, Any]] | None = None,
    vectors: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Fill per-token hidden digest columns on a mismatch candidate.

    If ``vectors`` (merged train/inference hidden vectors keyed by prefix_hash) is
    provided, geometry is computed directly from the vectors (cross-rank correct
    and includes per-side norms). Otherwise falls back to the pre-paired
    ``pair_stats`` lookup.
    """
    prefix_hashes = candidate.get("prefix_hash") or []
    token_indices = candidate.get("token_index") or []
    num_tokens = int(candidate.get("num_tokens") or len(token_indices))

    hidden_hash_match = []
    hidden_rel_l2 = []
    hidden_cosine = []
    train_hidden_hash = []
    inference_hidden_hash = []
    train_hidden_norm = []
    inference_hidden_norm = []
    hidden_norm_ratio = []
    for row_idx in range(num_tokens):
        prefix_hash = prefix_hashes[row_idx] if row_idx < len(prefix_hashes) else None
        if vectors is not None:
            entry = vectors.get(str(prefix_hash)) if prefix_hash is not None else None
            train_vec = entry.get("train") if entry else None
            inf_vec = entry.get("inf") if entry else None
            if train_vec is not None and inf_vec is not None:
                stats = _pair_stats(train_vec, inf_vec)
            elif train_vec is not None or inf_vec is not None:
                stats = {
                    "train_hidden_l2_norm": (
                        float(train_vec.norm().item()) if train_vec is not None else float("nan")
                    ),
                    "inference_hidden_l2_norm": (
                        float(inf_vec.norm().item()) if inf_vec is not None else float("nan")
                    ),
                }
            else:
                stats = {}
        else:
            stats = lookup_hidden_pair_stats(prefix_hash, pair_stats=pair_stats) or {}
        hidden_hash_match.append(stats.get("hidden_hash_match"))
        hidden_rel_l2.append(stats.get("hidden_rel_l2", float("nan")))
        hidden_cosine.append(stats.get("hidden_cosine", float("nan")))
        train_hidden_hash.append(stats.get("train_hidden_hash", ""))
        inference_hidden_hash.append(stats.get("inference_hidden_hash", ""))
        t_norm = stats.get("train_hidden_l2_norm", float("nan"))
        i_norm = stats.get("inference_hidden_l2_norm", float("nan"))
        train_hidden_norm.append(t_norm)
        inference_hidden_norm.append(i_norm)
        hidden_norm_ratio.append(
            float(i_norm / t_norm)
            if isinstance(t_norm, (int, float))
            and isinstance(i_norm, (int, float))
            and t_norm == t_norm
            and i_norm == i_norm
            and t_norm > 0
            else float("nan")
        )

    matched_pairs = [x for x in hidden_hash_match if x is not None]
    finite_rel_l2 = [
        x for x in hidden_rel_l2 if x is not None and isinstance(x, (int, float)) and x == x
    ]
    candidate.update(
        {
            "hidden_hash_match": hidden_hash_match,
            "hidden_rel_l2": hidden_rel_l2,
            "hidden_cosine": hidden_cosine,
            "train_hidden_hash": train_hidden_hash,
            "inference_hidden_hash": inference_hidden_hash,
            "train_hidden_norm": train_hidden_norm,
            "inference_hidden_norm": inference_hidden_norm,
            "hidden_norm_ratio": hidden_norm_ratio,
            "hidden_pairs_matched": len(matched_pairs),
            "hidden_hash_match_fraction": (
                float(sum(1 for x in matched_pairs if x) / len(matched_pairs))
                if matched_pairs
                else None
            ),
            "max_hidden_rel_l2": max(finite_rel_l2) if finite_rel_l2 else None,
        }
    )
    return candidate


def compute_candidate_logit_diagnostics(
    selected: list[dict[str, Any]],
    vectors: dict[str, dict[str, torch.Tensor]],
    output_weight: torch.Tensor,
    output_bias: torch.Tensor | None = None,
    row_chunk: int = 128,
) -> None:
    """Fill linearized / recomputed logit-gap columns on selected candidates.

    For each generated token with both train and inference hidden captured, using
    the shared output projection W (and optional bias b):

      - delta_logit_sampled = w_t . (h_train - h_inf)
      - recompute_train_logprob = log_softmax(W h_train + b)[t]
      - recompute_inference_logprob = log_softmax(W h_inf + b)[t]
      - recompute_logprob_delta = recompute_train_logprob - recompute_inference_logprob
      - linearized_logprob_delta = (w_t - E_p[w]) . (h_train - h_inf),
        with p = softmax(W h_train + b)

    The output projection is computed in **true fp32** (weight cast to fp32 one
    vocab chunk at a time), so this is a single common fp32 output layer applied
    to both sides -- distinct from the native bf16 training/inference output
    layers. Comparing recompute_* to the stored train/inference logprobs isolates
    the bf16-vs-fp32 output-layer contribution; comparing recompute_logprob_delta
    to the stored logprob_delta isolates the hidden-state contribution.

    Comparing recompute_logprob_delta (and its linearization) to the stored
    logprob_delta tells whether the probability mismatch is explained by the
    hidden-state difference passed through the readout, versus arising downstream.
    """
    device = output_weight.device
    weight = output_weight
    vocab_size, hidden_dim = weight.shape[0], weight.shape[1]
    bias_f = output_bias.float() if output_bias is not None else None
    # The output projection is run in true fp32 (a "common fp32 output layer").
    # To avoid a full fp32 copy of the (vocab x hidden) weight, we cast and matmul
    # one vocab chunk at a time.
    vocab_chunk = 16384

    def _fp32_logits(h_fp32: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (h_fp32.shape[0], vocab_size), dtype=torch.float32, device=h_fp32.device
        )
        for v0 in range(0, vocab_size, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab_size)
            out[:, v0:v1] = h_fp32 @ weight[v0:v1].float().t()
        if bias_f is not None:
            out += bias_f.unsqueeze(0)
        return out

    def _fp32_wbar(probs_fp32: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(
            (probs_fp32.shape[0], hidden_dim), dtype=torch.float32, device=probs_fp32.device
        )
        for v0 in range(0, vocab_size, vocab_chunk):
            v1 = min(v0 + vocab_chunk, vocab_size)
            acc += probs_fp32[:, v0:v1] @ weight[v0:v1].float()
        return acc

    for candidate in selected:
        num_tokens = int(candidate.get("num_tokens") or 0)
        prefix_hashes = candidate.get("prefix_hash") or []
        token_ids = candidate.get("token_id") or []

        delta_logit_sampled = [float("nan")] * num_tokens
        recompute_train_logprob = [float("nan")] * num_tokens
        recompute_inference_logprob = [float("nan")] * num_tokens
        recompute_logprob_delta = [float("nan")] * num_tokens
        linearized_logprob_delta = [float("nan")] * num_tokens

        rows: list[int] = []
        h_train_rows: list[torch.Tensor] = []
        h_inf_rows: list[torch.Tensor] = []
        sampled_ids: list[int] = []
        for row_idx in range(num_tokens):
            prefix_hash = prefix_hashes[row_idx] if row_idx < len(prefix_hashes) else None
            entry = vectors.get(str(prefix_hash)) if prefix_hash is not None else None
            if not entry:
                continue
            train_vec = entry.get("train")
            inf_vec = entry.get("inf")
            if train_vec is None or inf_vec is None:
                continue
            if train_vec.numel() != hidden_dim or inf_vec.numel() != hidden_dim:
                continue
            rows.append(row_idx)
            h_train_rows.append(train_vec)
            h_inf_rows.append(inf_vec)
            sampled_ids.append(int(token_ids[row_idx]) if row_idx < len(token_ids) else 0)

        for start in range(0, len(rows), row_chunk):
            chunk_rows = rows[start : start + row_chunk]
            h_train = torch.stack(h_train_rows[start : start + row_chunk]).to(
                device=device, dtype=torch.float32
            )
            h_inf = torch.stack(h_inf_rows[start : start + row_chunk]).to(
                device=device, dtype=torch.float32
            )
            ids = torch.tensor(
                sampled_ids[start : start + row_chunk], dtype=torch.long, device=device
            )

            logits_train = _fp32_logits(h_train)
            logits_inf = _fp32_logits(h_inf)
            logp_train = torch.log_softmax(logits_train, dim=-1)
            logp_inf = torch.log_softmax(logits_inf, dim=-1)
            probs_train = logp_train.exp()

            gather_ids = ids.unsqueeze(1)
            lp_train_t = logp_train.gather(1, gather_ids).squeeze(1)
            lp_inf_t = logp_inf.gather(1, gather_ids).squeeze(1)

            delta_h = h_train - h_inf
            w_t = weight[ids].float()  # [chunk, hidden]
            delta_logit_t = (w_t * delta_h).sum(dim=-1)
            # E_p[w] = probs_train @ W ; linearized = (w_t - E_p[w]) . delta_h
            w_bar = _fp32_wbar(probs_train)  # [chunk, hidden]
            lin_delta = ((w_t - w_bar) * delta_h).sum(dim=-1)

            for j, row_idx in enumerate(chunk_rows):
                delta_logit_sampled[row_idx] = float(delta_logit_t[j].item())
                recompute_train_logprob[row_idx] = float(lp_train_t[j].item())
                recompute_inference_logprob[row_idx] = float(lp_inf_t[j].item())
                recompute_logprob_delta[row_idx] = float(
                    (lp_train_t[j] - lp_inf_t[j]).item()
                )
                linearized_logprob_delta[row_idx] = float(lin_delta[j].item())

        finite_recompute = [
            x for x in recompute_logprob_delta if isinstance(x, float) and x == x
        ]
        candidate.update(
            {
                "delta_logit_sampled": delta_logit_sampled,
                "recompute_train_logprob": recompute_train_logprob,
                "recompute_inference_logprob": recompute_inference_logprob,
                "recompute_logprob_delta": recompute_logprob_delta,
                "linearized_logprob_delta": linearized_logprob_delta,
                "max_abs_recompute_logprob_delta": (
                    max(abs(x) for x in finite_recompute) if finite_recompute else None
                ),
            }
        )
