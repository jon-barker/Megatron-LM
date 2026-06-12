#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Analyze train vs inference logprob calibration on determinism probe data.

This script tests whether logprob mismatch is mostly full-vocab tail mass vs
within-top-k calibration by comparing:

  - full-vocab selected-token logprobs and importance ratios
  - top-k renormalized logprobs/ratios on a fixed support set
  - top-k entropy and top-1 margin

Typical use on a node with PyTorch + this repo:

  # 1) One-time weight extract (needs full Megatron env via run_job.sh):
  ./experiments/extract_output_weight_qwen.sh

  # 2) Calibration analysis (run on a compute node; large single-pass probes need RAM):
  PROBE_DIR=/path/to/determinism_probe ./experiments/analyze_logprob_calibration.sh

Or submit as a batch job:
  sbatch --mem=128G -t 02:00:00 experiments/analyze_logprob_calibration.sh

Or run the analyzer directly:

  PYTHONPATH=megatron-rl python megatron-rl/tools/rl_analyze_logprob_calibration.py \\
    --probe-dir /path/to/determinism_probe \\
    --tokenizer-json /path/to/tokenizer.json \\
    --output-weight-npy /path/to/output_weight.npy \\
    --output-json /tmp/logprob_calibration.json

If inference ``lm_logits`` probe rows are populated, ``--output-weight-npy`` is
optional. Training ``lm_logits`` are still used when present; otherwise logits
are recomputed from ``lm_final_hidden``.

Key report fields (per cohort: all, above_p50, above_p90, above_p99,
above_p90_top1_stable, above_p90_top1_disagree):

  - full_importance_ratio vs renormalized_importance_ratio
  - abs_log_ratio_reduction_full_minus_renorm (positive => renorm shrinks |log ratio|)
  - full_entropy_* vs renormalized_entropy_* (train sharper => lower entropy)
  - train_more_confident_*_prob_fraction
  - probed_vs_recomputed_* (does runtime lm_selected_logprob match log_softmax(logits)?)
  - probed_delta_minus_recomputed_delta (unexplained train/inf gap after logits agree)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    torch = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _record_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    token = record.get("token") or {}
    prefix_hash = token.get("prefix_hash")
    if prefix_hash is not None:
        return (int(record.get("iteration", token.get("iteration", 0))), str(prefix_hash))
    routing_dump_id = token.get("routing_dump_id")
    target_token_index = token.get("target_token_index")
    if routing_dump_id is None or target_token_index is None:
        return None
    return (
        int(record.get("iteration", token.get("iteration", 0))),
        str(routing_dump_id),
        int(target_token_index),
    )


def _record_allowed(record: dict[str, Any], *, generated_only: bool) -> bool:
    if not generated_only:
        return True
    token = record.get("token") or {}
    if token.get("is_padding"):
        return False
    if "is_generated_target" in token:
        return bool(token["is_generated_target"])
    return True


def _load_probe_records(
    probe_dir: Path,
    phase: str,
    module_name: str,
    *,
    keys_wanted: set[tuple[Any, ...]] | None = None,
    generated_only: bool = False,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted(probe_dir.glob(f"probe_{phase}_iter*_rank*.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if f'"module_name": "{module_name}"' not in line:
                    continue
                record = json.loads(line)
                if record.get("hash") is None or not record.get("token") or "value" not in record:
                    continue
                if not _record_allowed(record, generated_only=generated_only):
                    continue
                value = record.get("value") or {}
                if not value.get("data"):
                    continue
                key = _record_key(record)
                if key is None:
                    continue
                if keys_wanted is not None and key not in keys_wanted:
                    continue
                result[key] = record
    return result


def _load_topk_records(
    probe_dir: Path,
    phase: str,
    *,
    keys_wanted: set[tuple[Any, ...]] | None = None,
    generated_only: bool = False,
    max_complete: int = 0,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Load only selected-token top-k rows.

    This is the fast path for single-pass probes that wrote ``lm_topk_logprobs``
    and ``lm_topk_token_ids`` for every generated token. It avoids parsing
    4096-float ``lm_final_hidden`` records and avoids any full-vocab matmul.
    """
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    complete: set[tuple[Any, ...]] = set()
    wanted_modules = ('"module_name": "lm_topk_logprobs"', '"module_name": "lm_topk_token_ids"')

    for path in sorted(probe_dir.glob(f"probe_{phase}_iter*_rank*.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if wanted_modules[0] not in line and wanted_modules[1] not in line:
                    continue
                record = json.loads(line)
                if not record.get("token") or "value" not in record:
                    continue
                if not _record_allowed(record, generated_only=generated_only):
                    continue
                key = _record_key(record)
                if key is None:
                    continue
                if keys_wanted is not None and key not in keys_wanted:
                    continue
                value = (record.get("value") or {}).get("data")
                if not value:
                    continue

                entry = result.setdefault(key, {"token": record.get("token") or {}})
                if record.get("module_name") == "lm_topk_logprobs":
                    entry["logprobs"] = [float(x) for x in value]
                elif record.get("module_name") == "lm_topk_token_ids":
                    entry["token_ids"] = [int(x) for x in value]

                if "logprobs" in entry and "token_ids" in entry:
                    complete.add(key)
                    if keys_wanted is not None and len(complete) >= len(keys_wanted):
                        break
                    if keys_wanted is None and max_complete > 0 and len(complete) >= max_complete:
                        break
        if keys_wanted is not None and len(complete) >= len(keys_wanted):
            break
        if keys_wanted is None and max_complete > 0 and len(complete) >= max_complete:
            break

    return {
        key: entry
        for key, entry in result.items()
        if "logprobs" in entry and "token_ids" in entry
    }


def _probe_module_present(probe_dir: Path, phase: str, module_name: str) -> bool:
    for path in sorted(probe_dir.glob(f"probe_{phase}_iter*_rank*.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if f'"module_name": "{module_name}"' not in line:
                    continue
                record = json.loads(line)
                if record.get("hash") is not None and record.get("token") and (record.get("value") or {}).get(
                    "data"
                ):
                    return True
    return False


def _value_tensor(record: dict[str, Any], device: torch.device) -> torch.Tensor:
    data = (record.get("value") or {}).get("data")
    return torch.tensor(data, dtype=torch.float32, device=device)


def _scalar_probe_value(record: dict[str, Any], device: torch.device) -> float:
    return float(_value_tensor(record, device).reshape(-1)[0].item())


def _load_tokenizer(tokenizer_json: Path) -> dict[int, str]:
    payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    id_to_token: dict[int, str] = {}
    for item in payload.get("added_tokens", []):
        id_to_token[int(item["id"])] = str(item["content"])
    vocab = (payload.get("model") or {}).get("vocab") or payload.get("vocab") or {}
    for token, index in vocab.items():
        id_to_token[int(index)] = str(token)
    return id_to_token


def _decode_token(id_to_token: dict[int, str], token_id: int) -> str:
    text = id_to_token.get(int(token_id), f"<id:{token_id}>")
    return text.replace("Ġ", " ").replace("Ċ", "\\n")


def _inference_aligned_logprob(logits: torch.Tensor) -> torch.Tensor:
    """Match RL inference / aligned training path: fp32 log_softmax over last dim."""
    return F.log_softmax(logits.float(), dim=-1)


def _entropy(probs: torch.Tensor) -> float:
    probs = probs.clamp_min(1e-30)
    return float((-probs * probs.log()).sum().item())


def _percentiles(values: list[float], ps=(0.5, 0.9, 0.99)) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    out = {"mean": float(arr.mean()), "max": float(arr.max())}
    for p in ps:
        out[f"p{int(p * 100):02d}"] = float(np.quantile(arr, p))
    return out


def _load_output_weight_npy(path: Path, device: torch.device) -> torch.Tensor:
    weight = np.load(path)
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D output weight [vocab, hidden], got shape {weight.shape}")
    return torch.tensor(weight, dtype=torch.float32, device=device)


def _logits_from_hidden(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    vocab_chunk: int = 16384,
) -> torch.Tensor:
    """Column-parallel output layer with gathered logits: [vocab] from [hidden]."""
    hidden_f = hidden.float()
    if hidden_f.numel() != output_weight.shape[1]:
        raise ValueError(
            f"Hidden size {hidden_f.numel()} != output weight hidden dim {output_weight.shape[1]}"
        )
    vocab_size = output_weight.shape[0]
    logits = torch.empty(vocab_size, dtype=torch.float32, device=hidden_f.device)
    for v0 in range(0, vocab_size, vocab_chunk):
        v1 = min(v0 + vocab_chunk, vocab_size)
        logits[v0:v1] = output_weight[v0:v1].float().matmul(hidden_f)
    return logits


def _logits_from_record_or_hidden(
    *,
    record: dict[str, Any] | None,
    hidden_record: dict[str, Any] | None,
    output_weight: torch.Tensor | None,
    device: torch.device,
    vocab_size: int,
) -> tuple[torch.Tensor | None, str]:
    if record is not None:
        row = _value_tensor(record, device)
        meta = record.get("value") or {}
        original_numel = int(meta.get("original_numel", row.numel()))
        if row.numel() == original_numel == vocab_size:
            return row, "probed_lm_logits"
    if hidden_record is not None and output_weight is not None:
        hidden = _value_tensor(hidden_record, device)
        return _logits_from_hidden(hidden, output_weight), "recomputed_from_hidden"
    return None, "missing"


def _topk_indices(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.numel())
    return torch.topk(logits, k=k, dim=-1).indices


def _support_indices(
    *,
    logits_inf: torch.Tensor,
    logits_train: torch.Tensor,
    selected_id: int,
    top_k: int,
    support_mode: str,
) -> tuple[torch.Tensor, str]:
    inf_topk = _topk_indices(logits_inf, top_k)
    train_topk = _topk_indices(logits_train, top_k)
    selected = torch.tensor([selected_id], device=logits_inf.device, dtype=torch.long)
    if support_mode == "inference_topk":
        support = inf_topk
        label = f"inference_top{top_k}"
    elif support_mode == "training_topk":
        support = train_topk
        label = f"training_top{top_k}"
    elif support_mode == "union":
        support = torch.unique(torch.cat([inf_topk, train_topk, selected]))
        label = f"union_top{top_k}"
    else:
        raise ValueError(f"Unknown support_mode: {support_mode}")
    return support, label


def _renormalized_logprobs(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    subset = logits.index_select(0, support)
    return _inference_aligned_logprob(subset)


def _analyze_row(
    *,
    key: tuple[Any, ...],
    token_id: int,
    token_text: str,
    logits_inf: torch.Tensor,
    logits_train: torch.Tensor,
    logit_source_inf: str,
    logit_source_train: str,
    probed_logp_inf: float,
    probed_logp_train: float,
    top_k: int,
    support_mode: str,
) -> dict[str, Any]:
    logprobs_inf = _inference_aligned_logprob(logits_inf)
    logprobs_train = _inference_aligned_logprob(logits_train)
    probs_inf = logprobs_inf.exp()
    probs_train = logprobs_train.exp()

    inf_top1 = int(torch.argmax(logits_inf).item())
    train_top1 = int(torch.argmax(logits_train).item())
    inf_top2 = int(torch.topk(logits_inf, k=min(2, logits_inf.numel())).indices[-1].item())
    train_top2 = int(torch.topk(logits_train, k=min(2, logits_train.numel())).indices[-1].item())

    full_logp_inf = float(logprobs_inf[token_id].item())
    full_logp_train = float(logprobs_train[token_id].item())
    full_ratio = float(math.exp(full_logp_train - full_logp_inf))
    probed_logprob_delta = abs(probed_logp_train - probed_logp_inf)
    recomputed_logprob_delta = abs(full_logp_train - full_logp_inf)
    probed_vs_recomputed_inf = abs(probed_logp_inf - full_logp_inf)
    probed_vs_recomputed_train = abs(probed_logp_train - full_logp_train)

    support, support_label = _support_indices(
        logits_inf=logits_inf,
        logits_train=logits_train,
        selected_id=token_id,
        top_k=top_k,
        support_mode=support_mode,
    )
    selected_pos = int((support == token_id).nonzero(as_tuple=True)[0][0].item())
    renorm_logp_inf = _renormalized_logprobs(logits_inf, support)
    renorm_logp_train = _renormalized_logprobs(logits_train, support)
    renorm_probs_inf = renorm_logp_inf.exp()
    renorm_probs_train = renorm_logp_train.exp()

    renorm_logp_inf_sel = float(renorm_logp_inf[selected_pos].item())
    renorm_logp_train_sel = float(renorm_logp_train[selected_pos].item())
    renorm_ratio = float(math.exp(renorm_logp_train_sel - renorm_logp_inf_sel))

    topk_slice_inf = _topk_indices(logits_inf, top_k)
    topk_slice_train = _topk_indices(logits_train, top_k)
    topk_logp_inf = _inference_aligned_logprob(logits_inf.index_select(0, topk_slice_inf))
    topk_logp_train = _inference_aligned_logprob(logits_train.index_select(0, topk_slice_train))

    return {
        "key": key,
        "target_token_id": token_id,
        "target_token_text": token_text,
        "probed_selected_logprob_delta": probed_logprob_delta,
        "logit_source_inference": logit_source_inf,
        "logit_source_training": logit_source_train,
        "selected_is_inference_top1": token_id == inf_top1,
        "selected_is_training_top1": token_id == train_top1,
        "inference_top1_id": inf_top1,
        "training_top1_id": train_top1,
        "top1_agree": inf_top1 == train_top1,
        "full_vocab": {
            "logprob_inference": full_logp_inf,
            "logprob_training": full_logp_train,
            "prob_inference": float(probs_inf[token_id].item()),
            "prob_training": float(probs_train[token_id].item()),
            "importance_ratio_train_over_inf": full_ratio,
            "entropy_inference": _entropy(probs_inf),
            "entropy_training": _entropy(probs_train),
            "top1_margin_inference": float((logits_inf[inf_top1] - logits_inf[inf_top2]).item()),
            "top1_margin_training": float((logits_train[train_top1] - logits_train[train_top2]).item()),
        },
        "topk": {
            "k": top_k,
            "support_mode": support_label,
            "support_size": int(support.numel()),
            "entropy_inference": _entropy(topk_logp_inf.exp()),
            "entropy_training": _entropy(topk_logp_train.exp()),
            "top1_logprob_inference": float(topk_logp_inf[0].item()),
            "top1_logprob_training": float(topk_logp_train[0].item()),
        },
        "renormalized_on_support": {
            "logprob_inference": renorm_logp_inf_sel,
            "logprob_training": renorm_logp_train_sel,
            "prob_inference": float(renorm_probs_inf[selected_pos].item()),
            "prob_training": float(renorm_probs_train[selected_pos].item()),
            "importance_ratio_train_over_inf": renorm_ratio,
            "entropy_inference": _entropy(renorm_probs_inf),
            "entropy_training": _entropy(renorm_probs_train),
        },
        "consistency": {
            "probed_selected_logprob_inference": probed_logp_inf,
            "probed_selected_logprob_training": probed_logp_train,
            "recomputed_selected_logprob_inference": full_logp_inf,
            "recomputed_selected_logprob_training": full_logp_train,
            "probed_vs_recomputed_abs_delta_inference": probed_vs_recomputed_inf,
            "probed_vs_recomputed_abs_delta_training": probed_vs_recomputed_train,
            "probed_train_inf_delta": probed_logprob_delta,
            "recomputed_train_inf_delta": recomputed_logprob_delta,
            "probed_delta_minus_recomputed_delta": probed_logprob_delta - recomputed_logprob_delta,
            "dominant_consistency_gap_phase": (
                "inference"
                if probed_vs_recomputed_inf >= probed_vs_recomputed_train
                else "training"
            ),
        },
    }


def _cohort_masks(rows: list[dict[str, Any]], probed_deltas: np.ndarray) -> dict[str, np.ndarray]:
    p50, p90, p99 = [float(np.quantile(probed_deltas, q)) for q in (0.5, 0.9, 0.99)]
    selected_is_train_top1 = np.array([r["selected_is_training_top1"] for r in rows], dtype=bool)
    return {
        "all": np.ones(len(rows), dtype=bool),
        "above_p50": probed_deltas > p50,
        "above_p90": probed_deltas > p90,
        "above_p99": probed_deltas > p99,
        "above_p90_top1_stable": (probed_deltas > p90) & selected_is_train_top1,
        "above_p90_top1_disagree": (probed_deltas > p90) & (~selected_is_train_top1),
        "thresholds": np.array([p50, p90, p99]),
    }


def _summarize_cohort(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"count": 0}

    def collect(path: list[str]) -> list[float]:
        out = []
        for item in items:
            cur: Any = item
            for key in path:
                cur = cur[key]
            out.append(float(cur))
        return out

    full_ratio = collect(["full_vocab", "importance_ratio_train_over_inf"])
    renorm_ratio = collect(["renormalized_on_support", "importance_ratio_train_over_inf"])
    full_prob_inf = collect(["full_vocab", "prob_inference"])
    full_prob_train = collect(["full_vocab", "prob_training"])
    renorm_prob_inf = collect(["renormalized_on_support", "prob_inference"])
    renorm_prob_train = collect(["renormalized_on_support", "prob_training"])
    full_entropy_inf = collect(["full_vocab", "entropy_inference"])
    full_entropy_train = collect(["full_vocab", "entropy_training"])
    renorm_entropy_inf = collect(["renormalized_on_support", "entropy_inference"])
    renorm_entropy_train = collect(["renormalized_on_support", "entropy_training"])
    topk_entropy_inf = collect(["topk", "entropy_inference"])
    topk_entropy_train = collect(["topk", "entropy_training"])

    ratio_reduction = [
        abs(math.log(full_ratio)) - abs(math.log(renorm_ratio))
        for full_ratio, renorm_ratio in zip(full_ratio, renorm_ratio)
    ]

    return {
        "count": len(items),
        "top1_agree_fraction": float(np.mean([item["top1_agree"] for item in items])),
        "selected_is_inference_top1_fraction": float(
            np.mean([item["selected_is_inference_top1"] for item in items])
        ),
        "selected_is_training_top1_fraction": float(
            np.mean([item["selected_is_training_top1"] for item in items])
        ),
        "full_importance_ratio": _percentiles(full_ratio),
        "renormalized_importance_ratio": _percentiles(renorm_ratio),
        "abs_log_ratio_reduction_full_minus_renorm": _percentiles(ratio_reduction),
        "full_prob_inference": _percentiles(full_prob_inf),
        "full_prob_training": _percentiles(full_prob_train),
        "renormalized_prob_inference": _percentiles(renorm_prob_inf),
        "renormalized_prob_training": _percentiles(renorm_prob_train),
        "full_entropy_inference": _percentiles(full_entropy_inf),
        "full_entropy_training": _percentiles(full_entropy_train),
        "renormalized_entropy_inference": _percentiles(renorm_entropy_inf),
        "renormalized_entropy_training": _percentiles(renorm_entropy_train),
        "topk_entropy_inference": _percentiles(topk_entropy_inf),
        "topk_entropy_training": _percentiles(topk_entropy_train),
        "train_more_confident_full_prob_fraction": float(
            np.mean([p_train > p_inf for p_train, p_inf in zip(full_prob_train, full_prob_inf)])
        ),
        "train_more_confident_renorm_prob_fraction": float(
            np.mean([p_train > p_inf for p_train, p_inf in zip(renorm_prob_train, renorm_prob_inf)])
        ),
        **_summarize_consistency_fields(items),
    }


def _summarize_consistency_fields(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}

    def collect_consistency(field: str) -> list[float]:
        return [float(item["consistency"][field]) for item in items]

    probed_inf_gap = collect_consistency("probed_vs_recomputed_abs_delta_inference")
    probed_train_gap = collect_consistency("probed_vs_recomputed_abs_delta_training")
    probed_delta = collect_consistency("probed_train_inf_delta")
    recomputed_delta = collect_consistency("recomputed_train_inf_delta")
    unexplained_delta = collect_consistency("probed_delta_minus_recomputed_delta")
    dominant_phase = Counter(item["consistency"]["dominant_consistency_gap_phase"] for item in items)

    return {
        "probed_vs_recomputed_abs_delta_inference": _percentiles(probed_inf_gap),
        "probed_vs_recomputed_abs_delta_training": _percentiles(probed_train_gap),
        "probed_train_inf_delta": _percentiles(probed_delta),
        "recomputed_train_inf_delta": _percentiles(recomputed_delta),
        "probed_delta_minus_recomputed_delta": _percentiles(unexplained_delta),
        "dominant_consistency_gap_phase": dict(dominant_phase),
        "inference_probed_matches_recomputed_fraction": float(
            np.mean([gap < 1e-3 for gap in probed_inf_gap])
        ),
        "training_probed_matches_recomputed_fraction": float(
            np.mean([gap < 1e-3 for gap in probed_train_gap])
        ),
        "both_probed_match_recomputed_fraction": float(
            np.mean([inf_gap < 1e-3 and train_gap < 1e-3 for inf_gap, train_gap in zip(probed_inf_gap, probed_train_gap)])
        ),
    }


def _token_histogram(items: list[dict[str, Any]], limit: int = 20) -> list[tuple[str, int]]:
    counter = Counter(item["target_token_text"] for item in items)
    return counter.most_common(limit)


def _selected_topk_logprob(entry: dict[str, Any], token_id: int) -> tuple[float | None, int | None]:
    for rank, (candidate_id, logprob) in enumerate(
        zip(entry.get("token_ids", []), entry.get("logprobs", [])), start=1
    ):
        if int(candidate_id) == int(token_id):
            return float(logprob), rank
    return None, None


def _summarize_fast_cohort(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"count": 0}
    deltas = [float(item["topk_selected_logprob_delta"]) for item in items]
    return {
        "count": len(items),
        "topk_selected_logprob_delta": _percentiles(deltas),
        "selected_inference_topk_rank": _percentiles(
            [float(item["selected_inference_topk_rank"]) for item in items]
        ),
        "selected_training_topk_rank": _percentiles(
            [float(item["selected_training_topk_rank"]) for item in items]
        ),
        "top_tokens": _token_histogram(items),
    }


def _build_fast_topk_report(
    *,
    probe_dir: Path,
    inference_phase: str,
    training_phase: str,
    generated_only: bool,
    id_to_token: dict[int, str],
    max_rows: int,
) -> dict[str, Any]:
    print("Loading training top-k rows...", flush=True)
    train_topk = _load_topk_records(
        probe_dir,
        training_phase,
        generated_only=generated_only,
        max_complete=max_rows,
    )
    if not train_topk:
        raise SystemExit(f"No training top-k rows found under {probe_dir}")

    print(f"Loading inference top-k rows for {len(train_topk)} training keys...", flush=True)
    inf_topk = _load_topk_records(
        probe_dir,
        inference_phase,
        keys_wanted=set(train_topk),
        generated_only=generated_only,
    )

    shared_keys = sorted(set(train_topk) & set(inf_topk))
    if max_rows > 0:
        shared_keys = shared_keys[:max_rows]
    if not shared_keys:
        raise SystemExit(f"No matched top-k rows found under {probe_dir}")

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for key in shared_keys:
        token = train_topk[key].get("token") or {}
        token_id = token.get("target_token_id")
        if token_id is None:
            skipped["missing_target_token_id"] += 1
            continue
        token_id = int(token_id)
        inf_logprob, inf_rank = _selected_topk_logprob(inf_topk[key], token_id)
        train_logprob, train_rank = _selected_topk_logprob(train_topk[key], token_id)
        if inf_logprob is None:
            skipped["selected_missing_from_inference_topk"] += 1
            continue
        if train_logprob is None:
            skipped["selected_missing_from_training_topk"] += 1
            continue
        rows.append(
            {
                "key": key,
                "target_token_id": token_id,
                "target_token_text": _decode_token(id_to_token, token_id),
                "topk_selected_logprob_inference": inf_logprob,
                "topk_selected_logprob_training": train_logprob,
                "topk_selected_logprob_delta": abs(train_logprob - inf_logprob),
                "selected_inference_topk_rank": inf_rank,
                "selected_training_topk_rank": train_rank,
                "inference_top1_id": int(inf_topk[key]["token_ids"][0]),
                "training_top1_id": int(train_topk[key]["token_ids"][0]),
                "selected_is_inference_top1": token_id == int(inf_topk[key]["token_ids"][0]),
                "selected_is_training_top1": token_id == int(train_topk[key]["token_ids"][0]),
            }
        )

    if not rows:
        raise SystemExit(f"No analyzable top-k rows after loading probe data. skipped={dict(skipped)}")

    deltas = np.array([row["topk_selected_logprob_delta"] for row in rows], dtype=np.float64)
    masks = _cohort_masks(rows, deltas)
    p50, p90, p99 = [float(x) for x in masks.pop("thresholds")]
    report = {
        "probe_dir": str(probe_dir),
        "analysis_mode": "fast_topk_only",
        "matched_rows": len(shared_keys),
        "analyzed_rows": len(rows),
        "skipped": dict(skipped),
        "generated_only": generated_only,
        "has_probed_selected_logprob": False,
        "cohort_metric": "topk_selected",
        "cohort_delta_thresholds": {"p50": p50, "p90": p90, "p99": p99},
        "cohorts": {},
        "examples": {},
    }
    for cohort_name, mask in masks.items():
        cohort_items = [row for row, keep in zip(rows, mask) if keep]
        report["cohorts"][cohort_name] = _summarize_fast_cohort(cohort_items)
    for cohort_name in ("above_p90", "all"):
        cohort_items = [row for row, keep in zip(rows, masks[cohort_name]) if keep]
        report["examples"][cohort_name] = sorted(
            cohort_items,
            key=lambda item: item["topk_selected_logprob_delta"],
            reverse=True,
        )[:20]
    return report


def _cohort_delta(row: dict[str, Any], *, metric: str) -> float:
    if metric == "probed":
        return float(row["probed_selected_logprob_delta"])
    if metric == "recomputed":
        return float(row["consistency"]["recomputed_train_inf_delta"])
    raise ValueError(f"Unknown cohort metric: {metric}")


def _print_summary(report: dict[str, Any]) -> None:
    thresholds = report.get("cohort_delta_thresholds") or report.get(
        "probed_selected_logprob_delta_thresholds"
    ) or {}
    metric = report.get("cohort_metric", "probed")
    print(
        f"\n=== logprob calibration summary ({metric} delta) ===",
        flush=True,
    )
    if thresholds:
        print(
            f"Thresholds: p50={thresholds.get('p50', float('nan')):.4f}  "
            f"p90={thresholds.get('p90', float('nan')):.4f}  "
            f"p99={thresholds.get('p99', float('nan')):.4f}",
            flush=True,
        )
    print(
        f"Matched={report.get('matched_rows')}  analyzed={report.get('analyzed_rows')}  "
        f"skipped={report.get('skipped')}",
        flush=True,
    )
    for cohort_name in ("all", "above_p50", "above_p90", "above_p99"):
        cohort = (report.get("cohorts") or {}).get(cohort_name) or {}
        count = cohort.get("count", 0)
        if not count:
            continue
        delta_stats = (
            cohort.get("topk_selected_logprob_delta")
            or cohort.get("probed_train_inf_delta")
            or cohort.get("recomputed_train_inf_delta")
            or {}
        )
        print(f"\n[{cohort_name}] count={count}", flush=True)
        if delta_stats:
            print(
                f"  |logprob_delta| p50={delta_stats.get('p50', float('nan')):.4f}  "
                f"p90={delta_stats.get('p90', float('nan')):.4f}  "
                f"p99={delta_stats.get('p99', float('nan')):.4f}",
                flush=True,
            )
        top_tokens = cohort.get("top_tokens") or []
        if top_tokens:
            print("  top tokens:", flush=True)
            for token_text, token_count in top_tokens[:15]:
                print(f"    {token_count:5d}  {token_text!r}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True, help="Determinism probe directory")
    parser.add_argument(
        "--training-phase",
        default="training_old_logprobs",
        help="Training probe phase name (default: training_old_logprobs)",
    )
    parser.add_argument("--inference-phase", default="inference", help="Inference probe phase name")
    parser.add_argument("--tokenizer-json", type=Path, required=True, help="HuggingFace tokenizer.json")
    parser.add_argument(
        "--output-weight-npy",
        type=Path,
        default=None,
        help="Output layer weight as float32 .npy with shape [vocab, hidden]. "
        "Required to recompute inference logits when inference lm_logits probes are empty.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k size for support sets")
    parser.add_argument(
        "--support-mode",
        choices=("union", "inference_topk", "training_topk"),
        default="union",
        help="Support set for renormalized probabilities",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write full JSON report",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on analyzed rows (0 = all matched rows)",
    )
    parser.add_argument(
        "--generated-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only analyze generated-token targets (default: true)",
    )
    parser.add_argument(
        "--cohort-metric",
        choices=("auto", "probed", "recomputed"),
        default="auto",
        help="Metric for p50/p90/p99 cohort thresholds (default: auto)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print human-readable cohort summary instead of full JSON",
    )
    parser.add_argument(
        "--fast-topk-only",
        action="store_true",
        help=(
            "Use captured lm_topk_logprobs/lm_topk_token_ids only. This is much faster for "
            "single-pass probes and reports selected-token p50/p90/p99 cohorts, but does not "
            "run full-vocab recompute or calibration diagnostics."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    probe_dir = args.probe_dir

    id_to_token = _load_tokenizer(args.tokenizer_json)
    if args.fast_topk_only:
        report = _build_fast_topk_report(
            probe_dir=probe_dir,
            inference_phase=args.inference_phase,
            training_phase=args.training_phase,
            generated_only=args.generated_only,
            id_to_token=id_to_token,
            max_rows=args.max_rows,
        )
        if not args.summary_only:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        _print_summary(report)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nWrote {args.output_json}", flush=True)
        return

    if torch is None:
        raise SystemExit(
            "Full-vocab analysis requires PyTorch. Use --fast-topk-only for the fast "
            "selected-token top-k report, or run in an environment with torch installed."
        ) from _TORCH_IMPORT_ERROR

    device = torch.device(args.device)
    output_weight = _load_output_weight_npy(args.output_weight_npy, device) if args.output_weight_npy else None

    has_inf_logprobs = _probe_module_present(probe_dir, args.inference_phase, "lm_selected_logprob")
    has_train_logprobs = _probe_module_present(probe_dir, args.training_phase, "lm_selected_logprob")
    has_probed_logprobs = has_inf_logprobs and has_train_logprobs

    # Load the smaller training side fully, then filter inference to matching keys.
    train_hidden = _load_probe_records(
        probe_dir,
        args.training_phase,
        "lm_final_hidden",
        generated_only=args.generated_only,
    )
    if not train_hidden:
        raise SystemExit(f"No training lm_final_hidden rows found under {probe_dir}")

    train_keys = set(train_hidden)
    inf_hidden = _load_probe_records(
        probe_dir,
        args.inference_phase,
        "lm_final_hidden",
        keys_wanted=train_keys,
        generated_only=args.generated_only,
    )
    shared_keys = sorted(set(inf_hidden) & train_keys)
    if args.max_rows > 0:
        shared_keys = shared_keys[: args.max_rows]
    if not shared_keys:
        raise SystemExit(f"No matched probe rows found under {probe_dir}")

    shared_key_set = set(shared_keys)
    inf_logprobs = (
        _load_probe_records(
            probe_dir,
            args.inference_phase,
            "lm_selected_logprob",
            keys_wanted=shared_key_set,
            generated_only=args.generated_only,
        )
        if has_inf_logprobs
        else {}
    )
    train_logprobs = (
        _load_probe_records(
            probe_dir,
            args.training_phase,
            "lm_selected_logprob",
            keys_wanted=shared_key_set,
            generated_only=args.generated_only,
        )
        if has_train_logprobs
        else {}
    )
    inf_logits = _load_probe_records(
        probe_dir,
        args.inference_phase,
        "lm_logits",
        keys_wanted=shared_key_set,
        generated_only=args.generated_only,
    )
    train_logits = _load_probe_records(
        probe_dir,
        args.training_phase,
        "lm_logits",
        keys_wanted=shared_key_set,
        generated_only=args.generated_only,
    )

    if output_weight is None and not inf_logits:
        raise SystemExit(
            "Inference lm_logits probes are empty and --output-weight-npy was not provided. "
            "Provide an output weight to recompute inference logits from lm_final_hidden."
        )

    vocab_size = int(output_weight.shape[0]) if output_weight is not None else None
    if vocab_size is None and train_logits:
        sample = next(iter(train_logits.values()))
        vocab_size = int((sample.get("value") or {}).get("original_numel", 0))
    if not vocab_size:
        raise SystemExit("Could not determine vocab size from output weight or training lm_logits.")

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for key in shared_keys:
        token = train_hidden[key].get("token") or {}
        token_id = token.get("target_token_id")
        if token_id is None:
            skipped["missing_target_token_id"] += 1
            continue
        token_id = int(token_id)

        logits_train, source_train = _logits_from_record_or_hidden(
            record=train_logits.get(key),
            hidden_record=train_hidden.get(key),
            output_weight=output_weight,
            device=device,
            vocab_size=vocab_size,
        )
        logits_inf, source_inf = _logits_from_record_or_hidden(
            record=inf_logits.get(key),
            hidden_record=inf_hidden.get(key),
            output_weight=output_weight,
            device=device,
            vocab_size=vocab_size,
        )
        if logits_train is None or logits_inf is None:
            skipped["missing_logits"] += 1
            continue

        logprobs_inf = _inference_aligned_logprob(logits_inf)
        logprobs_train = _inference_aligned_logprob(logits_train)
        recomputed_logp_inf = float(logprobs_inf[token_id].item())
        recomputed_logp_train = float(logprobs_train[token_id].item())

        if key in inf_logprobs and key in train_logprobs:
            probed_logp_inf = _scalar_probe_value(inf_logprobs[key], device)
            probed_logp_train = _scalar_probe_value(train_logprobs[key], device)
        else:
            probed_logp_inf = recomputed_logp_inf
            probed_logp_train = recomputed_logp_train

        analyzed = _analyze_row(
            key=key,
            token_id=token_id,
            token_text=_decode_token(id_to_token, token_id),
            logits_inf=logits_inf,
            logits_train=logits_train,
            logit_source_inf=source_inf,
            logit_source_train=source_train,
            probed_logp_inf=probed_logp_inf,
            probed_logp_train=probed_logp_train,
            top_k=args.top_k,
            support_mode=args.support_mode,
        )
        rows.append(analyzed)

    if not rows:
        raise SystemExit(f"No analyzable rows after loading probe data. skipped={dict(skipped)}")

    cohort_metric = args.cohort_metric
    if cohort_metric == "auto":
        cohort_metric = "probed" if has_probed_logprobs else "recomputed"

    cohort_deltas = np.array([_cohort_delta(row, metric=cohort_metric) for row in rows], dtype=np.float64)
    masks = _cohort_masks(rows, cohort_deltas)
    p50, p90, p99 = [float(x) for x in masks.pop("thresholds")]

    report = {
        "probe_dir": str(probe_dir),
        "matched_rows": len(shared_keys),
        "analyzed_rows": len(rows),
        "skipped": dict(skipped),
        "generated_only": args.generated_only,
        "has_probed_selected_logprob": has_probed_logprobs,
        "cohort_metric": cohort_metric,
        "vocab_size": vocab_size,
        "top_k": args.top_k,
        "support_mode": args.support_mode,
        "cohort_delta_thresholds": {"p50": p50, "p90": p90, "p99": p99},
        "probed_selected_logprob_delta_thresholds": {"p50": p50, "p90": p90, "p99": p99},
        "logit_sources": {
            "inference": dict(Counter(row["logit_source_inference"] for row in rows)),
            "training": dict(Counter(row["logit_source_training"] for row in rows)),
        },
        "cohorts": {},
        "examples": {},
    }

    for cohort_name, mask in masks.items():
        cohort_items = [row for row, keep in zip(rows, mask) if keep]
        report["cohorts"][cohort_name] = _summarize_cohort(cohort_items)
        report["cohorts"][cohort_name]["top_tokens"] = _token_histogram(cohort_items)

    sort_key = lambda item: _cohort_delta(item, metric=cohort_metric)
    for cohort_name in ("above_p90_top1_stable", "above_p90", "all"):
        cohort_items = [row for row, keep in zip(rows, masks[cohort_name]) if keep]
        report["examples"][cohort_name] = sorted(cohort_items, key=sort_key, reverse=True)[:20]

    report["examples"]["consistency_outliers"] = sorted(
        rows,
        key=lambda item: max(
            item["consistency"]["probed_vs_recomputed_abs_delta_inference"],
            item["consistency"]["probed_vs_recomputed_abs_delta_training"],
        ),
        reverse=True,
    )[:20]

    if args.summary_only:
        _print_summary(report)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        _print_summary(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
