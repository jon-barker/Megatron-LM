# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Output-layer perturbation sensitivity diagnostics for loaded GPT models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.utils import unwrap_model
from megatron.training import print_rank_0


def _percentiles(values: torch.Tensor, percentiles=(0.5, 0.9, 0.95, 0.99)) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    values = values.detach().float().cpu()
    result = {
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }
    for p in percentiles:
        result[f"p{int(p * 100):02d}"] = float(torch.quantile(values, p).item())
    return result


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() < 2:
        return None
    x = x.detach().float()
    y = y.detach().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) == 0.0:
        return None
    return float((x * y).sum().item() / denom.item())


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() < 2:
        return None
    return _pearson_corr(_rankdata(x.detach().float()), _rankdata(y.detach().float()))


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _selected_values(logits: torch.Tensor, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    row_ids = torch.arange(logits_f.shape[0], device=logits_f.device)
    selected_logits = logits_f[row_ids, token_ids]
    selected_logprobs = F.log_softmax(logits_f, dim=-1)[row_ids, token_ids]
    return selected_logits, selected_logprobs


def _run_output_layer(
    output_layer,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # ColumnParallelLinear expects [sequence, batch, hidden].
    logits, _ = output_layer(hidden_states.unsqueeze(1), runtime_gather_output=True)
    return logits.squeeze(1)


def _top_right_singular_vectors(
    output_weight: torch.Tensor,
    *,
    num_components: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top right singular vectors of output_weight [vocab, hidden]."""
    hidden_size = output_weight.shape[1]
    gram = torch.zeros(hidden_size, hidden_size, device=output_weight.device, dtype=torch.float32)
    for start in range(0, output_weight.shape[0], chunk_size):
        chunk = output_weight[start : start + chunk_size].float()
        gram.add_(chunk.t().matmul(chunk))
    evals, evecs = torch.linalg.eigh(gram)
    top = torch.argsort(evals, descending=True)[:num_components]
    singular_values = evals[top].clamp_min(0).sqrt()
    right_vectors = evecs[:, top].t().contiguous()
    return singular_values, right_vectors


def _pair_direction_analysis(
    *,
    name: str,
    delta_hidden: torch.Tensor,
    output_weight: torch.Tensor,
    logits_inf: torch.Tensor,
    logits_train: torch.Tensor,
    token_a: torch.Tensor,
    token_b: torch.Tensor,
    selected_logit_abs_delta: torch.Tensor,
    selected_logprob_abs_delta: torch.Tensor,
) -> dict[str, Any]:
    valid = token_a != token_b
    if not valid.any():
        return {
            "name": name,
            "count": 0,
        }

    delta_v = delta_hidden[valid].float()
    a = token_a[valid].long()
    b = token_b[valid].long()
    pair_direction = output_weight[a].float() - output_weight[b].float()
    pair_norm = pair_direction.norm(dim=-1).clamp_min(1e-12)
    delta_norm = delta_v.norm(dim=-1).clamp_min(1e-12)
    projection = (delta_v * pair_direction).sum(dim=-1)
    cosine = projection / (delta_norm * pair_norm)

    row_ids = torch.arange(logits_inf.shape[0], device=logits_inf.device)[valid]
    inf_margin = logits_inf.float()[row_ids, a] - logits_inf.float()[row_ids, b]
    train_margin = logits_train.float()[row_ids, a] - logits_train.float()[row_ids, b]
    margin_delta = train_margin - inf_margin
    projection_error = projection - margin_delta

    selected_logit_abs = selected_logit_abs_delta[valid].float()
    selected_logprob_abs = selected_logprob_abs_delta[valid].float()
    return {
        "name": name,
        "count": int(valid.sum().item()),
        "token_a_equals_token_b_count": int((~valid).sum().item()),
        "pair_direction_norm": _percentiles(pair_norm),
        "signed_projection": _percentiles(projection),
        "abs_projection": _percentiles(projection.abs()),
        "signed_cosine": _percentiles(cosine),
        "abs_cosine": _percentiles(cosine.abs()),
        "inference_pair_margin": _percentiles(inf_margin),
        "training_pair_margin": _percentiles(train_margin),
        "pair_margin_delta": _percentiles(margin_delta),
        "projection_minus_margin_delta_abs": _percentiles(projection_error.abs()),
        "corr_abs_projection_vs_selected_logit_abs_delta": _pearson_corr(
            projection.abs(), selected_logit_abs
        ),
        "spearman_abs_projection_vs_selected_logit_abs_delta": _spearman_corr(
            projection.abs(), selected_logit_abs
        ),
        "corr_abs_projection_vs_selected_logprob_abs_delta": _pearson_corr(
            projection.abs(), selected_logprob_abs
        ),
        "spearman_abs_projection_vs_selected_logprob_abs_delta": _spearman_corr(
            projection.abs(), selected_logprob_abs
        ),
        "corr_abs_cosine_vs_selected_logit_abs_delta": _pearson_corr(
            cosine.abs(), selected_logit_abs
        ),
        "corr_abs_cosine_vs_selected_logprob_abs_delta": _pearson_corr(
            cosine.abs(), selected_logprob_abs
        ),
    }


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


def _load_probe_records(probe_dir: str | None, phase: str, module_name: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not probe_dir:
        return {}
    result = {}
    for path in Path(probe_dir).glob(f"probe_{phase}_iter*_rank*.jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if f'"module_name": "{module_name}"' not in line:
                    continue
                record = json.loads(line)
                if record.get("hash") is None or not record.get("token") or "value" not in record:
                    continue
                key = _record_key(record)
                if key is not None:
                    result[key] = record
    return result


def _value_tensor(record: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    value = record.get("value") or {}
    data = value.get("data")
    if data is None:
        return None
    return torch.tensor(data, dtype=torch.float32, device=device)


def _analyze_observed_probe_hidden_deltas(
    *,
    output_layer,
    output_weight: torch.Tensor,
    probe_dir: str | None,
    inference_phase: str,
    training_phase: str,
    svd_components: int = 0,
    svd_chunk_size: int = 8192,
) -> dict[str, Any] | None:
    if not probe_dir:
        return None

    inf_hidden = _load_probe_records(probe_dir, inference_phase, "lm_final_hidden")
    train_hidden = _load_probe_records(probe_dir, training_phase, "lm_final_hidden")
    inf_logprob = _load_probe_records(probe_dir, inference_phase, "lm_selected_logprob")
    train_logprob = _load_probe_records(probe_dir, training_phase, "lm_selected_logprob")
    shared = sorted(set(inf_hidden) & set(train_hidden))
    if not shared:
        return {
            "probe_dir": probe_dir,
            "matched_hidden_records": 0,
            "inference_hidden_records": len(inf_hidden),
            "training_hidden_records": len(train_hidden),
        }

    device = output_weight.device
    rows = []
    inf_rows = []
    train_rows = []
    token_ids = []
    keys = []
    for key in shared:
        inf = _value_tensor(inf_hidden[key], device)
        train = _value_tensor(train_hidden[key], device)
        if inf is None or train is None or inf.numel() != output_weight.shape[1] or train.numel() != output_weight.shape[1]:
            continue
        token_id = (train_hidden[key].get("token") or {}).get("target_token_id")
        if token_id is None or int(token_id) < 0 or int(token_id) >= output_weight.shape[0]:
            continue
        inf_rows.append(inf)
        train_rows.append(train)
        token_ids.append(int(token_id))
        keys.append(key)

    if not inf_rows:
        return {
            "probe_dir": probe_dir,
            "matched_hidden_records": 0,
            "inference_hidden_records": len(inf_hidden),
            "training_hidden_records": len(train_hidden),
            "reason": "No matched full hidden rows with valid token IDs.",
        }

    inf_h = torch.stack(inf_rows)
    train_h = torch.stack(train_rows)
    token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
    delta = train_h - inf_h
    selected_weight = output_weight[token_ids_t].float()

    delta_norm = delta.norm(dim=-1)
    train_norm = train_h.norm(dim=-1).clamp_min(1e-12)
    weight_norm = selected_weight.norm(dim=-1).clamp_min(1e-12)
    alignment = (delta * selected_weight).sum(dim=-1) / (delta_norm.clamp_min(1e-12) * weight_norm)
    projected_logit_delta = (delta * selected_weight).sum(dim=-1)

    with torch.no_grad():
        logits_inf = _run_output_layer(output_layer, inf_h.to(output_weight.dtype))
        logits_train = _run_output_layer(output_layer, train_h.to(output_weight.dtype))
        selected_logits_inf, selected_logprobs_inf = _selected_values(logits_inf, token_ids_t)
        selected_logits_train, selected_logprobs_train = _selected_values(logits_train, token_ids_t)

    selected_logit_delta = selected_logits_train - selected_logits_inf
    selected_logprob_delta = selected_logprobs_train - selected_logprobs_inf
    selected_logit_abs = selected_logit_delta.abs()
    selected_logprob_abs = selected_logprob_delta.abs()
    inf_top2 = torch.topk(logits_inf.float(), k=2, dim=-1)
    train_top2 = torch.topk(logits_train.float(), k=2, dim=-1)
    inf_top1 = inf_top2.indices[:, 0]
    train_top1 = train_top2.indices[:, 0]
    inf_top2_token = inf_top2.indices[:, 1]
    train_top2_token = train_top2.indices[:, 1]
    inf_top1_margin = inf_top2.values[:, 0] - inf_top2.values[:, 1]
    train_top1_margin = train_top2.values[:, 0] - train_top2.values[:, 1]
    probed_logprob_delta = []
    for key in keys:
        if key in inf_logprob and key in train_logprob:
            inf_lp = _value_tensor(inf_logprob[key], device)
            train_lp = _value_tensor(train_logprob[key], device)
            if inf_lp is not None and train_lp is not None and inf_lp.numel() and train_lp.numel():
                probed_logprob_delta.append((train_lp.flatten()[0] - inf_lp.flatten()[0]).float())
    probed_logprob_delta_t = torch.stack(probed_logprob_delta) if probed_logprob_delta else torch.empty(0, device=device)

    selected_logit_abs = selected_logit_delta.abs()
    selected_logprob_abs = selected_logprob_delta.abs()
    result = {
        "probe_dir": probe_dir,
        "matched_hidden_records": len(keys),
        "inference_hidden_records": len(inf_hidden),
        "training_hidden_records": len(train_hidden),
        "relative_l2": _percentiles(delta_norm / train_norm),
        "delta_hidden_l2": _percentiles(delta_norm),
        "selected_weight_norm": _percentiles(weight_norm),
        "cos_delta_hidden_selected_weight": _percentiles(alignment.abs()),
        "signed_cos_delta_hidden_selected_weight": _percentiles(alignment),
        "projected_selected_logit_delta": _percentiles(projected_logit_delta.abs()),
        "selected_logit_delta_recomputed": _percentiles(selected_logit_delta.abs()),
        "selected_logprob_delta_recomputed": _percentiles(selected_logprob_delta.abs()),
        "selected_logprob_delta_probed": _percentiles(probed_logprob_delta_t.abs()),
        "inference_top1_margin": _percentiles(inf_top1_margin),
        "training_top1_margin": _percentiles(train_top1_margin),
        "top1_disagreement_count": int((inf_top1 != train_top1).sum().item()),
        "top1_agreement_count": int((inf_top1 == train_top1).sum().item()),
        "selected_token_is_inference_top1_count": int((token_ids_t == inf_top1).sum().item()),
        "selected_token_is_training_top1_count": int((token_ids_t == train_top1).sum().item()),
        "pair_direction_analysis": [
            _pair_direction_analysis(
                name="training_top1_minus_inference_top1",
                delta_hidden=delta,
                output_weight=output_weight,
                logits_inf=logits_inf,
                logits_train=logits_train,
                token_a=train_top1,
                token_b=inf_top1,
                selected_logit_abs_delta=selected_logit_abs,
                selected_logprob_abs_delta=selected_logprob_abs,
            ),
            _pair_direction_analysis(
                name="selected_minus_inference_top1",
                delta_hidden=delta,
                output_weight=output_weight,
                logits_inf=logits_inf,
                logits_train=logits_train,
                token_a=token_ids_t,
                token_b=inf_top1,
                selected_logit_abs_delta=selected_logit_abs,
                selected_logprob_abs_delta=selected_logprob_abs,
            ),
            _pair_direction_analysis(
                name="selected_minus_training_top1",
                delta_hidden=delta,
                output_weight=output_weight,
                logits_inf=logits_inf,
                logits_train=logits_train,
                token_a=token_ids_t,
                token_b=train_top1,
                selected_logit_abs_delta=selected_logit_abs,
                selected_logprob_abs_delta=selected_logprob_abs,
            ),
            _pair_direction_analysis(
                name="inference_top1_minus_inference_top2",
                delta_hidden=delta,
                output_weight=output_weight,
                logits_inf=logits_inf,
                logits_train=logits_train,
                token_a=inf_top1,
                token_b=inf_top2_token,
                selected_logit_abs_delta=selected_logit_abs,
                selected_logprob_abs_delta=selected_logprob_abs,
            ),
            _pair_direction_analysis(
                name="training_top1_minus_training_top2",
                delta_hidden=delta,
                output_weight=output_weight,
                logits_inf=logits_inf,
                logits_train=logits_train,
                token_a=train_top1,
                token_b=train_top2_token,
                selected_logit_abs_delta=selected_logit_abs,
                selected_logprob_abs_delta=selected_logprob_abs,
            ),
        ],
    }
    if svd_components > 0:
        print_rank_0(
            f"[Output-layer-sensitivity] computing top {svd_components} output-weight SVD directions"
        )
        singular_values, right_vectors = _top_right_singular_vectors(
            output_weight,
            num_components=svd_components,
            chunk_size=svd_chunk_size,
        )
        projections = delta.float().matmul(right_vectors.t())
        abs_projections = projections.abs()
        abs_cosines = abs_projections / delta_norm.clamp_min(1e-12).unsqueeze(-1)
        weighted_abs_projections = abs_projections * singular_values.unsqueeze(0)
        topk_projection_l2 = projections.norm(dim=-1)
        topk_energy_fraction = topk_projection_l2 / delta_norm.clamp_min(1e-12)
        max_abs_projection = abs_projections.max(dim=-1).values
        max_abs_cosine = abs_cosines.max(dim=-1).values
        max_weighted_abs_projection = weighted_abs_projections.max(dim=-1).values

        per_component = []
        for idx in range(svd_components):
            per_component.append(
                {
                    "component": idx,
                    "singular_value": float(singular_values[idx].item()),
                    "abs_projection": _percentiles(abs_projections[:, idx]),
                    "abs_cosine": _percentiles(abs_cosines[:, idx]),
                    "corr_abs_projection_vs_selected_logit_abs_delta": _pearson_corr(
                        abs_projections[:, idx], selected_logit_abs
                    ),
                    "spearman_abs_projection_vs_selected_logit_abs_delta": _spearman_corr(
                        abs_projections[:, idx], selected_logit_abs
                    ),
                    "corr_abs_projection_vs_selected_logprob_abs_delta": _pearson_corr(
                        abs_projections[:, idx], selected_logprob_abs
                    ),
                    # "corr_abs_projection_vs_selected_logit_abs_delta_to_idx": _pearson_corr(
                    #     abs_projections[:, :idx].norm(dim=-1), selected_logit_abs
                    # ),
                    # "spearman_abs_projection_vs_selected_logit_abs_delta_to_idx": _spearman_corr(
                    #     abs_projections[:, :idx].norm(dim=-1), selected_logit_abs
                    # ),
                    # "corr_abs_projection_vs_selected_logprob_abs_delta_to_idx": _pearson_corr(
                    #     abs_projections[:, :idx].norm(dim=-1), selected_logprob_abs
                    # ),
                }
            )

        result["output_weight_svd_alignment"] = {
            "num_components": int(svd_components),
            "singular_values": [float(x.item()) for x in singular_values],
            "topk_projection_l2": _percentiles(topk_projection_l2),
            "topk_energy_fraction": _percentiles(topk_energy_fraction),
            "max_abs_projection": _percentiles(max_abs_projection),
            "max_abs_cosine": _percentiles(max_abs_cosine),
            "max_weighted_abs_projection": _percentiles(max_weighted_abs_projection),
            "corr_topk_projection_l2_vs_selected_logit_abs_delta": _pearson_corr(
                topk_projection_l2, selected_logit_abs
            ),
            "corr_topk_energy_fraction_vs_selected_logit_abs_delta": _pearson_corr(
                topk_energy_fraction, selected_logit_abs
            ),
            "corr_max_abs_projection_vs_selected_logit_abs_delta": _pearson_corr(
                max_abs_projection, selected_logit_abs
            ),
            "corr_max_abs_cosine_vs_selected_logit_abs_delta": _pearson_corr(
                max_abs_cosine, selected_logit_abs
            ),
            "corr_max_weighted_abs_projection_vs_selected_logit_abs_delta": _pearson_corr(
                max_weighted_abs_projection, selected_logit_abs
            ),
            "spearman_max_weighted_abs_projection_vs_selected_logit_abs_delta": _spearman_corr(
                max_weighted_abs_projection, selected_logit_abs
            ),
            "per_component": per_component,
        }
    return result


def _simulate_mode(
    *,
    output_layer,
    output_weight: torch.Tensor,
    num_samples: int,
    relative_l2: float,
    hidden_rms: float,
    mode: str,
    generator: torch.Generator,
) -> dict[str, Any]:
    device = output_weight.device
    hidden_size = output_weight.shape[1]
    vocab_size = output_weight.shape[0]

    base = torch.randn(num_samples, hidden_size, device=device, generator=generator)
    base = _normalize_rows(base) * (hidden_rms * (hidden_size**0.5))

    token_ids = torch.randint(0, vocab_size, (num_samples,), device=device, generator=generator)
    if mode == "random_isotropic":
        direction = _normalize_rows(torch.randn(num_samples, hidden_size, device=device, generator=generator))
    elif mode == "token_aligned":
        direction = _normalize_rows(output_weight[token_ids].float())
    elif mode == "top_weight_norm_aligned":
        top_ids = torch.topk(output_weight.float().norm(dim=1), k=min(num_samples, vocab_size)).indices
        if top_ids.numel() < num_samples:
            repeats = (num_samples + top_ids.numel() - 1) // top_ids.numel()
            top_ids = top_ids.repeat(repeats)
        token_ids = top_ids[:num_samples]
        direction = _normalize_rows(output_weight[token_ids].float())
    else:
        raise ValueError(f"Unknown output-layer sensitivity mode: {mode}")

    delta = direction * (relative_l2 * base.norm(dim=-1, keepdim=True))
    perturbed = base + delta

    with torch.no_grad():
        logits_base = _run_output_layer(output_layer, base.to(output_weight.dtype))
        logits_perturbed = _run_output_layer(output_layer, perturbed.to(output_weight.dtype))
        selected_logits_base, selected_logprobs_base = _selected_values(logits_base, token_ids)
        selected_logits_perturbed, selected_logprobs_perturbed = _selected_values(
            logits_perturbed, token_ids
        )

    selected_logit_delta = (selected_logits_perturbed - selected_logits_base).abs()
    selected_logprob_delta = (selected_logprobs_perturbed - selected_logprobs_base).abs()
    actual_relative_l2 = (perturbed - base).norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-12)

    return {
        "mode": mode,
        "num_samples": int(num_samples),
        "target_relative_l2": float(relative_l2),
        "hidden_rms": float(hidden_rms),
        "actual_relative_l2": _percentiles(actual_relative_l2),
        "selected_logit_abs_delta": _percentiles(selected_logit_delta),
        "selected_logprob_abs_delta": _percentiles(selected_logprob_delta),
    }


def run_output_layer_sensitivity(model: list, args) -> None:
    """Run synthetic perturbations through the loaded model's output layer."""
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    module = unwrap_model(model[0])
    if not hasattr(module, "output_layer"):
        raise RuntimeError("Loaded model has no output_layer on this rank.")

    output_layer = module.output_layer
    weight = output_layer.weight.detach()
    if weight.device.type != "cuda":
        output_layer = output_layer.cuda()
        weight = output_layer.weight.detach()

    generator = torch.Generator(device=weight.device)
    sensitivity_seed = getattr(args, "output_layer_sensitivity_seed", None)
    if sensitivity_seed is None:
        sensitivity_seed = getattr(args, "seed", 1234)
    generator.manual_seed(int(sensitivity_seed))

    rels = [
        float(x)
        for x in str(getattr(args, "output_layer_sensitivity_relative_l2", "0.005,0.01,0.015,0.02")).split(",")
        if x.strip()
    ]
    modes = [
        x.strip()
        for x in str(getattr(args, "output_layer_sensitivity_modes", "random_isotropic,token_aligned,top_weight_norm_aligned")).split(",")
        if x.strip()
    ]
    num_samples = int(getattr(args, "output_layer_sensitivity_num_samples", 256))
    hidden_rms = float(getattr(args, "output_layer_sensitivity_hidden_rms", 1.0))

    results = {
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype).replace("torch.", ""),
        "weight_row_norm": _percentiles(weight.float().norm(dim=1)),
        "experiments": [],
    }
    observed = _analyze_observed_probe_hidden_deltas(
        output_layer=output_layer,
        output_weight=weight,
        probe_dir=getattr(args, "output_layer_sensitivity_probe_dir", None),
        inference_phase=getattr(args, "output_layer_sensitivity_inference_phase", "inference"),
        training_phase=getattr(args, "output_layer_sensitivity_training_phase", "training_old_logprobs"),
        svd_components=int(getattr(args, "output_layer_sensitivity_svd_components", 0) or 0),
        svd_chunk_size=int(getattr(args, "output_layer_sensitivity_svd_chunk_size", 8192) or 8192),
    )
    if observed is not None:
        results["observed_probe_hidden_delta"] = observed
        print_rank_0(
            "[Output-layer-sensitivity] observed probe hidden records: "
            f"{observed.get('matched_hidden_records', 0)}"
        )
    for mode in modes:
        for rel in rels:
            print_rank_0(f"[Output-layer-sensitivity] mode={mode} relative_l2={rel}")
            results["experiments"].append(
                _simulate_mode(
                    output_layer=output_layer,
                    output_weight=weight,
                    num_samples=num_samples,
                    relative_l2=rel,
                    hidden_rms=hidden_rms,
                    mode=mode,
                    generator=generator,
                )
            )

    output_path = Path(getattr(args, "output_layer_sensitivity_results_file", "results/output_layer_sensitivity.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print_rank_0(f"[Output-layer-sensitivity] wrote {output_path}")
