#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare RL determinism probe JSONL logs from inference and training."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - only needed when rollout npz files are present.
    np = None


def _iter_records(paths: list[str]):
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
                if record.get("hash") is not None and record.get("token"):
                    yield record


def _phase_paths(probe_dir: Path, phase: str) -> list[str]:
    return sorted(glob.glob(str(probe_dir / f"probe_{phase}_iter*_rank*.jsonl")))


def _rollout_prompt_lengths(probe_dir: Path) -> dict[str, int]:
    if np is None:
        return {}
    prompt_lengths = {}
    for path in probe_dir.glob("rollout_*.npz"):
        with np.load(path) as data:
            dump_id = str(data.get("routing_dump_id", path.stem.replace("rollout_", "")))
            if "prompt_tokens" in data:
                prompt_lengths[dump_id] = int(len(data["prompt_tokens"]))
    return prompt_lengths


def _join_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    token = record["token"]
    routing_dump_id = token.get("routing_dump_id")
    target_token_index = token.get("target_token_index")
    if routing_dump_id is None or target_token_index is None:
        return None
    return (
        record.get("iteration"),
        routing_dump_id,
        int(target_token_index),
        record.get("module_name"),
        record.get("io_kind"),
        record.get("tensor_path"),
    )


def _token_key(record: dict[str, Any]) -> tuple[Any, ...]:
    token = record["token"]
    return (
        record.get("iteration"),
        token.get("routing_dump_id"),
        int(token.get("target_token_index")),
    )


def _value_array(record: dict[str, Any]):
    value = record.get("value")
    if np is None or not value:
        return None
    data = value.get("data")
    if data is None:
        return None
    return np.asarray(data, dtype=np.float32)


def _numeric_delta(inference_record: dict[str, Any], training_record: dict[str, Any]) -> dict[str, Any] | None:
    inf = _value_array(inference_record)
    train = _value_array(training_record)
    if inf is None or train is None or inf.size == 0 or train.size == 0:
        return None
    inf_size = inf.size
    train_size = train.size
    n = min(inf.size, train.size)
    inf = inf[:n]
    train = train[:n]
    diff = inf - train
    abs_diff = np.abs(diff)
    denom = np.maximum(np.maximum(np.abs(inf), np.abs(train)), 1e-12)
    inf_norm = float(np.linalg.norm(inf))
    train_norm = float(np.linalg.norm(train))
    diff_norm = float(np.linalg.norm(diff))
    scale_norm = max((inf_norm + train_norm) * 0.5, 1e-12)
    train_rms = float(np.sqrt(np.mean(train * train)))
    inf_rms = float(np.sqrt(np.mean(inf * inf)))
    rms_diff = float(np.sqrt(np.mean(diff * diff)))
    train_rms_denom = max(train_rms, 1e-12)
    cosine = None
    if inf_norm > 0.0 and train_norm > 0.0:
        cosine = float(np.dot(inf, train) / (inf_norm * train_norm))
    return {
        "compared_elements": int(n),
        "value_truncated": bool(
            inference_record.get("value", {}).get("truncated")
            or training_record.get("value", {}).get("truncated")
            or inf_size != train_size
        ),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": rms_diff,
        "max_rel_diff": float((abs_diff / denom).max()),
        "delta_l2": diff_norm,
        "relative_l2": diff_norm / max(train_norm, 1e-12),
        "relative_l2_symmetric": diff_norm / scale_norm,
        "train_rms": train_rms,
        "inference_rms": inf_rms,
        "rms_diff_over_train_rms": rms_diff / train_rms_denom,
        "max_abs_over_train_rms": float(abs_diff.max()) / train_rms_denom,
        "cosine": cosine,
        "inference_l2": inf_norm,
        "training_l2": train_norm,
    }


def _load_phase_records(
    probe_dir: Path,
    phase: str,
    *,
    generated_only: bool,
    prompt_lengths: dict[str, int],
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[str, int], Counter]:
    records = {}
    module_order = {}
    dropped = Counter()
    for order, record in enumerate(_iter_records(_phase_paths(probe_dir, phase))):
        token = record["token"]
        if token.get("is_padding"):
            dropped["padding"] += 1
            continue
        if generated_only:
            if "is_generated_target" in token and not token["is_generated_target"]:
                dropped["non_generated_target"] += 1
                continue
            prompt_len = prompt_lengths.get(token.get("routing_dump_id"))
            target_token_index = token.get("target_token_index")
            if prompt_len is not None and target_token_index is not None:
                if int(target_token_index) < prompt_len:
                    dropped["prompt_target"] += 1
                    continue
        key = _join_key(record)
        if key is None:
            dropped["missing_join_key"] += 1
            continue
        records[key] = record
        module_order.setdefault(record.get("module_name"), order)
    return records, module_order, dropped


def compare_probe_dir(
    probe_dir: Path,
    *,
    inference_phase: str,
    training_phase: str,
    generated_only: bool,
) -> dict[str, Any]:
    prompt_lengths = _rollout_prompt_lengths(probe_dir)
    inference, inference_order, inference_dropped = _load_phase_records(
        probe_dir,
        inference_phase,
        generated_only=generated_only,
        prompt_lengths=prompt_lengths,
    )
    training, training_order, training_dropped = _load_phase_records(
        probe_dir,
        training_phase,
        generated_only=generated_only,
        prompt_lengths=prompt_lengths,
    )

    shared_keys = sorted(set(inference) & set(training))
    mismatches_by_token = defaultdict(list)
    numeric_all_by_module = defaultdict(list)
    matched = 0
    for key in shared_keys:
        inf_record = inference[key]
        train_record = training[key]
        if inf_record["hash"] == train_record["hash"]:
            matched += 1
            continue
        module_name = inf_record.get("module_name")
        numeric_delta = _numeric_delta(inf_record, train_record)
        if numeric_delta is not None:
            numeric_all_by_module[module_name].append(numeric_delta)
        order = min(
            inference_order.get(module_name, 10**12),
            training_order.get(module_name, 10**12),
        )
        mismatches_by_token[_token_key(inf_record)].append(
            {
                "order": order,
                "module_name": module_name,
                "io_kind": inf_record.get("io_kind"),
                "tensor_path": inf_record.get("tensor_path"),
                "inference_hash": inf_record["hash"],
                "training_hash": train_record["hash"],
                "numeric_delta": numeric_delta,
            }
        )

    first_mismatches = []
    histogram = Counter()
    numeric_by_module = defaultdict(list)
    for token_key, mismatches in mismatches_by_token.items():
        first = min(
            mismatches,
            key=lambda item: (item["order"], item["module_name"] or "", item["tensor_path"] or ""),
        )
        histogram[first["module_name"]] += 1
        if first.get("numeric_delta") is not None:
            numeric_by_module[first["module_name"]].append(first["numeric_delta"])
        first_mismatches.append(
            {
                "iteration": token_key[0],
                "routing_dump_id": token_key[1],
                "target_token_index": token_key[2],
                **{k: v for k, v in first.items() if k != "order"},
            }
        )

    first_mismatches.sort(
        key=lambda item: (
            item["iteration"],
            item["routing_dump_id"],
            item["target_token_index"],
            item["module_name"] or "",
        )
    )

    numeric_histogram = []
    for module_name, rows in numeric_by_module.items():
        numeric_histogram.append(
            {
                "module_name": module_name,
                "count": len(rows),
                "max_abs_diff_max": max(row["max_abs_diff"] for row in rows),
                "max_abs_diff_mean": sum(row["max_abs_diff"] for row in rows) / len(rows),
                "mean_abs_diff_mean": sum(row["mean_abs_diff"] for row in rows) / len(rows),
                "rms_diff_mean": sum(row["rms_diff"] for row in rows) / len(rows),
                "delta_l2_max": max(row["delta_l2"] for row in rows),
                "delta_l2_mean": sum(row["delta_l2"] for row in rows) / len(rows),
                "relative_l2_max": max(row["relative_l2"] for row in rows),
                "relative_l2_mean": sum(row["relative_l2"] for row in rows) / len(rows),
                "rms_diff_over_train_rms_mean": sum(
                    row["rms_diff_over_train_rms"] for row in rows
                ) / len(rows),
                "max_abs_over_train_rms_max": max(
                    row["max_abs_over_train_rms"] for row in rows
                ),
                "max_rel_diff_max": max(row["max_rel_diff"] for row in rows),
                "cosine_min": min(
                    row["cosine"] for row in rows if row.get("cosine") is not None
                )
                if any(row.get("cosine") is not None for row in rows)
                else None,
            }
        )
    numeric_histogram.sort(key=lambda row: (-row["count"], row["module_name"] or ""))
    numeric_all_histogram = _summarize_numeric_by_module(numeric_all_by_module)

    return {
        "probe_dir": str(probe_dir),
        "inference_phase": inference_phase,
        "training_phase": training_phase,
        "inference_records": len(inference),
        "training_records": len(training),
        "inference_dropped": dict(inference_dropped),
        "training_dropped": dict(training_dropped),
        "shared_records": len(shared_keys),
        "matched_records": matched,
        "mismatched_records": len(shared_keys) - matched,
        "tokens_with_mismatches": len(first_mismatches),
        "first_mismatches": first_mismatches,
        "module_histogram": histogram.most_common(),
        "numeric_histogram": numeric_histogram,
        "numeric_all_histogram": numeric_all_histogram,
    }


def _summarize_numeric_by_module(rows_by_module: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    summary = []
    for module_name, rows in rows_by_module.items():
        summary.append(
            {
                "module_name": module_name,
                "count": len(rows),
                "max_abs_diff_max": max(row["max_abs_diff"] for row in rows),
                "max_abs_diff_mean": sum(row["max_abs_diff"] for row in rows) / len(rows),
                "mean_abs_diff_mean": sum(row["mean_abs_diff"] for row in rows) / len(rows),
                "rms_diff_mean": sum(row["rms_diff"] for row in rows) / len(rows),
                "delta_l2_max": max(row["delta_l2"] for row in rows),
                "delta_l2_mean": sum(row["delta_l2"] for row in rows) / len(rows),
                "relative_l2_max": max(row["relative_l2"] for row in rows),
                "relative_l2_mean": sum(row["relative_l2"] for row in rows) / len(rows),
                "rms_diff_over_train_rms_mean": sum(
                    row["rms_diff_over_train_rms"] for row in rows
                ) / len(rows),
                "max_abs_over_train_rms_max": max(
                    row["max_abs_over_train_rms"] for row in rows
                ),
                "max_rel_diff_max": max(row["max_rel_diff"] for row in rows),
                "cosine_min": min(
                    row["cosine"] for row in rows if row.get("cosine") is not None
                )
                if any(row.get("cosine") is not None for row in rows)
                else None,
            }
        )
    summary.sort(
        key=lambda row: (
            -(row["max_abs_diff_max"] if row["max_abs_diff_max"] is not None else 0.0),
            row["module_name"] or "",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir", type=Path, help="Directory containing probe_*.jsonl logs.")
    parser.add_argument("--inference-phase", default="inference")
    parser.add_argument("--training-phase", default="training_old_logprobs")
    parser.add_argument("--include-prompt-targets", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = compare_probe_dir(
        args.probe_dir,
        inference_phase=args.inference_phase,
        training_phase=args.training_phase,
        generated_only=not args.include_prompt_targets,
    )

    print(
        "Compared {shared_records} shared records: {matched_records} matched, "
        "{mismatched_records} mismatched.".format(**summary)
    )
    if summary["shared_records"] == 0:
        print(f"Inference dropped records: {summary['inference_dropped']}")
        print(f"Training dropped records: {summary['training_dropped']}")
    print(f"Tokens with mismatches: {summary['tokens_with_mismatches']}")
    if summary["module_histogram"]:
        print("First divergent modules:")
        for module_name, count in summary["module_histogram"][:20]:
            print(f"  {count:6d}  {module_name}")
    if summary["numeric_histogram"]:
        print("Numeric deltas for first divergent modules:")
        for row in summary["numeric_histogram"][:20]:
            cosine = row["cosine_min"]
            cosine_str = "n/a" if cosine is None else f"{cosine:.8f}"
            print(
                "  {count:6d}  {module_name}  "
                "max_abs<= {max_abs_diff_max:.6g}  "
                "mean_abs(avg)= {mean_abs_diff_mean:.6g}  "
                "rms(avg)= {rms_diff_mean:.6g}  "
                "delta_l2(avg)= {delta_l2_mean:.6g}  "
                "rel_l2(avg)= {relative_l2_mean:.6g}  "
                "rms/train_rms(avg)= {rms_over_rms:.6g}  "
                "cos_min= {cosine}".format(
                    count=row["count"],
                    module_name=row["module_name"],
                    max_abs_diff_max=row["max_abs_diff_max"],
                    mean_abs_diff_mean=row["mean_abs_diff_mean"],
                    rms_diff_mean=row["rms_diff_mean"],
                    delta_l2_mean=row["delta_l2_mean"],
                    relative_l2_mean=row["relative_l2_mean"],
                    rms_over_rms=row["rms_diff_over_train_rms_mean"],
                    cosine=cosine_str,
                )
            )
    if summary["numeric_all_histogram"]:
        print("Largest numeric deltas across all mismatched modules:")
        for row in summary["numeric_all_histogram"][:20]:
            cosine = row["cosine_min"]
            cosine_str = "n/a" if cosine is None else f"{cosine:.8f}"
            print(
                "  {count:6d}  {module_name}  "
                "max_abs<= {max_abs_diff_max:.6g}  "
                "mean_abs(avg)= {mean_abs_diff_mean:.6g}  "
                "rms(avg)= {rms_diff_mean:.6g}  "
                "delta_l2(avg)= {delta_l2_mean:.6g}  "
                "rel_l2(avg)= {relative_l2_mean:.6g}  "
                "rms/train_rms(avg)= {rms_over_rms:.6g}  "
                "cos_min= {cosine}".format(
                    count=row["count"],
                    module_name=row["module_name"],
                    max_abs_diff_max=row["max_abs_diff_max"],
                    mean_abs_diff_mean=row["mean_abs_diff_mean"],
                    rms_diff_mean=row["rms_diff_mean"],
                    delta_l2_mean=row["delta_l2_mean"],
                    relative_l2_mean=row["relative_l2_mean"],
                    rms_over_rms=row["rms_diff_over_train_rms_mean"],
                    cosine=cosine_str,
                )
            )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
