#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Plot rollout reward against selected-token train/inference probability deltas.

The determinism probe writes per-generated-token ``lm_topk_logprobs`` and
``lm_topk_token_ids`` rows for inference and training. This script joins those
rows by token, aggregates to one point per rollout, and plots reward against:

  - max per-token absolute logprob delta for the selected/target token
  - max per-token absolute probability delta for the selected/target token

It also bins rollouts by delta range and plots average reward per bin.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _phase_paths(probe_dir: Path, phase: str) -> list[Path]:
    return sorted(probe_dir.glob(f"probe_{phase}_iter*_rank*.jsonl"))


def _record_allowed(record: dict[str, Any]) -> bool:
    token = record.get("token") or {}
    if token.get("is_padding"):
        return False
    if "is_generated_target" in token:
        return bool(token["is_generated_target"])
    return True


def _token_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
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


def _rollout_key(token: dict[str, Any]) -> tuple[Any, ...]:
    routing_dump_id = token.get("routing_dump_id")
    if routing_dump_id is not None:
        return (token.get("iteration", 0), str(routing_dump_id))
    return (
        token.get("iteration", 0),
        token.get("group_index"),
        token.get("rollout_index", token.get("global_rollout_index")),
        token.get("turn_index", 0),
    )


def _load_topk_records(probe_dir: Path, phase: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    wanted = ('"module_name": "lm_topk_logprobs"', '"module_name": "lm_topk_token_ids"')

    for path in _phase_paths(probe_dir, phase):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if wanted[0] not in line and wanted[1] not in line:
                    continue
                record = json.loads(line)
                if not _record_allowed(record) or "value" not in record:
                    continue
                key = _token_key(record)
                if key is None:
                    continue
                value = (record.get("value") or {}).get("data")
                if not value:
                    continue

                entry = result.setdefault(key, {"token": record.get("token") or {}})
                if record.get("module_name") == "lm_topk_logprobs":
                    entry["logprobs"] = [float(x) for x in value]
                elif record.get("module_name") == "lm_topk_token_ids":
                    entry["token_ids"] = [int(x) for x in value]

    return {
        key: entry
        for key, entry in result.items()
        if "logprobs" in entry and "token_ids" in entry
    }


def _logprob_map(entry: dict[str, Any]) -> dict[int, float]:
    return {
        int(token_id): float(logprob)
        for token_id, logprob in zip(entry.get("token_ids", []), entry.get("logprobs", []))
    }


def _prob_map(entry: dict[str, Any]) -> dict[int, float]:
    return {token_id: math.exp(logprob) for token_id, logprob in _logprob_map(entry).items()}


def _token_deltas(
    inference_entry: dict[str, Any],
    training_entry: dict[str, Any],
) -> tuple[float | None, float | None]:
    token = training_entry.get("token") or inference_entry.get("token") or {}
    target_token_id = token.get("target_token_id")
    if target_token_id is None:
        return None, None
    target_token_id = int(target_token_id)

    inf_logp = _logprob_map(inference_entry)
    train_logp = _logprob_map(training_entry)
    if target_token_id not in inf_logp or target_token_id not in train_logp:
        return None, None

    logprob_delta = abs(train_logp[target_token_id] - inf_logp[target_token_id])

    inf_prob = _prob_map(inference_entry)
    train_prob = _prob_map(training_entry)
    prob_delta = abs(train_prob[target_token_id] - inf_prob[target_token_id])
    return logprob_delta, prob_delta


def _aggregate_rollouts(
    inference: dict[tuple[Any, ...], dict[str, Any]],
    training: dict[tuple[Any, ...], dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter]:
    dropped = Counter()
    rollouts: dict[tuple[Any, ...], dict[str, Any]] = {}

    for key in sorted(set(inference) & set(training)):
        inf_entry = inference[key]
        train_entry = training[key]
        token = train_entry.get("token") or inf_entry.get("token") or {}
        reward = token.get("reward")
        if reward is None:
            dropped["missing_reward"] += 1
            continue

        logprob_delta, prob_delta = _token_deltas(inf_entry, train_entry)
        if logprob_delta is None:
            dropped["selected_token_missing_from_topk"] += 1
            continue

        rollout_key = _rollout_key(token)
        entry = rollouts.setdefault(
            rollout_key,
            {
                "iteration": token.get("iteration", 0),
                "routing_dump_id": token.get("routing_dump_id"),
                "group_index": token.get("group_index"),
                "rollout_index": token.get("rollout_index"),
                "global_rollout_index": token.get("global_rollout_index"),
                "turn_index": token.get("turn_index"),
                "env_id": token.get("env_id"),
                "problem_id": token.get("problem_id"),
                "reward": float(reward),
                "max_logprob_delta": 0.0,
                "max_prob_delta": 0.0,
                "num_compared_tokens": 0,
            },
        )
        entry["num_compared_tokens"] += 1
        entry["max_prob_delta"] = max(float(entry["max_prob_delta"]), float(prob_delta))
        entry["max_logprob_delta"] = max(float(entry["max_logprob_delta"]), float(logprob_delta))

    return list(rollouts.values()), dropped


def _bin_edges(values: list[float], bins: int) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.05, 1e-12)
        low -= pad
        high += pad
    step = (high - low) / bins
    return [low + step * i for i in range(bins + 1)]


def _bin_summary(rows: list[dict[str, Any]], delta_field: str, bins: int) -> list[dict[str, Any]]:
    values = [float(row[delta_field]) for row in rows if row.get(delta_field) is not None]
    edges = _bin_edges(values, bins)
    if not edges:
        return []

    rewards_by_bin: list[list[float]] = [[] for _ in range(bins)]
    for row in rows:
        value = row.get(delta_field)
        if value is None:
            continue
        value = float(value)
        index = bins - 1 if value >= edges[-1] else int((value - edges[0]) / (edges[-1] - edges[0]) * bins)
        index = max(0, min(bins - 1, index))
        rewards_by_bin[index].append(float(row["reward"]))

    summary = []
    for index, rewards in enumerate(rewards_by_bin):
        left = edges[index]
        right = edges[index + 1]
        summary.append(
            {
                "bin_index": index,
                "left": left,
                "right": right,
                "label": f"[{left:.3g}, {right:.3g}{']' if index == bins - 1 else ')'}",
                "count": len(rewards),
                "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            }
        )
    return summary


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _threshold_summary(
    rows: list[dict[str, Any]],
    *,
    delta_field: str,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    summary = []
    for threshold in thresholds:
        below = [
            float(row["reward"])
            for row in rows
            if row.get(delta_field) is not None and float(row[delta_field]) < threshold
        ]
        above = [
            float(row["reward"])
            for row in rows
            if row.get(delta_field) is not None and float(row[delta_field]) >= threshold
        ]
        summary.append(
            {
                "threshold": float(threshold),
                "below_count": len(below),
                "below_mean_reward": _mean(below),
                "above_count": len(above),
                "above_mean_reward": _mean(above),
                "above_minus_below_mean_reward": (
                    _mean(above) - _mean(below)
                    if _mean(above) is not None and _mean(below) is not None
                    else None
                ),
            }
        )
    return summary


def _write_threshold_summary_csv(summary: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "threshold",
        "below_count",
        "below_mean_reward",
        "above_count",
        "above_mean_reward",
        "above_minus_below_mean_reward",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({field: row.get(field) for field in fields})


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required to create plots") from exc
    return plt


def _plot_scatter(rows: list[dict[str, Any]], delta_field: str, ylabel: str, path: Path) -> None:
    plt = _require_matplotlib()
    xs = [float(row[delta_field]) for row in rows if row.get(delta_field) is not None]
    ys = [float(row["reward"]) for row in rows if row.get(delta_field) is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, alpha=0.55, s=18)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("reward")
    ax.set_title(f"Rollout reward vs {ylabel}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_bar(summary: list[dict[str, Any]], ylabel: str, path: Path) -> None:
    plt = _require_matplotlib()
    labels = [item["label"] for item in summary]
    means = [float(item["mean_reward"]) if item["mean_reward"] is not None else 0.0 for item in summary]
    counts = [int(item["count"]) for item in summary]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, means)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("average reward")
    ax.set_title(f"Average reward by {ylabel} range")
    ax.tick_params(axis="x", labelrotation=35)
    for idx, (mean, count) in enumerate(zip(means, counts)):
        ax.text(idx, mean, f"n={count}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_threshold_bar(summary: list[dict[str, Any]], path: Path) -> None:
    plt = _require_matplotlib()
    labels = [f"{item['threshold']:.3g}" for item in summary]
    below_means = [
        float(item["below_mean_reward"]) if item["below_mean_reward"] is not None else 0.0
        for item in summary
    ]
    above_means = [
        float(item["above_mean_reward"]) if item["above_mean_reward"] is not None else 0.0
        for item in summary
    ]
    below_counts = [int(item["below_count"]) for item in summary]
    above_counts = [int(item["above_count"]) for item in summary]
    x_positions = list(range(len(summary)))
    width = 0.42

    fig, ax = plt.subplots(figsize=(11, 5))
    below_bars = ax.bar(
        [x - width / 2 for x in x_positions],
        below_means,
        width,
        label="below threshold",
    )
    above_bars = ax.bar(
        [x + width / 2 for x in x_positions],
        above_means,
        width,
        label="at/above threshold",
    )
    ax.set_xlabel("max selected-token probability delta threshold")
    ax.set_ylabel("average reward")
    ax.set_title("Average reward below vs at/above probability-delta thresholds")
    ax.set_xticks(x_positions, labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    for bars, counts in ((below_bars, below_counts), (above_bars, above_counts)):
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "iteration",
        "routing_dump_id",
        "group_index",
        "rollout_index",
        "global_rollout_index",
        "turn_index",
        "env_id",
        "problem_id",
        "reward",
        "max_logprob_delta",
        "max_prob_delta",
        "num_compared_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def analyze_probe_dir(
    probe_dir: Path,
    output_dir: Path,
    *,
    inference_phase: str,
    training_phase: str,
    bins: int,
    prob_thresholds: list[float],
) -> dict[str, Any]:
    inference = _load_topk_records(probe_dir, inference_phase)
    training = _load_topk_records(probe_dir, training_phase)
    rows, dropped = _aggregate_rollouts(inference, training)
    rows.sort(key=lambda row: (row.get("iteration", 0), str(row.get("routing_dump_id"))))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "rollout_reward_deltas.csv")

    logprob_bins = _bin_summary(rows, "max_logprob_delta", bins)
    prob_bins = _bin_summary(rows, "max_prob_delta", bins)
    prob_threshold_summary = _threshold_summary(
        rows,
        delta_field="max_prob_delta",
        thresholds=prob_thresholds,
    )
    _write_threshold_summary_csv(
        prob_threshold_summary,
        output_dir / "avg_reward_by_prob_delta_threshold.csv",
    )
    _plot_threshold_bar(
        prob_threshold_summary,
        output_dir / "avg_reward_by_prob_delta_threshold.png",
    )
    if any(row.get("max_logprob_delta") is not None for row in rows):
        _plot_scatter(
            rows,
            "max_logprob_delta",
            "max token |train - inference| selected-token logprob delta",
            output_dir / "reward_vs_max_logprob_delta.png",
        )
        _plot_bar(
            logprob_bins,
            "max token |train - inference| selected-token logprob delta",
            output_dir / "avg_reward_by_logprob_delta_bin.png",
        )
    _plot_scatter(
        rows,
        "max_prob_delta",
        "max token |train - inference| selected-token probability delta",
        output_dir / "reward_vs_max_prob_delta.png",
    )
    _plot_bar(
        prob_bins,
        "max token |train - inference| selected-token probability delta",
        output_dir / "avg_reward_by_prob_delta_bin.png",
    )

    report = {
        "probe_dir": str(probe_dir),
        "inference_phase": inference_phase,
        "training_phase": training_phase,
        "num_inference_tokens": len(inference),
        "num_training_tokens": len(training),
        "num_shared_token_keys": len(set(inference) & set(training)),
        "num_rollouts": len(rows),
        "dropped": dict(dropped),
        "logprob_bins": logprob_bins,
        "prob_bins": prob_bins,
        "prob_threshold_summary": prob_threshold_summary,
        "outputs": {
            "rollout_csv": str(output_dir / "rollout_reward_deltas.csv"),
            "prob_threshold_csv": str(output_dir / "avg_reward_by_prob_delta_threshold.csv"),
            "prob_threshold_bar": str(output_dir / "avg_reward_by_prob_delta_threshold.png"),
            "logprob_scatter": str(output_dir / "reward_vs_max_logprob_delta.png"),
            "logprob_bar": str(output_dir / "avg_reward_by_logprob_delta_bin.png"),
            "prob_scatter": str(output_dir / "reward_vs_max_prob_delta.png"),
            "prob_bar": str(output_dir / "avg_reward_by_prob_delta_bin.png"),
        },
    }
    with (output_dir / "reward_delta_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir", type=Path, help="Path to determinism_probe directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and summaries. Defaults to <probe_dir>/reward_delta_plots.",
    )
    parser.add_argument("--inference-phase", default="inference")
    parser.add_argument("--training-phase", default="training_old_logprobs")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--prob-thresholds",
        type=float,
        nargs="+",
        default=[0.1 * i for i in range(1, 10)],
        help="Probability-delta thresholds for above/below average reward summaries.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.probe_dir / "reward_delta_plots")
    report = analyze_probe_dir(
        args.probe_dir,
        output_dir,
        inference_phase=args.inference_phase,
        training_phase=args.training_phase,
        bins=args.bins,
        prob_thresholds=args.prob_thresholds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
