# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Activation hashing for RL inference/training determinism debugging."""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist


_PROBE_ATTR = "_rl_determinism_probe"
_ACTIVE_SCOPE = contextvars.ContextVar("rl_determinism_probe_scope", default=None)


@dataclass
class ProbeConfig:
    output_dir: str
    phases: set[str]
    module_filter: str | None = None
    targets_file: str | None = None
    target_keys: set[tuple[int, str, int]] | None = None
    target_content_keys: set[tuple[int, str]] | None = None
    max_rollouts: int = 0
    max_generated_tokens: int = 0
    hash_inputs: bool = True
    store_values: bool = False
    value_max_elements: int = 0
    topk: int = 0

    @classmethod
    def from_args(cls, args: Any) -> "ProbeConfig | None":
        output_dir = getattr(args, "rl_determinism_probe_dir", None)
        if not output_dir:
            return None

        phases_arg = getattr(
            args, "rl_determinism_probe_phases", "inference,training_old_logprobs"
        )
        if isinstance(phases_arg, str):
            phases = {phase.strip() for phase in phases_arg.split(",") if phase.strip()}
        else:
            phases = set(phases_arg or [])
        if not phases:
            phases = {"inference", "training_old_logprobs"}

        targets_file = getattr(args, "rl_determinism_probe_targets_file", None) or None
        return cls(
            output_dir=output_dir,
            phases=phases,
            module_filter=getattr(args, "rl_determinism_probe_module_filter", None) or None,
            targets_file=targets_file,
            target_keys=_load_target_keys(targets_file),
            target_content_keys=_load_target_content_keys(targets_file),
            max_rollouts=int(getattr(args, "rl_determinism_probe_max_rollouts", 0) or 0),
            max_generated_tokens=int(
                getattr(args, "rl_determinism_probe_max_generated_tokens", 0) or 0
            ),
            hash_inputs=bool(getattr(args, "rl_determinism_probe_hash_inputs", True)),
            store_values=bool(getattr(args, "rl_determinism_probe_store_values", False)),
            value_max_elements=int(
                getattr(args, "rl_determinism_probe_value_max_elements", 0) or 0
            ),
            topk=int(getattr(args, "rl_logprob_mismatch_top_k", 0) or 0),
        )


@dataclass
class ProbeScope:
    phase: str
    iteration: int
    token_metadata: list[dict[str, Any]]
    batch_size: int | None = None
    seq_length: int | None = None
    extra: dict[str, Any] | None = None
    probe: "DeterminismProbe | None" = None


class DeterminismProbe:
    """Register module hooks and write activation summaries for active probe scopes."""

    def __init__(self, model: torch.nn.Module, config: ProbeConfig):
        self.model = _unwrap_model(model)
        self.config = config
        self.rank = _rank()
        self._module_re = re.compile(config.module_filter) if config.module_filter else None
        self._handles = []
        self._call_counts: dict[str, int] = {}
        self._writers: dict[tuple[str, int], Any] = {}
        self._register_hooks()

    def update_config(self, config: ProbeConfig) -> None:
        self.config = config
        self._module_re = re.compile(config.module_filter) if config.module_filter else None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if not name:
                continue
            if any(True for _ in module.children()):
                continue
            if self._module_re is not None and self._module_re.search(name) is None:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, module_name: str):
        def _hook(module, inputs, output):
            scope = _ACTIVE_SCOPE.get()
            if scope is None or scope.phase not in self.config.phases:
                return
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return

            call_index = self._call_counts.get(module_name, 0)
            self._call_counts[module_name] = call_index + 1

            if self.config.hash_inputs:
                self._write_tensors(
                    scope=scope,
                    module=module,
                    module_name=module_name,
                    call_index=call_index,
                    io_kind="input",
                    value=inputs,
                )
            self._write_tensors(
                scope=scope,
                module=module,
                module_name=module_name,
                call_index=call_index,
                io_kind="output",
                value=output,
            )

        return _hook

    def record_tensor_point(
        self,
        name: str,
        value: Any,
        *,
        tensor_path: str = "tensor",
        token_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        scope = _ACTIVE_SCOPE.get()
        if scope is None or scope.phase not in self.config.phases:
            return
        if self._module_re is not None and self._module_re.search(name) is None:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return

        if token_metadata is not None:
            scope = ProbeScope(
                phase=scope.phase,
                iteration=scope.iteration,
                token_metadata=token_metadata,
                batch_size=None,
                seq_length=None,
                extra=scope.extra,
                probe=scope.probe,
            )

        self._write_tensors(
            scope=scope,
            module=None,
            module_name=name,
            call_index=0,
            io_kind="manual",
            value={tensor_path: value},
        )

    def _write_tensors(
        self,
        *,
        scope: ProbeScope,
        module: torch.nn.Module | None,
        module_name: str,
        call_index: int,
        io_kind: str,
        value: Any,
    ) -> None:
        for tensor_path, tensor in _iter_tensors(value):
            base = {
                "phase": scope.phase,
                "iteration": scope.iteration,
                "rank": self.rank,
                "module_name": module_name,
                "module_class": module.__class__.__name__ if module is not None else "ManualProbePoint",
                "module_call_index": call_index,
                "io_kind": io_kind,
                "tensor_path": tensor_path,
                "tensor_stats": _tensor_stats(tensor),
                "scope": scope.extra or {},
                "time": time.time(),
            }

            rows = _token_rows(tensor, scope)
            if rows is None:
                base["token"] = None
                base["hash"] = None
                self._write(scope.phase, scope.iteration, base)
                continue

            token_metadata, row_tensor = rows
            for meta, row in zip(token_metadata, row_tensor):
                if not self._allow_token(meta):
                    continue
                record = dict(base)
                record["token"] = meta
                record["hash"] = _tensor_hash(row)
                if self.config.store_values:
                    record["value"] = _tensor_values(row, self.config.value_max_elements)
                self._write(scope.phase, scope.iteration, record)

    def _allow_token(self, meta: dict[str, Any]) -> bool:
        if meta.get("is_padding"):
            return False
        if "is_generated_target" in meta and not meta["is_generated_target"]:
            return False
        if self.config.target_content_keys is not None:
            prefix_hash = meta.get("prefix_hash")
            if prefix_hash is not None:
                iteration = int(meta.get("iteration", _ACTIVE_SCOPE.get().iteration))
                return (iteration, str(prefix_hash)) in self.config.target_content_keys
        if self.config.target_keys is not None:
            routing_dump_id = meta.get("routing_dump_id")
            target_token_index = meta.get("target_token_index")
            if routing_dump_id is None or target_token_index is None:
                return False
            iteration = int(meta.get("iteration", _ACTIVE_SCOPE.get().iteration))
            return (iteration, str(routing_dump_id), int(target_token_index)) in self.config.target_keys
        if self.config.max_rollouts > 0:
            rollout_idx = meta.get("global_rollout_index", meta.get("request_id"))
            if rollout_idx is not None and int(rollout_idx) >= self.config.max_rollouts:
                return False
        if self.config.max_generated_tokens > 0:
            gen_offset = meta.get("gen_offset")
            if gen_offset is not None and int(gen_offset) >= self.config.max_generated_tokens:
                return False
        return True

    def _write(self, phase: str, iteration: int, record: dict[str, Any]) -> None:
        writer = self._writer(phase, iteration)
        writer.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")

    def _writer(self, phase: str, iteration: int):
        key = (phase, iteration)
        writer = self._writers.get(key)
        if writer is not None:
            return writer

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"probe_{phase}_iter{iteration:06d}_rank{self.rank:04d}.jsonl"
        writer = open(path, "a", encoding="utf-8", buffering=1)
        self._writers[key] = writer
        return writer


def probe_enabled(args: Any) -> bool:
    return bool(getattr(args, "rl_determinism_probe_dir", None))


def probe_target_discovery_enabled(args: Any) -> bool:
    return bool(getattr(args, "rl_determinism_probe_write_targets_file", None))


def probe_needs_rollout_ids(args: Any) -> bool:
    return probe_enabled(args) or probe_target_discovery_enabled(args)


def ensure_determinism_probe(model: torch.nn.Module, args: Any) -> DeterminismProbe | None:
    config = ProbeConfig.from_args(args)
    if config is None:
        return None

    root = _unwrap_model(model)
    probe = getattr(root, _PROBE_ATTR, None)
    if probe is None:
        probe = DeterminismProbe(root, config)
        setattr(root, _PROBE_ATTR, probe)
    else:
        probe.update_config(config)
    return probe


def get_determinism_probe(model: torch.nn.Module) -> DeterminismProbe | None:
    return getattr(_unwrap_model(model), _PROBE_ATTR, None)


@contextmanager
def probe_scope(
    *,
    phase: str,
    iteration: int,
    token_metadata: list[dict[str, Any]] | None,
    batch_size: int | None = None,
    seq_length: int | None = None,
    extra: dict[str, Any] | None = None,
    probe: DeterminismProbe | None = None,
):
    if not token_metadata:
        with nullcontext():
            yield
        return
    token = _ACTIVE_SCOPE.set(
        ProbeScope(
            phase=phase,
            iteration=int(iteration),
            token_metadata=token_metadata,
            batch_size=batch_size,
            seq_length=seq_length,
            extra=extra,
            probe=probe,
        )
    )
    try:
        yield
    finally:
        _ACTIVE_SCOPE.reset(token)


def probe_tensor_point(
    name: str,
    value: Any,
    *,
    tensor_path: str = "tensor",
    token_metadata: list[dict[str, Any]] | None = None,
) -> None:
    """Record a manually named tensor boundary in the active probe scope."""
    scope = _ACTIVE_SCOPE.get()
    if scope is None or scope.probe is None:
        return
    scope.probe.record_tensor_point(name, value, tensor_path=tensor_path, token_metadata=token_metadata)


def active_probe_topk() -> int:
    """Return the configured top-k for the active probe scope, or 0 if not probing.

    Used to gate (and size) per-token top-k logprob probe points so the expensive
    full-vocab top-k is only computed while a determinism-probe scope is active.
    """
    scope = _ACTIVE_SCOPE.get()
    if scope is None or scope.probe is None:
        return 0
    return int(getattr(scope.probe.config, "topk", 0) or 0)


def active_generated_token_metadata() -> list[dict[str, Any]]:
    """Return generated-token metadata for the active probe scope."""
    scope = _ACTIVE_SCOPE.get()
    if scope is None:
        return []
    return [
        meta for meta in scope.token_metadata
        if not meta.get("is_padding") and meta.get("is_generated_target", True)
    ]


def active_nonpadding_token_metadata() -> list[dict[str, Any]]:
    """Return non-padding token metadata for the active probe scope."""
    scope = _ACTIVE_SCOPE.get()
    if scope is None:
        return []
    return [meta for meta in scope.token_metadata if not meta.get("is_padding")]


def active_token_metadata_for_first_dim(
    tensor: torch.Tensor,
    *,
    tp_group: Any | None = None,
) -> list[dict[str, Any]] | None:
    """Return active token metadata aligned to a tensor sharded on dim 0.

    Final hidden states are recorded before the output layer. With sequence parallelism,
    the first dimension is sharded across TP ranks, while the probe scope metadata covers
    the full padded active-token sequence. Slice the metadata to the local TP shard so
    manual probe points can still emit per-token rows and stored values.
    """
    scope = _ACTIVE_SCOPE.get()
    if scope is None or tensor.ndim == 0:
        return None

    token_metadata = scope.token_metadata
    local_rows = int(tensor.shape[0])
    if len(token_metadata) == local_rows:
        return token_metadata

    if tp_group is None:
        return None
    try:
        world_size = tp_group.size()
        rank = tp_group.rank()
    except Exception:
        return None

    if world_size <= 1 or len(token_metadata) % world_size != 0:
        return None
    rows_per_rank = len(token_metadata) // world_size
    if rows_per_rank != local_rows:
        return None
    start = rank * rows_per_rank
    return token_metadata[start : start + rows_per_rank]


def build_inference_selected_logprob_metadata(
    *,
    context: Any,
    new_tokens: torch.Tensor,
    collection_id: str,
    rank: int,
    only_last_token_logits: bool,
) -> list[dict[str, Any]]:
    """Build per-row metadata aligned with ``DynamicContext.calculate_log_probs``.

    ``calculate_log_probs`` indexes selected logprobs in request order for decode
    steps (``only_last_token_logits``) or packed row order for mixed prefill/decode.
    The older ``active_generated_token_metadata()`` ordering does not match that
    indexing, which mislabels probe rows even when tensor/value counts agree.
    """
    iteration = int(collection_id) if str(collection_id).isdigit() else 0
    active_slice = slice(context.paused_request_count, context.total_request_count)
    request_ids = context.request_ids[active_slice].detach().cpu().tolist()

    if only_last_token_logits or context.is_decode_only():
        query_lengths = context.request_query_lengths[active_slice].detach().cpu().tolist()
        cumulative = 0
        last_row_indices: list[int] = []
        for query_length in query_lengths:
            cumulative += int(query_length)
            last_row_indices.append(cumulative - 1)

        metadata: list[dict[str, Any]] = []
        for local_idx, request_id in enumerate(request_ids):
            row_index = last_row_indices[local_idx]
            input_token_index = int(
                context.token_to_position_in_request[row_index].detach().cpu().item()
            )
            target_token_id = int(new_tokens[local_idx].detach().cpu().item())
            metadata.append(
                _inference_token_meta(
                    iteration=iteration,
                    collection_id=collection_id,
                    rank=rank,
                    request_id=int(request_id),
                    row_index=row_index,
                    input_token_index=input_token_index,
                    target_token_id=target_token_id,
                    is_generated_target=True,
                    is_padding=False,
                )
            )
        return metadata

    active_token_count = int(context.active_token_count)
    token_to_input_ids = context.token_to_input_ids[:active_token_count].clone()
    token_to_position = context.token_to_position_in_request[:active_token_count]
    token_to_request_idx = context.token_to_request_idx[:active_token_count]
    request_query_lengths = context.request_query_lengths[active_slice].detach().cpu().tolist()
    request_in_prefill = (
        context.request_in_prefill_status_tensor[active_slice].detach().cpu().tolist()
    )

    active_token_ids = token_to_input_ids.roll(-1, 0)
    new_token_idx = context.request_query_lengths[active_slice].cumsum(0) - 1
    active_token_ids[new_token_idx] = new_tokens.to(active_token_ids.device)

    metadata = []
    rows_seen_by_request: dict[int, int] = {}
    for row_index in range(active_token_count):
        request_idx = int(token_to_request_idx[row_index].detach().cpu().item())
        local_request_offset = request_idx - context.paused_request_count
        request_id = int(request_ids[local_request_offset])
        input_token_index = int(token_to_position[row_index].detach().cpu().item())
        target_token_id = int(active_token_ids[row_index].detach().cpu().item())
        row_offset = rows_seen_by_request.get(request_id, 0)
        rows_seen_by_request[request_id] = row_offset + 1
        is_prefill = bool(request_in_prefill[local_request_offset])
        is_generated_target = (not is_prefill) or (
            row_offset == int(request_query_lengths[local_request_offset]) - 1
        )
        metadata.append(
            _inference_token_meta(
                iteration=iteration,
                collection_id=collection_id,
                rank=rank,
                request_id=request_id,
                row_index=row_index,
                input_token_index=input_token_index,
                target_token_id=target_token_id,
                is_generated_target=is_generated_target,
                is_padding=not is_generated_target,
            )
        )
    return metadata


def build_inference_token_metadata(
    *,
    context: Any,
    collection_id: str,
    rank: int,
) -> list[dict[str, Any]]:
    active_slice = slice(context.paused_request_count, context.total_request_count)
    request_ids = context.request_ids[active_slice].detach().cpu().tolist()
    request_query_lengths = context.request_query_lengths[active_slice].detach().cpu().tolist()
    request_in_prefill = (
        context.request_in_prefill_status_tensor[active_slice].detach().cpu().tolist()
    )
    token_to_request_idx = context.token_to_request_idx[: context.active_token_count]
    token_to_position = context.token_to_position_in_request[: context.active_token_count]
    token_to_input_ids = context.token_to_input_ids[: context.active_token_count]
    padded_token_count = int(getattr(context, "padded_active_token_count", context.active_token_count))

    metadata = []
    rows_seen_by_request: dict[int, int] = {}
    for row_index, request_idx in enumerate(token_to_request_idx.detach().cpu().tolist()):
        if request_idx < context.paused_request_count or request_idx >= context.total_request_count:
            continue
        local_request_offset = request_idx - context.paused_request_count
        request_id = int(request_ids[local_request_offset])
        input_token_index = int(token_to_position[row_index].detach().cpu().item())
        row_offset = rows_seen_by_request.get(request_id, 0)
        rows_seen_by_request[request_id] = row_offset + 1
        is_prefill = bool(request_in_prefill[local_request_offset])
        # During prefill, only the final row can predict a generated token. Decode
        # rows all predict generated tokens.
        is_generated_target = (not is_prefill) or (
            row_offset == int(request_query_lengths[local_request_offset]) - 1
        )
        metadata.append(
            _inference_token_meta(
                iteration=int(collection_id) if str(collection_id).isdigit() else 0,
                collection_id=collection_id,
                rank=rank,
                request_id=request_id,
                row_index=row_index,
                input_token_index=input_token_index,
                input_token_id=int(token_to_input_ids[row_index].detach().cpu().item()),
                is_generated_target=is_generated_target,
                is_padding=not is_generated_target,
            )
        )
    for row_index in range(len(metadata), padded_token_count):
        metadata.append(
            {
                "phase": "inference",
                "row_index": row_index,
                "is_padding": True,
            }
        )
    return metadata


def build_training_token_metadata(
    *,
    tokens: torch.Tensor,
    generation_masks: torch.Tensor | None,
    seq_indices: torch.Tensor | Iterable[int] | None,
    turn_metadata: list[dict[str, Any]] | None,
    iteration: int,
    phase: str,
) -> list[dict[str, Any]]:
    if tokens.ndim != 2:
        return []

    batch_size, seq_length = tokens.shape
    if seq_indices is None:
        seq_indices_list = list(range(batch_size))
    elif isinstance(seq_indices, torch.Tensor):
        seq_indices_list = [int(x) for x in seq_indices.detach().cpu().flatten().tolist()]
    else:
        seq_indices_list = [int(x) for x in seq_indices]

    token_ids = tokens.detach().cpu()
    token_ids_list = token_ids.tolist()
    masks = generation_masks.detach().cpu() if generation_masks is not None else None

    # Precompute, per batch row, the generation mask aligned to its seq_index and
    # the cumulative count of generated tokens (so gen_offset is O(1) per token
    # instead of an O(seq) nonzero scan).
    mask_rows: list[list[int] | None] = []
    cumsum_rows: list[list[int] | None] = []
    for seq_index in seq_indices_list:
        if masks is not None and 0 <= seq_index < masks.shape[0]:
            m = masks[seq_index].to(torch.long)
            mask_rows.append(m.tolist())
            cumsum_rows.append(torch.cumsum(m, dim=0).tolist())
        else:
            mask_rows.append(None)
            cumsum_rows.append(None)

    turn_records = [
        {
            k: _json_default(v)
            for k, v in (
                turn_metadata[seq_index]
                if turn_metadata is not None and 0 <= seq_index < len(turn_metadata)
                else {}
            ).items()
            if k != "inference_top_logprobs"
        }
        for seq_index in seq_indices_list
    ]

    # Incremental prefix hash per batch row: rolling blake2b updated one token at a
    # time, snapshotted (copy + hexdigest) only at generated target positions. This
    # makes the whole pass O(seq * batch) instead of O(seq**2 * batch).
    running = [hashlib.blake2b(digest_size=16) for _ in seq_indices_list]

    metadata = []
    # Megatron transformer internals are sequence-major for most activation tensors.
    for token_index in range(seq_length):
        for batch_row, seq_index in enumerate(seq_indices_list):
            row_tokens = token_ids_list[batch_row]
            # Advance the rolling hash with the input token at this position; the
            # digest now covers tokens[:token_index + 1] == prefix of the target.
            running[batch_row].update(
                int(row_tokens[token_index]).to_bytes(8, byteorder="little", signed=True)
            )

            target_token_index = token_index + 1
            is_generated_target = False
            gen_offset = None
            target_token_id = None
            prefix_hash = None
            if target_token_index < seq_length:
                target_token_id = int(row_tokens[target_token_index])
                mrow = mask_rows[batch_row]
                if mrow is not None:
                    is_generated_target = bool(mrow[target_token_index])
                    if is_generated_target:
                        gen_offset = int(cumsum_rows[batch_row][target_token_index]) - 1
                        prefix_hash = running[batch_row].copy().hexdigest()

            row = {
                "phase": phase,
                "iteration": int(iteration),
                "row_index": len(metadata),
                "batch_row": int(batch_row),
                "seq_index": int(seq_index),
                "input_token_index": int(token_index),
                "target_token_index": int(target_token_index),
                "input_token_id": int(row_tokens[token_index]),
                "target_token_id": target_token_id,
                "prefix_hash": prefix_hash,
                "is_generated_target": is_generated_target,
                "gen_offset": gen_offset,
            }
            row.update(turn_records[batch_row])
            metadata.append(row)
    return metadata


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _load_target_keys(path: str | None) -> set[tuple[int, str, int]] | None:
    if not path:
        return None
    target_paths = _target_paths(path)
    if not target_paths:
        return set()
    keys = set()
    for target_path in target_paths:
        with open(target_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        targets = payload.get("targets", payload) if isinstance(payload, dict) else payload
        for item in targets or []:
            routing_dump_id = item.get("routing_dump_id")
            target_token_index = item.get("target_token_index")
            if routing_dump_id is None or target_token_index is None:
                continue
            keys.add(
                (
                    int(item.get("iteration", 0)),
                    str(routing_dump_id),
                    int(target_token_index),
                )
            )
    return keys


def _load_target_content_keys(path: str | None) -> set[tuple[int, str]] | None:
    if not path:
        return None
    target_paths = _target_paths(path)
    if not target_paths:
        return set()
    keys = set()
    for target_path in target_paths:
        with open(target_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        targets = payload.get("targets", payload) if isinstance(payload, dict) else payload
        for item in targets or []:
            prefix_hash = item.get("prefix_hash")
            if prefix_hash is None:
                continue
            keys.add((int(item.get("iteration", 0)), str(prefix_hash)))
    return keys


def _target_paths(path: str) -> list[Path]:
    target_path = Path(path)
    paths = []
    if target_path.exists():
        paths.append(target_path)
    paths.extend(sorted(target_path.parent.glob(f"{target_path.stem}.rank*{target_path.suffix}")))
    return paths


def hash_token_ids(token_ids: Iterable[int]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for token_id in token_ids:
        h.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


_INFERENCE_REQUEST_REGISTRY: dict[int, dict[str, Any]] = {}


def register_inference_request(
    *, request_id: int, routing_dump_id: str | None, prompt_tokens: Iterable[int]
) -> None:
    prompt = [int(x) for x in prompt_tokens]
    _INFERENCE_REQUEST_REGISTRY[int(request_id)] = {
        "routing_dump_id": routing_dump_id,
        "prompt_tokens": prompt,
        "generated_tokens": [],
        "prompt_hash": hash_token_ids(prompt),
    }


def update_inference_request_generated_tokens(
    *, request_id: int, generated_tokens: Iterable[int]
) -> None:
    entry = _INFERENCE_REQUEST_REGISTRY.get(int(request_id))
    if entry is None:
        return
    entry["generated_tokens"] = [int(x) for x in generated_tokens]


def _inference_token_meta(
    *,
    iteration: int,
    collection_id: str,
    rank: int,
    request_id: int,
    row_index: int,
    input_token_index: int,
    is_generated_target: bool,
    is_padding: bool,
    input_token_id: int | None = None,
    target_token_id: int | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "phase": "inference",
        "iteration": iteration,
        "row_index": row_index,
        "request_id": request_id,
        "routing_dump_id": f"{collection_id}_{rank:04d}_{request_id:08d}",
        "input_token_index": input_token_index,
        "target_token_index": input_token_index + 1,
        "prefix_hash": _inference_prefix_hash(request_id, input_token_index + 1),
        "is_generated_target": is_generated_target,
        "is_padding": is_padding,
    }
    if input_token_id is not None:
        meta["input_token_id"] = input_token_id
    if target_token_id is not None:
        meta["target_token_id"] = target_token_id
    return meta


def _inference_prefix_hash(request_id: int, target_token_index: int) -> str | None:
    entry = _INFERENCE_REQUEST_REGISTRY.get(int(request_id))
    if entry is None:
        return None
    tokens = entry["prompt_tokens"] + entry["generated_tokens"]
    if target_token_index > len(tokens):
        return None
    return hash_token_ids(tokens[:target_token_index])


def _iter_tensors(value: Any, prefix: str = ""):
    if torch.is_tensor(value):
        yield prefix or "tensor", value
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}.{idx}" if prefix else str(idx))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}.{key}" if prefix else str(key))


def _token_rows(tensor: torch.Tensor, scope: ProbeScope):
    token_count = len(scope.token_metadata)
    if tensor.ndim == 0 or token_count == 0:
        return None

    detached = tensor.detach()
    if detached.shape[0] == token_count:
        return scope.token_metadata, detached.reshape(token_count, -1)

    if (
        scope.batch_size is not None
        and scope.seq_length is not None
        and detached.ndim >= 2
        and detached.shape[0] == scope.seq_length
        and detached.shape[1] == scope.batch_size
        and scope.seq_length * scope.batch_size == token_count
    ):
        return scope.token_metadata, detached.reshape(token_count, -1)

    if (
        scope.batch_size is not None
        and scope.seq_length is not None
        and detached.ndim >= 2
        and detached.shape[0] == scope.batch_size
        and detached.shape[1] == scope.seq_length
        and scope.seq_length * scope.batch_size == token_count
    ):
        permute_order = [1, 0] + list(range(2, detached.ndim))
        return scope.token_metadata, detached.permute(*permute_order).reshape(token_count, -1)

    return None


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    stats = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        return stats

    values = tensor.detach()
    if not values.is_floating_point() and not values.is_complex():
        values = values.to(torch.float32)
    elif values.is_complex():
        values = values.abs().to(torch.float32)
    else:
        values = values.to(torch.float32)

    values = values.flatten()
    finite = torch.isfinite(values)
    stats["finite_count"] = int(finite.sum().item())
    if stats["finite_count"] == 0:
        return stats
    finite_values = values[finite]
    stats.update(
        {
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
        }
    )
    return stats


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    try:
        h.update(cpu.view(torch.uint8).numpy().tobytes())
    except (TypeError, RuntimeError):
        h.update(cpu.to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def _tensor_values(tensor: torch.Tensor, max_elements: int = 0) -> dict[str, Any]:
    values = tensor.detach().flatten()
    original_numel = int(values.numel())
    truncated = False
    if max_elements > 0 and original_numel > max_elements:
        values = values[:max_elements]
        truncated = True
    return {
        "dtype": "float32",
        "shape": [int(values.numel())],
        "original_numel": original_numel,
        "truncated": truncated,
        "data": values.to(torch.float32).cpu().tolist(),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    return str(value)
