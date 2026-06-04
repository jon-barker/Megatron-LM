# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc

import asyncio
import copy
from functools import partial
# Keep this to make the env registered.
import itertools
import math
import logging
import json
import os
import time
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional 

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import is_pp_last_stage, get_pp_last_rank
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import HuggingFaceTokenizer
from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.utils import (
    toggle_cuda_graphs,
    transition_moe_cudagraphs,
)
from megatron.core.inference.utils import set_decode_expert_padding
from megatron.core.resharding.refit import swap_model_weights
from megatron.core.inference.unified_memory import (
    advise_managed_module_parameters_preferred_location,
    prefetch_managed_module_parameters,
)
from megatron.core.inference.utils import device_memory_summary
from megatron.core.utils import get_asyncio_loop, log_single_rank
from megatron.rl.sequence_packing_utils import (
    get_microbatch_dataloader,
    pack_inference_logprobs,
    compute_packed_inference_logprobs_stats,
    pack_all_trajectories,
    load_packed_data_by_index,
    get_sequence_packing_tensorboard_metrics,
    get_sequence_packing_log_info,
    get_default_packed_seq_params,
    update_microbatch_calculator,
)
from megatron.rl.agent.api import (
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutRequest,
    GroupedRollouts,
    RewardEvaluationResult,
    Rollout,
    RolloutGroup,
    Rollouts,
    TokenRollout,
)
from megatron.rl.agent.weighted_multi_task import WeightedMultiTask
from megatron.rl.inference.megatron import MegatronLocal
from megatron.rl.logging import LOG_DIR as lang_rl_log_dir
from megatron.rl.logging import log as lang_rl_log
from megatron.rl.server.inference.inference_interface_server import InferenceInterfaceServer
from megatron.training.global_vars import (
    get_args,
    get_tensorboard_writer,
    get_tokenizer,
    get_wandb_writer,
)
from megatron.training.utils import (
    get_ltor_masks_and_position_ids,
    get_nvtx_range,
    print_rank_0,
    unwrap_model,
)
from megatron.core.utils import get_pg_rank, get_pg_size, get_attr_wrapped_model
from megatron.core.process_groups_config import ProcessGroupCollection
from wandb import wandb_run
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    is_batch_invariant_mode_enabled,
)
from megatron.rl.determinism_probe import (
    build_training_token_metadata,
    ensure_determinism_probe,
    get_determinism_probe,
    hash_token_ids,
    probe_enabled,
    probe_needs_rollout_ids,
    probe_scope,
    probe_tensor_point,
)

from megatron.core.inference.contexts.dynamic_context import HAVE_TORCH_MEMORY_SAVER
if HAVE_TORCH_MEMORY_SAVER:
    from torch_memory_saver import torch_memory_saver

logger = logging.getLogger(__name__)

# Global variable to store packing context for forward_step
_GLOBAL_PACKING_CONTEXT = None


# Track whether the inference model is currently paused (offloaded to CPU).
# Model starts on GPU after creation and is used immediately, so starts as False.
_INFERENCE_MODEL_IS_PAUSED = False

# Side channel for optional diagnostic top-k logprobs.  The Megatron pipeline
# schedules expect forward output to be tensor-like, so get_logprobs() must not
# return a tuple when diagnostics are enabled.
_LOGPROBS_TOPK_BUFFER = []


def _torch_saver_swap_inference_model(*, to_cpu: bool) -> None:
    """Swap RL inference model weights between CPU and GPU using torch_memory_saver.

    Uses torch_memory_saver.pause()/resume() to transfer inference model weights
    that were allocated within a torch_memory_saver.region() context.

    Args:
        to_cpu: If True, move weights to CPU (pause). If False, restore weights to GPU (resume).
    """
    global _INFERENCE_MODEL_IS_PAUSED

    if not HAVE_TORCH_MEMORY_SAVER:
        raise RuntimeError(
            "torch_memory_saver is required for inference model offloading when not using UVM. "
            "Please install it: pip install torch_memory_saver "
            "(see https://github.com/fzyzcjy/torch_memory_saver)"
        )

    tag = "rl_inference_model"
    if to_cpu:
        if not _INFERENCE_MODEL_IS_PAUSED:
            print_rank_0(f"torch_memory_saver: pausing {tag}, before: {device_memory_summary()}")
            torch_memory_saver.pause(tag)
            _INFERENCE_MODEL_IS_PAUSED = True
            print_rank_0(f"torch_memory_saver: paused  {tag}, after:  {device_memory_summary()}")
    else:
        if _INFERENCE_MODEL_IS_PAUSED:
            print_rank_0(f"torch_memory_saver: resuming {tag}, before: {device_memory_summary()}")
            torch_memory_saver.resume(tag)
            _INFERENCE_MODEL_IS_PAUSED = False
            print_rank_0(f"torch_memory_saver: resumed  {tag}, after:  {device_memory_summary()}")


def _maybe_prefetch_separate_inference_model_weights(model_core, *, to_cpu: bool) -> None:
    """Prefetch RL *separate inference model* weights to CPU/GPU.

    Supports two modes:
    1. UVM-based offloading (when --rl-inference-model-unified-memory-level=1)
    2. torch_memory_saver-based offloading (when offloading is enabled but UVM is not)

    Gated by user args; this assumes the separate inference model was allocated
    with UVM or torch_memory_saver when enabled.
    """
    args = get_args()
    if not args.rl_offload_inference_model_weights_when_idle:
        return

    # Check for torch_memory_saver path (when offloading is enabled but UVM is not)
    if args.rl_inference_model_unified_memory_level != 1:
        _torch_saver_swap_inference_model(to_cpu=to_cpu)
        return

    # UVM-based path (when UVM level is 1)
    device = -1 if to_cpu else int(torch.cuda.current_device())
    # Note: include_buffers=False because buffers created with explicit device= in register_buffer()
    # are not allocated via the UVM mempool and will fail UVM operations. Only parameters are UVM-allocated.
    advise_managed_module_parameters_preferred_location(model_core, device=device, include_buffers=False)
    nbytes = prefetch_managed_module_parameters(model_core, device=device, include_buffers=False)
    # Ensure pages are resident before we enter CUDA-graph capture / inference, or before training continues.
    torch.cuda.synchronize()

    if to_cpu:
        print_rank_0(f"[Rank 0] offloaded {nbytes / 1024**2:.2f} MB of separate RL inference model weights to CPU (other ranks may vary)")
    else:
        print_rank_0(f"[Rank 0] prefetched {nbytes / 1024**2:.2f} MB of separate RL inference model weights to GPU (other ranks may vary)")


def verify_model_weights_swap(
    train_model: LanguageModule,
    inference_model: LanguageModule,
    seq_len: int = 8,
    batch_size: int = 2,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Verify that the inference model produces the same forward pass outputs
    as the training model after the weights have been swapped.

    This function should be called after swap_model_weights to ensure the weight
    transfer was successful. It runs a forward pass on both models and asserts
    the outputs match.  This is meant for debugging purposes only.

    Args:
        train_model: The training model (source of weights).
        inference_model: The inference model (target of weights).
        seq_len: Sequence length for test input.
        batch_size: Batch size for test input.
        atol: Absolute tolerance for comparing outputs.
        rtol: Relative tolerance for comparing outputs.

    Raises:
        AssertionError: If forward pass outputs do not match within tolerance.
    """
    args = get_args()

    # Unwrap models to get the core module
    train_lm = train_model[0] if isinstance(train_model, (list, tuple)) else train_model
    inf_lm = inference_model[0] if isinstance(inference_model, (list, tuple)) else inference_model

    train_core = unwrap_model(train_lm)
    inf_core = unwrap_model(inf_lm)

    actual_vocab_size = getattr(args, 'padded_vocab_size', 128256)
    actual_seq_len = min(seq_len, getattr(args, 'seq_length', seq_len))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Generate deterministic test input - same across ALL ranks
    torch.manual_seed(1234)
    test_tokens = torch.randint(
        low=0, high=actual_vocab_size, size=(batch_size, actual_seq_len),
        device=device, dtype=torch.long
    )
    test_position_ids = (
        torch.arange(actual_seq_len, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    test_attention_mask = torch.ones(
        (batch_size, 1, actual_seq_len, actual_seq_len), device=device, dtype=torch.bool
    )

    # Save and restore training state
    train_was_training = train_core.training
    inf_was_training = inf_core.training

    train_core.eval()
    inf_core.eval()

    try:
        with torch.no_grad():
            train_output = train_lm(
                test_tokens, test_position_ids, test_attention_mask,
                runtime_gather_output=True
            )

            inf_output = inf_lm(
                test_tokens, test_position_ids, test_attention_mask,
                runtime_gather_output=True
            )

        # Only check on ranks that have output (last PP stage)
        if train_output is not None and inf_output is not None:
            assert train_output.shape == inf_output.shape, (
                f"Output shape mismatch: train={train_output.shape}, infer={inf_output.shape}"
            )
            
            max_diff = (train_output - inf_output).abs().max().item()
            assert torch.allclose(train_output, inf_output, atol=atol, rtol=rtol), (
                f"Forward pass outputs do not match: max_diff={max_diff:.6e}, atol={atol}, rtol={rtol}"
            )

    finally:
        # Restore training state
        if train_was_training:
            train_core.train()
        if inf_was_training:
            inf_core.train()



@dataclass(slots=True)
class RolloutStats:
    rewards: list[list[float]] # inner list is for a group
    env_ids: list[str] # same length as len(rewards)
    turn_lens: list[list[int]] # token lengths of turns, grouped.
    traj_lens: list[list[int]] # all turns comprise one trajectory.
    num_turns: None | list[list[int]] # num_turns per traj
    advantages: None | list[list[float]]
    min_piold_to_inf_prob: None | float
    max_piold_to_inf_prob: None | float
    mean_piold_to_inf_prob: None | float
    min_inf_train_prob_abs_diff: None | float
    max_inf_train_prob_abs_diff: None | float
    mean_inf_train_prob_abs_diff: None | float
    min_inf_prob: None | float
    max_inf_prob: None | float
    mean_inf_prob: None | float
    policy_epoch: list[list[int]]
    kv_cache_epoch: list[list[int]]
    completed_epochs: list[list[int]]
    num_evictions: list[list[int]]


# Runtime state container for RL-specific data that shouldn't be checkpointed
class RLRuntimeState:
    """Container for runtime state that is not checkpointed, tracking state between rollout collections"""

    def __init__(self):
        self.packing_context = None
        self.last_collection_iteration = 0
        self.sequences_this_iteration_on_rank = 0
        self.latest_batch_num_sequences = 0
        self.probe_turn_metadata = None
        self.probe_generation_masks = None
        self.probe_tokens = None
        self.probe_iteration = 0

    def reset_iteration_counters(self, iteration):
        """Reset per-iteration counters."""
        self.sequences_this_iteration_on_rank = 0
        self.last_collection_iteration = iteration

    def increment_sequences(self, count):
        """Increment the sequence counter."""
        self.sequences_this_iteration_on_rank += count
        self.latest_batch_num_sequences = count


# Global runtime state instance
_rl_runtime_state = RLRuntimeState()


def get_rl_runtime_state():
    """Get the global RL runtime state."""
    return _rl_runtime_state


def update_inference_logprobs_group_stats(
    old_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor,
    mask: torch.Tensor,
    group_stats: Any,
    gather_for_diag: bool = False,
) -> None:
    """Update group statistics with inference/train logprobs comparison metrics.

    This is the common statistics computation used by both packed and unpacked cases.

    Args:
        old_logprobs: Old logprobs tensor (train side)
        inference_logprobs: Inference logprobs tensor (aligned to match old_logprobs shape)
        mask: Boolean mask indicating valid positions for statistics
        group_stats: Statistics object to update with computed metrics
        gather_for_diag: If True, all_gather lp_delta across the DP group before printing
            [IS-diag].  Must be True consistently on ALL TP-rank-0 DP nodes simultaneously
            (i.e., only set from the unpacked training path, not the packed path).
    """
    n_elems = mask.sum()

    # Collect raw values for the IS-diag gather (empty list = this rank has no tokens).
    _lp_list = _inf_list = _trn_list = []

    if n_elems > 0:
        lp_delta = (old_logprobs - inference_logprobs)[mask]  # train_lp - inf_lp [nats]
        ratios = lp_delta.exp()
        abs_diffs = (old_logprobs.exp() - inference_logprobs.exp()).abs()[mask]

        group_stats.min_piold_to_inf_prob = ratios.min().item()
        group_stats.max_piold_to_inf_prob = ratios.max().item()
        group_stats.mean_piold_to_inf_prob = (ratios.sum() / n_elems).item()
        group_stats.min_inf_train_prob_abs_diff = abs_diffs.min().item()
        group_stats.max_inf_train_prob_abs_diff = abs_diffs.max().item()
        group_stats.mean_inf_train_prob_abs_diff = (abs_diffs.sum() / n_elems).item()

        inf_probs = inference_logprobs.exp()[mask]
        group_stats.min_inf_prob = inf_probs.min().item()
        group_stats.max_inf_prob = inf_probs.max().item()
        group_stats.mean_inf_prob = inf_probs.mean().item()

        if gather_for_diag:
            _lp_list  = lp_delta.cpu().tolist()
            _inf_list = inference_logprobs[mask].cpu().tolist()
            _trn_list = old_logprobs[mask].cpu().tolist()

    # IS-diag: gather across the DP group (TP-rank-0 only to avoid double-counting;
    # unconditional within that guard so all shards participate even when n_elems==0).
    if gather_for_diag:
        if dist.is_initialized() and mpu.get_tensor_model_parallel_rank() == 0:
            dp_group = mpu.get_data_parallel_group()
            _all_lp  = [None] * dist.get_world_size(dp_group)
            _all_inf = [None] * dist.get_world_size(dp_group)
            _all_trn = [None] * dist.get_world_size(dp_group)
            dist.all_gather_object(_all_lp,  _lp_list,  group=dp_group)
            dist.all_gather_object(_all_inf, _inf_list, group=dp_group)
            dist.all_gather_object(_all_trn, _trn_list, group=dp_group)
            if dist.get_rank() == 0:
                lp_g  = torch.tensor([v for lst in _all_lp  for v in lst], dtype=torch.float32)
                inf_g = torch.tensor([v for lst in _all_inf for v in lst], dtype=torch.float32)
                trn_g = torch.tensor([v for lst in _all_trn for v in lst], dtype=torch.float32)
                if len(lp_g) > 0:
                    print(
                        f"[IS-diag] train_lp - inf_lp (nats): "
                        f"mean={lp_g.mean().item():.3f}  "
                        f"p50={lp_g.median().item():.3f}  "
                        #f"p95={lp_g.float().quantile(0.95).item():.3f}  "
                        f"n={len(lp_g)}  "
                        f"mean_inf_lp={inf_g.mean().item():.3f}  "
                        f"mean_train_lp={trn_g.mean().item():.3f}"
                    )
        elif not dist.is_initialized() and n_elems > 0:
            lp_delta_t = torch.tensor(_lp_list, dtype=torch.float32)
            inf_t = torch.tensor(_inf_list, dtype=torch.float32)
            trn_t = torch.tensor(_trn_list, dtype=torch.float32)
            print(
                f"[IS-diag] train_lp - inf_lp (nats): "
                f"mean={lp_delta_t.mean().item():.3f}  "
                f"p50={lp_delta_t.median().item():.3f}  "
                #f"p95={lp_delta_t.float().quantile(0.95).item():.3f}  "
                f"n={len(lp_delta_t)}  "
                f"mean_inf_lp={inf_t.mean().item():.3f}  "
                f"mean_train_lp={trn_t.mean().item():.3f}"
            )


def align_unpacked_inference_logprobs(
    inference_logprobs: List[torch.Tensor],
    old_logprobs_for_data: torch.Tensor,
    generation_masks: torch.Tensor,
    group_stats: Any,
) -> torch.Tensor:
    """Align inference logprobs with old_logprobs for unpacked sequences and compute statistics.

    Args:
        inference_logprobs: List of inference logprobs tensors for each sequence
        old_logprobs_for_data: Template tensor with correct shape for alignment
        generation_masks: Tensor indicating which tokens were generated
        group_stats: Statistics object to update with computed metrics

    Returns:
        Aligned inference logprobs tensor
    """
    # Get first occurrence of a generation token
    # In get_logprobs() we chop off the first token -> the generation mask is shifted by one
    gen_masks_for_alignment = generation_masks
    first_gen_tok = gen_masks_for_alignment.int().argmax(dim=1) - 1

    # Align inference logprobs with old_logprobs
    # Note: We use old_logprobs_for_data as template since it has correct shape
    padded_inference_logprobs = old_logprobs_for_data.clone()

    # We need to align old_logprobs and inference logprobs as the latter are only for generations
    for i, inf_logprobs in enumerate(inference_logprobs):
        first_gen_idx = first_gen_tok[i]
        # We subtract -1 here because we append eod token on the train side, and we do not
        # get it from the inference. For the eod token, we reuse old_logprobs value.
        end_idx = min(first_gen_idx + len(inf_logprobs), padded_inference_logprobs.shape[1])
        actual_len = end_idx - first_gen_idx
        if actual_len > 0:
            padded_inference_logprobs[i, first_gen_idx:end_idx] = inf_logprobs[:actual_len]

    # Create truncated mask for statistics
    if old_logprobs_for_data.shape[1] + 1 < gen_masks_for_alignment.shape[1]:
        gen_masks_for_alignment = gen_masks_for_alignment[:, : old_logprobs_for_data.shape[1] + 1]

    truncated_mask = gen_masks_for_alignment[:, 1:].bool()

    # Final safety check
    if truncated_mask.shape != old_logprobs_for_data.shape:
        if truncated_mask.shape[1] > old_logprobs_for_data.shape[1]:
            truncated_mask = truncated_mask[:, : old_logprobs_for_data.shape[1]]
        elif truncated_mask.shape[1] < old_logprobs_for_data.shape[1]:
            pad_size = old_logprobs_for_data.shape[1] - truncated_mask.shape[1]
            truncated_mask = torch.nn.functional.pad(truncated_mask, (0, pad_size), value=False)

    # Sanity check: Two probability values cannot be more than 1.0 apart
    abs_diffs = (old_logprobs_for_data.exp() - padded_inference_logprobs.exp()).abs()[truncated_mask]
    assert all(abs_diffs <= 1.0)

    # Update group statistics using common helper; gather across DP for IS-diag.
    update_inference_logprobs_group_stats(
        old_logprobs=old_logprobs_for_data,
        inference_logprobs=padded_inference_logprobs,
        mask=truncated_mask,
        group_stats=group_stats,
        gather_for_diag=True,
    )

    return padded_inference_logprobs


def _first_generated_token_index(generation_mask: torch.Tensor) -> int | None:
    """Return the full-sequence index of the first generated token."""
    generated_positions = torch.nonzero(generation_mask.bool(), as_tuple=False).flatten()
    if generated_positions.numel() == 0:
        return None
    return int(generated_positions[0].item())


def _rollout_reward_scalar(reward: list[float] | float | None) -> float | None:
    if reward is None:
        return None
    if isinstance(reward, list):
        return float(np.mean(reward)) if reward else None
    return float(reward)


def _build_logprob_mismatch_turn_metadata(
    rollouts: Rollouts,
    rollout_metadata: list[dict[str, Any]] | None = None,
    inference_top_logprobs_by_turn: list | None = None,
) -> list[dict[str, Any]]:
    """Build per-turn metadata aligned with prepare_trajectories() row order."""
    args = get_args()
    include_top_logprobs = getattr(args, "rl_logprob_mismatch_top_k", 0) > 0
    turn_metadata = []
    rollout_metadata = rollout_metadata or [{} for _ in rollouts]
    for local_rollout_idx, rollout in enumerate(rollouts):
        metadata = rollout_metadata[local_rollout_idx] if local_rollout_idx < len(rollout_metadata) else {}
        num_turns = len(rollout.trajectory)
        for turn_idx in range(num_turns):
            inference_top_logprobs = None
            if include_top_logprobs:
                if inference_top_logprobs_by_turn is not None and len(turn_metadata) < len(
                    inference_top_logprobs_by_turn
                ):
                    inference_top_logprobs = inference_top_logprobs_by_turn[len(turn_metadata)]
                rollout_top_logprobs = getattr(rollout, "top_logprobs", None)
                if inference_top_logprobs is None and rollout_top_logprobs is not None and turn_idx < len(rollout_top_logprobs):
                    inference_top_logprobs = rollout_top_logprobs[turn_idx]
            turn_metadata.append(
                {
                    "group_index": metadata.get("group_index"),
                    "rollout_index": metadata.get("rollout_index", local_rollout_idx),
                    "global_rollout_index": metadata.get("global_rollout_index", local_rollout_idx),
                    "turn_index": turn_idx,
                    "env_id": getattr(rollout, "env_id", ""),
                    "problem_id": getattr(rollout, "problem_id", None),
                    "reward": _rollout_reward_scalar(getattr(rollout, "reward", None)),
                    "routing_dump_id": (
                        rollout.routing_dump_id[turn_idx]
                        if isinstance(rollout, TokenRollout)
                        and rollout.routing_dump_id is not None
                        and turn_idx < len(rollout.routing_dump_id)
                        else None
                    ),
                    "inference_top_logprobs": inference_top_logprobs,
                }
            )
    return turn_metadata


def _gather_logprob_mismatch_turn_metadata(
    local_turn_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not dist.is_initialized():
        return local_turn_metadata

    dp_group = mpu.get_data_parallel_group()
    gathered = [None] * dist.get_world_size(dp_group)
    dist.all_gather_object(gathered, local_turn_metadata, group=dp_group)
    return [item for rank_items in gathered for item in (rank_items or [])]


def _prepare_inference_logit_moments(rollouts: Rollouts, seq_length: int):
    means_rows, stds_rows = [], []
    found = False
    for rollout in rollouts:
        rollout_means = getattr(rollout, "logit_means", None)
        rollout_stds = getattr(rollout, "logit_stds", None)
        if rollout_means is None or rollout_stds is None:
            return None, None
        for turn_idx, generation_mask in enumerate(rollout.generation_mask):
            means_row = torch.full((seq_length - 1,), float("nan"), dtype=torch.float32)
            stds_row = torch.full((seq_length - 1,), float("nan"), dtype=torch.float32)
            turn_means = rollout_means[turn_idx] if turn_idx < len(rollout_means) else None
            turn_stds = rollout_stds[turn_idx] if turn_idx < len(rollout_stds) else None
            if turn_means is not None and turn_stds is not None and any(generation_mask):
                first_gen_idx = max(0, generation_mask.index(True) - 1)
                actual_len = min(len(turn_means), len(turn_stds), seq_length - 1 - first_gen_idx)
                if actual_len > 0:
                    means_row[first_gen_idx:first_gen_idx + actual_len] = torch.tensor(
                        turn_means[:actual_len], dtype=torch.float32
                    )
                    stds_row[first_gen_idx:first_gen_idx + actual_len] = torch.tensor(
                        turn_stds[:actual_len], dtype=torch.float32
                    )
                    found = True
            means_rows.append(means_row)
            stds_rows.append(stds_row)
    if not found:
        return None, None
    return torch.stack(means_rows), torch.stack(stds_rows)


def match_logits_to_inference_moments(
    logits: torch.Tensor,
    inference_logit_means: torch.Tensor | None,
    inference_logit_stds: torch.Tensor | None,
) -> torch.Tensor:
    if inference_logit_means is None or inference_logit_stds is None:
        return logits
    target_means = inference_logit_means.to(device=logits.device, dtype=torch.float32)
    target_stds = inference_logit_stds.to(device=logits.device, dtype=torch.float32)
    valid = torch.isfinite(target_means) & torch.isfinite(target_stds) & (target_stds > 0)
    if not valid.any():
        return logits

    logits_float = logits.float()
    train_means = logits_float.mean(dim=-1)
    train_stds = logits_float.std(dim=-1, unbiased=False).clamp_min(1e-6)
    matched = (
        (logits_float - train_means.unsqueeze(-1))
        / train_stds.unsqueeze(-1)
        * target_stds.unsqueeze(-1)
        + target_means.unsqueeze(-1)
    )
    return torch.where(valid.unsqueeze(-1), matched, logits_float)


def _detokenize_single_token(tokenizer: MegatronTokenizer | None, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        detok = tokenizer.detokenize([int(token_id)])
        if isinstance(detok, list):
            return "".join(str(part) for part in detok)
        return str(detok)
    except Exception:
        return str(token_id)


def _normalize_inference_top_logprobs(top_logprobs_row) -> tuple[list[str], list[float]]:
    if not top_logprobs_row:
        return [], []

    tokens = []
    logprobs = []
    if isinstance(top_logprobs_row, dict):
        iterable = [{"token": token, "logprob": logprob} for token, logprob in top_logprobs_row.items()]
    else:
        iterable = top_logprobs_row

    for item in iterable:
        if isinstance(item, dict):
            token = item.get("token")
            logprob = item.get("logprob")
        else:
            token = getattr(item, "token", None)
            logprob = getattr(item, "logprob", None)
        if token is None or logprob is None:
            continue
        tokens.append(str(token))
        logprobs.append(float(logprob))
    return tokens, logprobs


def _rank_in_tokens(token: str, tokens: list[str]) -> int | None:
    try:
        return tokens.index(token) + 1
    except ValueError:
        return None


def _top2_margin(logprobs: list[float]) -> float | None:
    if len(logprobs) < 2:
        return None
    return float(logprobs[0] - logprobs[1])


def _empty_topk_token_fields(num_tokens: int) -> dict[str, list[Any]]:
    return {
        "train_topk_tokens": [None] * num_tokens,
        "train_topk_logprobs": [None] * num_tokens,
        "train_topk_available": [False] * num_tokens,
        "train_top1_token": [""] * num_tokens,
        "train_top1_logprob": [float("nan")] * num_tokens,
        "train_top2_token": [""] * num_tokens,
        "train_top2_logprob": [float("nan")] * num_tokens,
        "train_top2_margin": [float("nan")] * num_tokens,
        "inference_topk_tokens": [None] * num_tokens,
        "inference_topk_logprobs": [None] * num_tokens,
        "inference_topk_available": [False] * num_tokens,
        "inference_top1_token": [""] * num_tokens,
        "inference_top1_logprob": [float("nan")] * num_tokens,
        "inference_top2_token": [""] * num_tokens,
        "inference_top2_logprob": [float("nan")] * num_tokens,
        "inference_top2_margin": [float("nan")] * num_tokens,
        "topk_token_overlap": [0] * num_tokens,
        "topk_token_overlap_frac": [float("nan")] * num_tokens,
        "sampled_token_train_rank": [0] * num_tokens,
        "sampled_token_inference_rank": [0] * num_tokens,
    }


def _build_logprob_mismatch_candidate(
    *,
    old_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor,
    generation_mask: torch.Tensor,
    tokens: torch.Tensor,
    seq_index: int,
    metadata: dict[str, Any] | None,
    max_tokens: int,
    train_topk_logprobs: torch.Tensor | None = None,
    train_topk_indices: torch.Tensor | None = None,
    topk_positions: int = 0,
    tokenizer: MegatronTokenizer | None = None,
) -> dict[str, Any] | None:
    first_generated = _first_generated_token_index(generation_mask)
    if first_generated is None:
        return None

    generated_positions = torch.nonzero(generation_mask.bool(), as_tuple=False).flatten()
    generated_positions = generated_positions[generated_positions > 0]
    if max_tokens > 0:
        generated_positions = generated_positions[:max_tokens]
    if generated_positions.numel() == 0:
        return None

    logprob_positions = generated_positions - 1
    valid = (logprob_positions >= 0) & (logprob_positions < old_logprobs.numel())
    valid &= logprob_positions < inference_logprobs.numel()
    generated_positions = generated_positions[valid]
    logprob_positions = logprob_positions[valid]
    if generated_positions.numel() == 0:
        return None

    old_vals = old_logprobs.detach().float().cpu()[logprob_positions.cpu()]
    inf_vals = inference_logprobs.detach().float().cpu()[logprob_positions.cpu()]
    delta = old_vals - inf_vals
    old_probs = old_vals.exp()
    inf_probs = inf_vals.exp()
    prob_abs_diff = (old_probs - inf_probs).abs()
    prob_ratio = torch.exp(torch.clamp(delta, min=-80.0, max=80.0))

    token_positions_cpu = generated_positions.cpu()
    token_ids = tokens.detach().cpu()[token_positions_cpu].to(torch.long)
    gen_offsets = token_positions_cpu - first_generated
    phases = ["prefill" if int(pos.item()) == first_generated else "decode" for pos in token_positions_cpu]

    metadata = metadata or {}
    topk_fields = _empty_topk_token_fields(int(generated_positions.numel()))
    if (
        topk_positions > 0
        and train_topk_logprobs is not None
        and train_topk_indices is not None
    ):
        inference_top_logprobs = metadata.get("inference_top_logprobs") or []
        num_topk_rows = min(topk_positions, prob_abs_diff.numel())
        topk_row_indices = torch.topk(prob_abs_diff, k=num_topk_rows).indices.tolist()
        for row_idx in topk_row_indices:
            logprob_pos = int(logprob_positions[row_idx].item())
            sampled_token_id = int(token_ids[row_idx].item())
            sampled_token = _detokenize_single_token(tokenizer, sampled_token_id)

            train_ids = train_topk_indices.detach().cpu()[logprob_pos].to(torch.long).tolist()
            train_lps = train_topk_logprobs.detach().cpu()[logprob_pos].float().tolist()
            train_tokens = [_detokenize_single_token(tokenizer, token_id) for token_id in train_ids]

            gen_offset = int(gen_offsets[row_idx].item())
            inf_tokens, inf_lps = (
                _normalize_inference_top_logprobs(inference_top_logprobs[gen_offset])
                if 0 <= gen_offset < len(inference_top_logprobs)
                else ([], [])
            )
            if not inf_tokens:
                # Some inference providers return only the sampled token logprob.
                # Keep the inference top-k availability flag false, but expose the
                # sampled token as a useful one-token proxy instead of leaving all
                # inference-side fields blank.
                inf_tokens = [sampled_token]
                inf_lps = [float(inf_vals[row_idx].item())]

            overlap = len(set(train_tokens) & set(inf_tokens))
            denom = min(len(train_tokens), len(inf_tokens))
            topk_fields["train_topk_tokens"][row_idx] = train_tokens
            topk_fields["train_topk_logprobs"][row_idx] = [float(x) for x in train_lps]
            topk_fields["train_topk_available"][row_idx] = bool(train_tokens)
            topk_fields["train_top1_token"][row_idx] = train_tokens[0] if train_tokens else ""
            topk_fields["train_top1_logprob"][row_idx] = (
                float(train_lps[0]) if train_lps else float("nan")
            )
            topk_fields["train_top2_token"][row_idx] = (
                train_tokens[1] if len(train_tokens) > 1 else ""
            )
            topk_fields["train_top2_logprob"][row_idx] = (
                float(train_lps[1]) if len(train_lps) > 1 else float("nan")
            )
            topk_fields["train_top2_margin"][row_idx] = _top2_margin(train_lps)
            topk_fields["inference_topk_tokens"][row_idx] = inf_tokens
            topk_fields["inference_topk_logprobs"][row_idx] = [float(x) for x in inf_lps]
            topk_fields["inference_topk_available"][row_idx] = bool(
                0 <= gen_offset < len(inference_top_logprobs)
                and inference_top_logprobs[gen_offset]
            )
            topk_fields["inference_top1_token"][row_idx] = (
                inf_tokens[0] if inf_tokens else ""
            )
            topk_fields["inference_top1_logprob"][row_idx] = (
                float(inf_lps[0]) if inf_lps else float("nan")
            )
            topk_fields["inference_top2_token"][row_idx] = (
                inf_tokens[1] if len(inf_tokens) > 1 else ""
            )
            topk_fields["inference_top2_logprob"][row_idx] = (
                float(inf_lps[1]) if len(inf_lps) > 1 else float("nan")
            )
            topk_fields["inference_top2_margin"][row_idx] = _top2_margin(inf_lps)
            topk_fields["topk_token_overlap"][row_idx] = overlap
            topk_fields["topk_token_overlap_frac"][row_idx] = (
                float(overlap / denom) if denom > 0 else None
            )
            topk_fields["sampled_token_train_rank"][row_idx] = _rank_in_tokens(
                sampled_token, train_tokens
            )
            topk_fields["sampled_token_inference_rank"][row_idx] = _rank_in_tokens(
                sampled_token, inf_tokens
            )

    candidate = {
        "seq_index": int(seq_index),
        "rank": int(dist.get_rank()) if dist.is_initialized() else 0,
        "group_index": metadata.get("group_index"),
        "rollout_index": metadata.get("rollout_index"),
        "global_rollout_index": metadata.get("global_rollout_index"),
        "turn_index": metadata.get("turn_index"),
        "env_id": metadata.get("env_id", ""),
        "problem_id": metadata.get("problem_id"),
        "reward": metadata.get("reward"),
        "first_generated_token_index": int(first_generated),
        "decode_start_token_index": int(first_generated + 1),
        "token_index": [int(x.item()) for x in token_positions_cpu],
        "gen_offset": [int(x.item()) for x in gen_offsets],
        "token_id": [int(x.item()) for x in token_ids],
        "phase": phases,
        "train_logprob": old_vals.tolist(),
        "inference_logprob": inf_vals.tolist(),
        "logprob_delta": delta.tolist(),
        "train_prob": old_probs.tolist(),
        "inference_prob": inf_probs.tolist(),
        "prob_abs_diff": prob_abs_diff.tolist(),
        "prob_ratio": prob_ratio.tolist(),
        "max_abs_logprob_delta": float(delta.abs().max().item()),
        "max_prob_abs_diff": float(prob_abs_diff.max().item()),
        "mean_abs_logprob_delta": float(delta.abs().mean().item()),
        "mean_prob_abs_diff": float(prob_abs_diff.mean().item()),
        "num_tokens": int(generated_positions.numel()),
        **topk_fields,
    }
    return candidate


def _extract_unpacked_logprob_mismatch_candidates(
    *,
    old_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor | None,
    generation_masks: torch.Tensor,
    trajs: torch.Tensor,
    turn_metadata: list[dict[str, Any]] | None,
    max_tokens: int,
    train_topk_logprobs: torch.Tensor | None = None,
    train_topk_indices: torch.Tensor | None = None,
    topk_positions: int = 0,
    tokenizer: MegatronTokenizer | None = None,
) -> list[dict[str, Any]]:
    if inference_logprobs is None:
        return []

    candidates = []
    for seq_index in range(generation_masks.shape[0]):
        metadata = turn_metadata[seq_index] if turn_metadata and seq_index < len(turn_metadata) else None
        candidate = _build_logprob_mismatch_candidate(
            old_logprobs=old_logprobs[seq_index],
            inference_logprobs=inference_logprobs[seq_index],
            generation_mask=generation_masks[seq_index],
            tokens=trajs[seq_index],
            seq_index=seq_index,
            metadata=metadata,
            max_tokens=max_tokens,
            train_topk_logprobs=(
                train_topk_logprobs[seq_index] if train_topk_logprobs is not None else None
            ),
            train_topk_indices=(
                train_topk_indices[seq_index] if train_topk_indices is not None else None
            ),
            topk_positions=topk_positions,
            tokenizer=tokenizer,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _extract_packed_logprob_mismatch_candidates(
    *,
    old_logprobs: torch.Tensor,
    packed_inference_logprobs: torch.Tensor | None,
    packing_context: Any,
    turn_metadata: list[dict[str, Any]] | None,
    max_tokens: int,
    train_topk_logprobs: torch.Tensor | None = None,
    train_topk_indices: torch.Tensor | None = None,
    topk_positions: int = 0,
    tokenizer: MegatronTokenizer | None = None,
) -> list[dict[str, Any]]:
    if packed_inference_logprobs is None:
        return []

    candidates = []
    packing_info = packing_context.packing_info
    old_logprobs = old_logprobs.cpu()
    packed_inference_logprobs = packed_inference_logprobs.cpu()
    if train_topk_logprobs is not None:
        train_topk_logprobs = train_topk_logprobs.cpu()
    if train_topk_indices is not None:
        train_topk_indices = train_topk_indices.cpu()
    for local_bin_idx, seq_indices in enumerate(packing_info.bin_seq_indices):
        seq_starts = packing_info.seq_starts[local_bin_idx]
        for seq_pos_in_bin, seq_index in enumerate(seq_indices):
            seq_start = seq_starts[seq_pos_in_bin]
            seq_len = packing_info.seq_lengths[seq_index]
            if seq_len <= 1:
                continue

            seq_slice = slice(seq_start, seq_start + seq_len - 1)
            metadata = (
                turn_metadata[seq_index]
                if turn_metadata is not None and seq_index < len(turn_metadata)
                else None
            )
            candidate = _build_logprob_mismatch_candidate(
                old_logprobs=old_logprobs[local_bin_idx, seq_slice],
                inference_logprobs=packed_inference_logprobs[local_bin_idx, seq_slice],
                generation_mask=packing_context.original_generation_masks[seq_index, :seq_len],
                tokens=packing_context.original_trajs[seq_index, :seq_len],
                seq_index=seq_index,
                metadata=metadata,
                max_tokens=max_tokens,
                train_topk_logprobs=(
                    train_topk_logprobs[local_bin_idx, seq_slice]
                    if train_topk_logprobs is not None
                    else None
                ),
                train_topk_indices=(
                    train_topk_indices[local_bin_idx, seq_slice]
                    if train_topk_indices is not None
                    else None
                ),
                topk_positions=topk_positions,
                tokenizer=tokenizer,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _select_logprob_mismatch_candidates(
    candidates: list[dict[str, Any]],
    num_examples: int,
    selection: str,
) -> list[dict[str, Any]]:
    if num_examples <= 0:
        return []
    if selection == "first":
        return candidates[:num_examples]

    per_group = selection.endswith("_per_group")
    base_selection = selection.removesuffix("_per_group")
    score_key = "max_prob_abs_diff" if base_selection == "top_prob_abs_diff" else "max_abs_logprob_delta"
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.get(score_key, 0.0),
            candidate.get("rank", 0),
            candidate.get("seq_index", 0),
        ),
    )
    if not per_group:
        return sorted_candidates[:num_examples]

    selected = []
    selected_groups = set()
    for candidate in sorted_candidates:
        group_key = candidate.get("group_index")
        if group_key is None:
            group_key = ("ungrouped", candidate.get("rank", 0), candidate.get("seq_index", 0))
        if group_key in selected_groups:
            continue
        selected.append(candidate)
        selected_groups.add(group_key)
        if len(selected) == num_examples:
            return selected

    # If fewer groups exist than requested examples, fill remaining slots with the next-best candidates.
    selected_ids = {id(candidate) for candidate in selected}
    for candidate in sorted_candidates:
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        if len(selected) == num_examples:
            break
    return selected


def _is_logprob_mismatch_canonical_rank() -> bool:
    if not dist.is_initialized():
        return True
    return (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == 0
    )


def _select_global_logprob_mismatch_candidates(
    local_candidates: list[dict[str, Any]],
    num_examples: int,
    selection: str,
) -> list[dict[str, Any]]:
    if not dist.is_initialized():
        return _select_logprob_mismatch_candidates(local_candidates, num_examples, selection)

    selected = None
    if _is_logprob_mismatch_canonical_rank():
        dp_group = mpu.get_data_parallel_group()
        gathered = [None] * dist.get_world_size(dp_group)
        dist.all_gather_object(gathered, local_candidates, group=dp_group)
        if dist.get_rank() == 0:
            all_candidates = [item for rank_items in gathered for item in (rank_items or [])]
            selected = _select_logprob_mismatch_candidates(
                all_candidates, num_examples, selection
            )

    broadcast_payload = [selected]
    dist.broadcast_object_list(broadcast_payload, src=0)
    return broadcast_payload[0] or []


def _binned_median_line(x_values: list[int], y_values: list[float], max_bins: int = 256):
    if len(x_values) < 128:
        return None, None
    x_array = np.asarray(x_values)
    y_array = np.asarray(y_values)
    num_bins = min(max_bins, max(1, len(x_values) // 64))
    bins = np.linspace(x_array.min(), x_array.max() + 1, num_bins + 1)
    x_medians = []
    y_medians = []
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (x_array >= start) & (x_array < end)
        if np.any(mask):
            x_medians.append(float(np.median(x_array[mask])))
            y_medians.append(float(np.median(y_array[mask])))
    return x_medians, y_medians


def _make_logprob_mismatch_figure(candidate: dict[str, Any], iteration: int):
    import matplotlib.pyplot as plt

    plt.switch_backend('agg')
    x_values = candidate["gen_offset"]
    delta_values = candidate["logprob_delta"]
    prob_diff_values = candidate["prob_abs_diff"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    title_bits = [
        f"iter={iteration}",
        f"rank={candidate.get('rank')}",
        f"seq={candidate.get('seq_index')}",
    ]
    if candidate.get("env_id"):
        title_bits.append(f"env={candidate['env_id']}")
    if candidate.get("problem_id") is not None:
        title_bits.append(f"problem={candidate['problem_id']}")
    fig.suptitle("Logprob mismatch: " + ", ".join(title_bits))

    axes[0].scatter(x_values, delta_values, s=2, alpha=0.45, rasterized=True)
    axes[0].axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel("train_lp - inf_lp")

    axes[1].scatter(x_values, prob_diff_values, s=2, alpha=0.45, rasterized=True)
    axes[1].set_ylabel("|train_p - inf_p|")
    axes[1].set_xlabel("generated token offset")

    for ax, y_values in zip(axes, [delta_values, prob_diff_values]):
        median_x, median_y = _binned_median_line(x_values, y_values)
        if median_x:
            ax.plot(median_x, median_y, color='red', linewidth=1.0, label='binned median')
            ax.legend(loc='best', fontsize=8)
        ax.axvline(0, color='green', linestyle='--', linewidth=1.0, label='prefill logprob')
        if len(x_values) > 1:
            ax.axvline(1, color='purple', linestyle=':', linewidth=1.0, label='decode start')
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def _json_table_cell(value):
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _wandb_log_logprob_mismatch_candidates(
    candidates: list[dict[str, Any]],
    iteration: int,
) -> None:
    wandb_writer = get_wandb_writer()
    if wandb_writer is None or not candidates:
        return

    try:
        import matplotlib.pyplot as plt
        import wandb as _wandb

        token_rows = []
        plot_rows = []
        for example_idx, candidate in enumerate(candidates):
            fig = _make_logprob_mismatch_figure(candidate, iteration)
            plot_rows.append(
                [
                    example_idx,
                    candidate.get("rank"),
                    candidate.get("seq_index"),
                    candidate.get("env_id"),
                    candidate.get("problem_id"),
                    candidate.get("reward"),
                    candidate.get("num_tokens"),
                    candidate.get("max_abs_logprob_delta"),
                    candidate.get("max_prob_abs_diff"),
                    _wandb.Image(fig),
                ]
            )
            plt.close(fig)

            for row_idx in range(candidate["num_tokens"]):
                token_rows.append(
                    [
                        iteration,
                        example_idx,
                        candidate.get("rank"),
                        candidate.get("group_index"),
                        candidate.get("rollout_index"),
                        candidate.get("turn_index"),
                        candidate.get("env_id"),
                        candidate.get("problem_id"),
                        candidate.get("reward"),
                        candidate["token_index"][row_idx],
                        candidate["gen_offset"][row_idx],
                        candidate["token_id"][row_idx],
                        candidate["phase"][row_idx],
                        candidate["train_logprob"][row_idx],
                        candidate["inference_logprob"][row_idx],
                        candidate["logprob_delta"][row_idx],
                        candidate["train_prob"][row_idx],
                        candidate["inference_prob"][row_idx],
                        candidate["prob_abs_diff"][row_idx],
                        candidate["prob_ratio"][row_idx],
                        _json_table_cell(candidate["train_topk_tokens"][row_idx]),
                        _json_table_cell(candidate["train_topk_logprobs"][row_idx]),
                        candidate["train_topk_available"][row_idx],
                        candidate["train_top1_token"][row_idx],
                        candidate["train_top1_logprob"][row_idx],
                        candidate["train_top2_token"][row_idx],
                        candidate["train_top2_logprob"][row_idx],
                        candidate["train_top2_margin"][row_idx],
                        _json_table_cell(candidate["inference_topk_tokens"][row_idx]),
                        _json_table_cell(candidate["inference_topk_logprobs"][row_idx]),
                        candidate["inference_topk_available"][row_idx],
                        candidate["inference_top1_token"][row_idx],
                        candidate["inference_top1_logprob"][row_idx],
                        candidate["inference_top2_token"][row_idx],
                        candidate["inference_top2_logprob"][row_idx],
                        candidate["inference_top2_margin"][row_idx],
                        candidate["topk_token_overlap"][row_idx],
                        candidate["topk_token_overlap_frac"][row_idx],
                        candidate["sampled_token_train_rank"][row_idx],
                        candidate["sampled_token_inference_rank"][row_idx],
                    ]
                )

        metrics = {
            "rl/logprob_mismatch/token_table": wandb_writer.Table(
                columns=[
                    "iteration",
                    "example_index",
                    "rank",
                    "group_index",
                    "rollout_index",
                    "turn_index",
                    "env_id",
                    "problem_id",
                    "reward",
                    "token_index",
                    "gen_offset",
                    "token_id",
                    "phase",
                    "train_logprob",
                    "inference_logprob",
                    "logprob_delta",
                    "train_prob",
                    "inference_prob",
                    "prob_abs_diff",
                    "prob_ratio",
                    "train_topk_tokens_json",
                    "train_topk_logprobs_json",
                    "train_topk_available",
                    "train_top1_token",
                    "train_top1_logprob",
                    "train_top2_token",
                    "train_top2_logprob",
                    "train_top2_margin",
                    "inference_topk_tokens_json",
                    "inference_topk_logprobs_json",
                    "inference_topk_available",
                    "inference_top1_token",
                    "inference_top1_logprob",
                    "inference_top2_token",
                    "inference_top2_logprob",
                    "inference_top2_margin",
                    "topk_token_overlap",
                    "topk_token_overlap_frac",
                    "sampled_token_train_rank",
                    "sampled_token_inference_rank",
                ],
                data=token_rows,
            ),
            "rl/logprob_mismatch/rollout_plots": wandb_writer.Table(
                columns=[
                    "example_index",
                    "rank",
                    "seq_index",
                    "env_id",
                    "problem_id",
                    "reward",
                    "num_tokens",
                    "max_abs_logprob_delta",
                    "max_prob_abs_diff",
                    "plot",
                ],
                data=plot_rows,
            ),
        }
        wandb_writer.log(metrics, step=iteration)
    except Exception as e:
        print_rank_0(f"[Logprob-mismatch] W&B plot creation failed: {e}")


def _maybe_log_logprob_mismatch_diagnostics(
    *,
    old_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor | None,
    generation_masks: torch.Tensor | None,
    trajs: torch.Tensor | None,
    packing_context: Any | None,
    turn_metadata: list[dict[str, Any]] | None,
    iteration: int,
    train_topk_logprobs: torch.Tensor | None = None,
    train_topk_indices: torch.Tensor | None = None,
) -> None:
    args = get_args()
    num_examples = getattr(args, "rl_logprob_mismatch_num_examples", 0)
    if num_examples <= 0:
        return

    local_candidates = []
    if _is_logprob_mismatch_canonical_rank() and inference_logprobs is not None:
        max_tokens = getattr(args, "rl_logprob_mismatch_max_tokens", 0)
        topk_positions = (
            getattr(args, "rl_logprob_mismatch_topk_positions", 0)
            if getattr(args, "rl_logprob_mismatch_top_k", 0) > 0
            else 0
        )
        tokenizer = get_tokenizer() if topk_positions > 0 else None
        if packing_context is None:
            local_candidates = _extract_unpacked_logprob_mismatch_candidates(
                old_logprobs=old_logprobs,
                inference_logprobs=inference_logprobs,
                generation_masks=generation_masks,
                trajs=trajs,
                turn_metadata=turn_metadata,
                max_tokens=max_tokens,
                train_topk_logprobs=train_topk_logprobs,
                train_topk_indices=train_topk_indices,
                topk_positions=topk_positions,
                tokenizer=tokenizer,
            )
        else:
            local_candidates = _extract_packed_logprob_mismatch_candidates(
                old_logprobs=old_logprobs,
                packed_inference_logprobs=inference_logprobs,
                packing_context=packing_context,
                turn_metadata=turn_metadata,
                max_tokens=max_tokens,
                train_topk_logprobs=train_topk_logprobs,
                train_topk_indices=train_topk_indices,
                topk_positions=topk_positions,
                tokenizer=tokenizer,
            )

    selected = _select_global_logprob_mismatch_candidates(
        local_candidates,
        num_examples,
        getattr(args, "rl_logprob_mismatch_selection", "top_abs_delta"),
    )
    _wandb_log_logprob_mismatch_candidates(selected, iteration)


def get_agent(args, parallel_generation_tasks: int | None = None):
    """Get an agent based on environment configuration.

    If args.langrl_env_config is provided, uses weighted environment selection.
    Otherwise falls back to legacy single environment selection.
    """
    with open(args.langrl_env_config, 'r') as f:
        config = yaml.safe_load(f)

    return WeightedMultiTask.from_config(
        config,
        parallel_generation_tasks=parallel_generation_tasks,
    )


_INFERENCE_INTERFACE = None


def get_inference_interface(args, loop, model):
    global _INFERENCE_INTERFACE
    if _INFERENCE_INTERFACE is None:
        _INFERENCE_INTERFACE = loop.run_until_complete(
            MegatronLocal.launch(
                model[0],
                host='0.0.0.0',
                port=8294,
                verbose=args.inference_text_gen_server_logging)
        )
    return _INFERENCE_INTERFACE


_ROLLOUT_GENERATOR = None


def get_rollout_generator(args, inference_interface, n_prompts, samples_per_group):
    global _ROLLOUT_GENERATOR
    if not (streaming := args.rl_partial_rollouts) or _ROLLOUT_GENERATOR is None:
        agent = get_agent(args, parallel_generation_tasks=args.rl_parallel_generation_tasks if streaming else n_prompts)
        request = GroupedRolloutRequest(
            num_groups=args.rl_generation_batch_size,
            streaming=streaming,
            rollouts_per_group=samples_per_group,
            inference_interface=inference_interface,
            generation_args={
                'temperature': args.rl_default_temperature,
                'max_tokens': args.inference_max_seq_length,
                'top_p': args.rl_default_top_p,
                'top_k': args.rl_default_top_k,
            },
            filter_groups_with_same_reward=args.grpo_filter_groups_with_same_reward,
            enforce_order=args.rl_enforce_generation_order,
        )
        _ROLLOUT_GENERATOR = agent.get_grouped_rollouts(request)
    return _ROLLOUT_GENERATOR


def get_environment_rollouts(
    model: LanguageModule, inference_model: LanguageModule, optimizer: MegatronOptimizer, n_prompts: int, samples_per_group: int
):
    """Sample environment rollouts from an LLM.

    Args:
        model: Model to sample from.
        inference_model: Inference model to use for inference.
        n_prompts: Number of prompts to sample for across *all* data parallel workers.
        samples_per_group: Amount of trajectories per prompt.

    Returns:
        GroupedRollouts object which is a nested list with each element being a list of rollouts of a group.
    """
    args = get_args()
    nvtx_range = get_nvtx_range()

    router_dump_dir = os.environ.get("ROUTER_STUDY_DUMP_DIR", "")
    if probe_needs_rollout_ids(args):
        os.environ["RL_DETERMINISM_PROBE_DIR"] = str(
            args.rl_determinism_probe_dir or args.rl_determinism_probe_write_targets_file
        )
        os.environ["ROUTER_STUDY_COLLECTION_ID"] = str(getattr(args, "curr_iteration", 0))
    if router_dump_dir:
        os.environ["ROUTER_STUDY_COLLECTION_ID"] = str(getattr(args, "curr_iteration", 0))
        topk_for_dump = getattr(args, "rl_logprob_mismatch_top_k", 0)
        if topk_for_dump > 0:
            os.environ["ROUTER_STUDY_DUMP_TOPK"] = str(topk_for_dump)
        if not dist.is_initialized() or dist.get_rank() == 0:
            dump_path = Path(router_dump_dir)
            dump_path.mkdir(parents=True, exist_ok=True)
            for pattern in ("rollout_*.npz", "rollout_*.npz.tmp.npz"):
                for old_dump in dump_path.glob(pattern):
                    try:
                        old_dump.unlink()
                    except FileNotFoundError:
                        pass
            print_rank_0(f"[Router-replay] cleared routing dump dir {router_dump_dir}")
        if dist.is_initialized():
            dist.barrier()

    if args.rl_offload_optimizer_during_inference:
        with nvtx_range("offload-optimizer-state-and-grad-buffers-during-inference"):
            if not args.rl_training_cuda_graphs:
                model[0].offload_grad_buffers()
            else:
                logger.warning(
                    "Gradient buffers will not be offloaded when training cudagraphs are used!"
                )
            optimizer.offload_to_cpu()

    # If we have separate training and inference models we to refit weights from the training model to the inference model.
    has_separate_inference_model = inference_model is not None
    if has_separate_inference_model:
        # If the separate inference model weights were prefetched to CPU while idle, bring them
        # back to GPU before refit/copy and before any CUDA-graph'd inference.
        with nvtx_range("prefetch-inference-model-weights-to-gpu"):
            inf_core = unwrap_model(inference_model[0])
            _maybe_prefetch_separate_inference_model_weights(inf_core, to_cpu=False)
        swap_model_weights(model, inference_model, args.refit_method)
        if args.rl_verify_model_weights_swap:
            verify_model_weights_swap(
                train_model=model,
                inference_model=inference_model,
                atol=.1,
                rtol=5e-4,
            )
    else:
        inference_model = model

    inference_pg_collection = get_attr_wrapped_model(inference_model[0], "pg_collection")
    pg_size = get_pg_size(inference_pg_collection.ep)
    assert (n_prompts % pg_size == 0), f"{n_prompts=} must be divisible by {pg_size=}"

    with nvtx_range("rollout-collection"):
        loop = get_asyncio_loop()
        with megatron_rl_inference_mode(
            inference_model,
            optimizer,
            args.cuda_graph_impl,
            False, # offload optimizer during rollout collection is handled above
            training_model=model if has_separate_inference_model else None,
        ) as inference_interface:

            with nvtx_range("inference-setup"):
                # Asyncronously run inference and rollout collection
                rollout_generator = get_rollout_generator(
                    args, inference_interface, n_prompts, samples_per_group
                )

            # NOTE(jbarker): we need to double check this when using PP>1
            rank = torch.distributed.get_rank()
            with nvtx_range("collect-rollouts"):
                if rank == 0:
                    log_single_rank(
                        logger,
                        logging.INFO,
                        f"Collecting rollouts, Iteration {args.curr_iteration}...",
                    )
                    rollouts = [
                        loop.run_until_complete(anext(rollout_generator)) for _ in range(n_prompts)
                    ]
                    # In deterministic mode, sort rollouts by problem_id for consistent ordering
                    # regardless of completion order due to system timing jitter.
                    if torch.are_deterministic_algorithms_enabled():
                        rollouts.sort(key=lambda group: group[0].problem_id if group and group[0].problem_id else "")
                    if not args.rl_partial_rollouts:
                        while True:
                            try:
                                loop.run_until_complete(anext(rollout_generator))
                                assert False, "Unexpected group left in generator."
                            except StopAsyncIteration:
                                break
                else:
                    # Just set up space to collect the rollouts
                    rollouts = [[None for _ in range(samples_per_group)] for _ in range(n_prompts)]

        with nvtx_range("sync-rollouts"):
            # Wait for Rollouts to be collected
            # TODO(jbarker): double check why this isn't causing rank 0 memory allocations
            torch.distributed.broadcast_object_list(rollouts, src=0)
        logger.debug(f"Got rollouts on rank {rank}")

    if args.rl_offload_optimizer_during_inference:
        with nvtx_range("restore-optimizer-state-and-grad-buffers-after-inference"):
            model[0].restore_grad_buffers()
            optimizer.restore_from_cpu()

    if lang_rl_log_dir and rank == get_pg_rank(inference_pg_collection.tp):
        with open(
            lang_rl_log_dir
            + f'/rollouts_rank{rank}_iteration{args.curr_iteration}_'
            + f'{Path(args.langrl_env_config).stem}.json',
            'w',
        ) as f:
            json.dump([[r.model_dump() for r in group] for group in rollouts], f)

    return rollouts


async def _score_prefill_rollouts_on_rank0(inference_interface, rollout_paths, args):
    rows = []
    per_rollout_delta_lp = []
    per_rollout_delta_p = []

    for file_idx, path in enumerate(rollout_paths):
        with np.load(path) as data:
            prompt_tokens = data["prompt_tokens"].astype(np.int64).tolist()
            generated_tokens = data["generated_tokens"].astype(np.int64).tolist()
            if "generated_log_probs" not in data:
                print_rank_0(
                    f"[Prefill-rescore] {file_idx + 1}/{len(rollout_paths)} "
                    f"skipping {os.path.basename(path)}: missing generated_log_probs"
                )
                continue
            decode_logprobs = data["generated_log_probs"].astype(np.float32)

        if not generated_tokens or len(decode_logprobs) == 0:
            print_rank_0(
                f"[Prefill-rescore] {file_idx + 1}/{len(rollout_paths)} "
                f"skipping {os.path.basename(path)}: empty generated tokens/logprobs"
            )
            continue
        full_tokens = prompt_tokens + generated_tokens
        print_rank_0(
            f"[Prefill-rescore] {file_idx + 1}/{len(rollout_paths)} scoring "
            f"{os.path.basename(path)} prompt_len={len(prompt_tokens)} "
            f"gen_len={len(generated_tokens)} total_len={len(full_tokens)}"
        )
        if len(full_tokens) + 1 > args.inference_max_seq_length:
            rows.append(
                {
                    "path": path,
                    "status": "skipped_too_long",
                    "num_tokens": len(full_tokens),
                    "inference_max_seq_length": args.inference_max_seq_length,
                }
            )
            print_rank_0(
                f"[Prefill-rescore] {file_idx + 1}/{len(rollout_paths)} skipped: "
                f"total_len={len(full_tokens)} exceeds inference_max_seq_length={args.inference_max_seq_length}"
            )
            continue

        try:
            prompt_logprobs = await asyncio.wait_for(
                inference_interface.score_prompt_logprobs(full_tokens),
                timeout=args.rl_prefill_rescore_timeout_seconds,
            )
        except asyncio.TimeoutError:
            rows.append(
                {
                    "path": path,
                    "status": "timeout",
                    "timeout_seconds": args.rl_prefill_rescore_timeout_seconds,
                    "prompt_len": len(prompt_tokens),
                    "generated_len": len(generated_tokens),
                    "total_len": len(full_tokens),
                }
            )
            print_rank_0(
                f"[Prefill-rescore] {file_idx + 1}/{len(rollout_paths)} timed out after "
                f"{args.rl_prefill_rescore_timeout_seconds}s; stopping diagnostic early"
            )
            break
        if prompt_logprobs is None:
            rows.append({"path": path, "status": "missing_prompt_logprobs"})
            print_rank_0(
                f"[Prefill-rescore] {file_idx + 1}/{len(rollout_paths)} failed: "
                "missing prompt_logprobs"
            )
            continue

        start = len(prompt_tokens) - 1
        end = start + len(generated_tokens)
        prefill_logprobs = np.asarray(prompt_logprobs[start:end], dtype=np.float32)
        n = min(len(prefill_logprobs), len(decode_logprobs), len(generated_tokens))
        if n == 0:
            rows.append({"path": path, "status": "empty_aligned_logprobs"})
            continue

        prefill_logprobs = prefill_logprobs[:n]
        decode_logprobs = decode_logprobs[:n]
        delta_lp = prefill_logprobs - decode_logprobs
        delta_p = np.abs(np.exp(prefill_logprobs) - np.exp(decode_logprobs))
        per_rollout_delta_lp.append(torch.tensor(delta_lp, dtype=torch.float32))
        per_rollout_delta_p.append(torch.tensor(delta_p, dtype=torch.float32))
        rows.append(
            {
                "path": path,
                "status": "ok",
                "prompt_len": len(prompt_tokens),
                "generated_len": n,
                "mean_abs_logprob_delta": float(np.mean(np.abs(delta_lp))),
                "max_abs_logprob_delta": float(np.max(np.abs(delta_lp))),
                "mean_prob_abs_diff": float(np.mean(delta_p)),
                "max_prob_abs_diff": float(np.max(delta_p)),
            }
        )
        print_rank_0(
            f"[Prefill-rescore] {file_idx + 1}/{len(rollout_paths)} done: "
            f"mean|Δlp|={np.mean(np.abs(delta_lp)):.6f} "
            f"max|Δlp|={np.max(np.abs(delta_lp)):.6f} "
            f"mean|Δp|={np.mean(delta_p):.6f} "
            f"max|Δp|={np.max(delta_p):.6f}"
        )
        await asyncio.sleep(0.5)

    return rows, per_rollout_delta_lp, per_rollout_delta_p


def _pad_1d_tensors_for_numpy(tensors: list[torch.Tensor]) -> np.ndarray:
    if not tensors:
        return np.empty((0, 0), dtype=np.float32)
    max_len = max(t.numel() for t in tensors)
    padded = []
    for tensor in tensors:
        if tensor.numel() < max_len:
            pad = tensor.new_full((max_len - tensor.numel(),), float("nan"))
            tensor = torch.cat([tensor, pad], dim=0)
        padded.append(tensor)
    return torch.stack(padded).numpy().astype(np.float32)


def run_prefill_rescore_diagnostic(
    model: list[LanguageModule],
    inference_model: list[LanguageModule] | None,
    optimizer: MegatronOptimizer,
    args,
) -> None:
    """Compare saved decode logprobs with full-prefill inference prompt logprobs."""
    import glob as _glob

    if not hasattr(args, "curr_iteration"):
        args.curr_iteration = 0

    rollout_file = args.rl_prefill_rescore_rollout_file
    rollout_dir = args.rl_prefill_rescore_rollout_dir
    results_dir = (
        args.rl_prefill_rescore_results_dir
        or (os.path.dirname(rollout_file) if rollout_file else rollout_dir)
    )
    if torch.distributed.get_rank() == 0:
        os.makedirs(results_dir, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    rollout_paths = []
    if torch.distributed.get_rank() == 0:
        if rollout_file:
            rollout_paths = [rollout_file]
        else:
            rollout_paths = sorted(_glob.glob(os.path.join(rollout_dir, "rollout_*.npz")))
        if not rollout_file and args.rl_prefill_rescore_max_rollouts > 0:
            rollout_paths = rollout_paths[: args.rl_prefill_rescore_max_rollouts]
        print_rank_0(
            f"[Prefill-rescore] scoring {len(rollout_paths)} rollout dump(s) from "
            f"{rollout_file or rollout_dir}"
        )

    original_return_log_probs = getattr(args, "return_log_probs", False)
    args.return_log_probs = True
    inference_model_to_use = inference_model if inference_model is not None else model
    if inference_model is not None:
        inf_core = unwrap_model(inference_model[0])
        _maybe_prefetch_separate_inference_model_weights(inf_core, to_cpu=False)
        swap_model_weights(model, inference_model, args.refit_method)

    with megatron_rl_inference_mode(
        inference_model_to_use,
        optimizer,
        args.cuda_graph_impl,
        False,
        training_model=model if inference_model is not None else None,
    ) as inference_interface:
        if torch.distributed.get_rank() == 0:
            loop = get_asyncio_loop()
            rows, delta_lp_tensors, delta_p_tensors = loop.run_until_complete(
                _score_prefill_rollouts_on_rank0(inference_interface, rollout_paths, args)
            )
        else:
            rows, delta_lp_tensors, delta_p_tensors = None, None, None
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    args.return_log_probs = original_return_log_probs

    if torch.distributed.get_rank() != 0:
        return

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    flat_lp = (
        torch.cat([tensor.abs() for tensor in delta_lp_tensors])
        if delta_lp_tensors
        else torch.tensor([], dtype=torch.float32)
    )
    flat_p = (
        torch.cat(delta_p_tensors)
        if delta_p_tensors
        else torch.tensor([], dtype=torch.float32)
    )
    summary = {
        "num_rollout_files": len(rollout_paths),
        "num_scored": len(ok_rows),
        "num_failed_or_skipped": len(rows) - len(ok_rows),
        "logprob_abs_delta": (
            {
                "mean": float(flat_lp.mean().item()),
                "p50": float(flat_lp.quantile(0.50).item()),
                "p95": float(flat_lp.quantile(0.95).item()),
                "p99": float(flat_lp.quantile(0.99).item()),
                "max": float(flat_lp.max().item()),
            }
            if flat_lp.numel() > 0
            else {}
        ),
        "prob_abs_diff": (
            {
                "mean": float(flat_p.mean().item()),
                "p50": float(flat_p.quantile(0.50).item()),
                "p95": float(flat_p.quantile(0.95).item()),
                "p99": float(flat_p.quantile(0.99).item()),
                "max": float(flat_p.max().item()),
            }
            if flat_p.numel() > 0
            else {}
        ),
        "rollouts": rows,
    }
    out_json = os.path.join(results_dir, "prefill_vs_decode_rescore.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    np.save(
        os.path.join(results_dir, "prefill_vs_decode_logprob_delta.npy"),
        _pad_1d_tensors_for_numpy(delta_lp_tensors),
    )
    np.save(
        os.path.join(results_dir, "prefill_vs_decode_prob_delta.npy"),
        _pad_1d_tensors_for_numpy(delta_p_tensors),
    )
    print_rank_0(f"[Prefill-rescore] saved → {out_json}")


def selective_log_softmax(logits, index):
    """Taken from: https://github.com/huggingface/trl/blob/26d86757a7c7e24e397ea44f57ecce6031dfac01/trl/trainer/utils.py#L1659.

    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    use_bik_logsoftmax = is_batch_invariant_mode_enabled()
    if logits.dtype in [torch.float32, torch.float64] and not use_bik_logsoftmax:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = torch.nn.functional.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(
                -1
            )
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def topk_log_softmax(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k token ids and logprobs without materializing full log-softmax output."""
    topk_logprobs = []
    topk_indices = []
    for row_logits in logits:
        row_topk_logits, row_topk_indices = torch.topk(row_logits, k, dim=-1)
        row_logsumexp = torch.logsumexp(row_logits.float(), dim=-1, keepdim=True)
        topk_logprobs.append(row_topk_logits.float() - row_logsumexp)
        topk_indices.append(row_topk_indices)
    return torch.stack(topk_logprobs), torch.stack(topk_indices)


def get_logprobs(
    model,
    tokens,
    position_ids,
    no_grad=False,
    sequence_packing=False,
    packed_seq_params=None,
    topk_logprobs: int = 0,
    inference_logit_means: torch.Tensor | None = None,
    inference_logit_stds: torch.Tensor | None = None,
):
    """Get sequence logprobs from their token ids.

    Args:
        model: model to predict with.
        tokens: inputs for which we want to get logprobs.
        position_ids: position ids that come with tokens.
        attention_mask: attention mask that comes with tokens.
        no_grad: whether to run in no_grad mode.
        packed_seq_params: Optional PackedSeqParams for sequence packing with TE.
            When provided with qkv_format='thd', the input tokens are sliced to
            remove padding before the forward pass, and outputs are padded back.
        packed_seq_len: Optional length of the packed sequence (excluding padding).
            Required when packed_seq_params is provided to avoid CPU-GPU synchronization.

    Returns:
        Logprobs of input sequences.

    """

    args = get_args()
    # Ensure packed_seq_params is always provided for CUDA graph signature consistency.
    # When sequence_packing is enabled, construct from packing config (max_sequences_per_bin).
    # When sequence_packing is disabled, construct a single-sequence default so the CUDA
    # graph signature matches the training forward_step in train_rl.py.
    # This is necessary because reference logprobs steps will reuse the training forward graph.
    if packed_seq_params is None:
        if sequence_packing:
            packed_seq_params = get_default_packed_seq_params(
                seq_length=tokens.shape[1],
                max_sequences_per_bin=args.rl_sequence_packing_max_sequences_per_bin,
                device=tokens.device,
            )
        else:
            cu_seqlens = torch.tensor([0, tokens.shape[1]], dtype=torch.int32, device=tokens.device)
            packed_seq_params = PackedSeqParams(
                qkv_format='thd',
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=tokens.shape[1],
                max_seqlen_kv=tokens.shape[1],
                total_tokens=tokens.shape[1],
            )

    nvtx_range = get_nvtx_range()

    with nvtx_range("get-logprobs", time=False):
        with nvtx_range("forward-pass", time=False):
            # TODO(vitalyk): use fp16/bf16 as a function argument. Do not use args.

            attention_mask_for_forward = None

            # This is a hack to fix megatron's behaviour when flash-decode affects the training code flow.
            flash_decode = model.config.flash_decode
            model.config.flash_decode = False
            fp32_output = not (args.fp16 or args.bf16)
            with torch.no_grad() if no_grad else nullcontext():
                logits_or_hidden_states = model(
                    tokens,
                    position_ids,
                    attention_mask_for_forward,
                    packed_seq_params=packed_seq_params,
                    runtime_gather_output=True,
                    fp32_output=fp32_output,
                )
            model.config.flash_decode = flash_decode

        pg_collection = get_attr_wrapped_model(model, "pg_collection")
        pp_group = pg_collection.pp

        if not is_pp_last_stage(pp_group):
            return logits_or_hidden_states
        else:
            logits = logits_or_hidden_states
            with nvtx_range("log-softmax", time=False):
                # We do not need logprobs for the n+1 token.
                logits_for_targets = logits[:, :-1, :]
                logits_for_targets = match_logits_to_inference_moments(
                    logits_for_targets, inference_logit_means, inference_logit_stds
                )
                logprobs = selective_log_softmax(logits_for_targets, tokens[:, 1:])
                selected_logprobs_for_probe = torch.full(
                    tokens.shape,
                    float("nan"),
                    dtype=logprobs.dtype,
                    device=logprobs.device,
                )
                selected_logprobs_for_probe[:, :-1] = logprobs
                probe_tensor_point(
                    "lm_selected_logprob",
                    selected_logprobs_for_probe.unsqueeze(-1),
                )
                if topk_logprobs > 0:
                    topk_values, topk_indices = topk_log_softmax(
                        logits_for_targets, topk_logprobs
                    )
                    _LOGPROBS_TOPK_BUFFER.append((topk_values.detach(), topk_indices.detach()))
            return logprobs


def calculate_grpo_advantages(rewards: list[list[float]], num_turns: list[list[int]], skip_std_normalization: bool = False, advantage_baseline_type: str = 'mean') -> np.ndarray:
    """Calculate GRPO advantages from rewards/num_turns.

    For multiturn rollouts, the logic is a bit more involved.
    # For training, we'll be turning each turn into a trajectory with the same reward
    # within a trajectory, e.g. if [[a,b],[c,d,e]] trajectory has reward 1.0, we will
    # get [a,b] with 1.0 and [c,d,e] with 1.0 when doing updates.
    """

    rewards = np.array(rewards)

    num_turns = np.array(num_turns)
    # Each outer dimension of num_turns is a group. Sum of those gives total num_turns per group.
    # Let's use this to calculate advantage.
    # mean/std should be repeated based on group lens
    group_turns = num_turns.sum(axis=-1)
    if advantage_baseline_type == 'mean':
        reward_baseline = rewards.mean(axis=1, keepdims=True).repeat(group_turns)
    elif advantage_baseline_type == 'median':
        reward_baseline = np.median(rewards, axis=1, keepdims=True).repeat(group_turns)
    else:
        raise ValueError(f"Advantage baseline type can only be mean or median. You pass {advantage_baseline_type}.")
    reward_stds = rewards.std(axis=1, keepdims=True).repeat(group_turns)

    # rewards are originally [g, group_size]
    # Making an assumption that all groups are of the same size!
    # @vitalyk: this will go away when we start sending env-based sample reqs.
    rewards = rewards.flatten().repeat(num_turns.flatten())

    advs = (rewards - reward_baseline)
    if not skip_std_normalization:
        advs = advs / (1e-4 + reward_stds)

    return advs.tolist()


def compute_group_stats(
    rollouts: GroupedRollouts, tokenizer: MegatronTokenizer, seq_len: int, skip_adv_std_normalization: bool = False, advantage_baseline_type: str = 'mean'
) -> RolloutStats:
    """Add group-based rollout stats for logging.

    Args:
        rollouts: Rollouts to generate the stats for. Each inner list is a group (as in GRPO group), i.e. all rollouts are for the same prompt.
        tokenizer: Tokenizer to tokenize the rollouts in case they are raw strings.
        seq_len: Maximum sequence length.

    Returns:
       RolloutStats object containing all the stats.
    """
    # TODO (rkirby) Maybe do some of this after the tensor building
    group_reward_means = []
    group_reward_stds = []
    turn_lens = []
    traj_lens = []
    rewards = []
    env_ids = []
    group_reward_ids = []
    num_turns = [] # num_turns per traj
    all_policy_epoch = []
    all_kv_cache_epoch = []
    all_completed_epochs = []
    all_num_evictions = []
    for group in rollouts:
        group_rewards = []
        group_traj_lengths = []
        group_turn_lengths = []
        group_num_turns = []
        group_policy_epoch = []
        group_kv_epoch = []
        group_completed_epochs = []
        group_num_evictions = []
        for rollout in group:
            if isinstance(rollout, TokenRollout):
                for turn_traj in rollout.trajectory:
                    detokenized_traj = tokenizer.detokenize(turn_traj)
                    lang_rl_log(
                        f"Rollout: [{rollout.env_id}] [{rollout.reward} : {len(rollout.trajectory)} tokens] {detokenized_traj}"
                    )
                    # TODO(vitalyk): how does multiturn change EOD/EOT?
                    assert (len(turn_traj) == seq_len) or (
                        turn_traj[-1] == tokenizer.eod
                    ), f"Rollout is not the correct length: {len(turn_traj)} {turn_traj[-1]}\n{detokenized_traj}"
            else:
                lang_rl_log(
                    f"Rollout: [{rollout.env_id}] [{rollout.reward} : {len(rollout.trajectory)} chars] {rollout.trajectory}"
                )
            group_num_turns.append(len(rollout.trajectory))
            group_rewards.append(rollout.reward)
            roll_turn_lens = [len(t) for t in rollout.trajectory]
            group_turn_lengths.extend(roll_turn_lens)
            group_traj_lengths.append(sum(roll_turn_lens))
            assert rollout.policy_epoch, "Rollout has no policy_epoch data"
            assert rollout.kv_cache_epoch, "Rollout has no kv_cache_epoch data"
            group_policy_epoch.append(min(turn[0][1] for turn in rollout.policy_epoch))
            group_kv_epoch.append(min(turn[0][1] for turn in rollout.kv_cache_epoch))
            group_completed_epochs.extend(turn[-1][1] for turn in rollout.policy_epoch)
            group_num_evictions.append(sum(rollout.num_evictions))
        all_policy_epoch.append(group_policy_epoch)
        all_kv_cache_epoch.append(group_kv_epoch)
        all_completed_epochs.append(group_completed_epochs)
        all_num_evictions.append(group_num_evictions)
        traj_lens.append(group_traj_lengths)
        turn_lens.append(group_turn_lengths)
        env_ids.append(group[0].env_id) # All rollouts in a group share the env_id by design.
        rewards.append(group_rewards)
        # https://arxiv.org/abs/2504.21233 reports that lens variance hurts.
        # Let's track this.
        num_turns.append(group_num_turns)

    stats = RolloutStats(
        traj_lens=traj_lens,
        turn_lens=turn_lens,
        rewards=rewards,
        # --------
        # Everything above is per-group, i.e. it is a list of lists,
        # with the inner list being the group data.
        env_ids=env_ids,
        num_turns=num_turns,
        advantages=calculate_grpo_advantages(rewards, num_turns, skip_std_normalization=skip_adv_std_normalization, advantage_baseline_type=advantage_baseline_type),
        min_piold_to_inf_prob=None,
        max_piold_to_inf_prob=None,
        mean_piold_to_inf_prob=None,
        min_inf_train_prob_abs_diff=None,
        max_inf_train_prob_abs_diff=None,
        mean_inf_train_prob_abs_diff=None,
        min_inf_prob=None,
        max_inf_prob=None,
        mean_inf_prob=None,
        policy_epoch=all_policy_epoch,
        kv_cache_epoch=all_kv_cache_epoch,
        completed_epochs=all_completed_epochs,
        num_evictions=all_num_evictions,
    )
    return stats



def prep_wandb_metrics(
        wandb_writer: wandb_run.Run,
        traj_lens: List[List[int]],
        turn_lens: List[List[int]],
        rewards: List[List[float]],
        num_turns: List[List[int]],
        advantages: List[float],
        policy_epoch: List[List[int]],
        kv_cache_epoch: List[List[int]],
        completed_epochs: List[List[int]],
        num_evictions: List[List[int]],
        current_iteration: int,
        example_group: list[TokenRollout | Rollout] | None = None,
        tokenizer: MegatronTokenizer | None = None,
    ):

    """Make a wandb-parseable dictionary of metrics for logging.

    Args:
        wandb_writer: Wandb run to log to.
        traj_lens: Grouped list of trajectory lengths.
        turn_lens: Grouped list of turn lengths.
        rewards: Grouped list of rewards.
        num_turns: Grouped list of number of turns in the trajectories.
        advantages: Flattened list of advantages.
        policy_epoch: Grouped list of per-rollout min policy epoch stamps.
        kv_cache_epoch: Grouped list of per-rollout min KV cache epoch stamps.
        completed_epochs: Grouped list of per-turn max policy epoch stamps.
        num_evictions: Grouped list of per-rollout number of evictions.
        current_iteration: Current training iteration.
        example_group: A list of rollouts of one group to log examples of trajectories.
        tokenizer: Tokenizer to untokenize trajectories for logging.
    """

    group_table = wandb_writer.Table(
        columns=['group_means', 'group_stds'],
        data=[[np.mean(g), np.std(g)] for g in rewards],
    )

    true_policy_staleness = [current_iteration - s for g in policy_epoch for s in g]
    true_kv_staleness = [current_iteration - s for g in kv_cache_epoch for s in g]

    metrics = {
            'group_means_hist': wandb_writer.plot.histogram(
                group_table, 'group_means', 'Group Means'
            ),
            'group_stds_hist': wandb_writer.plot.histogram(
                group_table, 'group_stds', 'Group STDs'
            ),
            'rewards_hist': wandb_writer.plot.histogram(
                wandb_writer.Table(
                    columns=['reward'], data=[[r] for g in rewards for r in g]
                ),
                'reward', 'All Rewards'
            ),
            'advantages_hist': wandb_writer.plot.histogram(
                wandb_writer.Table(
                    columns=['advantages'], data=[[x] for x in advantages]
                ),
                'advantages', 'Advantages'
            ),
            'rollout_table': wandb_writer.Table(
                columns=['reward', 'traj_length', 'num_evictions'],
                data=list(zip(
                    [r for g in rewards for r in g],
                    [l for g in traj_lens for l in g],
                    [e for g in num_evictions for e in g],
                )),
            ),
            'mean_turn_length': np.mean([np.mean(g) for g in turn_lens]),
            'mean_turn_length_std': np.mean([np.std(g) for g in turn_lens]),
            'max_turn_length': max([max(g) for g in turn_lens]),
            'min_turn_length': min([min(g) for g in turn_lens]),
            'mean_traj_length': np.mean([np.mean(g) for g in traj_lens]),
            'mean_traj_length_std': np.mean([np.std(g) for g in traj_lens]),
            'max_traj_length': max([max(g) for g in traj_lens]),
            'min_traj_length': min([min(g) for g in traj_lens]),
            'mean_num_turns': np.mean([np.mean(g) for g in num_turns]),
            'max_num_turns': max([max(g) for g in num_turns]),
            'min_num_turns': min([min(g) for g in num_turns]),
            'mean_reward': np.mean([np.mean(g) for g in rewards]),
            'mean_advantage': np.mean(advantages),
            'nonzero_groups_ratio': np.count_nonzero(advantages)
            / len(advantages),
            'mean_policy_staleness': np.mean(true_policy_staleness),
            'max_policy_staleness': max(true_policy_staleness),
            'min_policy_staleness': min(true_policy_staleness),
            'mean_kv_cache_staleness': np.mean(true_kv_staleness),
            'max_kv_cache_staleness': max(true_kv_staleness),
            'min_kv_cache_staleness': min(true_kv_staleness),
            'total_eviction_count': sum([sum(g) for g in num_evictions]),
            'max_num_evictions': max([max(g) for g in num_evictions]),
            'mean_completion_gap': np.mean([current_iteration - s for g in completed_epochs for s in g]),
    }
    if example_group:
        if tokenizer is None:
            raise ValueError("If you provide an example group to log, you need to provide a tokenizer too.")
        metrics['rollouts'] = wandb_writer.Table(
            columns=['Trajectories', 'Tokens', 'Rewards'],
            rows=[
                [
                    tokenizer.detokenize(turn) if isinstance(r, TokenRollout) else turn,
                    r.trajectory,
                    r.reward,
                ]
                for r in example_group for turn in r.trajectory
            ],
        )
    return metrics


def maybe_log_training_metrics(
    group_stats: RolloutStats,
    current_iteration: int,
    tokenizer: MegatronTokenizer,
    example_groups: dict[str, list[TokenRollout | Rollout]],
):
    """Log training metrics if writers are available.

    Args:
        group_stats: RolloutStats object to pass to writers.
        current_iteration: Current training iteration.
        tokenizer: Tokenizer to untokenize trajectories for logging.
        example_groups: A dict with values as list of rollouts of one group to log examples of trajectories. Keys are env names.
    """

    wandb_writer = get_wandb_writer()
    tb_writer = get_tensorboard_writer()
    if tb_writer:
        tb_writer.add_scalar('mean_reward', np.mean([np.mean(g) for g in group_stats.rewards]), current_iteration)
    if not wandb_writer:
        return

    # We log these metrics for the aggregated data, no split per env.
    metrics = {
        'min_piold_to_inf_prob': group_stats.min_piold_to_inf_prob,
        'max_piold_to_inf_prob': group_stats.max_piold_to_inf_prob,
        'mean_piold_to_inf_prob': group_stats.mean_piold_to_inf_prob,
        'min_inf_train_prob_abs_diff': group_stats.min_inf_train_prob_abs_diff,
        'max_inf_train_prob_abs_diff': group_stats.max_inf_train_prob_abs_diff,
        'mean_inf_train_prob_abs_diff': group_stats.mean_inf_train_prob_abs_diff,
        'min_inf_prob': group_stats.min_inf_prob,
        'max_inf_prob': group_stats.max_inf_prob,
        'mean_inf_prob': group_stats.mean_inf_prob,
    }

    traj_lens = group_stats.traj_lens
    turn_lens = group_stats.turn_lens
    rewards = group_stats.rewards
    num_turns = group_stats.num_turns
    advantages = group_stats.advantages
    policy_epoch = group_stats.policy_epoch
    kv_cache_epoch = group_stats.kv_cache_epoch
    completed_epochs = group_stats.completed_epochs
    num_evictions = group_stats.num_evictions

    metrics = metrics | prep_wandb_metrics(wandb_writer=wandb_writer,
        traj_lens=traj_lens, turn_lens=turn_lens, rewards=rewards, num_turns=num_turns, advantages=advantages,
        policy_epoch=policy_epoch, kv_cache_epoch=kv_cache_epoch, completed_epochs=completed_epochs,
        num_evictions=num_evictions, current_iteration=current_iteration)
    env_stats = lambda cont, idx: [cont[i] for i in idx]
    group_turn_counts = [sum(nt) for nt in num_turns]

    for env_id in set(group_stats.env_ids):
        env_idx = [i for i, eidx in enumerate(group_stats.env_ids) if eidx == env_id]

        # Advantages are flattened, we need to be more careful with those.
        env_advantages = []
        for i in env_idx:
            st = sum(group_turn_counts[:i])
            end = st + group_turn_counts[i]
            env_advantages.extend(advantages[st:end])

        env_metrics = prep_wandb_metrics(wandb_writer=wandb_writer, traj_lens=env_stats(traj_lens, env_idx),
            turn_lens=env_stats(turn_lens, env_idx),
            rewards=env_stats(rewards, env_idx),
            num_turns=env_stats(num_turns, env_idx),
            advantages=env_advantages,
            policy_epoch=env_stats(policy_epoch, env_idx),
            kv_cache_epoch=env_stats(kv_cache_epoch, env_idx),
            completed_epochs=env_stats(completed_epochs, env_idx),
            num_evictions=env_stats(num_evictions, env_idx),
            current_iteration=current_iteration,
            example_group=example_groups[env_id],
            tokenizer=tokenizer,
        )
        for k, v in env_metrics.items():
            metrics[f"{env_id}_{k}"] = v

    wandb_writer.log(metrics, step=current_iteration)


def prepare_trajectories(
    rollouts: Rollouts, tokenizer: MegatronTokenizer, seq_length: int, sequence_packing: bool, skip_bos_token: bool
):
    """Pad trajectories and extract the generation masks.
    Args:
        rollouts: Rollouts to extract trajectories from.
        tokenizer: Tokenizer to get the padding token and potentially tokenize.
        seq_length:  Maximum sequence length to pad to.

    Returns:
        Trajectories and their generation masks.

    Raises:
        ValueError:
    """
    # Track counts for each environment ID
    env_id_counts = Counter()

    DEFAULT_PAD_TOKENS = ['<|finetune_right_pad_id|>', '<SPECIAL_999>']

    if tokenizer.library == "huggingface":
        tokenizer : HuggingFaceTokenizer
        if not tokenizer.pad:
            for pad_token in DEFAULT_PAD_TOKENS:
                if pad_token in tokenizer._tokenizer.tokenizer.get_vocab():
                    log_single_rank(
                        logger, logging.INFO, f"Updating tokenizer pad token to {pad_token}"
                    )
                    tokenizer._tokenizer.pad_token = pad_token
                    break
            else:
                raise ValueError("No pad token found in tokenizer vocabulary")
    elif tokenizer.library == "tiktoken":
        assert "<SPECIAL_233>" in tokenizer.vocab, "Pad token is NOT in the tokenizer"
        tokenizer._pad_id = tokenizer.vocab["<SPECIAL_233>"]

    log_single_rank(logger, logging.INFO, f"Tokenizer vocab size: {tokenizer.vocab_size}")
    log_single_rank(
        logger,
        logging.INFO,
        f"Tokenizer PAD: '{tokenizer.detokenize([tokenizer.pad])} ({tokenizer.pad})'",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"Tokenizer EOD: '{tokenizer.detokenize([tokenizer.eod])} ({tokenizer.eod})'",
    )

    trajs = []
    generation_masks = []
    inference_logprobs = []
    inference_routing = []
    routing_dump_ids = []
    for rollout in rollouts:
        # traj, gen mask and logprobs are lists now.
        # each list entry is a turn, single-turn environments just have a single-element list.
        # We assume that all lengths of the structs above have the same lengths (number of turns).

        all_turns_trajectories = (
            copy.deepcopy(rollout.trajectory)
            if isinstance(rollout, TokenRollout)
            else tokenizer.tokenize(rollout.trajectory)
        )
        for turn_idx, trajectory in enumerate(all_turns_trajectories):
            inf_logprobs = rollout.logprobs[turn_idx]
            generation_mask = rollout.generation_mask[turn_idx] if isinstance(rollout, TokenRollout) else None
            length = len(trajectory)
            assert length <= seq_length, "Rollout too long, how did this happen?"
            if len(trajectory) < seq_length:
                assert (
                    trajectory[-1] == tokenizer.eod
                ), "Trajectories under a seq_length limit should have eod token at the end."

            if length < seq_length:
                trajectory.extend([tokenizer.pad] * (seq_length - length))
                if generation_mask:
                    generation_mask.extend([False] * (seq_length - length))
            trajs.append(trajectory)
            generation_masks.append(generation_mask)

            if inf_logprobs is not None:
                inf_logprobs_tensor = torch.Tensor(inf_logprobs)
                # Don't pad individual logprobs here - padding happens later if needed
                inference_logprobs.append(inf_logprobs_tensor)
            else:
                inference_logprobs.append(None)

            # Routing indices: [G, L, top_k] int list from inference engine (may be None).
            ri = None
            if isinstance(rollout, TokenRollout) and rollout.routing_indices is not None:
                ri = rollout.routing_indices[turn_idx]
            inference_routing.append(ri)

            dump_id = None
            if isinstance(rollout, TokenRollout) and rollout.routing_dump_id is not None:
                dump_id = rollout.routing_dump_id[turn_idx]
            routing_dump_ids.append(dump_id)

        env_id_counts[rollout.env_id] += 1

    if torch.distributed.is_initialized():
        logger.info(f"[{dist.get_rank()}] Rollout counts:")
        for env_id, count in env_id_counts.items():
            logger.info(f"[{dist.get_rank()}] \t{env_id}: {count}")

    generation_masks = torch.tensor(generation_masks, dtype=torch.bool, device='cpu')
    trajs = torch.tensor(trajs, device='cpu')

    # Only process if we have inference_logprobs
    if inference_logprobs and any(lp is not None for lp in inference_logprobs):
        # We need to pad all logprobs to the same size for sequence packing.
        # For non-packing mode, keep as list of tensors (unpadded)
        # This preserves the original behavior where each sequence can have different lengths
        if sequence_packing:
            inference_logprobs = _pad_nonnull_with_zeros(inference_logprobs, seq_length)
    else:
        inference_logprobs = None

    # Some sanity checks regarding the tokenization
    if not skip_bos_token:
        assert (
            tokenizer.bos is None or (trajs[:, 0] == tokenizer.bos).all()
        ), "First token should be bos"
    else:
        assert (
            tokenizer.bos is None or (trajs[:, 0] != tokenizer.bos).all()
        ), "First token should not be bos"  
    assert (
        tokenizer.bos is None or (trajs[:, 1] != tokenizer.bos).all()
    ), "Second token should not be bos"
    assert (
        (trajs * generation_masks.int() == tokenizer.eod).sum(axis=1) <= 1
    ).all(), "Only one eod per trajectory in generated tokens."
    # TODO(rkirby):
    # We should avoid the tokenizer pad token being the same as the eod token for proper loss masking,
    # But now the deepseek tokenizer has the pad token set to eod, we need to handle this.
    # assert (tokenizer.pad != tokenizer.eod), "Pad and eod should be different"
    has_routing = any(r is not None for r in inference_routing)
    has_routing_dump_ids = any(r is not None for r in routing_dump_ids)
    return (
        trajs,
        generation_masks,
        inference_logprobs,
        (inference_routing if has_routing else None),
        (routing_dump_ids if has_routing_dump_ids else None),
    )


def logprobs_forward_step(
    data_iterator,
    model,
    is_correction,
    packing_context=None,
    replay_enabled=False,
    topk_logprobs: int = 0,
    match_logit_moments: bool = False,
    probe_turn_metadata=None,
    probe_generation_masks=None,
    probe_iteration: int = 0,
    probe_phase: str = "training_old_logprobs",
):
    # Avoid self.training checks which will trigger cudagraph capture; this path reuses
    # the forward pass from training after it has been captured on the 1st iteration.
    model.eval()
    b_logit_means, b_logit_stds = None, None
    b_probe_seq_indices = None

    if packing_context is not None:
        # When using sequence packing, the data iterator returns a tuple with a single element, the bin index.
        bin_tensor = next(data_iterator)[0]
        #TODO(jalbericiola): change for named tuple
        (b_trajs, _, _, _, b_posids, _, _, _, _, _, b_packed_seq_params) = (
            load_packed_data_by_index(bin_tensor.item(), packing_context, is_correction)
        )
    else:
        if replay_enabled:
            batch = next(data_iterator)
            if len(batch) in (5, 7):
                b_probe_seq_indices = batch[-1]
                batch = batch[:-1]
            if len(batch) == 6:
                b_trajs, b_posids, b_routing, b_seq_mask, b_logit_means, b_logit_stds = batch
            else:
                b_trajs, b_posids, b_routing, b_seq_mask = batch
                b_logit_means, b_logit_stds = None, None
            from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
            replay_mask = b_seq_mask.view(-1).cuda()
            flat = b_routing.view(-1, b_routing.shape[2], b_routing.shape[3])
            layer_tensors = [flat[replay_mask.cpu(), l, :].cuda() for l in range(flat.shape[1])]
            RouterReplay.set_replay_data(layer_tensors, replay_mask)
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        else:
            batch = next(data_iterator)
            if len(batch) in (5, 7):
                b_probe_seq_indices = batch[-1]
                batch = batch[:-1]
            if len(batch) == 6:
                b_trajs, b_posids, _, _, b_logit_means, b_logit_stds = batch
            else:
                b_trajs, b_posids, _, _ = batch
                b_logit_means, b_logit_stds = None, None
        b_packed_seq_params = None
    if not match_logit_moments:
        b_logit_means, b_logit_stds = None, None

    probe_context = nullcontext()
    if (
        packing_context is None
        and b_probe_seq_indices is not None
        and probe_turn_metadata is not None
        and probe_generation_masks is not None
    ):
        token_metadata = build_training_token_metadata(
            tokens=b_trajs,
            generation_masks=probe_generation_masks,
            seq_indices=b_probe_seq_indices,
            turn_metadata=probe_turn_metadata,
            iteration=probe_iteration,
            phase=probe_phase,
        )
        probe_context = probe_scope(
            phase=probe_phase,
            iteration=probe_iteration,
            token_metadata=token_metadata,
            batch_size=int(b_trajs.shape[0]),
            seq_length=int(b_trajs.shape[1]),
            extra={"seq_indices": b_probe_seq_indices.detach().cpu().tolist()},
            probe=get_determinism_probe(model),
        )

    with probe_context:
        logprobs_value = get_logprobs(
            model,
            b_trajs.cuda(),
            b_posids.cuda(),
            no_grad=True,
            sequence_packing=packing_context is not None,
            packed_seq_params=b_packed_seq_params,
            topk_logprobs=topk_logprobs,
            inference_logit_means=b_logit_means.cuda() if b_logit_means is not None else None,
            inference_logit_stds=b_logit_stds.cuda() if b_logit_stds is not None else None,
        )
    logprobs = (logprobs_value, None)

    if replay_enabled and packing_context is None:
        from megatron.core.transformer.moe.router_replay import RouterReplay
        RouterReplay.clear_global_router_replay_action()
        RouterReplay.clear_global_indices()

    model.train()
    return logprobs


def compute_logprobs_batch(
    model,
    data_loader,
    forward_backward_func,
    packing_context,
    trajs_batch_size, # n_bins for seq packing, and batch_size for non seq packing
    seq_length,
    logprobs_batch_size,
    decoder_seq_length,
    dtype,
    pp_group,
    is_correction,
    collect_non_loss_data=False,
    replay_enabled=False,
    topk_logprobs: int = 0,
    match_logit_moments: bool = False,
    probe_turn_metadata=None,
    probe_generation_masks=None,
    probe_iteration: int = 0,
    probe_phase: str = "training_old_logprobs",
):
    """Compute logprobs for all batches in the data loader."""
    global _LOGPROBS_TOPK_BUFFER
    _LOGPROBS_TOPK_BUFFER = []
    args = get_args()
    if probe_enabled(args):
        model_for_probe = model[0] if isinstance(model, list) else model
        ensure_determinism_probe(model_for_probe, args)
    if replay_enabled:
        from megatron.core.transformer.moe.router_replay import RouterReplay
        RouterReplay.clear_global_replay_stats()

    logprobs_list = []
    topk_logprobs_list = []
    topk_indices_list = []
    data_iterator = iter(data_loader)
    for i in range(len(data_loader)):
        output_tensor = forward_backward_func(
            forward_step_func=partial(
                logprobs_forward_step,
                is_correction=is_correction,
                packing_context=packing_context,
                replay_enabled=replay_enabled,
                topk_logprobs=topk_logprobs,
                match_logit_moments=match_logit_moments,
                probe_turn_metadata=probe_turn_metadata,
                probe_generation_masks=probe_generation_masks,
                probe_iteration=probe_iteration,
                probe_phase=probe_phase,
            ),
            data_iterator=data_iterator,
            model=model,
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=logprobs_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=True,
            adjust_tensor_shapes_fn=None,
            collect_non_loss_data=collect_non_loss_data,
        )
        if is_pp_last_stage(pp_group):
            batch_logprobs = output_tensor[0]
            if topk_logprobs > 0:
                batch_topk_logprobs, batch_topk_indices = _LOGPROBS_TOPK_BUFFER[-1]
                topk_logprobs_list.append(batch_topk_logprobs.detach())
                topk_indices_list.append(batch_topk_indices.detach())
            logprobs_list.append(batch_logprobs.detach())

    if is_pp_last_stage(pp_group):
        logprobs = torch.concat(logprobs_list, dim=0)
        expected_dtype = torch.float32 if match_logit_moments else dtype
        assert logprobs.dtype == expected_dtype
        if topk_logprobs > 0:
            all_topk_logprobs = torch.concat(topk_logprobs_list, dim=0)
            all_topk_indices = torch.concat(topk_indices_list, dim=0)
    else:
        logprobs = torch.empty(
            trajs_batch_size,
            seq_length-1,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )
        if topk_logprobs > 0:
            all_topk_logprobs = torch.empty(
                trajs_batch_size,
                seq_length - 1,
                topk_logprobs,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            all_topk_indices = torch.empty(
                trajs_batch_size,
                seq_length - 1,
                topk_logprobs,
                dtype=torch.long,
                device=torch.cuda.current_device(),
            )

    # Only PP>1 needs a broadcast from the last stage; for PP=1 the output is already local.
    if get_pg_size(pp_group) > 1:
        dist.broadcast(logprobs, src=get_pp_last_rank(pp_group), group=pp_group)
        if topk_logprobs > 0:
            dist.broadcast(all_topk_logprobs, src=get_pp_last_rank(pp_group), group=pp_group)
            dist.broadcast(all_topk_indices, src=get_pp_last_rank(pp_group), group=pp_group)
    if topk_logprobs > 0:
        return logprobs.cpu(), all_topk_logprobs.cpu(), all_topk_indices.cpu()
    return logprobs.cpu()


def _log_router_replay_correctness(iteration=None):
    """Log whether RouterReplay returned exactly the requested top-k indices."""
    from megatron.core.transformer.moe.router_replay import RouterReplay

    stats = RouterReplay.get_global_replay_stats()
    if not stats:
        return

    local_rows = []
    for layer_idx, layer_stats in enumerate(stats):
        local_rows.append(
            [
                float(layer_stats["matching_slots"]),
                float(layer_stats["total_slots"]),
                float(layer_stats["exact_tokens"]),
                float(layer_stats["total_tokens"]),
                float(layer_stats["shape_mismatches"]),
                float(layer_stats["natural_matching_slots"]),
                float(layer_stats["natural_total_slots"]),
                float(layer_stats["natural_exact_tokens"]),
                float(layer_stats["natural_total_tokens"]),
                float(layer_stats["natural_shape_mismatches"]),
                float(layer_idx),
            ]
        )
    counts = torch.tensor(local_rows, dtype=torch.float64, device=torch.cuda.current_device())
    if dist.is_initialized():
        # Do not reduce layer_idx (last column); restore after summing counts.
        layer_ids = counts[:, 10].clone()
        counts[:, 10].zero_()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        counts[:, 10] = layer_ids

    counts_cpu = counts.cpu()
    total_slots = float(counts_cpu[:, 1].sum().item())
    total_tokens = float(counts_cpu[:, 3].sum().item())
    if total_tokens == 0:
        return

    matching_slots = float(counts_cpu[:, 0].sum().item())
    exact_tokens = float(counts_cpu[:, 2].sum().item())
    shape_mismatches = int(counts_cpu[:, 4].sum().item())
    slot_rate = matching_slots / total_slots if total_slots else 0.0
    exact_rate = exact_tokens / total_tokens
    natural_total_slots = float(counts_cpu[:, 6].sum().item())
    natural_total_tokens = float(counts_cpu[:, 8].sum().item())
    natural_matching_slots = float(counts_cpu[:, 5].sum().item())
    natural_exact_tokens = float(counts_cpu[:, 7].sum().item())
    natural_shape_mismatches = int(counts_cpu[:, 9].sum().item())
    natural_slot_rate = (
        natural_matching_slots / natural_total_slots if natural_total_slots else 0.0
    )
    natural_exact_rate = (
        natural_exact_tokens / natural_total_tokens if natural_total_tokens else 0.0
    )

    print_rank_0(
        "[Router-replay] correctness: "
        f"exact_token_match={exact_rate:.6f} "
        f"slot_match={slot_rate:.6f} "
        f"tokens={int(total_tokens)} "
        f"shape_mismatches={shape_mismatches}"
    )
    print_rank_0(
        "[Router-diag] natural routing without replay: "
        f"exact_token_match={natural_exact_rate:.6f} "
        f"slot_match={natural_slot_rate:.6f} "
        f"tokens={int(natural_total_tokens)} "
        f"shape_mismatches={natural_shape_mismatches}"
    )

    metrics = {
        "router_replay/exact_token_match": exact_rate,
        "router_replay/slot_match": slot_rate,
        "router_replay/replayed_tokens": total_tokens,
        "router_replay/shape_mismatches": shape_mismatches,
        "router_diag/replayed_inf_train_mean": exact_rate,
        "router_diag/replayed_inf_train_slot_match": slot_rate,
        "router_diag/natural_inf_train_mean": natural_exact_rate,
        "router_diag/natural_inf_train_slot_match": natural_slot_rate,
        "router_diag/natural_inf_train_tokens": natural_total_tokens,
        "router_diag/natural_inf_train_shape_mismatches": natural_shape_mismatches,
    }
    for row in counts_cpu:
        layer_tokens = float(row[3].item())
        natural_layer_tokens = float(row[8].item())
        if layer_tokens == 0:
            continue
        layer_idx = int(row[10].item())
        layer_slots = float(row[1].item())
        metrics[f"router_replay/layer_{layer_idx}_exact_token_match"] = float(row[2].item()) / layer_tokens
        metrics[f"router_replay/layer_{layer_idx}_slot_match"] = (
            float(row[0].item()) / layer_slots if layer_slots else 0.0
        )
        metrics[f"router_replay/layer_{layer_idx}_replayed_tokens"] = layer_tokens
        metrics[f"router_diag/replayed_inf_train_layer_{layer_idx}"] = (
            float(row[2].item()) / layer_tokens
        )
        if natural_layer_tokens > 0:
            natural_layer_slots = float(row[6].item())
            metrics[f"router_diag/natural_inf_train_layer_{layer_idx}"] = (
                float(row[7].item()) / natural_layer_tokens
            )
            metrics[f"router_diag/natural_inf_train_layer_{layer_idx}_slot_match"] = (
                float(row[5].item()) / natural_layer_slots if natural_layer_slots else 0.0
            )

    if dist.is_initialized():
        # WandB is owned by the last global rank, while replay stats are printed on rank 0.
        # Broadcast rank-0's payload so the writer rank can log the same values.
        payload = [metrics if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        metrics = payload[0]

    wandb_writer = get_wandb_writer()
    if wandb_writer is not None and iteration is not None and metrics:
        wandb_writer.log(metrics, step=iteration)


# ---------------------------------------------------------------------------
# Router diagnostic helpers
# ---------------------------------------------------------------------------

def _summarize_router_scores(router):
    """Summarize router scores used for expert selection for one forward call."""
    logits = getattr(router, "_last_logits", None)
    if logits is None:
        return None

    logits = logits.detach().view(-1, router.config.num_moe_experts)
    score_function = router.config.moe_router_score_function
    if score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if getattr(router, "expert_bias", None) is not None:
            scores_for_routing = scores + router.expert_bias.detach().to(
                device=scores.device, dtype=scores.dtype
            )
        else:
            scores_for_routing = scores
    elif score_function == "softmax":
        if router.config.moe_router_pre_softmax:
            scores_for_routing = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        else:
            scores_for_routing = logits
    else:
        return None

    if scores_for_routing.numel() == 0:
        return None

    num_tokens, num_experts = scores_for_routing.shape
    topk = router.config.moe_router_topk
    flat_scores = scores_for_routing.float().reshape(-1)
    sorted_scores, _ = torch.sort(scores_for_routing.float(), dim=-1, descending=True)
    top1_margin = sorted_scores[:, 0] - sorted_scores[:, 1] if num_experts > 1 else None
    if num_experts > topk:
        boundary_margin = sorted_scores[:, topk - 1] - sorted_scores[:, topk]
    else:
        boundary_margin = None

    score_probs = torch.softmax(scores_for_routing.float(), dim=-1)
    normalized_entropy = (
        -(score_probs * torch.log(score_probs + 1e-20)).sum(dim=-1) / math.log(num_experts)
    )
    topk_mass = torch.topk(score_probs, k=min(topk, num_experts), dim=-1).values.sum(dim=-1)

    def _quantiles(tensor, qs):
        return torch.quantile(tensor.float(), torch.tensor(qs, device=tensor.device)).detach().cpu()

    summary = {
        "score_count": int(flat_scores.numel()),
        "score_sum": float(flat_scores.sum().item()),
        "score_sumsq": float((flat_scores * flat_scores).sum().item()),
        "score_min": float(flat_scores.min().item()),
        "score_max": float(flat_scores.max().item()),
        "score_quantiles": _quantiles(flat_scores, [0.01, 0.50, 0.99]).tolist(),
        "token_count": int(num_tokens),
        "entropy_sum": float(normalized_entropy.sum().item()),
        "topk_mass_sum": float(topk_mass.sum().item()),
    }
    if top1_margin is not None:
        summary["top1_margin_sum"] = float(top1_margin.sum().item())
        summary["top1_margin_quantiles"] = _quantiles(top1_margin, [0.10, 0.50]).tolist()
    if boundary_margin is not None:
        summary["boundary_margin_sum"] = float(boundary_margin.sum().item())
        summary["boundary_margin_quantiles"] = _quantiles(boundary_margin, [0.10, 0.50]).tolist()
    return summary


def _register_routing_hooks(model):
    """Register forward hooks on TopKRouter modules to capture routing diagnostics.

    Returns (store, handles).
      store[layer_idx] = list of [S, num_experts] bool tensors, one per micro-batch call.
      score_store[layer_idx] = list of compact router score summaries.
      handles = list of RemovableHandle objects.
    """
    routers = []
    m = model
    while hasattr(m, 'module'):
        m = m.module
    for _name, module in m.named_modules():
        if module.__class__.__name__ == 'TopKRouter':
            routers.append(module)

    store = [[] for _ in routers]
    score_store = [[] for _ in routers]
    handles = []
    for layer_idx, router in enumerate(routers):
        def _make_hook(li):
            def _hook(module, input, output):
                _, routing_map = output
                store[li].append(routing_map.bool().detach().cpu())
                score_summary = _summarize_router_scores(module)
                if score_summary is not None:
                    score_store[li].append(score_summary)
            return _hook
        handles.append(router.register_forward_hook(_make_hook(layer_idx)))
    return store, score_store, handles



def _routing_dump_key(prompt_tokens, generated_tokens):
    """Build a stable rollout key that disambiguates repeated completions."""
    return (
        tuple(int(x) for x in prompt_tokens),
        tuple(int(x) for x in generated_tokens),
    )


def _trajectory_routing_key(traj, gen_mask):
    """Build the same key from a padded training trajectory and generation mask."""
    gen_start = int(gen_mask.int().argmax().item())
    prompt_toks = traj[:gen_start].tolist()
    gen_toks = traj[gen_mask].tolist()
    return _routing_dump_key(prompt_toks, gen_toks)


def _routing_dump_npz_path(dump_dir, routing_dump_id):
    return os.path.join(dump_dir, f"rollout_{routing_dump_id}.npz")


def _load_exact_routing_dump(dump_dir, routing_dump_id, require_prompt=False, timeout_s=60.0):
    """Load one atomically written routing dump by its unique lightweight id."""
    if routing_dump_id is None:
        return None

    path = _routing_dump_npz_path(dump_dir, routing_dump_id)
    deadline = time.time() + timeout_s
    while not os.path.exists(path):
        if time.time() >= deadline:
            print_rank_0(f"[Router-replay] missing routing dump {path}")
            return None
        time.sleep(0.1)

    with np.load(path) as data:
        if "routing_indices" not in data:
            return None
        if require_prompt and "prompt_routing_indices" not in data:
            return None
        payload = {"routing_indices": data["routing_indices"].copy()}
        if "prompt_routing_indices" in data:
            payload["prompt_routing_indices"] = data["prompt_routing_indices"].copy()
        return payload


def _load_top_logprobs_from_npz(dump_dir, routing_dump_ids=None, timeout_s=60.0):
    if routing_dump_ids is None:
        return None

    result = []
    n_matched = 0
    for dump_id in routing_dump_ids:
        if dump_id is None:
            result.append(None)
            continue
        path = _routing_dump_npz_path(dump_dir, dump_id)
        deadline = time.time() + timeout_s
        while not os.path.exists(path):
            if time.time() >= deadline:
                break
            time.sleep(0.1)
        if not os.path.exists(path):
            result.append(None)
            continue
        with np.load(path) as data:
            if "generated_topk_tokens" not in data or "generated_topk_logprobs" not in data:
                result.append(None)
                continue
            tokens = data["generated_topk_tokens"]
            logprobs = data["generated_topk_logprobs"]
            rows = []
            for token_row, logprob_row in zip(tokens, logprobs):
                row = []
                for token, logprob in zip(token_row.tolist(), logprob_row.tolist()):
                    if token == "" or not np.isfinite(logprob):
                        continue
                    row.append({"token": str(token), "logprob": float(logprob)})
                rows.append(row)
            result.append(rows)
            n_matched += 1
    print_rank_0(
        f"[Logprob-mismatch] matched inference top-k for {n_matched}/{len(routing_dump_ids)} "
        f"local rollouts by routing_dump_id in {dump_dir}"
    )
    return result if n_matched > 0 else None


def _load_router_diag_from_npz(dump_dir, routing_dump_ids=None, trajs=None, generation_masks=None):
    """Load inference routing from npz dump files, matched by prompt and generated tokens."""
    if dist.is_initialized() and mpu.get_tensor_model_parallel_rank() != 0:
        return None

    if routing_dump_ids is not None:
        result = []
        n_matched = 0
        for dump_id in routing_dump_ids:
            payload = _load_exact_routing_dump(dump_dir, dump_id, require_prompt=False)
            if payload is None:
                result.append(None)
                continue
            n_matched += 1
            result.append(payload["routing_indices"])
        print_rank_0(
            f"[Router-diag] matched {n_matched}/{len(routing_dump_ids)} local rollouts "
            f"by routing_dump_id in {dump_dir}"
        )
        return result if n_matched > 0 else None

    if trajs is None or generation_masks is None:
        return None

    import glob as _glob
    npz_files = sorted(_glob.glob(os.path.join(dump_dir, "rollout_*.npz")))
    if not npz_files:
        print_rank_0(f"[Router-diag] no rollout_*.npz files found in {dump_dir}")
        return None

    routing_by_gentoks = {}
    n_collisions = 0
    for path in npz_files:
        data = np.load(path)
        if "routing_indices" not in data:
            continue
        if "prompt_tokens" not in data:
            continue
        key = _routing_dump_key(data["prompt_tokens"], data["generated_tokens"])
        if key in routing_by_gentoks:
            n_collisions += 1
        routing_by_gentoks[key] = data["routing_indices"]

    if not routing_by_gentoks:
        print_rank_0(
            f"[Router-diag] {len(npz_files)} npz files found but none contain routing_indices"
        )
        return None

    result = []
    n_matched = 0
    for seq_i in range(trajs.shape[0]):
        gen_mask = generation_masks[seq_i]
        key = _trajectory_routing_key(trajs[seq_i], gen_mask)
        rt = routing_by_gentoks.get(key)
        if rt is not None:
            n_matched += 1
        result.append(rt)

    print_rank_0(
        f"[Router-diag] matched {n_matched}/{trajs.shape[0]} local rollouts "
        f"from {len(npz_files)} npz files in {dump_dir}  key_collisions={n_collisions}"
    )
    return result if n_matched > 0 else None


def _load_routing_for_replay(dump_dir, routing_dump_ids=None, trajs=None, generation_masks=None):
    """Load full-sequence (prompt + generated) routing from npz files for replay.

    Unlike _load_router_diag_from_npz, this runs on ALL ranks (no TP-rank gate)
    because the result flows into the DataLoader which is identical across TP ranks.

    Returns a list of np.ndarray [P_i+G_i, L, top_k] or None per sequence,
    or None if no files were found / none had prompt_routing_indices.
    """
    if routing_dump_ids is not None:
        result = []
        n_matched = 0
        n_missing_prompt = 0
        for dump_id in routing_dump_ids:
            payload = _load_exact_routing_dump(dump_dir, dump_id, require_prompt=True)
            if payload is None:
                result.append(None)
                n_missing_prompt += 1
                continue
            n_matched += 1
            full = np.concatenate(
                [payload["prompt_routing_indices"], payload["routing_indices"]], axis=0
            )
            result.append(full)

        if n_missing_prompt > 0:
            print_rank_0(
                f"[Router-replay] WARNING: {n_missing_prompt} routing dumps were missing "
                f"or lacked prompt_routing_indices"
            )
        print_rank_0(
            f"[Router-replay] matched {n_matched}/{len(routing_dump_ids)} sequences "
            f"by routing_dump_id in {dump_dir}"
        )
        return result if n_matched > 0 else None

    if trajs is None or generation_masks is None:
        return None

    import glob as _glob
    npz_files = sorted(_glob.glob(os.path.join(dump_dir, "rollout_*.npz")))
    if not npz_files:
        print_rank_0(f"[Router-replay] no rollout_*.npz files found in {dump_dir}")
        return None

    routing_by_gentoks = {}
    n_collisions = 0
    n_missing_prompt = 0
    for path in npz_files:
        data = np.load(path)
        if "routing_indices" not in data:
            continue
        if "prompt_routing_indices" not in data:
            n_missing_prompt += 1
            continue
        if "prompt_tokens" not in data:
            n_missing_prompt += 1
            continue
        key = _routing_dump_key(data["prompt_tokens"], data["generated_tokens"])
        if key in routing_by_gentoks:
            n_collisions += 1
        full = np.concatenate([data["prompt_routing_indices"], data["routing_indices"]], axis=0)
        routing_by_gentoks[key] = full

    if n_missing_prompt > 0:
        print_rank_0(
            f"[Router-replay] WARNING: {n_missing_prompt} npz files lack prompt_routing_indices "
            f"(old format) — re-run inference to capture prompt routing for full-sequence replay"
        )
    if not routing_by_gentoks:
        print_rank_0(f"[Router-replay] no usable routing found in {len(npz_files)} npz files")
        return None

    result = []
    n_matched = 0
    for seq_i in range(trajs.shape[0]):
        gen_mask = generation_masks[seq_i]
        key = _trajectory_routing_key(trajs[seq_i], gen_mask)
        rt = routing_by_gentoks.get(key)
        if rt is not None:
            n_matched += 1
        result.append(rt)

    print_rank_0(
        f"[Router-replay] matched {n_matched}/{trajs.shape[0]} sequences "
        f"from {len(npz_files)} npz files in {dump_dir}  key_collisions={n_collisions}"
    )
    return result if n_matched > 0 else None


def _build_replay_routing_tensor(routing_list, trajs, seq_len):
    """Build padded routing tensor and replay mask for the DataLoader.

    Args:
        routing_list: list of np.ndarray [P_i+G_i, L, top_k] or None per sequence.
        trajs: [N, seq_len] token tensor (used only to get N).
        seq_len: padded sequence length.

    Returns:
        (routing_padded, replay_mask) where:
          routing_padded: [N, seq_len, L, top_k] int32 tensor
          replay_mask:    [N, seq_len] bool tensor — True for non-padding positions with routing
        or (None, None) if no sequences have routing.
    """
    first = next((r for r in routing_list if r is not None), None)
    if first is None:
        return None, None

    N = trajs.shape[0]
    num_layers = first.shape[1]
    top_k = first.shape[2]

    routing_padded = torch.zeros(N, seq_len, num_layers, top_k, dtype=torch.int32)
    replay_mask = torch.zeros(N, seq_len, dtype=torch.bool)

    for i, rt in enumerate(routing_list):
        if rt is None:
            continue
        seq_tokens = rt.shape[0]  # P_i + G_i
        if seq_tokens > seq_len:
            print_rank_0(
                f"[Router-replay] WARNING: sequence {i} routing length {seq_tokens} "
                f"exceeds seq_len {seq_len}, truncating"
            )
            seq_tokens = seq_len
        routing_padded[i, :seq_tokens] = torch.tensor(rt[:seq_tokens], dtype=torch.int32)
        replay_mask[i, :seq_tokens] = True

    return routing_padded, replay_mask


def _gather_sequence_parallel_routing_store(train_routing_store):
    """Gather per-TP sequence shards into full routing maps on TP rank 0."""
    if not dist.is_initialized() or mpu.get_tensor_model_parallel_world_size() == 1:
        return train_routing_store

    tp_group = mpu.get_tensor_model_parallel_group()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    gathered_store = [[] for _ in train_routing_store]
    for layer_idx, layer_data in enumerate(train_routing_store):
        for local_map in layer_data:
            gathered = [None] * tp_size
            dist.all_gather_object(gathered, local_map, group=tp_group)
            if tp_rank == 0:
                gathered_store[layer_idx].append(torch.cat(gathered, dim=0))
    return gathered_store if tp_rank == 0 else None


def _find_microbatch_position(train_routing_store, generation_masks, seq_i, pos):
    """Map a global sequence/token position into a flattened router hook row."""
    seq_len = generation_masks.shape[1]
    seq_cursor = 0
    for microbatch_idx, first_layer_map in enumerate(train_routing_store[0]):
        rows = first_layer_map.shape[0]
        if rows % seq_len != 0:
            # Fallback for legacy microbatch=1 / sharded diagnostics.
            return microbatch_idx, pos
        batch_size = rows // seq_len
        if seq_i < seq_cursor + batch_size:
            seq_in_microbatch = seq_i - seq_cursor
            return microbatch_idx, pos * batch_size + seq_in_microbatch
        seq_cursor += batch_size
    return None, None


def _log_router_diag(
    train_routing_store,
    inference_routing,
    generation_masks,
    iteration=None,
    sequence_parallel=False,
):
    """Compare per-token inference routing to training routing.

    Must be called on every TP-rank-0 node (all DP ranks).  Gathers agree_vals
    across the data-parallel group before printing so the stats cover all 512
    rollouts, not just the local DP shard.

    Returns a dict of raw data for wandb logging on rank 0 only; returns None on
    all other TP-rank-0 ranks.  Non-TP-rank-0 ranks never call this function.
    The caller is responsible for broadcasting the result and calling
    _wandb_log_router_metrics() on all world ranks.

    Args:
        train_routing_store: store[layer_idx][micro_batch_i] = [S, num_experts] bool.
        inference_routing: list (one per local sequence) of [G', L, top_k] int arrays
            or None entries for unmatched sequences.  Pass [] when nothing matched.
        generation_masks: [N, seq_length] bool tensor — True for generated token positions.
        iteration: current training iteration for WandB logging (None = skip WandB).
    """
    _rank = dist.get_rank() if dist.is_initialized() else 0
    if sequence_parallel:
        train_routing_store = _gather_sequence_parallel_routing_store(train_routing_store)
        if train_routing_store is None:
            return None

    n_layers = len(train_routing_store)
    print(f"[Router-diag] rank {_rank}: _log_router_diag start  n_layers={n_layers}  "
          f"n_inf_seqs={len(inference_routing) if inference_routing else 0}", flush=True)

    # Compute local agreement values (may be empty for unmatched shards).
    # Each element is (layer_idx, agree_float) so we can compute per-layer stats.
    agree_pairs = []  # list of (layer_idx, 1.0|0.0)
    if n_layers > 0:
        for seq_i, inf_rt_raw in enumerate(inference_routing or []):
            if inf_rt_raw is None:
                continue
            inf_rt = torch.tensor(inf_rt_raw, dtype=torch.int32)  # [G', L, top_k]
            G_prime, L = inf_rt.shape[0], inf_rt.shape[1]
            if G_prime == 0:
                continue

            gen_mask = generation_masks[seq_i]   # [seq_length] bool
            gen_start = int(gen_mask.int().argmax().item())  # first generated token position

            for l in range(min(L, n_layers)):
                microbatch_idx, row_pos = _find_microbatch_position(
                    train_routing_store, generation_masks, seq_i, gen_start
                )
                if microbatch_idx is None:
                    continue
                train_map = train_routing_store[l][microbatch_idx]  # [S*B, num_experts] bool
                S = train_map.shape[0]
                # routing_indices[d] is routing for generated token g_d as the input token.
                # Inference stores routing for each decode step contiguously: inf_rt[d, l, :].
                for d in range(G_prime):
                    _microbatch_idx, row_pos = _find_microbatch_position(
                        train_routing_store, generation_masks, seq_i, gen_start + d
                    )
                    if _microbatch_idx != microbatch_idx or row_pos is None or row_pos >= S:
                        continue
                    train_experts = set(train_map[row_pos].nonzero(as_tuple=True)[0].tolist())
                    inf_experts = set(inf_rt[d, l].tolist())
                    agree_pairs.append((l, 1.0 if train_experts == inf_experts else 0.0))

    print(f"[Router-diag] rank {_rank}: local agree_pairs computed  n={len(agree_pairs)}", flush=True)

    # Always gather across the data-parallel group — every TP-rank-0 node must
    # participate unconditionally to avoid hangs when some shards have 0 matches.
    if dist.is_initialized():
        dp_group = mpu.get_data_parallel_group()
        dp_size = dist.get_world_size(dp_group)
        print(f"[Router-diag] rank {_rank}: entering all_gather_object  dp_size={dp_size}", flush=True)
        all_pair_lists = [None] * dp_size
        dist.all_gather_object(all_pair_lists, agree_pairs, group=dp_group)
        print(f"[Router-diag] rank {_rank}: all_gather_object done", flush=True)
        if dist.get_rank() != 0:
            return None
        agree_pairs = [p for lst in all_pair_lists for p in lst]

    if not agree_pairs:
        print_rank_0("[Router-diag] no comparable token positions found")
        return None

    agree_layer_ids = [p[0] for p in agree_pairs]
    agree_vals = [p[1] for p in agree_pairs]

    agree_t = torch.tensor(agree_vals, dtype=torch.float32)
    n = len(agree_t)
    mean_agree = agree_t.mean().item()
    p50_agree = agree_t.median().item()
    #p5_agree = agree_t.float().quantile(0.05).item()
    #p95_agree = agree_t.float().quantile(0.95).item()

    # Per-layer agreement means.
    layer_agree: dict[int, list[float]] = {}
    for l, v in zip(agree_layer_ids, agree_vals):
        layer_agree.setdefault(l, []).append(v)
    layer_means = {l: float(np.mean(vs)) for l, vs in sorted(layer_agree.items())}

    print_rank_0(
        f"[Router-diag] inf vs train routing agree: "
        f"mean={mean_agree:.3f}  p50={p50_agree:.3f}  " #p5={p5_agree:.3f}  p95={p95_agree:.3f}  "
        f"n={n}  n_layers={n_layers}"
    )

    # Return raw data for the caller to broadcast and log.  WandB logging is
    # handled by _wandb_log_router_metrics() after a world-group broadcast.
    return {
        'scalars': {
            'router_diag/inf_train_mean': mean_agree,
            #'router_diag/inf_train_p5': p5_agree,
            'router_diag/inf_train_p50': p50_agree,
            #'router_diag/inf_train_p95': p95_agree,
            **{f'router_diag/inf_train_layer_{l}': lmean for l, lmean in layer_means.items()},
        },
        'chart': {'layer_means': layer_means, 'mean_agree': mean_agree},
    }


def _log_expert_load(routing_store, iteration):
    """Compute per-layer expert load distribution.

    Must be called on every TP-rank-0 node (all DP ranks). Gathers token counts across
    the data-parallel group so the load fractions reflect all rollouts.

    Returns a dict of raw data for wandb logging on rank 0 only; returns None on
    all other TP-rank-0 ranks.  The caller is responsible for broadcasting the
    result and calling _wandb_log_router_metrics() on all world ranks.

    Args:
        routing_store: store[layer_idx][micro_batch_i] = [S, num_experts] bool tensor.
        iteration: current training iteration for WandB step.
    """
    _rank = dist.get_rank() if dist.is_initialized() else 0
    if iteration is None:
        print(f"[Router-diag] rank {_rank}: _log_expert_load skipped (iteration=None)", flush=True)
        return None

    n_layers = len(routing_store)
    print(f"[Router-diag] rank {_rank}: _log_expert_load start  n_layers={n_layers}", flush=True)

    # Accumulate local per-layer expert token counts and total token counts.
    local_counts: list = []  # [n_layers] each is [n_experts] float array or None
    local_totals: list = []  # [n_layers] each is int

    for li, layer_data in enumerate(routing_store):
        if not layer_data:
            local_counts.append(None)
            local_totals.append(0)
            continue
        print(f"[Router-diag] rank {_rank}: layer {li} torch.cat n_microbatches={len(layer_data)}", flush=True)
        all_maps = torch.cat(layer_data, dim=0).float()  # [total_S, n_experts]
        local_counts.append(all_maps.sum(dim=0).numpy())
        local_totals.append(int(all_maps.shape[0]))

    print(f"[Router-diag] rank {_rank}: local load computed  "
          f"n_nonempty={sum(c is not None for c in local_counts)}", flush=True)

    # All-gather across the data-parallel group (unconditional to avoid hangs).
    if dist.is_initialized():
        dp_group = mpu.get_data_parallel_group()
        dp_size = dist.get_world_size(dp_group)
        print(f"[Router-diag] rank {_rank}: entering counts all_gather_object  dp_size={dp_size}", flush=True)
        all_rank_counts = [None] * dp_size
        dist.all_gather_object(all_rank_counts, local_counts, group=dp_group)
        print(f"[Router-diag] rank {_rank}: counts all_gather_object done", flush=True)

        print(f"[Router-diag] rank {_rank}: entering totals all_gather_object", flush=True)
        all_rank_totals = [None] * dp_size
        dist.all_gather_object(all_rank_totals, local_totals, group=dp_group)
        print(f"[Router-diag] rank {_rank}: totals all_gather_object done", flush=True)

        if dist.get_rank() != 0:
            return None
        global_counts = []
        global_totals = []
        for l in range(n_layers):
            combined_c = None
            combined_t = 0
            for rc, rt in zip(all_rank_counts, all_rank_totals):
                c = rc[l]
                if c is not None:
                    combined_c = c if combined_c is None else combined_c + c
                combined_t += rt[l]
            global_counts.append(combined_c)
            global_totals.append(combined_t)
    else:
        global_counts = local_counts
        global_totals = local_totals

    # Convert to load fractions.
    layer_loads = []
    valid_layers = []
    for l, (counts, total) in enumerate(zip(global_counts, global_totals)):
        if counts is None or total == 0:
            continue
        layer_loads.append(counts / total)
        valid_layers.append(l)

    if not layer_loads:
        print_rank_0("[Router-diag] _log_expert_load: no valid layers found after gather")
        return None

    loads_matrix = np.stack(layer_loads)          # [n_valid_layers, n_experts]
    n_valid_layers, n_experts = loads_matrix.shape
    load_entropy = -np.sum(loads_matrix * np.log(loads_matrix + 1e-10), axis=1)
    max_load = loads_matrix.max(axis=1)

    print_rank_0(
        f"[Router-diag] expert load: "
        f"max_load_mean={float(max_load.mean()):.3f}  "
        f"entropy_mean={float(load_entropy.mean()):.4f}  "
        f"n_layers={n_valid_layers}  n_experts={n_experts}"
    )

    # Return raw data for the caller to broadcast and log.
    scalars: dict = {}
    for i, l in enumerate(valid_layers):
        scalars[f'router/layer_{l}_load_entropy'] = float(load_entropy[i])
        scalars[f'router/layer_{l}_max_expert_load'] = float(max_load[i])

    return {
        'scalars': scalars,
        'chart': {
            'loads_matrix': loads_matrix.tolist(),
            'valid_layers': valid_layers,
        },
    }


def _combine_router_score_summaries(score_store):
    """Combine compact per-call router score summaries into scalar metrics."""
    summaries = [summary for layer in score_store for summary in layer]
    if not summaries:
        return None

    score_count = sum(s["score_count"] for s in summaries)
    token_count = sum(s["token_count"] for s in summaries)
    if score_count == 0 or token_count == 0:
        return None

    score_sum = sum(s["score_sum"] for s in summaries)
    score_sumsq = sum(s["score_sumsq"] for s in summaries)
    score_mean = score_sum / score_count
    score_var = max(0.0, score_sumsq / score_count - score_mean * score_mean)
    score_quantiles = np.array([s["score_quantiles"] for s in summaries], dtype=np.float64)
    token_weights = np.array([s["token_count"] for s in summaries], dtype=np.float64)
    token_weights = token_weights / token_weights.sum()

    metrics = {
        "router_scores/scores_for_routing_mean": float(score_mean),
        "router_scores/scores_for_routing_std": float(math.sqrt(score_var)),
        "router_scores/scores_for_routing_min": float(min(s["score_min"] for s in summaries)),
        "router_scores/scores_for_routing_max": float(max(s["score_max"] for s in summaries)),
        "router_scores/scores_for_routing_p01": float((score_quantiles[:, 0] * token_weights).sum()),
        "router_scores/scores_for_routing_p50": float((score_quantiles[:, 1] * token_weights).sum()),
        "router_scores/scores_for_routing_p99": float((score_quantiles[:, 2] * token_weights).sum()),
        "router_scores/normalized_score_entropy_mean": float(
            sum(s["entropy_sum"] for s in summaries) / token_count
        ),
        "router_scores/topk_mass_mean": float(
            sum(s["topk_mass_sum"] for s in summaries) / token_count
        ),
    }

    top1_summaries = [s for s in summaries if "top1_margin_sum" in s]
    if top1_summaries:
        top1_weights = np.array([s["token_count"] for s in top1_summaries], dtype=np.float64)
        top1_weights = top1_weights / top1_weights.sum()
        top1_q = np.array([s["top1_margin_quantiles"] for s in top1_summaries], dtype=np.float64)
        top1_tokens = sum(s["token_count"] for s in top1_summaries)
        metrics.update(
            {
                "router_scores/top1_margin_mean": float(
                    sum(s["top1_margin_sum"] for s in top1_summaries) / top1_tokens
                ),
                "router_scores/top1_margin_p10": float((top1_q[:, 0] * top1_weights).sum()),
                "router_scores/top1_margin_p50": float((top1_q[:, 1] * top1_weights).sum()),
            }
        )

    boundary_summaries = [s for s in summaries if "boundary_margin_sum" in s]
    if boundary_summaries:
        boundary_weights = np.array([s["token_count"] for s in boundary_summaries], dtype=np.float64)
        boundary_weights = boundary_weights / boundary_weights.sum()
        boundary_q = np.array(
            [s["boundary_margin_quantiles"] for s in boundary_summaries], dtype=np.float64
        )
        boundary_tokens = sum(s["token_count"] for s in boundary_summaries)
        metrics.update(
            {
                "router_scores/topk_boundary_margin_mean": float(
                    sum(s["boundary_margin_sum"] for s in boundary_summaries) / boundary_tokens
                ),
                "router_scores/topk_boundary_margin_p10": float(
                    (boundary_q[:, 0] * boundary_weights).sum()
                ),
                "router_scores/topk_boundary_margin_p50": float(
                    (boundary_q[:, 1] * boundary_weights).sum()
                ),
            }
        )

    return metrics


def _log_router_score_stats(score_store, iteration):
    """Gather and summarize router score statistics for WandB logging."""
    _rank = dist.get_rank() if dist.is_initialized() else 0
    if iteration is None:
        print(f"[Router-diag] rank {_rank}: _log_router_score_stats skipped (iteration=None)", flush=True)
        return None

    local_summaries = [summary for layer in score_store for summary in layer]
    if dist.is_initialized():
        dp_group = mpu.get_data_parallel_group()
        all_rank_summaries = [None] * dist.get_world_size(dp_group)
        dist.all_gather_object(all_rank_summaries, local_summaries, group=dp_group)
        if dist.get_rank() != 0:
            return None
        combined_store = [[s for rank_summaries in all_rank_summaries for s in rank_summaries]]
    else:
        combined_store = [local_summaries]

    scalars = _combine_router_score_summaries(combined_store)
    if scalars is None:
        print_rank_0("[Router-diag] no router score stats found")
        return None

    print_rank_0(
        "[Router-diag] router score stats: "
        f"score_p50={scalars['router_scores/scores_for_routing_p50']:.4f} "
        f"score_p99={scalars['router_scores/scores_for_routing_p99']:.4f} "
        f"entropy={scalars['router_scores/normalized_score_entropy_mean']:.4f} "
        f"topk_margin_p50={scalars.get('router_scores/topk_boundary_margin_p50', float('nan')):.4f}"
    )
    return {"scalars": scalars}


def _wandb_log_router_metrics(diag_data, expert_data, score_data, iteration):
    """Log router diagnostics to WandB.  Call on ALL world ranks after a broadcast.

    Only the rank that holds the WandB writer (rank world_size-1 by Megatron
    convention) will actually log; all others return immediately.  This mirrors
    the pattern used by maybe_log_training_metrics().

    Args:
        diag_data:   return value of _log_router_diag() after world-group broadcast.
        expert_data: return value of _log_expert_load() after world-group broadcast.
        score_data:  return value of _log_router_score_stats() after world-group broadcast.
        iteration:   current training step.
    """
    wandb_writer = get_wandb_writer()
    if wandb_writer is None or iteration is None:
        return

    metrics: dict = {}

    if diag_data is not None:
        metrics.update(diag_data['scalars'])
        chart = diag_data['chart']
        try:
            import matplotlib.pyplot as plt
            import wandb as _wandb
            plt.switch_backend('agg')
            layer_means = chart['layer_means']
            mean_agree = chart['mean_agree']
            layers_sorted = sorted(layer_means.keys())
            fig, ax = plt.subplots(figsize=(max(6, len(layers_sorted) * 0.45), 4))
            ax.bar(layers_sorted, [layer_means[l] for l in layers_sorted])
            ax.axhline(mean_agree, color='red', linestyle='--', linewidth=1,
                       label=f'mean={mean_agree:.3f}')
            ax.set_ylim(0, 1)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Agreement rate')
            ax.set_title(f'Inf vs Train Routing Agreement by Layer (iter {iteration})')
            ax.legend(fontsize=8)
            fig.tight_layout()
            metrics['router_diag/per_layer_agree_chart'] = _wandb.Image(fig)
            plt.close(fig)
        except Exception as e:
            print_rank_0(f"[Router-diag] agreement chart creation failed: {e}")

    if expert_data is not None:
        metrics.update(expert_data['scalars'])
        chart = expert_data['chart']
        try:
            import matplotlib.pyplot as plt
            import wandb as _wandb
            plt.switch_backend('agg')
            loads_matrix = np.array(chart['loads_matrix'])
            valid_layers = chart['valid_layers']
            n_valid_layers, n_experts = loads_matrix.shape
            fig_w = max(8.0, n_experts * 0.25)
            fig_h = max(3.0, n_valid_layers * 0.35)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            # Use a fixed scale relative to uniform load so heatmaps are comparable
            # over time without compressing normal load variation near zero.
            uniform_load = 1.0 / n_experts
            vmax = min(1.0, uniform_load * 16.0)
            im = ax.imshow(loads_matrix, aspect='auto', cmap='hot_r', vmin=0.0, vmax=vmax)
            ax.set_xlabel('Expert ID')
            ax.set_ylabel('Layer')
            ax.set_xticks(np.arange(0, n_experts, max(1, n_experts // 16)))
            ax.set_yticks(range(n_valid_layers))
            ax.set_yticklabels(valid_layers)
            ax.set_title(f'Expert Load Distribution (iter {iteration})')
            plt.colorbar(
                im,
                ax=ax,
                label=f'Fraction of tokens routed (vmax={vmax:.3f})',
                extend='max',
            )
            fig.tight_layout()
            metrics['router/expert_load_heatmap'] = _wandb.Image(fig)
            plt.close(fig)
     
        except Exception as e:
            print_rank_0(f"[Router-diag] expert load heatmap creation failed: {e}")

    if score_data is not None:
        metrics.update(score_data['scalars'])

    if metrics:
        print_rank_0("[Router-diag] logging router metrics to wandb")
        wandb_writer.log(metrics, step=iteration)
        print_rank_0("[Router-diag] router metrics logged")


def _maybe_write_determinism_probe_targets(
    *,
    old_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor | None,
    generation_masks: torch.Tensor | None,
    trajs: torch.Tensor | None,
    turn_metadata: list[dict[str, Any]] | None,
    iteration: int,
    train_topk_logprobs: torch.Tensor | None = None,
    train_topk_indices: torch.Tensor | None = None,
) -> None:
    """Write worst train-vs-inference logprob token keys for a later targeted probe run."""
    args = get_args()
    output_path = getattr(args, "rl_determinism_probe_write_targets_file", None)
    if getattr(args, "rl_determinism_probe_targets_file", None):
        # Targeted probe runs consume a previously discovered key set. Do not
        # overwrite that file with the current run's targets after probe logging.
        return
    if not output_path or inference_logprobs is None or generation_masks is None or trajs is None:
        return

    max_targets = int(getattr(args, "rl_determinism_probe_num_targets", 64) or 0)
    if max_targets <= 0:
        return

    old_cpu = old_logprobs.detach().float().cpu()
    inf_cpu = inference_logprobs.detach().float().cpu()
    masks_cpu = generation_masks.detach().cpu().bool()
    trajs_cpu = trajs.detach().cpu()
    target_mask = masks_cpu[:, 1:].clone()
    target_mask &= torch.isfinite(old_cpu) & torch.isfinite(inf_cpu)
    if not target_mask.any():
        return

    gen_offsets_by_token = masks_cpu.long().cumsum(dim=1) - 1
    deltas = (old_cpu - inf_cpu).abs()
    deltas = deltas.masked_fill(~target_mask, float("-inf"))

    target_selection = getattr(args, "rl_determinism_probe_target_selection", "top_logprob_delta")
    min_margin = float(getattr(args, "rl_determinism_probe_target_min_margin", 0.5))
    tokenizer = get_tokenizer()

    def _build_target(seq_index, logprob_index, score, extra=None):
        token_index = int(logprob_index) + 1
        metadata = (
            turn_metadata[seq_index]
            if turn_metadata is not None and seq_index < len(turn_metadata)
            else {}
        )
        routing_dump_id = metadata.get("routing_dump_id")
        if routing_dump_id is None:
            return None
        old_lp = float(old_cpu[seq_index, logprob_index].item())
        inf_lp = float(inf_cpu[seq_index, logprob_index].item())
        result = {
            "iteration": int(iteration),
            "routing_dump_id": str(routing_dump_id),
            "target_token_index": token_index,
            "prefix_hash": hash_token_ids(trajs_cpu[seq_index, :token_index].tolist()),
            "gen_offset": int(gen_offsets_by_token[seq_index, token_index].item()),
            "target_token_id": int(trajs_cpu[seq_index, token_index].item()),
            "seq_index": int(seq_index),
            "global_rollout_index": metadata.get("global_rollout_index"),
            "group_index": metadata.get("group_index"),
            "rollout_index": metadata.get("rollout_index"),
            "turn_index": metadata.get("turn_index"),
            "old_logprob": old_lp,
            "inference_logprob": inf_lp,
            "abs_logprob_delta": abs(old_lp - inf_lp),
            "prob_abs_diff": abs(math.exp(old_lp) - math.exp(inf_lp)),
            "target_selection_score": float(score),
            "target_selection": target_selection,
        }
        if extra:
            result.update(extra)
        return result

    k = min(max_targets, int(target_mask.sum().item()))
    candidate_pool = k
    if target_selection == "top1_disagreement_high_margin":
        # Avoid scanning every generated token in Python: inspect a large pool of
        # high logprob-delta candidates, then keep the high-margin top-1 flips.
        candidate_pool = min(max(k * 100, 10000), int(target_mask.sum().item()))
    top_values, top_flat_indices = torch.topk(deltas.flatten(), k=candidate_pool)
    seq_indices = torch.div(top_flat_indices, deltas.shape[1], rounding_mode="floor")
    logprob_indices = top_flat_indices % deltas.shape[1]

    local_targets = []
    for rank_idx in range(candidate_pool):
        seq_index = int(seq_indices[rank_idx].item())
        logprob_index = int(logprob_indices[rank_idx].item())
        if target_selection == "top1_disagreement_high_margin":
            if train_topk_logprobs is None or train_topk_indices is None:
                continue
            train_row_lps = train_topk_logprobs[seq_index, logprob_index].detach().float().cpu()
            train_row_ids = train_topk_indices[seq_index, logprob_index].detach().cpu()
            if train_row_lps.numel() < 2 or train_row_ids.numel() < 2:
                continue
            train_top1 = _detokenize_single_token(tokenizer, int(train_row_ids[0].item()))
            train_top2 = _detokenize_single_token(tokenizer, int(train_row_ids[1].item()))
            train_margin = float((train_row_lps[0] - train_row_lps[1]).item())
            metadata = (
                turn_metadata[seq_index]
                if turn_metadata is not None and seq_index < len(turn_metadata)
                else {}
            )
            inf_rows = metadata.get("inference_top_logprobs")
            gen_offset = int(gen_offsets_by_token[seq_index, logprob_index + 1].item())
            if inf_rows is None or gen_offset < 0 or gen_offset >= len(inf_rows):
                continue
            inf_tokens, inf_lps = _normalize_inference_top_logprobs(inf_rows[gen_offset])
            inf_margin = _top2_margin(inf_lps)
            if not inf_tokens or inf_margin is None:
                continue
            inf_top1 = inf_tokens[0]
            if train_top1 == inf_top1:
                continue
            if train_margin < min_margin or float(inf_margin) < min_margin:
                continue
            score = min(train_margin, float(inf_margin)) * max(float(top_values[rank_idx].item()), 0.0)
            extra = {
                "train_top1_token": train_top1,
                "train_top2_token": train_top2,
                "train_top1_logprob": float(train_row_lps[0].item()),
                "train_top2_logprob": float(train_row_lps[1].item()),
                "train_top1_top2_margin": train_margin,
                "inference_top1_token": inf_top1,
                "inference_top2_token": inf_tokens[1] if len(inf_tokens) > 1 else None,
                "inference_top1_logprob": float(inf_lps[0]) if inf_lps else None,
                "inference_top2_logprob": float(inf_lps[1]) if len(inf_lps) > 1 else None,
                "inference_top1_top2_margin": float(inf_margin),
            }
            target = _build_target(seq_index, logprob_index, score, extra)
        else:
            target = _build_target(seq_index, logprob_index, float(top_values[rank_idx].item()))
        if target is not None:
            local_targets.append(target)
            if len(local_targets) >= max_targets:
                break

    if target_selection == "top1_disagreement_high_margin" and len(local_targets) < max_targets:
        print_rank_0(
            f"[RL-determinism-probe] selected {len(local_targets)}/{max_targets} "
            f"high-margin top-1 disagreement targets from {candidate_pool} candidates"
        )

    all_targets = sorted(
        local_targets, key=lambda item: item["target_selection_score"], reverse=True
    )[:max_targets]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        rank = dist.get_rank()
        path = path.with_name(f"{path.stem}.rank{rank:04d}{path.suffix}")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "iteration": int(iteration),
        "selection": "top_abs_train_inference_logprob_delta",
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "num_targets": len(all_targets),
        "targets": all_targets,
    }
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print_rank_0(
            f"[RL-determinism-probe] wrote per-rank target token shards under {path.parent}"
        )


def prepare_data_for_update(
    model: list[LanguageModule],
    ref_state_dict: Dict[str, Any],
    rollouts: GroupedRollouts,
    tokenizer: MegatronTokenizer,
    sequence_packing: bool,
    is_correction: bool,
    iteration: int = 0,
) -> tuple[RerunDataIterator, RolloutStats, dict]:
    """Extract data for the update from raw rollouts.

    Args:
        model: Current policy as the zero-eth element.
        ref_state_dict: Reference policy state dict.
        rollouts: Rollouts to extract the data from.
        tokenizer: Tokenizer to pad/tokenize data.
        sequence_packing: Use sequence packing if True.
        is_correction: Prepare data for IS correction if True.
        iteration: Current training iteration, used for WandB logging.

    Returns:
        Tuple of (cycled iterator over dataset batches, group stats, example groups per env).
    """
    args = get_args()
    nvtx_range = get_nvtx_range()
    runtime_state = get_rl_runtime_state()

    if args.cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
        lang_module = (
            model[0].module.module if hasattr(model[0].module, "module") else model[0].module
        )
        toggle_cuda_graphs(lang_module, "none")

    model = model[0]
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    with nvtx_range("prepare-data-for-update"):
        with nvtx_range("compute-group-stats"):
            group_stats = compute_group_stats(rollouts, tokenizer, args.seq_length, skip_adv_std_normalization = args.rl_skip_advantage_std_normalization, advantage_baseline_type=args.advantage_baseline_type)
            # TODO(vitalyk): why do we need global_advantages here? go inside packing
            advantages = global_advantages = torch.tensor(group_stats.advantages, dtype=dtype).cuda()

        # Now split the rollouts across the data parallel ranks for training
        # This needs to be done at this point because we are about to calculate logprobs
        # Note :- For EP, do not use the expert data parallel group here. Always 
        # use the regular data parallel group. 

        # Get example group per environment to log their rollouts.
        example_groups = {}
        for g in rollouts:
            if g[0].env_id not in example_groups:
                example_groups[g[0].env_id] = g

        # Let's expand rollouts getting rid of the groups.
        # We need this to correctly split the rollouts across dp groups.
        # Keep lightweight metadata aligned to flattened rollouts for diagnostics.
        flattened_rollouts = []
        flattened_rollout_metadata = []
        for group_idx, group in enumerate(rollouts):
            for rollout_idx, rollout in enumerate(group):
                flattened_rollouts.append(rollout)
                flattened_rollout_metadata.append(
                    {
                        "group_index": group_idx,
                        "rollout_index": rollout_idx,
                        "global_rollout_index": len(flattened_rollouts) - 1,
                    }
                )
        rollouts = flattened_rollouts
        num_turns = [nt for g in group_stats.num_turns for nt in g]
        total_turns_sampled = len(rollouts)

        # We might sample more than we consume in one step.
        samples_ratio_per_step = args.global_batch_size / (args.grpo_prompts_per_step * args.grpo_group_size)
        assert samples_ratio_per_step <= 1, "You cannot use more data than you sampled."

        if (data_parallel_world_size := mpu.get_data_parallel_world_size()) > 0:
            data_split_size = len(rollouts) // data_parallel_world_size
            data_split_range = (
                mpu.get_data_parallel_rank() * data_split_size,
                (mpu.get_data_parallel_rank() + 1) * data_split_size,
            )
            rollouts = rollouts[data_split_range[0] : data_split_range[1]]
            flattened_rollout_metadata = flattened_rollout_metadata[
                data_split_range[0] : data_split_range[1]
            ]
            local_num_turns = sum(num_turns[data_split_range[0] : data_split_range[1]])
            steps_before = sum(num_turns[:data_split_range[0]])
            advantages = advantages[steps_before:steps_before+local_num_turns]
            # First we calculate them on a global level and then we split and recalculate on a local level.
            # Sequence packing and reporting needs it global but non-packing wants it local.

        with nvtx_range("prepare_trajectories"):
            (
                trajs,
                generation_masks,
                inference_logprobs,
                inference_routing,
                routing_dump_ids,
            ) = prepare_trajectories(
                rollouts, tokenizer, args.seq_length, sequence_packing, args.rl_skip_bos_token
            )
            inference_top_logprobs_by_turn = None
            _topk_dump_dir = os.environ.get("ROUTER_STUDY_DUMP_DIR", "")
            if _topk_dump_dir and getattr(args, "rl_logprob_mismatch_top_k", 0) > 0:
                inference_top_logprobs_by_turn = _load_top_logprobs_from_npz(
                    _topk_dump_dir, routing_dump_ids
                )
            local_turn_metadata = _build_logprob_mismatch_turn_metadata(
                rollouts, flattened_rollout_metadata, inference_top_logprobs_by_turn
            )
            if probe_enabled(args):
                runtime_state.probe_turn_metadata = local_turn_metadata
                runtime_state.probe_generation_masks = generation_masks
                runtime_state.probe_tokens = trajs
                runtime_state.probe_iteration = iteration
                if sequence_packing:
                    print_rank_0(
                        "[RL-determinism-probe] sequence packing is enabled; "
                        "training-side activation probe scopes are disabled"
                    )
            all_turn_metadata = (
                _gather_logprob_mismatch_turn_metadata(local_turn_metadata)
                if sequence_packing and getattr(args, "rl_logprob_mismatch_num_examples", 0) > 0
                else None
            )
            inference_logit_means, inference_logit_stds = (
                _prepare_inference_logit_moments(rollouts, args.seq_length)
                if getattr(args, "rl_match_train_logit_moments_to_inference", False)
                and not sequence_packing
                else (None, None)
            )

        packing_context = None
        _replay_routing_tensor, _replay_seq_mask = None, None
        # Build trajectories based on sequence packing or standard processing
        if sequence_packing:
            with nvtx_range("sequence_packing", time=True):
                global_turn_metadata = all_turn_metadata
                runtime_state.packing_context = packing_context = pack_all_trajectories(
                    trajs, 
                    generation_masks, 
                    inference_logprobs, 
                    global_advantages, 
                    args.seq_length, 
                    args.rl_sequence_packing_max_sequences_per_bin,
                    args.rl_sequence_packing_algo
                    )
    
                compute_trajs = packing_context.packed_trajs
                compute_position_ids = packing_context.packed_position_ids
                # Use batch_size=1 for packed computation to enable proper attention masking
                # via PackedSeqParams (TE needs cu_seqlens per bin)
                dataset = TensorDataset(torch.arange(len(compute_trajs)))
                data_loader = DataLoader(dataset, batch_size=1)
                logprobs_batch_size = 1
        else:
            # Always compute standard masks for the original data (we'll need them later)
            with nvtx_range("get_ltor_masks_and_position_ids"):
                _, original_loss_mask, original_position_ids = get_ltor_masks_and_position_ids(
                    trajs,
                    tokenizer.eod,
                    tokenizer.pad,
                    args.reset_position_ids,
                    args.reset_attention_mask,
                    eod_mask_loss=False,
                    pad_mask_loss=True,
                )
                original_loss_mask[~generation_masks] = 0.0
                compute_trajs = trajs
                compute_position_ids = original_position_ids

                # Load full-sequence (prompt+generated) routing for replay if enabled.
                _replay_routing_tensor, _replay_seq_mask = None, None
                _router_diag_dump_dir = os.environ.get("ROUTER_STUDY_DUMP_DIR", "")
                _do_replay = (
                    getattr(args, 'moe_enable_routing_replay', False)
                    and bool(_router_diag_dump_dir)
                )
                if _do_replay:
                    _raw_replay = _load_routing_for_replay(
                        _router_diag_dump_dir,
                        routing_dump_ids=routing_dump_ids,
                        trajs=trajs,
                        generation_masks=generation_masks,
                    )
                    if _raw_replay is not None:
                        _replay_routing_tensor, _replay_seq_mask = _build_replay_routing_tensor(
                            _raw_replay, trajs, args.seq_length
                        )

                if _replay_routing_tensor is not None:
                    dataset_tensors = [
                        compute_trajs,
                        compute_position_ids,
                        _replay_routing_tensor,
                        _replay_seq_mask,
                    ]
                    if inference_logit_means is not None and inference_logit_stds is not None:
                        dataset_tensors.extend([inference_logit_means, inference_logit_stds])
                    if probe_enabled(args):
                        dataset_tensors.append(torch.arange(len(compute_trajs), dtype=torch.long))
                    data_loader = DataLoader(
                        TensorDataset(*dataset_tensors), batch_size=args.micro_batch_size
                    )
                else:
                    dataset_tensors = [
                        compute_trajs,
                        compute_position_ids,
                        torch.zeros_like(compute_trajs),
                        torch.zeros_like(compute_trajs, dtype=torch.bool),
                    ]
                    if inference_logit_means is not None and inference_logit_stds is not None:
                        dataset_tensors.extend([inference_logit_means, inference_logit_stds])
                    if probe_enabled(args):
                        dataset_tensors.append(torch.arange(len(compute_trajs), dtype=torch.long))
                    data_loader = DataLoader(
                        TensorDataset(*dataset_tensors), batch_size=args.micro_batch_size
                    )
                logprobs_batch_size = args.micro_batch_size

        with torch.no_grad(), nvtx_range("compute_logprobs", time=True):
            # Before we can update the model, we need to get the logprobs for the \pi_{old} model.

            forward_backward_func = get_forward_backward_func()
            if (
                args.cuda_graph_impl == "local"
                and CudaGraphScope.full_iteration in args.cuda_graph_scope
                and not probe_enabled(args)
            ):
                forward_backward_func = FullCudaGraphWrapper(
                    forward_backward_func, cuda_graph_warmup_steps=args.cuda_graph_warmup_steps
                )

            dtype = (
                torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
            )

            pg_collection = get_attr_wrapped_model(model, "pg_collection")
            pp_group = pg_collection.pp

            # Register router hooks for Router-diag if a dump dir is available (non-packed only).
            _routing_store = None
            _router_score_store = None
            _routing_handles = []
            _router_diag_dump_dir = os.environ.get("ROUTER_STUDY_DUMP_DIR", "")
            _do_router_diag = bool(_router_diag_dump_dir) and not sequence_packing
            print_rank_0(
                f"[Router-diag] guard: dump_dir={_router_diag_dump_dir!r}  "
                f"sequence_packing={sequence_packing}"
            )
            if _do_router_diag:
                _routing_store, _router_score_store, _routing_handles = _register_routing_hooks(model)

            with torch.no_grad(), nvtx_range("compute_old_logprobs", time=True):
                mismatch_top_k = (
                    getattr(args, "rl_logprob_mismatch_top_k", 0)
                    if getattr(args, "rl_logprob_mismatch_num_examples", 0) > 0
                    else 0
                )
                old_logprobs = compute_logprobs_batch(
                    model=model,
                    data_loader=data_loader,
                    forward_backward_func=forward_backward_func,
                    packing_context=packing_context,
                    trajs_batch_size=len(compute_trajs),
                    seq_length=args.seq_length,
                    logprobs_batch_size=logprobs_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    dtype=dtype,
                    pp_group=pp_group,
                    is_correction=args.rl_inference_logprobs_is_correction,
                    replay_enabled=(_replay_routing_tensor is not None),
                    topk_logprobs=mismatch_top_k,
                    match_logit_moments=(
                        inference_logit_means is not None and inference_logit_stds is not None
                    ),
                    probe_turn_metadata=local_turn_metadata if not sequence_packing else None,
                    probe_generation_masks=generation_masks if not sequence_packing else None,
                    probe_iteration=iteration,
                    probe_phase="training_old_logprobs",
                )
                old_topk_logprobs, old_topk_indices = None, None
                if mismatch_top_k > 0:
                    old_logprobs, old_topk_logprobs, old_topk_indices = old_logprobs
                if _replay_routing_tensor is not None:
                    _log_router_replay_correctness(iteration=iteration)

            for h in _routing_handles:
                h.remove()
            print_rank_0("[Router-diag] routing hooks removed")

            if _routing_store is not None:
                _diag_routing = _load_router_diag_from_npz(
                    _router_diag_dump_dir,
                    routing_dump_ids=routing_dump_ids,
                    trajs=trajs,
                    generation_masks=generation_masks,
                )
                _is_tp_rank0 = (
                    not dist.is_initialized()
                    or mpu.get_tensor_model_parallel_rank() == 0
                )
                print_rank_0(f"[Router-diag] calling _log_router_diag  is_tp_rank0={_is_tp_rank0}")
                _router_diag_data = None
                _expert_load_data = None
                _router_score_data = None
                _router_diag_data = _log_router_diag(
                    _routing_store, _diag_routing or [], generation_masks,
                    iteration=iteration,
                    sequence_parallel=getattr(args, "sequence_parallel", False),
                )
                if _is_tp_rank0:
                    print_rank_0("[Router-diag] _log_router_diag returned, calling _log_expert_load")
                    _expert_load_data = _log_expert_load(_routing_store, iteration=iteration)
                    print_rank_0("[Router-diag] _log_expert_load returned")
                    _router_score_data = _log_router_score_stats(_router_score_store, iteration=iteration)
                # Broadcast computed metrics from rank 0 to all world ranks so that
                # the rank holding the WandB writer (rank world_size-1) can log them.
                # This is the same pattern used by maybe_log_training_metrics().
                if dist.is_initialized():
                    _broadcast_container = [_router_diag_data, _expert_load_data, _router_score_data]
                    dist.broadcast_object_list(_broadcast_container, src=0)
                    _router_diag_data, _expert_load_data, _router_score_data = _broadcast_container
                _wandb_log_router_metrics(
                    _router_diag_data, _expert_load_data, _router_score_data, iteration
                )

            with torch.no_grad(), nvtx_range("compute_ref_logprobs", time=True):
                # We need to load the ref model state dict and compute the logprobs for the ref model
                cur_st_dict = {
                    k: (v.cpu() if v is not None else v) for k, v in model.state_dict().items()
                }
                model.load_state_dict(ref_state_dict)
                ref_logprobs = compute_logprobs_batch(
                    model=model,
                    data_loader=data_loader,
                    forward_backward_func=forward_backward_func,
                    packing_context=packing_context,
                    trajs_batch_size=len(compute_trajs),
                    seq_length=args.seq_length,
                    logprobs_batch_size=logprobs_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    dtype=dtype,
                    pp_group=pp_group,
                    is_correction=args.rl_inference_logprobs_is_correction,
                )

                # logprobs are [b, seq, h] now.
                model.load_state_dict(cur_st_dict)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()


        if sequence_packing:
            with nvtx_range("pack_logprobs", time=True):
                # Store logprobs on gpu in packing context
                # Since PackingContext is a dataclass, we add these as new attributes
                packing_context.old_logprobs = old_logprobs.cuda()
                packing_context.ref_logprobs = ref_logprobs.cuda()
                packed_inference_logprobs = None

                if inference_logprobs is not None:
                    # Pack the inference logprobs using the helper function
                    # We do this for logging purposes even if is_correction is disabled
                    packed_inference_logprobs = pack_inference_logprobs(
                        inference_logprobs=packing_context.original_inference_logprobs,
                        packing_info=packing_context.packing_info,
                        generation_masks=packing_context.original_generation_masks,
                        bin_size=args.seq_length,
                    )

                    # Compute statistics for logging using packed data
                    compute_packed_inference_logprobs_stats(
                        old_logprobs=old_logprobs,
                        packed_inference_logprobs=packed_inference_logprobs,
                        packed_loss_mask=packing_context.packed_loss_mask,
                        group_stats=group_stats,
                    )

                    # Store packed inference logprobs in packing context
                    packing_context.packed_inference_logprobs = packed_inference_logprobs.cuda()
                    # Only mark as having inference logprobs for IS correction if enabled
                    packing_context.has_inference_logprobs = args.rl_inference_logprobs_is_correction
                _maybe_log_logprob_mismatch_diagnostics(
                    old_logprobs=old_logprobs,
                    inference_logprobs=packed_inference_logprobs,
                    generation_masks=None,
                    trajs=None,
                    packing_context=packing_context,
                    turn_metadata=global_turn_metadata,
                    iteration=iteration,
                    train_topk_logprobs=old_topk_logprobs,
                    train_topk_indices=old_topk_indices,
                )
            with nvtx_range("create_dataloader"):
                # @vitalyk: This function also reconfigures the data loader to count the
                # global_batch_size in the bins frame of reference.
                # I think it will be a better design if we split the data loader creating and logic
                # that reconfigures the microbatch calculator.

                update_microbatch_calculator(
                    samples_ratio_per_step=samples_ratio_per_step,
                    num_bins_this_rank = len(packing_context.packed_trajs),
                    bin_seq_indices = packing_context.packing_info.bin_seq_indices,
                    global_batch_size=args.global_batch_size, 
                    rampup_batch_size=args.rampup_batch_size, 
                    micro_batch_size=args.micro_batch_size, 
                    decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
               )
                loader = get_microbatch_dataloader(len(packing_context.packed_trajs), args.micro_batch_size)
        else:
            with nvtx_range("align_inference_logprobs", time=True):
                aligned_inference_logprobs_for_diag = None
                if inference_logprobs is not None:
                    inference_logprobs = align_unpacked_inference_logprobs(
                        inference_logprobs=inference_logprobs,
                        old_logprobs_for_data=old_logprobs,
                        generation_masks=generation_masks,
                        group_stats=group_stats,
                    )
                    aligned_inference_logprobs_for_diag = inference_logprobs
                    # We run the above to fill in the inference/train side mismatch stats.
                    # We do the above for logging purposes.
                    # Nullify logprobs if not used in IS correction,
                    if not args.rl_inference_logprobs_is_correction:
                        inference_logprobs = None
                _maybe_log_logprob_mismatch_diagnostics(
                    old_logprobs=old_logprobs,
                    inference_logprobs=aligned_inference_logprobs_for_diag,
                    generation_masks=generation_masks,
                    trajs=trajs,
                    packing_context=None,
                    turn_metadata=local_turn_metadata,
                    iteration=iteration,
                    train_topk_logprobs=old_topk_logprobs,
                    train_topk_indices=old_topk_indices,
                )
                _maybe_write_determinism_probe_targets(
                    old_logprobs=old_logprobs,
                    inference_logprobs=aligned_inference_logprobs_for_diag,
                    generation_masks=generation_masks,
                    trajs=trajs,
                    turn_metadata=local_turn_metadata,
                    iteration=iteration,
                    train_topk_logprobs=old_topk_logprobs,
                    train_topk_indices=old_topk_indices,
                )
                if (
                    getattr(args, "rl_determinism_probe_write_targets_file", None)
                    and not getattr(args, "rl_determinism_probe_dir", None)
                    and not getattr(args, "rl_determinism_probe_targets_file", None)
                ):
                    print_rank_0(
                        "[RL-determinism-probe] target discovery complete; exiting before training"
                    )
                    raise SystemExit(0)
            with nvtx_range("create_dataloader"):
                # Because of multiturn, our batch sizes for non-sequence packed trajectories are not fixed anymore.
                # As in sequence packing above, we need to reconfigure it too.
                runtime_state.packing_context = None

                reconfigure_num_microbatches_calculator(
                    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                    global_batch_size=math.ceil(samples_ratio_per_step*total_turns_sampled), 
                    rampup_batch_size=args.rampup_batch_size, 
                    micro_batch_size=args.micro_batch_size, 
                    decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
                    data_parallel_size=mpu.get_data_parallel_world_size(),
                )

                dataset_tensors = [
                    compute_trajs,
                    advantages,
                    old_logprobs,
                    original_loss_mask,
                    original_position_ids,
                    ref_logprobs,
                ]
                if is_correction and inference_logprobs is not None:
                    dataset_tensors.append(inference_logprobs)
                else:
                    dataset_tensors.append(torch.zeros_like(old_logprobs))
                if _replay_routing_tensor is not None:
                    dataset_tensors.append(_replay_routing_tensor)
                    dataset_tensors.append(_replay_seq_mask)
                if probe_enabled(args):
                    dataset_tensors.append(torch.arange(len(compute_trajs), dtype=torch.long))
                data = TensorDataset(*dataset_tensors)
                loader = DataLoader(data, batch_size=args.micro_batch_size)


    return RerunDataIterator(itertools.cycle(loader)), group_stats, example_groups


def get_grpo_data_iterator(
    model: LanguageModule,
    inference_model: LanguageModule | None,
    optimizer: MegatronOptimizer,
    iteration: int,
    ref_state_dict: Dict[str, torch.Tensor],
    grpo_iterations: int,
    grpo_prompts_per_step: int,
    grpo_group_size: int,
    global_batch_size: int,
    sequence_packing: bool,
    is_correction: bool,
    buffered_rollouts: RerunDataIterator | None = None,
) -> RerunDataIterator:
    """
    Get the data iterator for GRPO training.

    Depending on the sampling parameters either performs data collections or returns
    the buffered_rollouts as is.

    Args:
        model: The language model
        optimizer: The Megatron optimizer
        iteration: Current training iteration
        ref_state_dict: Reference model state dict for GRPO
        grpo_iterations: How many steps we reuse the sampled data for.
        grpo_prompts_per_step: How many prompts we sample per data collection.
        grpo_group_size: How many samples we do per prompt.
        global_batch_size: Global batch size.
        sequence_packing: Use sequence packing if True.
        is_correction: Use IS correction if True.
        buffered_rollouts: Previously collected rollouts (if any)

    Returns:
        RerunDataIterator for the current training step
    """
    runtime_state = get_rl_runtime_state()
    tokenizer = get_tokenizer()

    # We collect new rollouts when we've gone over the collected data 'grpo_iterations' times.
    global_batches_per_collection = (grpo_prompts_per_step * grpo_group_size) // global_batch_size
    if (
        buffered_rollouts is None or
        iteration == runtime_state.last_collection_iteration +
        (grpo_iterations * global_batches_per_collection)
    ):

        rollouts = get_environment_rollouts(
            model, inference_model, optimizer, grpo_prompts_per_step, grpo_group_size
        )
        buffered_rollouts, group_stats, example_groups = prepare_data_for_update(
            model=model,
            ref_state_dict=ref_state_dict,
            rollouts=rollouts,
            tokenizer=tokenizer,
            sequence_packing=sequence_packing,
            is_correction=is_correction,
            iteration=iteration,
        )
        runtime_state.group_stats = group_stats
        runtime_state.example_groups = example_groups
        runtime_state.reset_iteration_counters(iteration)

    maybe_log_training_metrics(
        group_stats=runtime_state.group_stats,
        current_iteration=iteration,
        tokenizer=tokenizer,
        example_groups=runtime_state.example_groups,
    )

    return buffered_rollouts


def evaluate_and_print_results_rl(
    data_iterator: Iterator[TensorDataset],
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    iteration: int,
    write_to_tensorboard: bool = True,
    training_model: Optional[list[LanguageModule]] = None,
):
    """Helper function to evaluate and dump results on screen.

    Args:
        data_iterator: Iterator over batches of evaluation dataset.
        model: Model to evaluate with (may be separate inference model).
        iteration: Current training iteration.
        write_to_tensorboard: Dumpt stuff to tensorboard or not.
        training_model: Training model (if separate from inference model). Used to offload
            grad buffers and restore to train mode. If None, uses model parameter.
    """
    args = get_args()

    # TODO(vitalyk): I do not track eval loss as in training. We probably should.
    # megatron-lm uses forward_step_func to do the above.

    # Use context manager to temporarily disable sequence parallelism for evaluation

    with torch.no_grad():
        with megatron_rl_inference_mode(
            model,
            optimizer,
            args.cuda_graph_impl,
            args.rl_offload_optimizer_during_inference,
            training_model,
        ) as inference_interface:

            loop = get_asyncio_loop()

            rank = torch.distributed.get_rank()
            if rank == 0:
                logger.info("Collecting evaluation results...")
                agent = get_agent(args)
                request = EvaluationRequest(
                    inference_interface=inference_interface,
                    num_prompts=args.rl_prompts_per_eval,
                    validation=True,
                    rank_info=None,
                    generation_args={
                        'temperature': args.rl_default_temperature,
                        'max_tokens': args.seq_length,
                        'top_p': args.rl_default_top_p,
                        'top_k': args.rl_default_top_k,
                    },
                )
                evaluation_responses = loop.run_until_complete(agent.run_evaluation(request))
                if not isinstance(evaluation_responses, list):
                    evaluation_responses = [evaluation_responses]
            else:
                evaluation_responses = None

        dp_eval_results: list[None | list[EvaluationResponse]] = [
            None for _ in range(args.world_size)
        ]
        dist.gather_object(
            evaluation_responses,
            dp_eval_results if dist.get_rank() == (args.world_size - 1) else None,
            dst=args.world_size - 1,
        )

        if dist.get_rank() == args.world_size - 1:
            dp_eval_results = [x for x in dp_eval_results if x is not None]
            # TODO(rkirby): maybe factor this out into a function?
            eval_metrics = defaultdict(list)
            for responses in dp_eval_results:
                for response in responses:
                    if response is None:
                        continue
                    for k, v in response.metrics().items():
                        eval_metrics[f"{response.env_id}_eval_mean_{k}"].extend(v)
                    for result in response.results:
                        if isinstance(result, RewardEvaluationResult):
                            try:
                                lang_rl_log(
                                    f"Evaluation: [{response.env_id}] [{result.reward}] {result.prompt} {result.response}"
                                )
                            except Exception as e:
                                lang_rl_log(f"Error: {e}")
                                lang_rl_log(f"Result: {result}")
            logger.info(
                "Collected metrics:"
                + "".join([f"\n\t{k} count: {len(v)}" for k, v in eval_metrics.items()])
            )
            eval_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
            if write_to_tensorboard:
                tb_writer = get_tensorboard_writer()
                if tb_writer:
                    for k, v in eval_metrics.items():
                        tb_writer.add_scalar(k, v, iteration)
            wandb_writer = get_wandb_writer()
            if wandb_writer:
                wandb_writer.log(eval_metrics, step=iteration)
            logger.info(
                "Evaluation results:"
                + "".join([f"\n\t{k}: {v:0.4f}" for k, v in eval_metrics.items()])
            )
            if lang_rl_log_dir:
                with open(
                    lang_rl_log_dir
                    + f'/eval_rank{rank}_iteration{args.curr_iteration}_'
                    + f'{Path(args.langrl_env_config).stem}.json',
                    'w',
                ) as f:
                    json.dump([[r.model_dump() for r in group] for group in dp_eval_results], f)


def calculate_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clamp_eps_lower: float,
    clamp_eps_upper: float,
    kl_beta: float,
    entropy_weight: float,
    inference_logprobs: torch.Tensor | None = None,
    is_truncation_coef: float | None = None,
    seq_starts: list | None = None,
    seq_lengths: list | None = None,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get GRPO loss, the kl term of the loss and the pi/pi_{old} ratios.

    Args:
        current_logprobs: pi logprobs, [batch, seq] for unpacked or [1, bin_size] for packed.
        old_logprobs: pi_{old} logprobs, [batch, seq] for unpacked or [1, bin_size] for packed.
        ref_logprobs: pi_{ref} logprobs, [batch, seq] for unpacked or [1, bin_size] for packed.
        advantages: advantages tensor, [batch,] for unpacked or [num_sequences_in_bin,] for packed.
        clamp_eps_lower: eps to clamp ratios from below.
        clamp_eps_upper: eps to clamp ratios from above, if vanilla GRPO, this should be equal to clamp_eps_lower.
        kl_beta: weight for the KL penalty term measuring the distance between pi and pi_{ref}.
        entropy_weight: weight for the entropy term.
        inference_logprobs: pi_{old} logprobs calculated by the inference engine.
            If not None, importance sampling correction will be applied.
        is_truncation_coef: importance sampling truncation coefficient. Will be applied if it is not None and inference_logprobs are present.
        seq_starts: (optional) For packed sequences: start positions of each sequence in the bin.
        seq_lengths: (optional) For packed sequences: original lengths of each sequence.
        loss_mask: (optional) To calculate losses, that require sequence-level stats, e.g VESPO, if None, GRPO loss is used.

    Returns:
        total per-token GRPO loss [batch, seq] or [1, bin_size],
        kl_term of the loss [batch, seq] or [1, bin_size],
        pi/pi_{old} ratios [batch, seq] or [1, bin_size],
        entropy_term of the loss [batch, seq] or [1, bin_size],
        truncated_from_above [batch, seq] or [1, bin_size] (whether we clamped the ratios or not),
        truncated_from_below [batch, seq] or [1, bin_size] (whether we clamped the ratios or not).
    """
    # Ensure shapes match before computation
    if current_logprobs.shape != old_logprobs.shape:
        log_single_rank(
            logger,
            logging.WARNING,
            f"WARNING: Shape mismatch - current_logprobs: {current_logprobs.shape}, old_logprobs: {old_logprobs.shape}",
        )

    log_ratios = current_logprobs - old_logprobs
    ratios = log_ratios.exp()
    clamped_ratios = ratios.clamp(1 - clamp_eps_lower, 1 + clamp_eps_upper)
    truncated_from_above = torch.gt(ratios, 1 + clamp_eps_upper)
    truncated_from_below = torch.lt(ratios, 1 - clamp_eps_lower)

    # Handle advantages based on whether this is packed or unpacked
    if seq_starts is not None and seq_lengths is not None:
        if loss_mask is not None:
            raise ValueError("Sequence packing now only supports GRPO loss. Do not pass loss_mask.")
        # Packed sequences: map each sequence's advantage to its tokens
        bin_size = current_logprobs.shape[1]
        packed_advantages = torch.zeros(
            (1, bin_size), device=current_logprobs.device, dtype=current_logprobs.dtype
        )

        for seq_idx, (start, seq_len) in enumerate(zip(seq_starts, seq_lengths)):
            # Logprobs are 1 token shorter than sequences
            end = min(start + seq_len - 1, bin_size)
            if end > start:
                packed_advantages[0, start:end] = advantages[seq_idx].item()

        advantages = packed_advantages
    else:
        # Unpacked sequences: broadcast single advantage per sequence
        # Reshape to [batch, 1] to match logprobs shape [batch, seq]
        advantages = advantages.view(-1, 1)

    ref_diff = ref_logprobs - current_logprobs
    kl_term = ref_diff.exp() - ref_diff - 1
    entropy_term = -current_logprobs.exp() * current_logprobs
    if loss_mask is not None:
        # VESPO branch. https://arxiv.org/abs/2602.10693
        # TODO(vitalyk): add to arguments.py
        c_pos = (2, 3)
        c_neg = (3, 2)

        W = (log_ratios * loss_mask).sum(dim=-1, keepdim=True).exp()

        # Use separate hyperparameters for positive/negative advantages.
        pos_adv = (advantages >= 0).float()
        neg_adv = 1 - pos_adv

        c1 = pos_adv * c_pos[0] + neg_adv * c_neg[0]
        c2 = pos_adv * c_pos[1] + neg_adv * c_neg[1]
        log_phi = c2 + c1 * torch.log(W) - c2 * W
        phi = log_phi.exp().detach()

        loss = -phi * advantages * current_logprobs
    else:
        # Actual GRPO loss.
        # DONOTMERGE
        # TODO(vitalyk): make another loss func and send it as a forward_step from train_rl.py

        is_weights = torch.tensor(1.0, dtype=old_logprobs.dtype).to(old_logprobs.device)
        if inference_logprobs is not None:
            is_weights = (old_logprobs - inference_logprobs).exp()
            if is_truncation_coef is not None:
                is_weights = torch.min(
                    is_weights,
                    torch.tensor(is_truncation_coef, dtype=old_logprobs.dtype).to(old_logprobs.device),
                )

        loss = (
            -is_weights * torch.min(ratios * advantages, clamped_ratios * advantages)
            + kl_beta * kl_term
            - entropy_weight * entropy_term
        )

    return loss, kl_term, ratios, entropy_term, truncated_from_above, truncated_from_below


@contextmanager
def megatron_rl_inference_mode(
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    cuda_graph_impl: str,
    offload_optimizer_during_inference: bool,
    training_model: Optional[list[LanguageModule]] = None,
):
    """Manage the model inference context when collecting rollouts.

    Args:
        model: model to prepare for inference (may be separate inference model).
        optimizer: optimizer used to train the model.
        cuda_graph_impl: which cuda graph implementation to use.
        offload_optimizer_during_inference: move optimizer to cpu during inference or not.
        training_model: training model (if separate from inference model). Used to offload
            grad buffers and restore to train mode. If None, uses model parameter.

    Yields:
        None: this context manager does not return a value.

    """
    args = get_args()
    loop = get_asyncio_loop()
    nvtx_range = get_nvtx_range()

    logger.debug(f"[{dist.get_rank()}] Entering inference mode")

    # Change cudagraph scope for inference (empty list = full-layer capture)
    model[0].config.cuda_graph_scope = []
    if probe_enabled(args):
        ensure_determinism_probe(model[0], args)
        if cuda_graph_impl != "none":
            print_rank_0(
                "[RL-determinism-probe] disabling inference CUDA graphs so forward hooks run"
            )
            cuda_graph_impl = "none"
        model[0].config.cuda_graph_impl = "none"
    else:
        model[0].config.cuda_graph_impl = "local"

    # If we get a lower precision wrapper, we go one object deeper.
    lang_module = model[0].module.module if hasattr(model[0].module, "module") else model[0].module

    if probe_enabled(args):
        # Hooks do CPU transfers and scalar reductions, which are illegal while
        # CUDA graph capture is active. Remove existing layer graph managers too,
        # not just the top-level config flag.
        toggle_cuda_graphs(lang_module, "none")

    # Switch MoE layers to full CUDA graph capture for inference
    if not probe_enabled(args) and args.rl_training_cuda_graphs and args.num_experts is not None:
        transition_moe_cudagraphs(lang_module, 'full')

    lang_module.eval()
    # If this is a separate RL inference model with offloading enabled, ensure weights are on GPU
    # before any CUDA-graph capture/replay or inference. This is a no-op if already on GPU.
    model_core = unwrap_model(model[0])
    with nvtx_range("prefetch-inference-model-weights-to-gpu"):
        _maybe_prefetch_separate_inference_model_weights(model_core, to_cpu=False)

    rotary_module = getattr(lang_module, "rotary_pos_emb", None)
    # Vanilla RotaryEmbedding module has lru_cache decorator which breaks RL training
    # as it tries to reuse frequences tensors cached in inference mode.
    has_lru_cache = rotary_module is not None and hasattr(rotary_module.forward, "cache_parameters")
    if has_lru_cache:
        rotary_module.forward.cache_clear()

    with torch.no_grad():

        if offload_optimizer_during_inference:
            with nvtx_range("offload-optimizer-state-and-grad-buffers-before-inference"):
                if not args.rl_training_cuda_graphs:
                    # Offload grad buffers from the training model (if separate inference model is used)
                    # or from the inference model (if they're the same model)
                    model_for_grad_offload = training_model if training_model is not None else model
                    model_for_grad_offload[0].offload_grad_buffers()
                else:
                    logger.warning(
                        "Gradient buffers will not be offloaded when training cudagraphs are used!"
                    )
                optimizer.offload_to_cpu()

        if cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
            toggle_cuda_graphs(lang_module, cuda_graph_impl)

        inference_interface = get_inference_interface(args, loop, model)
        inference_interface.set_generation_epoch(get_args().curr_iteration)
        loop.run_until_complete(inference_interface.resume())

        logger.debug(f"[{dist.get_rank()}] Entered inference mode")
        yield inference_interface

        with nvtx_range("suspend-engine"):
            loop.run_until_complete(inference_interface.suspend())

        if cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
            toggle_cuda_graphs(lang_module, 'none')

        # Reset drop_and_pad leaked from inference decode
        set_decode_expert_padding(unwrap_model(model[0]), set_to=False)

        # Restore partial capture cudagraph scope for training if this is MoE
        if args.num_experts is not None:
            model[0].config.cuda_graph_scope = [
                CudaGraphScope.mamba,
                CudaGraphScope.attn,
                CudaGraphScope.moe_router,
                CudaGraphScope.moe_preprocess,
            ]

        # Switch MoE layers to partial CUDA graph capture for training
        if args.rl_training_cuda_graphs and args.num_experts is not None:
            transition_moe_cudagraphs(lang_module, 'partial')

        # If this is a separate RL inference model, prefetch weights back to CPU so they
        # don't consume GPU memory during training.
        with nvtx_range("prefetch-inference-model-weights-to-cpu"):
            _maybe_prefetch_separate_inference_model_weights(model_core, to_cpu=True)

        if offload_optimizer_during_inference:
            with nvtx_range("onload-optimizer-state-and-grad-buffers-after-inference"):
                # Restore grad buffers to the training model (if separate inference model is used)
                # or to the inference model (if they're the same model)
                model_for_grad_offload = training_model if training_model is not None else model
                model_for_grad_offload[0].restore_grad_buffers()
                optimizer.restore_from_cpu()

        # Set training model back to train mode (not inference model if they're separate)
        training_lang_module = unwrap_model(training_model[0]) if training_model is not None else lang_module
        training_lang_module.train()

        if has_lru_cache:
            rotary_module.forward.cache_clear()

        logger.debug(f"[{dist.get_rank()}] Exiting inference mode")


def rl_inference_interface_shutdown():
    global _INFERENCE_INTERFACE
    global _ROLLOUT_GENERATOR

    if _ROLLOUT_GENERATOR is not None:
        loop = get_asyncio_loop()
        loop.run_until_complete(_ROLLOUT_GENERATOR.aclose())
        _ROLLOUT_GENERATOR = None

    if _INFERENCE_INTERFACE is not None:
        loop = get_asyncio_loop()
        loop.run_until_complete(_INFERENCE_INTERFACE.kill())
        _INFERENCE_INTERFACE = None
    else:
        logger.warning("No inference interface to shutdown. This should not happen.")

    # TODO(rkirby): This is a hack to hard exit. There is a bug that is preventing us from using sys.exit(0).
    # It seem the Flask server has non-daemon threads that are preventing the program from exiting.
    # We need to find a way to gracefully complete all in progress requests and shutdown the Flask server.
    import os
    os._exit(0)


def get_iteration_sequence_count(args):
    """Get the total number of sequences processed in this iteration across all ranks."""
    runtime_state = get_rl_runtime_state()
    sequences_tensor = torch.tensor(
        runtime_state.sequences_this_iteration_on_rank, device='cuda', dtype=torch.long
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sequences_tensor, group=mpu.get_data_parallel_group())
    return int(sequences_tensor.item())
    
def _pad_nonnull_with_zeros(data: list[Optional[torch.Tensor]], max_len: int) -> torch.Tensor:
    """Pad each element of a list of tensors to the length required.
    Args:
        data: List of tensors to pad.
        max_len: Maximum length to pad to. Must be higher or equal than the max len of the data tensors.
    Returns:
        A padded tensor which is a stacked list of padded input tensors.

    """
    if all([el is None for el in data]):
        raise ValueError("At least one element of the data list should be not None.")
    padded_data = []
    for chunk in data:
        if chunk is not None:
            padding_size = max_len - len(chunk)
            if padding_size > 0:
                # Pad with zeros (these positions will be masked anyway)
                padded = torch.nn.functional.pad(chunk, (0, padding_size), value=0.0)
                padded_data.append(padded)
            elif padding_size == 0:
                padded_data.append(chunk)
            else:
                raise ValueError("One of the input tensors has larger length than padding max len.")
        else:
            # Create zero tensor for None logprobs
            padded_data.append(torch.zeros(max_len))
    return torch.stack(padded_data)

