# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel


def use_fp32_output_layer_logsoftmax(config=None) -> bool:
    """Return whether the experimental RL fp32 output-head path is enabled."""
    if config is not None and hasattr(config, "rl_fp32_output_layer_logsoftmax"):
        return bool(config.rl_fp32_output_layer_logsoftmax)
    try:
        from megatron.training import get_args

        args = get_args()
    except Exception:
        return False
    return bool(getattr(args, "rl_fp32_output_layer_logsoftmax", False))


def fp32_output_layer(
    output_layer,
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None = None,
    runtime_gather_output: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a ColumnParallelLinear output head with fp32 matmul inputs and math.

    This mirrors the tensor-parallel input copy / sequence-parallel gather / vocab gather
    behavior of ``ColumnParallelLinear.forward`` while keeping the output-head GEMM in fp32.
    """
    if weight is None:
        if output_layer.weight is None:
            raise RuntimeError(
                "weight was not supplied to fp32_output_layer and output_layer.weight is None."
            )
        weight = output_layer.weight
    else:
        expected_shape = (output_layer.output_size_per_partition, output_layer.input_size)
        if weight.shape != expected_shape:
            raise RuntimeError(
                f"supplied weight's shape is {tuple(weight.shape)}, "
                f"not {expected_shape} as expected"
            )

    bias = output_layer.bias if not output_layer.skip_bias_add else None

    if (
        output_layer.sequence_parallel
        or output_layer.explicit_expert_comm
        or output_layer.disable_grad_reduce
    ):
        input_parallel = hidden_states
    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(
            hidden_states, group=output_layer.tp_group
        )

    if output_layer.sequence_parallel and not output_layer.explicit_expert_comm:
        input_parallel = tensor_parallel.gather_from_sequence_parallel_region(
            input_parallel, tensor_parallel_output_grad=True, group=output_layer.tp_group
        )

    output_parallel = torch.matmul(input_parallel.float(), weight.float().t())
    if bias is not None:
        output_parallel = output_parallel + bias.float()

    gather_output = output_layer.gather_output
    if runtime_gather_output is not None:
        gather_output = runtime_gather_output

    if gather_output:
        output = tensor_parallel.gather_from_tensor_model_parallel_region(
            output_parallel, group=output_layer.tp_group
        )
    else:
        output = output_parallel
    output_bias = output_layer.bias if output_layer.skip_bias_add else None
    return output, output_bias


def fp32_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Run log-softmax after casting logits to fp32."""
    return F.log_softmax(logits.float(), dim=dim)


def fp32_selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather selected logprobs from an fp32 log-softmax over the final dimension."""
    log_probs = fp32_log_softmax(logits, dim=-1)
    if logits.dim() == 1:
        return log_probs[index]
    return torch.gather(log_probs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
