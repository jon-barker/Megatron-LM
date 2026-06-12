# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.models.common.output_layer import fp32_output_layer


def _fake_output_layer(weight: torch.Tensor):
    return SimpleNamespace(
        allreduce_dgrad=False,
        bias=None,
        disable_grad_reduce=True,
        explicit_expert_comm=False,
        gather_output=False,
        input_size=weight.shape[1],
        output_size_per_partition=weight.shape[0],
        sequence_parallel=False,
        skip_bias_add=False,
        tp_group=None,
        weight=weight,
    )


def test_fp32_output_layer_matches_explicit_fp32_reference_for_gpt_style_head():
    hidden_states = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    weight = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)
    output_layer = _fake_output_layer(weight)

    logits, bias = fp32_output_layer(output_layer, hidden_states)

    expected = torch.matmul(hidden_states.float(), weight.float().t())
    assert bias is None
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected)


def test_fp32_output_layer_matches_explicit_fp32_reference_for_mamba_style_head():
    hidden_states = torch.randn(5, 1, 12, dtype=torch.bfloat16)
    weight = torch.randn(20, 12, dtype=torch.bfloat16, requires_grad=True)
    output_layer = _fake_output_layer(weight)

    logits, _ = fp32_output_layer(output_layer, hidden_states, runtime_gather_output=False)

    expected = torch.matmul(hidden_states.float(), weight.float().t())
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected)
