# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from types import SimpleNamespace

import torch

from megatron.rl.determinism_probe import (
    active_token_metadata_for_first_dim,
    build_inference_selected_logprob_metadata,
    build_training_token_metadata,
    ensure_determinism_probe,
    hash_token_ids,
    probe_scope,
    register_inference_request,
)


def _probe_args(tmp_path, **kwargs):
    defaults = {
        "rl_determinism_probe_dir": str(tmp_path),
        "rl_determinism_probe_phases": "training_old_logprobs",
        "rl_determinism_probe_module_filter": "0",
        "rl_determinism_probe_max_rollouts": 0,
        "rl_determinism_probe_max_generated_tokens": 0,
        "rl_determinism_probe_hash_inputs": True,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_training_token_metadata_uses_causal_generated_alignment():
    tokens = torch.tensor([[10, 20, 30, 40]])
    generation_masks = torch.tensor([[False, False, True, True]])
    metadata = build_training_token_metadata(
        tokens=tokens,
        generation_masks=generation_masks,
        seq_indices=torch.tensor([0]),
        turn_metadata=[{"routing_dump_id": "1_0000_00000007", "global_rollout_index": 3}],
        iteration=1,
        phase="training_old_logprobs",
    )

    assert metadata[1]["input_token_index"] == 1
    assert metadata[1]["target_token_index"] == 2
    assert metadata[1]["target_token_id"] == 30
    assert metadata[1]["is_generated_target"] is True
    assert metadata[1]["gen_offset"] == 0
    assert metadata[1]["routing_dump_id"] == "1_0000_00000007"


def test_probe_writes_per_token_hash_records(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.ones_like(model[0].weight))

    args = _probe_args(tmp_path)
    probe = ensure_determinism_probe(model, args)
    assert probe is not None

    token_metadata = [
        {
            "routing_dump_id": "1_0000_00000000",
            "target_token_index": 1,
            "global_rollout_index": 0,
            "gen_offset": 0,
        },
        {
            "routing_dump_id": "1_0000_00000000",
            "target_token_index": 2,
            "global_rollout_index": 0,
            "gen_offset": 1,
        },
    ]
    x = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
    with probe_scope(
        phase="training_old_logprobs",
        iteration=1,
        token_metadata=token_metadata,
        batch_size=1,
        seq_length=2,
    ):
        model(x)
    probe.close()

    paths = list(tmp_path.glob("probe_training_old_logprobs_iter000001_rank0000.jsonl"))
    assert len(paths) == 1
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]

    output_records = [
        record for record in records
        if record["io_kind"] == "output" and record["module_name"] == "0"
    ]
    assert len(output_records) == 2
    assert all(record["hash"] for record in output_records)
    assert {record["token"]["target_token_index"] for record in output_records} == {1, 2}


def test_active_token_metadata_for_first_dim_slices_tp_sequence_shard():
    class _TPGroup:
        def size(self):
            return 2

        def rank(self):
            return 1

    token_metadata = [{"row": row} for row in range(8)]
    local_tensor = torch.zeros(4, 1, 3)

    with probe_scope(
        phase="training_old_logprobs",
        iteration=1,
        token_metadata=token_metadata,
        batch_size=1,
        seq_length=8,
    ):
        local_metadata = active_token_metadata_for_first_dim(local_tensor, tp_group=_TPGroup())

    assert local_metadata == token_metadata[4:8]


def test_decode_selected_logprob_metadata_matches_request_order():
    class _Context:
        paused_request_count = 0
        total_request_count = 2
        active_token_count = 2

        def is_decode_only(self):
            return True

    context = _Context()
    context.request_ids = torch.tensor([7, 9])
    context.request_query_lengths = torch.tensor([1, 1])
    context.token_to_position_in_request = torch.tensor([5, 3])
    context.token_to_request_idx = torch.tensor([0, 1])
    context.token_to_input_ids = torch.tensor([111, 222])
    context.request_in_prefill_status_tensor = torch.tensor([False, False])

    register_inference_request(request_id=7, routing_dump_id="c_0000_00000007", prompt_tokens=[1, 2, 3, 4, 5])
    register_inference_request(request_id=9, routing_dump_id="c_0000_00000009", prompt_tokens=[8, 9])
    update = __import__(
        "megatron.rl.determinism_probe",
        fromlist=["update_inference_request_generated_tokens"],
    ).update_inference_request_generated_tokens
    update(request_id=7, generated_tokens=[60])
    update(request_id=9, generated_tokens=[70, 71])

    new_tokens = torch.tensor([61, 72])
    metadata = build_inference_selected_logprob_metadata(
        context=context,
        new_tokens=new_tokens,
        collection_id="3",
        rank=0,
        only_last_token_logits=True,
    )

    assert len(metadata) == 2
    assert [item["request_id"] for item in metadata] == [7, 9]
    assert [item["row_index"] for item in metadata] == [0, 1]
    assert [item["target_token_id"] for item in metadata] == [61, 72]
    assert metadata[0]["prefix_hash"] == hash_token_ids([1, 2, 3, 4, 5, 60])
    assert metadata[1]["prefix_hash"] == hash_token_ids([8, 9, 70, 71])


def test_probe_respects_generated_token_cap(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    args = _probe_args(tmp_path, rl_determinism_probe_max_generated_tokens=1)
    probe = ensure_determinism_probe(model, args)

    token_metadata = [
        {"routing_dump_id": "1_0000_00000000", "target_token_index": 1, "gen_offset": 0},
        {"routing_dump_id": "1_0000_00000000", "target_token_index": 2, "gen_offset": 1},
    ]
    with probe_scope(
        phase="training_old_logprobs",
        iteration=1,
        token_metadata=token_metadata,
        batch_size=1,
        seq_length=2,
    ):
        model(torch.ones(2, 1, 1))
    probe.close()

    path = next(tmp_path.glob("probe_training_old_logprobs_iter000001_rank0000.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    output_records = [record for record in records if record["io_kind"] == "output"]
    assert len(output_records) == 1
    assert output_records[0]["token"]["gen_offset"] == 0
