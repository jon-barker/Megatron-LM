# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pydantic import BaseModel

from ..__init__ import Request


class LLMChatMessage(BaseModel):
    role: str
    content: str


class InferenceRequest(Request):
    prompt: list[LLMChatMessage]
    tools: list[dict] | None = None


class InferenceResponse(BaseModel):
    """The minimum required response for an inference interface."""

    response: LLMChatMessage
    raw_text: str | None = None
    token_ids: list[int] | None = None
    prompt_length: int | None = None
    logprobs: list[float] | None = None
    logit_means: list[float] | None = None
    logit_stds: list[float] | None = None
    top_logprobs: list[list[dict]] | None = None
    routing_indices: list | None = None  # [G, L, top_k] int, present when --moe-enable-routing-replay
    routing_dump_id: str | None = None  # Lightweight key for loading routing tensors from .npz
    policy_epoch: list[tuple[int, int]]
    kv_cache_epoch: list[tuple[int, int]]
    num_evictions: int
