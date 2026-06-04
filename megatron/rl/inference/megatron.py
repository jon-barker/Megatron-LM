# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging

import httpx
import torch.distributed as dist
from openai import AsyncOpenAI, DefaultAioHttpClient
from pydantic import PrivateAttr

try:
    import h2  # noqa: F401
    use_http2 = True
except ImportError:
    use_http2 = False

from megatron.core.inference.config import KVCacheManagementMode
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
from megatron.core.inference.inference_request import unwrap_serialized_tensors
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import log_single_rank
from megatron.training.global_vars import get_args, get_tokenizer

from ..inference.inference_interface import (
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)
from ..server.api import InferenceServer

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class MegatronLocal(InferenceServer, ReturnsTokens, ReturnsRaw):
    """Interface to use MCoreEngine directly as an inference engine."""

    host: str
    port: int

    _client: InferenceClient = PrivateAttr(None)
    _inference_engine: DynamicInferenceEngine = PrivateAttr(None)
    _rl_kv_cache_management_mode: KVCacheManagementMode = PrivateAttr(None)
    _openai_client: AsyncOpenAI = PrivateAttr(None)

    @staticmethod
    def _get_response_field(obj, field_name: str):
        if isinstance(obj, dict):
            return obj.get(field_name)
        value = getattr(obj, field_name, None)
        if value is not None:
            return value
        model_extra = getattr(obj, 'model_extra', None)
        if isinstance(model_extra, dict) and field_name in model_extra:
            return model_extra[field_name]
        if hasattr(obj, 'model_dump'):
            dumped = obj.model_dump()
            if field_name in dumped:
                return dumped[field_name]
        return None

    @classmethod
    def _extract_top_logprobs(cls, choice) -> list[list[dict]] | None:
        generated_top_n = cls._get_response_field(choice, 'generated_top_n_logprobs')
        if generated_top_n is None:
            generated_top_n = cls._get_response_field(
                getattr(choice, 'message', None), 'generated_top_n_logprobs'
            )
        if generated_top_n:
            return [
                [
                    {'token': str(token), 'logprob': float(logprob)}
                    for token, logprob in (
                        row.items()
                        if isinstance(row, dict)
                        else [
                            (
                                cls._get_response_field(item, 'token'),
                                cls._get_response_field(item, 'logprob'),
                            )
                            for item in row
                        ]
                    )
                    if token is not None and logprob is not None
                ]
                for row in generated_top_n
            ]

        logprobs = cls._get_response_field(choice, 'logprobs')
        content = cls._get_response_field(logprobs, 'content')
        if not content:
            return None

        rows = []
        for token_logprob in content:
            top_logprobs = cls._get_response_field(token_logprob, 'top_logprobs') or []
            row = []
            for item in top_logprobs:
                token = cls._get_response_field(item, 'token')
                logprob = cls._get_response_field(item, 'logprob')
                if token is not None and logprob is not None:
                    row.append({'token': token, 'logprob': float(logprob)})
            rows.append(row)
        return rows

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:
        tokenizer = get_tokenizer()
        args = get_args()

        # Use the shared, optimized client instead of spinning up a new one
        client = self._openai_client

        extra_body = {
            "skip_prompt_log_probs": True,
            "add_BOS": (not args.rl_skip_bos_token and tokenizer.bos is not None),
        }
        top_logprobs = getattr(args, 'rl_logprob_mismatch_top_k', 0)
        if top_logprobs > 0:
            extra_body["top_logprobs"] = top_logprobs
        if getattr(args, "rl_match_train_logit_moments_to_inference", False):
            extra_body["return_logit_stats"] = True

        # Things that may be problematic when doing this switch
        # - Add BOS token
        # - Skip prompt logprobs
        response = await client.chat.completions.create(
            model="",
            messages=[message.model_dump() for message in request.prompt],
            temperature=request.generation_args.temperature or 1.0,
            top_p=request.generation_args.top_p or 0.0,
            n=1,
            logprobs=True,
            top_logprobs=top_logprobs if top_logprobs > 0 else None,
            extra_body=extra_body,
        )

        choice = response.choices[0]

        return InferenceResponse(
            # TODO: Handle tool calls and reasoning in LLMChatMessage
            response=LLMChatMessage(**choice.message.model_dump(include={'role', 'content'})),
            raw_text=choice.raw_text,
            token_ids=choice.prompt_token_ids + choice.generation_token_ids,
            logprobs=choice.generation_log_probs,
            logit_means=(
                self._get_response_field(choice, "generation_logit_means")
                or self._get_response_field(choice.message, "generation_logit_means")
            ),
            logit_stds=(
                self._get_response_field(choice, "generation_logit_stds")
                or self._get_response_field(choice.message, "generation_logit_stds")
            ),
            top_logprobs=self._extract_top_logprobs(choice),
            routing_indices=getattr(choice, 'moe_topk_indices', None),
            routing_dump_id=getattr(choice, 'routing_dump_id', None),
            prompt_length=len(choice.prompt_token_ids),
            policy_epoch=choice.policy_epoch,
            kv_cache_epoch=choice.kv_cache_epoch,
            num_evictions=getattr(choice, 'num_evictions', 0),
        )

    async def score_prompt_logprobs(self, token_ids: list[int]) -> list[float] | None:
        """Score a fixed token sequence as a prompt and return logprobs for token_ids[1:]."""
        futures = self.submit_prompt_logprob_requests([token_ids])
        if not futures:
            return None
        result = await futures[0]
        return self._extract_prompt_logprobs_from_result(result)

    def submit_prompt_logprob_requests(self, token_ids_batch: list[list[int]]) -> list[asyncio.Future]:
        if self._client is None:
            return []
        futures = []
        for token_ids in token_ids_batch:
            sampling_params = SamplingParams(
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                return_log_probs=True,
                skip_prompt_log_probs=False,
                num_tokens_to_generate=1,
            )
            futures.append(self._client.add_request(token_ids, sampling_params))
        return futures

    @staticmethod
    def _extract_prompt_logprobs_from_result(result) -> list[float] | None:
        result = unwrap_serialized_tensors(result)
        prompt_log_probs = result.get("prompt_log_probs")
        if prompt_log_probs is None:
            return None
        return [float(x) for x in prompt_log_probs]

    @classmethod
    async def launch(cls, model: GPTModel, **kwargs):
        # Import here to avoid circular imports
        from megatron.inference.utils import get_dynamic_inference_engine

        args = get_args()
        tokenizer = get_tokenizer()

        if tokenizer.bos is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "WARNING: Tokenizer has no BOS token so prompt will not have BOS token",
            )

        inference_engine: DynamicInferenceEngine = get_dynamic_inference_engine(model=model)
        dp_addr = await inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=41521, launch_inference_coordinator=True,
        )

        if dist.get_rank() == 0:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import start_text_gen_server

            client = InferenceClient(inference_coordinator_address=dp_addr)
            client.start()

            start_text_gen_server(
                coordinator_addr=dp_addr,
                tokenizer=inference_engine.controller.tokenizer,
                rank=dist.get_rank(),
                server_port=kwargs.get('port', 8294),
                parsers=[],
                verbose=kwargs.get('verbose', False),
            )
        else:
            client = None

        launched_server = cls(**kwargs)
        launched_server._client = client
        launched_server._inference_engine = inference_engine
        launched_server._rl_kv_cache_management_mode = KVCacheManagementMode(
            args.rl_kv_cache_management_mode
        )

        concurrency_limit = args.grpo_prompts_per_step * args.grpo_group_size * args.rl_parallel_generation_tasks
        custom_limits = httpx.Limits(
            max_connections=concurrency_limit,
            max_keepalive_connections=concurrency_limit,
        )
        http_client = DefaultAioHttpClient(
            timeout=None,
            limits=custom_limits,
            http2=use_http2
        )

        launched_server._openai_client = AsyncOpenAI(
            base_url=f"http://{launched_server.host}:{launched_server.port}",
            api_key="NONE",
            http_client=http_client
        )

        return launched_server

    async def kill(self):
        # Gracefully close the shared OpenAI client connections
        if self._openai_client is not None:
            await self._openai_client.close()

        if dist.get_rank() == 0:
            self._client.pause_engines()
        await self._inference_engine.wait_until(EngineState.PAUSED)

        if dist.get_rank() == 0:
            self._client.stop_engines()
        await self._inference_engine.wait_until(EngineState.STOPPED)

        if dist.get_rank() == 0:
            self._client.shutdown_coordinator()
            self._client.stop()

        if dist.get_rank() == 0:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import stop_text_gen_server
            stop_text_gen_server()

    def set_generation_epoch(self, generation_epoch: int):
        if dist.get_rank() == 0:
            self._client.set_generation_epoch(generation_epoch)

    async def suspend(self):
        if dist.get_rank() == 0:
            self._client.pause_engines()
        await self._inference_engine.wait_until(EngineState.PAUSED)

        if dist.get_rank() == 0:
            self._client.suspend_engines()
        await self._inference_engine.wait_until(EngineState.SUSPENDED)

    async def resume(self):
        if self._inference_engine._state_events[EngineState.RUNNING].is_set():
            return

        if dist.get_rank() == 0:
            self._client.resume_engines()
        await self._inference_engine.wait_until(EngineState.RESUMED)

        if dist.get_rank() == 0:
            self._client.unpause_engines()
        await self._inference_engine.wait_until(EngineState.RUNNING)
