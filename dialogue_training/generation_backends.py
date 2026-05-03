from __future__ import annotations

from typing import Any, List, Optional

from dialogue_training.model_loader_profiles import resolve_model_loader_profile
from dialogue_training.run_local_memory_eval import (
    _configure_tokenizer_for_generation,
    _extract_first_complete_json_object,
    _generate_batch,
    _load_local_model,
)
class HuggingFaceGenerationBackend:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        **_: Any,
    ):
        tokenizer, model = _load_local_model(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            adapter_path=adapter_path,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = tokenizer
        self.model = model

    def generate_batch(
        self,
        prompts: List[str],
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> List[str]:
        return _generate_batch(
            tokenizer=self.tokenizer,
            model=self.model,
            prompts=prompts,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )


class VLLMGenerationBackend:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        max_input_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: int = 1,
        vllm_tokenizer_mode: str = "auto",
        vllm_trust_remote_code: Optional[bool] = None,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_max_model_len: Optional[int] = None,
        vllm_max_num_seqs: Optional[int] = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_seed: int = 0,
        vllm_disable_eager: bool = False,
        **_: Any,
    ):
        if adapter_path:
            raise NotImplementedError(
                "adapter_path is not supported for backend=vllm in this first pass."
            )

        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError(
                "backend=vllm requires working transformers and vllm installations."
            ) from exc

        profile = resolve_model_loader_profile(None, model_name_or_path)
        trust_remote_code = (
            profile.trust_remote_code
            if vllm_trust_remote_code is None
            else bool(vllm_trust_remote_code)
        )
        effective_tokenizer_path = tokenizer_name_or_path or model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            effective_tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = _configure_tokenizer_for_generation(tokenizer)
        self._SamplingParams = SamplingParams
        self._repetition_penalty = profile.repetition_penalty

        effective_max_model_len = _resolve_vllm_max_model_len(
            vllm_max_model_len=vllm_max_model_len,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )
        effective_max_num_seqs = max(
            1,
            int(vllm_max_num_seqs if vllm_max_num_seqs is not None else batch_size),
        )

        llm_kwargs = {
            "model": model_name_or_path,
            "tokenizer": effective_tokenizer_path,
            "trust_remote_code": trust_remote_code,
            "tokenizer_mode": vllm_tokenizer_mode,
            "tensor_parallel_size": int(vllm_tensor_parallel_size),
            "gpu_memory_utilization": float(vllm_gpu_memory_utilization),
            "max_num_seqs": effective_max_num_seqs,
            "enforce_eager": not bool(vllm_disable_eager),
            "seed": int(vllm_seed),
            "dtype": torch_dtype,
        }
        if effective_max_model_len is not None:
            llm_kwargs["max_model_len"] = effective_max_model_len

        self.llm = LLM(**llm_kwargs)

    def generate_batch(
        self,
        prompts: List[str],
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> List[str]:
        prompt_inputs = _build_vllm_prompt_inputs(
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_input_tokens=max_input_tokens,
        )
        sampling_params = _build_vllm_sampling_params(
            SamplingParams=self._SamplingParams,
            max_new_tokens=max_new_tokens,
            repetition_penalty=self._repetition_penalty,
        )
        request_outputs = self.llm.generate(
            prompt_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        responses: List[str] = []
        for request_output in request_outputs:
            text = ""
            if getattr(request_output, "outputs", None):
                text = str(getattr(request_output.outputs[0], "text", "") or "")
            responses.append(_normalize_generated_text(text))
        return responses


def _resolve_vllm_max_model_len(
    vllm_max_model_len: Optional[int],
    max_input_tokens: Optional[int],
    max_new_tokens: Optional[int],
) -> Optional[int]:
    if vllm_max_model_len is not None:
        return int(vllm_max_model_len)
    if max_input_tokens is None and max_new_tokens is None:
        return None
    return max(1, int(max_input_tokens or 0) + int(max_new_tokens or 0))


def _build_vllm_sampling_params(
    SamplingParams,
    max_new_tokens: int,
    repetition_penalty: Optional[float],
):
    kwargs = {
        "temperature": 0.0,
        "max_tokens": int(max_new_tokens),
    }
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    return SamplingParams(**kwargs)


def _build_vllm_prompt_inputs(
    tokenizer,
    prompts: List[str],
    max_input_tokens: int,
) -> List[dict[str, List[int]]]:
    encoded = tokenizer(
        prompts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    return [{"prompt_token_ids": token_ids} for token_ids in encoded["input_ids"]]


def _normalize_generated_text(text: str) -> str:
    raw = str(text or "").strip()
    return _extract_first_complete_json_object(raw) or raw


def build_generation_backend(
    *,
    backend: str,
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    max_input_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    batch_size: int = 1,
    vllm_tokenizer_mode: str = "auto",
    vllm_trust_remote_code: Optional[bool] = None,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: Optional[int] = None,
    vllm_max_num_seqs: Optional[int] = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_seed: int = 0,
    vllm_disable_eager: bool = False,
):
    resolved_backend = str(backend or "hf").strip().lower()
    common_kwargs = {
        "model_name_or_path": model_name_or_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "adapter_path": adapter_path,
        "torch_dtype": torch_dtype,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "vllm_tokenizer_mode": vllm_tokenizer_mode,
        "vllm_trust_remote_code": vllm_trust_remote_code,
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        "vllm_max_model_len": vllm_max_model_len,
        "vllm_max_num_seqs": vllm_max_num_seqs,
        "vllm_tensor_parallel_size": vllm_tensor_parallel_size,
        "vllm_seed": vllm_seed,
        "vllm_disable_eager": vllm_disable_eager,
    }
    if resolved_backend == "hf":
        return HuggingFaceGenerationBackend(**common_kwargs)
    if resolved_backend == "vllm":
        return VLLMGenerationBackend(**common_kwargs)
    raise ValueError(f"Unsupported backend: {backend}")
