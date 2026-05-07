#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from fact_track.evaluation import (
    IncrementalWriter,
    collect_prompt_files,
    evaluate_file,
    load_jobs,
    print_summary,
    safe_name,
    write_run_config,
)


DEFAULT_ENDPOINT = "http://localhost:8000/v1"
DEFAULT_INPUT_TOKENS = 1024 * 128
DEFAULT_OUTPUT_TOKENS = 16 * 1024
DEFAULT_TOTAL_CONTEXT = DEFAULT_INPUT_TOKENS + DEFAULT_OUTPUT_TOKENS


@dataclass(frozen=True)
class VllmModelSpec:
    key: str
    aliases: Tuple[str, ...]
    model_id: str
    official_context_tokens: int
    default_max_model_len: int
    tensor_parallel_size: int
    trust_remote_code: bool = False
    reasoning_parser: str = ""
    tokenizer_mode: str = ""
    config_format: str = ""
    load_format: str = ""
    extra_serve_args: Tuple[str, ...] = ()
    extra_body: Dict[str, Any] = field(default_factory=dict)
    prompt_prefix: str = ""
    notes: Tuple[str, ...] = ()
    sources: Tuple[str, ...] = ()

    @property
    def all_names(self) -> Tuple[str, ...]:
        return (self.key, self.model_id, *self.aliases)


def _qwen35_spec(key: str, model_id: str, tp: int, *aliases: str) -> VllmModelSpec:
    return VllmModelSpec(
        key=key,
        aliases=aliases,
        model_id=model_id,
        official_context_tokens=262144,
        default_max_model_len=262144,
        tensor_parallel_size=tp,
        reasoning_parser="qwen3",
        extra_serve_args=("--language-model-only",),
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        notes=(
            "Qwen3.5 thinks by default; vLLM requests should pass "
            "chat_template_kwargs.enable_thinking=false for non-thinking mode.",
            "Official Qwen3.5 vLLM examples use --reasoning-parser qwen3 and max model length 262144.",
        ),
        sources=(
            "https://huggingface.co/Qwen/Qwen3.5-9B",
            "https://huggingface.co/Qwen/Qwen3.5-27B",
            "https://huggingface.co/Qwen/Qwen3.5-35B-A3B",
            "https://docs.vllm.ai/en/stable/features/reasoning_outputs/",
        ),
    )


VLLM_MODEL_SPECS: Tuple[VllmModelSpec, ...] = (
    VllmModelSpec(
        key="qwen2.5-7b-instruct-1m",
        aliases=("Qwen2.5-7B-Instruct-1M", "qwen2.5-7b-1m", "qwen25-7b-1m"),
        model_id="Qwen/Qwen2.5-7B-Instruct-1M",
        official_context_tokens=1010000,
        default_max_model_len=DEFAULT_TOTAL_CONTEXT,
        tensor_parallel_size=4,
        extra_serve_args=(
            "--enable-chunked-prefill",
            "--max-num-batched-tokens",
            str(DEFAULT_INPUT_TOKENS),
            "--enforce-eager",
            "--max-num-seqs",
            "1",
        ),
        notes=(
            "Official card says full context is 1010000 tokens and recommends custom vLLM for ultra-long text.",
            "The card's generation length is 8192; this evaluator still uses 16384 because the benchmark setting requires it.",
        ),
        sources=("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M",),
    ),
    VllmModelSpec(
        key="glm-4-9b-chat-1m",
        aliases=("GLM-4-9B-Chat-1M", "glm4-9b-chat-1m"),
        model_id="zai-org/glm-4-9b-chat-1m",
        official_context_tokens=1048576,
        default_max_model_len=DEFAULT_TOTAL_CONTEXT,
        tensor_parallel_size=1,
        trust_remote_code=True,
        extra_serve_args=(
            "--enforce-eager",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens",
            "8192",
        ),
        notes=(
            "Official vLLM sample uses trust_remote_code, enforce_eager, and suggests chunked prefill for OOM.",
            "This GLM-4-9B-Chat-1M model is not documented as a default thinking model.",
        ),
        sources=("https://huggingface.co/zai-org/glm-4-9b-chat-1m",),
    ),
    _qwen35_spec("qwen3.5-9b", "Qwen/Qwen3.5-9B", 1, "Qwen3.5-9B", "qwen35-9b"),
    VllmModelSpec(
        key="hunyuan-a13b-instruct",
        aliases=("Hunyuan-A13B-Instruct", "hunyuan-a13b"),
        model_id="tencent/Hunyuan-A13B-Instruct",
        official_context_tokens=262144,
        default_max_model_len=262144,
        tensor_parallel_size=4,
        trust_remote_code=True,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        prompt_prefix="/no_think\n",
        notes=(
            "Hunyuan-A13B defaults to slow-thinking mode.",
            "Official card documents two non-thinking controls: apply_chat_template(enable_thinking=false) and /no_think.",
            "vLLM reasoning parser support for Hunyuan A13B is documented as under development.",
        ),
        sources=(
            "https://huggingface.co/tencent/Hunyuan-A13B-Instruct",
            "https://recipes.vllm.ai/tencent/Hunyuan-A13B-Instruct",
        ),
    ),
    _qwen35_spec("qwen3.5-27b", "Qwen/Qwen3.5-27B", 8, "Qwen3.5-27B", "qwen35-27b"),
    VllmModelSpec(
        key="gemma-4-31b-it",
        aliases=("gemma-4-31B-it", "google/gemma-4-31B-it"),
        model_id="google/gemma-4-31B-it",
        official_context_tokens=262144,
        default_max_model_len=262144,
        tensor_parallel_size=2,
        reasoning_parser="gemma4",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        notes=(
            "Gemma 4 31B has a 256K context window and configurable thinking mode.",
            "The official model card disables thinking with enable_thinking=false in apply_chat_template.",
        ),
        sources=(
            "https://huggingface.co/google/gemma-4-31B-it",
            "https://recipes.vllm.ai/Google/gemma-4-31B-it",
        ),
    ),
    VllmModelSpec(
        key="glm-4.7-flash",
        aliases=("GLM-4.7-Flash", "glm47-flash"),
        model_id="zai-org/GLM-4.7-Flash",
        official_context_tokens=200000,
        default_max_model_len=200000,
        tensor_parallel_size=1,
        trust_remote_code=True,
        reasoning_parser="glm45",
        extra_body={
            "thinking": {"type": "disabled"},
        },
        notes=(
            "Z.AI documents GLM-4.7 thinking as enabled by default and disables it with thinking.type=disabled.",
            "Z.AI documents the GLM-4.7 series context length as 200K and maximum output tokens as 128K.",
            "vLLM GLM-4.7 recipes use the glm45 reasoning parser and glm47 tool parser for tool use.",
        ),
        sources=(
            "https://docs.z.ai/guides/capabilities/thinking-mode",
            "https://recipes.vllm.ai/zai-org/GLM-4.7",
            "https://huggingface.co/zai-org/GLM-4.7-Flash",
        ),
    ),
    _qwen35_spec(
        "qwen3.5-35b-a3b",
        "Qwen/Qwen3.5-35B-A3B",
        8,
        "Qwen3.5-35B-A3B",
        "qwen35-35b-a3b",
        "Qwen/Qwen3.5-35B-A3B",
    ),
    VllmModelSpec(
        key="ministral-3-8b-instruct-2512",
        aliases=("mistralai/Ministral-3-8B-Instruct-2512", "Ministral-3-8B-Instruct-2512"),
        model_id="mistralai/Ministral-3-8B-Instruct-2512",
        official_context_tokens=262144,
        default_max_model_len=262144,
        tensor_parallel_size=1,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        notes=(
            "Official card says Ministral 3 8B Instruct supports a 256K context window.",
            "Official vLLM launch uses tokenizer_mode/config_format/load_format=mistral.",
            "This is the Instruct model, not the Reasoning variant, so no thinking switch is documented.",
        ),
        sources=("https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512",),
    ),
    VllmModelSpec(
        key="gemma-4-26b-a4b-it",
        aliases=("google/gemma-4-26B-A4B-it", "gemma-4-26B-A4B-it"),
        model_id="google/gemma-4-26B-A4B-it",
        official_context_tokens=262144,
        default_max_model_len=262144,
        tensor_parallel_size=1,
        reasoning_parser="gemma4",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        notes=(
            "Gemma 4 26B A4B has a 256K context window and configurable thinking mode.",
            "The official model card disables thinking with enable_thinking=false in apply_chat_template.",
        ),
        sources=(
            "https://huggingface.co/google/gemma-4-26B-A4B-it",
            "https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it",
        ),
    ),
)

MODEL_SPECS_BY_NAME: Dict[str, VllmModelSpec] = {
    name.lower(): spec for spec in VLLM_MODEL_SPECS for name in spec.all_names
}


def compact_json(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def resolve_model_spec(model_key: str, model_path: str) -> VllmModelSpec:
    key = (model_key or model_path).strip()
    if not key:
        names = ", ".join(spec.key for spec in VLLM_MODEL_SPECS)
        raise ValueError(f"Missing --model-key or --model-path. Known model keys: {names}")
    spec = MODEL_SPECS_BY_NAME.get(key.lower())
    if spec:
        return spec

    return VllmModelSpec(
        key=safe_name(key),
        aliases=(key,),
        model_id=key,
        official_context_tokens=0,
        default_max_model_len=DEFAULT_TOTAL_CONTEXT,
        tensor_parallel_size=1,
        notes=("Custom model path; no built-in model-specific thinking control is known.",),
    )


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def parse_json_object(value: str, *, name: str) -> Dict[str, Any]:
    if not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return parsed


def normalize_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value[: -len("/chat/completions")]
    return value


class PromptTokenGuard:
    def __init__(self, *, model_id: str, max_input_tokens: int, mode: str, trust_remote_code: bool) -> None:
        self.max_input_tokens = max_input_tokens
        self.mode = mode
        self.tokenizer = None
        self.disabled_reason = ""
        self.warned_load_failure = False
        if mode == "off":
            return

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception as exc:
            self.disabled_reason = str(exc)

    def check(self, prompt: str) -> None:
        if self.mode == "off":
            return
        if self.tokenizer is None:
            if not self.warned_load_failure:
                print(f"[warn] token check disabled: {self.disabled_reason}", file=sys.stderr)
                self.warned_load_failure = True
            return

        token_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if token_count <= self.max_input_tokens:
            return

        message = f"prompt has {token_count} tokens, exceeding --max-input-tokens={self.max_input_tokens}"
        if self.mode == "error":
            raise RuntimeError(message)
        print(f"[warn] {message}", file=sys.stderr)


class VllmOpenAIClient:
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        timeout_s: int,
        max_retries: int,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        min_p: Optional[float],
        top_k: Optional[int],
        extra_body: Dict[str, Any],
        prompt_prefix: str,
        token_guard: PromptTokenGuard,
    ) -> None:
        self.endpoint = normalize_endpoint(endpoint)
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max(1, max_retries)
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.top_k = top_k
        self.extra_body = extra_body
        self.prompt_prefix = prompt_prefix
        self.token_guard = token_guard

    def generate(self, prompt: str) -> str:
        user_prompt = f"{self.prompt_prefix}{prompt}" if self.prompt_prefix else prompt
        self.token_guard.check(user_prompt)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        if self.presence_penalty != 0.0:
            payload["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        payload = merge_dict(payload, self.extra_body)

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.endpoint}/chat/completions"
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            req = request.Request(url, data=data, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_s) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                return self._extract_content(json.loads(raw))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                last_error = RuntimeError(f"vllm HTTP {exc.code}: {body[:1000]}")
                if exc.code in (400, 401, 403, 404):
                    raise last_error from exc
                if attempt < self.max_retries - 1:
                    wait_s = min(30.0, 2.0 * (2 ** attempt))
                    print(f"[warn] vllm HTTP {exc.code}; retry after {wait_s:.1f}s", file=sys.stderr)
                    time.sleep(wait_s)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait_s = min(30.0, 2.0 * (2 ** attempt))
                    print(f"[warn] vllm request error; retry after {wait_s:.1f}s: {exc}", file=sys.stderr)
                    time.sleep(wait_s)

        raise RuntimeError(f"vllm request failed: {last_error}")

    @staticmethod
    def _extract_content(obj: Dict[str, Any]) -> str:
        message = (obj.get("choices", [{}])[0].get("message", {}) or {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content or "").strip()


def build_vllm_command(
    spec: VllmModelSpec,
    *,
    host: str,
    port: int,
    max_model_len: int,
    tensor_parallel_size: int,
    served_model_name: str,
    gpu_memory_utilization: float,
    extra_serve_args: List[str],
) -> List[str]:
    cmd = [
        "vllm",
        "serve",
        spec.model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if spec.trust_remote_code:
        cmd.append("--trust-remote-code")
    if spec.reasoning_parser:
        cmd.extend(["--reasoning-parser", spec.reasoning_parser])
    if spec.tokenizer_mode:
        cmd.extend(["--tokenizer-mode", spec.tokenizer_mode])
    if spec.config_format:
        cmd.extend(["--config-format", spec.config_format])
    if spec.load_format:
        cmd.extend(["--load-format", spec.load_format])
    chat_template_kwargs = spec.extra_body.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict) and chat_template_kwargs:
        cmd.extend(["--default-chat-template-kwargs", compact_json(chat_template_kwargs)])
    cmd.extend(spec.extra_serve_args)
    cmd.extend(extra_serve_args)
    return cmd


def print_known_models() -> None:
    print("Known vLLM model keys:")
    for spec in VLLM_MODEL_SPECS:
        alias_text = ", ".join(spec.aliases)
        print(
            f"  {spec.key:32s} -> {spec.model_id} "
            f"(ctx={spec.official_context_tokens}, tp={spec.tensor_parallel_size})"
        )
        if alias_text:
            print(f"    aliases: {alias_text}")
        if spec.notes:
            print(f"    note: {spec.notes[0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate schema_longest prompts with local vLLM models.")
    parser.add_argument("-i", "--input", default="fact_track_schema_longest", help="Prompt JSON file or directory.")
    parser.add_argument("-o", "--output-dir", default="eval_results/local", help="Directory for CSV/JSONL outputs.")
    parser.add_argument("--backend", choices=("vllm", "placeholder"), default="vllm")
    parser.add_argument("--model-key", default="", help="Known model key or alias. Use --list-models to inspect.")
    parser.add_argument("--model-path", default="", help="Custom model path or Hugging Face identifier.")
    parser.add_argument("--served-model-name", default="", help="Model name exposed by the vLLM server.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="OpenAI-compatible vLLM endpoint.")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--num-samples", type=int, default=0, help="Max samples per file. 0 means all.")
    parser.add_argument("--level-start", type=int, default=None)
    parser.add_argument("--level-end", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_INPUT_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 omits top_k from the request.")
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--extra-body-json", default="", help="Merge additional JSON object into every request body.")
    parser.add_argument("--disable-built-in-thinking-control", action="store_true")
    parser.add_argument("--disable-prompt-prefix", action="store_true")
    parser.add_argument("--token-check", choices=("off", "warn", "error"), default="warn")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only list selected prompt files and sample counts.")
    parser.add_argument("--list-models", action="store_true", help="Print supported vLLM model keys and exit.")
    parser.add_argument("--print-vllm-command", action="store_true", help="Print a recommended vLLM serve command and exit.")
    parser.add_argument("--serve-host", default="0.0.0.0")
    parser.add_argument("--serve-port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=0, help="For --print-vllm-command. 0 uses model default.")
    parser.add_argument("--tensor-parallel-size", type=int, default=0, help="For --print-vllm-command. 0 uses model default.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--serve-arg", action="append", default=[], help="Extra vLLM serve argument token; repeat as needed.")
    return parser.parse_args()


def count_samples(path: Path, max_samples: Optional[int]) -> int:
    return len(load_jobs(path, max_samples))


def main() -> None:
    args = parse_args()
    if args.list_models:
        print_known_models()
        return

    if args.backend == "placeholder":
        print("Local placeholder backend is not implemented. Use --backend vllm.", file=sys.stderr)
        raise SystemExit(2)

    spec = resolve_model_spec(args.model_key, args.model_path)
    served_model_name = args.served_model_name.strip() or spec.model_id
    max_model_len = args.max_model_len if args.max_model_len > 0 else spec.default_max_model_len
    tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size > 0 else spec.tensor_parallel_size

    if args.print_vllm_command:
        cmd = build_vllm_command(
            spec,
            host=args.serve_host,
            port=args.serve_port,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            served_model_name=served_model_name,
            gpu_memory_utilization=args.gpu_memory_utilization,
            extra_serve_args=args.serve_arg,
        )
        print(" ".join(shlex.quote(part) for part in cmd))
        return

    max_samples = args.num_samples if args.num_samples > 0 else None
    max_files = args.max_files if args.max_files > 0 else None
    prompt_files = collect_prompt_files(
        args.input,
        level_start=args.level_start,
        level_end=args.level_end,
        max_files=max_files,
    )

    if args.dry_run:
        total = 0
        print(f"Selected {len(prompt_files)} prompt files:")
        for file_path in prompt_files:
            samples = count_samples(file_path, max_samples)
            total += samples
            print(f"  {file_path.name}: {samples} samples")
        print(f"Total selected samples: {total}")
        print(f"model_key={spec.key} model_id={spec.model_id} endpoint={normalize_endpoint(args.endpoint)}")
        print(f"max_input_tokens={args.max_input_tokens} max_output_tokens={args.max_output_tokens}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_label = served_model_name
    prefix = f"eval_local_vllm_{safe_name(model_label)}"
    csv_path = output_dir / f"{prefix}.csv"
    jsonl_path = output_dir / f"{prefix}.jsonl"
    config_path = output_dir / f"{prefix}.config.json"

    built_in_extra_body = {} if args.disable_built_in_thinking_control else spec.extra_body
    extra_body = merge_dict(built_in_extra_body, parse_json_object(args.extra_body_json, name="--extra-body-json"))
    prompt_prefix = "" if args.disable_prompt_prefix else spec.prompt_prefix

    writer = IncrementalWriter(
        csv_path=csv_path,
        jsonl_path=jsonl_path,
        runner="local",
        provider="vllm",
        model=model_label,
        resume=not args.no_resume,
        overwrite=args.overwrite,
    )

    write_run_config(
        config_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "runner": "local",
            "provider": "vllm",
            "model_key": spec.key,
            "model_id": spec.model_id,
            "served_model_name": served_model_name,
            "endpoint": normalize_endpoint(args.endpoint),
            "input": str(args.input),
            "files": [file.name for file in prompt_files],
            "num_samples": max_samples,
            "official_context_tokens": spec.official_context_tokens,
            "max_input_tokens": args.max_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "recommended_max_model_len": max_model_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k if args.top_k > 0 else None,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "extra_body": extra_body,
            "prompt_prefix": prompt_prefix,
            "token_check": args.token_check,
            "concurrency": args.concurrency,
            "sleep": args.sleep,
            "resume": not args.no_resume,
            "model_notes": spec.notes,
            "model_sources": spec.sources,
        },
    )

    if spec.official_context_tokens and DEFAULT_TOTAL_CONTEXT > spec.official_context_tokens:
        print(
            f"[warn] requested input+output budget {DEFAULT_TOTAL_CONTEXT} exceeds "
            f"official context {spec.official_context_tokens} for {spec.model_id}",
            file=sys.stderr,
        )

    token_guard = PromptTokenGuard(
        model_id=args.model_path.strip() or spec.model_id,
        max_input_tokens=args.max_input_tokens,
        mode=args.token_check,
        trust_remote_code=spec.trust_remote_code,
    )
    client = VllmOpenAIClient(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=served_model_name,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        top_k=args.top_k if args.top_k > 0 else None,
        extra_body=extra_body,
        prompt_prefix=prompt_prefix,
        token_guard=token_guard,
    )

    print(
        f"provider=vllm model={served_model_name} model_key={spec.key} files={len(prompt_files)} "
        f"concurrency={args.concurrency} endpoint={normalize_endpoint(args.endpoint)} output={output_dir}"
    )
    evaluated = []
    for idx, file_path in enumerate(prompt_files, start=1):
        if writer.is_done(file_path.name):
            print(f"[{idx}/{len(prompt_files)}] {file_path.name} -> skip")
            continue

        started = time.time()
        print(f"[{idx}/{len(prompt_files)}] evaluating {file_path.name}", flush=True)
        result, records = evaluate_file(
            file_path,
            model_call=client.generate,
            max_samples=max_samples,
            concurrency=args.concurrency,
            sleep_s=args.sleep,
            include_prompt=args.include_prompt,
        )
        writer.save(result, records)
        elapsed = time.time() - started
        evaluated.append(result)
        print(
            f"  {result.level}: prompts={result.total_prompts} "
            f"prompt_em={result.prompt_exact_match}/{result.total_prompts} "
            f"schema_acc={result.schema_accuracy:.4f} parse={result.parse_rate:.4f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    print_summary(evaluated, provider="vllm", model=model_label)
    print(f"Saved:\n  {csv_path}\n  {jsonl_path}\n  {config_path}")


if __name__ == "__main__":
    main()
