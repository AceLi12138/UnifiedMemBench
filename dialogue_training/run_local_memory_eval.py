from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from dialogue_training.model_loader_profiles import (
    prepare_config_for_model_loading,
    prepare_model_class_for_loading,
    prepare_model_instance_for_loading,
    resolve_model_loader_profile,
)

_EVAL_DIR = Path(__file__).resolve().parent.parent / "dialogue_gen_api" / "evaluation"

DEFAULT_TASK_TYPE_MAX_NEW_TOKENS = {
    "Information Extraction": 128,
    "Temporal Reasoning": 128,
    "Knowledge Updating": 128,
    "Memory Arbitration": 192,
    "Multi-session Reasoning": 384,
    "Event Summarization": 448,
}
LOCAL_MEMORY_SYSTEM_PROMPT = (
    "You answer memory-grounded question answering tasks. "
    "Follow the user's required JSON schema exactly. "
    "Do not role-play or continue any dialogue transcript."
)
LOCAL_MEMORY_ANSWER_SUFFIX = (
    "[Answer]\n"
    "The original dialogue is not available now. Answer from memory only.\n"
    "Return ONLY one complete JSON object that satisfies the required fields.\n"
    "Do not write user: or assistant: turns.\n"
    "Do not add markdown or explanation.\n"
    "Ensure the JSON object is complete and closed before stopping."
)


def _chat_template_kwargs_for_generation(*, disable_thinking: bool) -> Dict[str, Any]:
    if not disable_thinking:
        return {}
    return {"enable_thinking": False}


def _sample_task_type(sample: Dict[str, Any]) -> str:
    return str(sample.get("metadata", {}).get("task_type", "")).strip()


def _resolve_sample_max_new_tokens(
    tokenizer,
    sample: Dict[str, Any],
    max_new_tokens: int,
    adaptive_max_new_tokens: bool,
    adaptive_multiplier: float,
    adaptive_min_new_tokens: int,
    adaptive_max_new_tokens_cap: Optional[int],
    use_task_type_budgets: bool = True,
    task_type_budgets: Optional[Dict[str, int]] = None,
) -> int:
    hard_cap = max_new_tokens
    if adaptive_max_new_tokens and adaptive_max_new_tokens_cap is not None:
        hard_cap = min(hard_cap, adaptive_max_new_tokens_cap)

    sample_budget: Optional[int] = None
    budget_profile = task_type_budgets or DEFAULT_TASK_TYPE_MAX_NEW_TOKENS

    if use_task_type_budgets:
        task_type = _sample_task_type(sample)
        task_budget = int(budget_profile.get(task_type, hard_cap))
        sample_budget = min(hard_cap, max(1, task_budget))

    if adaptive_max_new_tokens:
        reference = (sample.get("reference") or "").strip()
        ref_tokens = len(tokenizer.encode(reference, add_special_tokens=False))
        scaled_budget = math.ceil(ref_tokens * adaptive_multiplier)
        adaptive_budget = min(hard_cap, max(adaptive_min_new_tokens, scaled_budget))
        sample_budget = adaptive_budget if sample_budget is None else max(sample_budget, adaptive_budget)

    return sample_budget if sample_budget is not None else hard_cap


def _estimate_prompt_tokens(tokenizer, sample: Dict[str, Any], max_input_tokens: int) -> int:
    prompt = str(sample.get("prompt", ""))
    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    return min(tokens, max_input_tokens)


def _filter_samples_for_shard(
    samples: List[Dict[str, Any]],
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if num_shards == 1:
        return samples

    filtered: List[Dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % num_shards
        if bucket == shard_index:
            filtered.append(sample)
    return filtered


def _build_generation_batches(
    tokenizer,
    samples: List[Dict[str, Any]],
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    adaptive_max_new_tokens: bool,
    adaptive_multiplier: float,
    adaptive_min_new_tokens: int,
    adaptive_max_new_tokens_cap: Optional[int],
    use_task_type_budgets: bool = True,
    task_type_budgets: Optional[Dict[str, int]] = None,
) -> List[List[Dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not samples:
        return []

    grouped: Dict[Tuple[int, str], List[Tuple[int, Dict[str, Any], int]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        sample_budget = _resolve_sample_max_new_tokens(
            tokenizer=tokenizer,
            sample=sample,
            max_new_tokens=max_new_tokens,
            adaptive_max_new_tokens=adaptive_max_new_tokens,
            adaptive_multiplier=adaptive_multiplier,
            adaptive_min_new_tokens=adaptive_min_new_tokens,
            adaptive_max_new_tokens_cap=adaptive_max_new_tokens_cap,
            use_task_type_budgets=use_task_type_budgets,
            task_type_budgets=task_type_budgets,
        )
        prompt_tokens = _estimate_prompt_tokens(tokenizer, sample, max_input_tokens=max_input_tokens)
        task_type = _sample_task_type(sample)
        grouped[(sample_budget, task_type)].append((idx, sample, prompt_tokens))

    ordered_batches: List[List[Dict[str, Any]]] = []
    for key in sorted(grouped.keys()):
        rows = sorted(grouped[key], key=lambda item: (item[2], item[0]))
        for start in range(0, len(rows), batch_size):
            ordered_batches.append([item[1] for item in rows[start: start + batch_size]])
    return ordered_batches


def _load_memory_eval_utils():
    if __package__ in {None, ""}:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    module = importlib.import_module("dialogue_training.memory_eval_utils")
    return module.load_memory_eval_jsonl, module.score_result_rows


def _load_judge_llm_module():
    if str(_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(_EVAL_DIR))
    return importlib.import_module("llm")


def _render_local_memory_prompt(
    tokenizer,
    sample: Dict[str, Any],
    disable_thinking: bool = False,
) -> str:
    base_prompt = str(sample.get("prompt", "")).rstrip()
    user_prompt = f"{base_prompt}\n\n{LOCAL_MEMORY_ANSWER_SUFFIX}"
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(
            [
                {"role": "system", "content": LOCAL_MEMORY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs_for_generation(disable_thinking=disable_thinking),
        )
    return (
        f"System: {LOCAL_MEMORY_SYSTEM_PROMPT}\n\n"
        f"User: {user_prompt}\n\n"
        "Assistant:\n"
    )


def _prepare_memory_samples_for_local_generation(
    tokenizer,
    samples: List[Dict[str, Any]],
    disable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for sample in samples:
        prepared_sample = dict(sample)
        prepared_sample["raw_prompt"] = sample.get("prompt", "")
        prepared_sample["prompt"] = _render_local_memory_prompt(
            tokenizer,
            sample,
            disable_thinking=disable_thinking,
        )
        prepared.append(prepared_sample)
    return prepared


def _configure_tokenizer_for_generation(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _extract_first_complete_json_object(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[idx:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
    return None


def _build_first_json_stopping_criteria(tokenizer, prompt_len: int):
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        return None

    class _StopOnFirstJSONObject(StoppingCriteria):
        def __init__(self, current_tokenizer, current_prompt_len: int):
            self.tokenizer = current_tokenizer
            self.prompt_len = current_prompt_len

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            generated_ids = input_ids[:, self.prompt_len:]
            for seq in generated_ids:
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                if _extract_first_complete_json_object(text) is None:
                    return False
            return True

    return StoppingCriteriaList([_StopOnFirstJSONObject(tokenizer, prompt_len)])


def _load_local_model(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    torch_dtype: str = "bfloat16",
):
    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Local memory eval requires a working transformers installation. "
            "Please fix the current environment and retry."
        ) from exc

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    resolved_dtype = dtype_map.get(torch_dtype.lower(), torch.bfloat16)

    effective_tokenizer_path = tokenizer_name_or_path or model_name_or_path
    initial_profile = resolve_model_loader_profile(None, effective_tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(
        effective_tokenizer_path,
        trust_remote_code=initial_profile.trust_remote_code,
    )
    tokenizer = _configure_tokenizer_for_generation(tokenizer)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=initial_profile.trust_remote_code,
    )
    config, profile, _ = prepare_config_for_model_loading(config, model_name_or_path)
    resolved_model_class, profile, _ = prepare_model_class_for_loading(
        config,
        model_name_or_path,
        profile.auto_model_class_name,
    )

    auto_model_classes = {
        "AutoModel": AutoModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
    }
    auto_model_class = resolved_model_class or auto_model_classes.get(
        profile.auto_model_class_name,
        AutoModelForCausalLM,
    )

    model = auto_model_class.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=resolved_dtype,
        device_map=profile.device_map,
        trust_remote_code=profile.trust_remote_code,
    )
    if adapter_path:
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError(
                "Adapter loading requires a working peft installation. "
                "Please fix the current environment and retry."
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    model, profile, _ = prepare_model_instance_for_loading(model, model_name_or_path)

    if profile.post_load_device and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        model = model.to(profile.post_load_device)

    setattr(model, "_umb_loader_profile", profile)
    model.eval()
    return tokenizer, model


def _resolve_batch_max_new_tokens(
    tokenizer,
    batch: List[Dict[str, Any]],
    max_new_tokens: int,
    adaptive_max_new_tokens: bool,
    adaptive_multiplier: float,
    adaptive_min_new_tokens: int,
    adaptive_max_new_tokens_cap: Optional[int],
    use_task_type_budgets: bool = True,
    task_type_budgets: Optional[Dict[str, int]] = None,
) -> int:
    budgets: List[int] = []
    for sample in batch:
        budgets.append(
            _resolve_sample_max_new_tokens(
                tokenizer=tokenizer,
                sample=sample,
                max_new_tokens=max_new_tokens,
                adaptive_max_new_tokens=adaptive_max_new_tokens,
                adaptive_multiplier=adaptive_multiplier,
                adaptive_min_new_tokens=adaptive_min_new_tokens,
                adaptive_max_new_tokens_cap=adaptive_max_new_tokens_cap,
                use_task_type_budgets=use_task_type_budgets,
                task_type_budgets=task_type_budgets,
            )
        )

    return max(budgets, default=max_new_tokens)


def _build_generation_kwargs(
    tokenizer,
    model,
    max_new_tokens: int,
    stopping_criteria,
) -> Dict[str, Any]:
    profile = getattr(model, "_umb_loader_profile", None)
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "stopping_criteria": stopping_criteria,
    }

    omit_temperature = bool(getattr(profile, "omit_temperature_when_greedy", False))
    if not omit_temperature:
        kwargs["temperature"] = 0.0

    if bool(getattr(profile, "use_generation_config_eos_token_id", False)):
        generation_config = getattr(model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id

    repetition_penalty = getattr(profile, "repetition_penalty", None)
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty

    return kwargs


def _generate_batch(
    tokenizer,
    model,
    prompts: List[str],
    max_input_tokens: int,
    max_new_tokens: int,
) -> List[str]:
    import torch

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    stopping_criteria = _build_first_json_stopping_criteria(tokenizer, prompt_len)

    generation_kwargs = _build_generation_kwargs(
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
    )

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            **generation_kwargs,
        )

    decoded: List[str] = []
    for seq in generated:
        raw_text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        decoded.append(_extract_first_complete_json_object(raw_text) or raw_text)
    return decoded


def _load_existing_results(results_path: Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not results_path.exists():
        return existing
    with results_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            existing[row["id"]] = row
    return existing


def run_local_memory_eval(
    dataset_path: Path,
    model_name_or_path: str,
    output_dir: Path,
    tokenizer_name_or_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    batch_size: int = 1,
    max_input_tokens: int = 2048,
    max_new_tokens: int = 512,
    torch_dtype: str = "bfloat16",
    judge_model: Optional[str] = "mimo-v2-flash",
    force_new: bool = False,
    adaptive_max_new_tokens: bool = False,
    adaptive_max_new_tokens_multiplier: float = 2.0,
    adaptive_max_new_tokens_min: int = 64,
    adaptive_max_new_tokens_cap: Optional[int] = 256,
    use_task_type_budgets: bool = True,
    num_shards: int = 1,
    shard_index: int = 0,
    backend: str = "hf",
    vllm_tokenizer_mode: str = "auto",
    vllm_trust_remote_code: Optional[bool] = None,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: Optional[int] = None,
    vllm_max_num_seqs: Optional[int] = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_seed: int = 0,
    vllm_disable_eager: bool = False,
    disable_thinking: bool = False,
) -> Dict[str, Any]:
    load_memory_eval_jsonl, score_result_rows = _load_memory_eval_utils()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "detailed_results.jsonl"
    existing = {} if force_new else _load_existing_results(results_path)
    all_samples = load_memory_eval_jsonl(dataset_path)
    samples = _filter_samples_for_shard(all_samples, num_shards=num_shards, shard_index=shard_index)
    pending = [sample for sample in samples if sample["id"] not in existing]

    from dialogue_training.generation_backends import build_generation_backend

    generation_backend = build_generation_backend(
        backend=backend,
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        adapter_path=adapter_path,
        torch_dtype=torch_dtype,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        vllm_tokenizer_mode=vllm_tokenizer_mode,
        vllm_trust_remote_code=vllm_trust_remote_code,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_seed=vllm_seed,
        vllm_disable_eager=vllm_disable_eager,
    )
    tokenizer = generation_backend.tokenizer
    pending = _prepare_memory_samples_for_local_generation(
        tokenizer,
        pending,
        disable_thinking=disable_thinking,
    )

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    shard_note = ""
    if num_shards > 1:
        shard_note = f" Shard={shard_index + 1}/{num_shards}, shard_samples={len(samples)}, total_dataset_samples={len(all_samples)}."
    print(
        f"Loaded {len(samples)} eval samples from {dataset_path}. "
        f"Cached={len(existing)}, pending={len(pending)}.{shard_note}"
    )

    batches = _build_generation_batches(
        tokenizer=tokenizer,
        samples=pending,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        adaptive_max_new_tokens=adaptive_max_new_tokens,
        adaptive_multiplier=adaptive_max_new_tokens_multiplier,
        adaptive_min_new_tokens=adaptive_max_new_tokens_min,
        adaptive_max_new_tokens_cap=adaptive_max_new_tokens_cap,
        use_task_type_budgets=use_task_type_budgets,
    )

    mode = "w" if force_new else "a"
    with results_path.open(mode, encoding="utf-8") as fh:
        progress = tqdm(total=len(pending), desc="Generating memory-eval responses", unit="sample") if (tqdm and pending) else None
        completed = 0
        for batch in batches:
            batch_max_new_tokens = _resolve_batch_max_new_tokens(
                tokenizer=tokenizer,
                batch=batch,
                max_new_tokens=max_new_tokens,
                adaptive_max_new_tokens=adaptive_max_new_tokens,
                adaptive_multiplier=adaptive_max_new_tokens_multiplier,
                adaptive_min_new_tokens=adaptive_max_new_tokens_min,
                adaptive_max_new_tokens_cap=adaptive_max_new_tokens_cap,
                use_task_type_budgets=use_task_type_budgets,
            )
            outputs = generation_backend.generate_batch(
                prompts=[sample["prompt"] for sample in batch],
                max_input_tokens=max_input_tokens,
                max_new_tokens=batch_max_new_tokens,
            )
            for sample, response in zip(batch, outputs):
                row = {
                    "id": sample["id"],
                    "response": response,
                    "reference": sample["reference"],
                    "metadata": sample.get("metadata", {}),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            completed += len(batch)
            if progress is not None:
                progress.update(len(batch))
                progress.set_postfix_str(
                    f"max_new_tokens<={batch_max_new_tokens}, task={_sample_task_type(batch[0]) or 'mixed'}"
                )
            else:
                print(
                    f"Generated {completed}/{len(pending)} samples "
                    f"(batch max_new_tokens={batch_max_new_tokens}, task={_sample_task_type(batch[0]) or 'mixed'})."
                )
        if progress is not None:
            progress.close()

    rows = list(_load_existing_results(results_path).values())
    judge_llm_module = _load_judge_llm_module() if judge_model else None
    judge_llm = judge_llm_module.get_llm(judge_model) if judge_llm_module else None
    print(f"Scoring {len(rows)} responses...")
    scored_rows, aggregate = score_result_rows(rows, judge_llm=judge_llm)

    with results_path.open("w", encoding="utf-8") as fh:
        for row in scored_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "scores.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run memory QA evaluation on a local checkpoint with an hf or vllm backend."
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--judge_model", type=str, default="mimo-v2-flash")
    parser.add_argument("--force_new", action="store_true")
    parser.add_argument("--adaptive_max_new_tokens", action="store_true")
    parser.add_argument("--adaptive_max_new_tokens_multiplier", type=float, default=2.0)
    parser.add_argument("--adaptive_max_new_tokens_min", type=int, default=64)
    parser.add_argument("--adaptive_max_new_tokens_cap", type=int, default=256)
    parser.add_argument(
        "--disable_task_type_budgets",
        action="store_true",
        help="Disable the default task-aware max_new_tokens profile and rely only on fixed/adaptive budgeting.",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--backend", type=str, default="hf", choices=("hf", "vllm"))
    parser.add_argument("--vllm_tokenizer_mode", type=str, default="auto")
    parser.add_argument(
        "--vllm_trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_seed", type=int, default=0)
    parser.add_argument("--vllm_disable_eager", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    args = parser.parse_args()

    aggregate = run_local_memory_eval(
        dataset_path=args.dataset_path,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        judge_model=args.judge_model,
        force_new=args.force_new,
        adaptive_max_new_tokens=args.adaptive_max_new_tokens,
        adaptive_max_new_tokens_multiplier=args.adaptive_max_new_tokens_multiplier,
        adaptive_max_new_tokens_min=args.adaptive_max_new_tokens_min,
        adaptive_max_new_tokens_cap=args.adaptive_max_new_tokens_cap,
        use_task_type_budgets=not args.disable_task_type_budgets,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        backend=args.backend,
        vllm_tokenizer_mode=args.vllm_tokenizer_mode,
        vllm_trust_remote_code=args.vllm_trust_remote_code,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_seed=args.vllm_seed,
        vllm_disable_eager=args.vllm_disable_eager,
        disable_thinking=args.disable_thinking,
    )
    print(json.dumps({"overall_equal_weighted_score": aggregate.get("overall_equal_weighted_score")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
