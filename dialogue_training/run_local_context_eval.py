from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dialogue_training.generation_backends import build_generation_backend
from dialogue_training.memory_eval_utils import score_result_rows
from dialogue_training.run_local_memory_eval import (
    _build_generation_batches,
    _filter_samples_for_shard,
    _load_existing_results,
    _load_judge_llm_module,
    _resolve_batch_max_new_tokens,
    _sample_task_type,
)

_EVAL_DIR = Path(__file__).resolve().parent.parent / "dialogue_gen_api" / "evaluation"
DEFAULT_CONTEXT_MAX_INPUT_TOKENS = 262144
DEFAULT_CONTEXT_MAX_NEW_TOKENS = 1024
DEFAULT_CONTEXT_TASK_TYPE_MAX_NEW_TOKENS = {
    "Information Extraction": 256,
    "Temporal Reasoning": 256,
    "Knowledge Updating": 256,
    "Memory Arbitration": 384,
    "Multi-session Reasoning": 640,
    "Event Summarization": 768,
}
LOCAL_CONTEXT_SYSTEM_PROMPT = (
    "You answer dialogue-grounded question answering tasks. "
    "Follow the user's required JSON schema exactly. "
    "Do not continue the dialogue transcript or role-play it."
)
LOCAL_CONTEXT_ANSWER_SUFFIX = (
    "[Answer]\n"
    "The dialogue transcript above has ended. Answer the question now.\n"
    "Return ONLY one complete JSON object that satisfies the required fields.\n"
    "Do not continue the dialogue transcript.\n"
    "Do not write user: or assistant: turns.\n"
    "Do not add markdown or explanation.\n"
    "Ensure the JSON object is complete and closed before stopping."
)


def _load_context_eval_module():
    if str(_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(_EVAL_DIR))
    import run_eval  # type: ignore

    return run_eval


def _placeholder_scalar_for_field(field_name: str, value: Any) -> Any:
    placeholder_map = {
        "answer": "(answer for the query)",
        "evidence_snippets": "(supporting evidence from the dialogue)",
        "final_answer": "(final answer for the query)",
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "latest_state": "(latest state for the query)",
        "as_of_time": "(time for the latest state if stated)",
        "deprecated_state": "(older state mentioned in the dialogue)",
        "event_chain": "(chronological step from the dialogue)",
        "final_outcome": "(outcome for the query)",
        "time_span": "(time span asked by the query)",
        "key_turning_points": "(key turning point from the asked period)",
        "summary": "(summary for the asked period)",
        "premise_verdict": "correct",
        "premise_error": "(what premise is wrong, if any)",
        "corrected_facts": "(correct fact from the dialogue)",
    }
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    if isinstance(value, str):
        return placeholder_map.get(field_name, f"({field_name.replace('_', ' ')} for the query)")
    return placeholder_map.get(field_name, "(value for the query)")


def _placeholderize_example_output(value: Any, field_name: str = "") -> Any:
    if isinstance(value, dict):
        return {key: _placeholderize_example_output(item, key) for key, item in value.items()}
    if isinstance(value, list):
        if not value:
            return [_placeholder_scalar_for_field(field_name, "")]
        return [_placeholderize_example_output(item, field_name) for item in value]
    return _placeholder_scalar_for_field(field_name, value)


def _build_local_context_prompt_catalog(prompt_catalog: Dict[str, Any]) -> Dict[str, Any]:
    sanitized_catalog = copy.deepcopy(prompt_catalog or {})
    tasks = sanitized_catalog.get("tasks") or {}
    for task_item in tasks.values():
        example_output = task_item.get("example_output")
        if isinstance(example_output, dict):
            task_item["example_output"] = _placeholderize_example_output(example_output)
    return sanitized_catalog


def load_context_eval_samples(
    dataset_path: Path,
    max_dialogues: Optional[int] = None,
    max_samples: Optional[int] = None,
    eval_profile: str = "umb_tasklight_v1",
    response_format: str = "json",
    prompt_config_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    run_eval = _load_context_eval_module()
    prompt_catalog = _build_local_context_prompt_catalog(
        run_eval._load_prompt_catalog(prompt_config_path)
    )
    samples = run_eval.load_dataset(
        str(dataset_path),
        max_dialogues=max_dialogues,
        eval_profile=eval_profile,
        response_format=response_format,
        prompt_catalog=prompt_catalog,
    )
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def _chat_template_kwargs_for_generation(*, disable_thinking: bool) -> Dict[str, Any]:
    if not disable_thinking:
        return {}
    return {"enable_thinking": False}


def _render_local_context_prompt(
    tokenizer,
    sample: Dict[str, Any],
    disable_thinking: bool = False,
) -> str:
    base_prompt = str(sample.get("prompt", "")).rstrip()
    user_prompt = f"{base_prompt}\n\n{LOCAL_CONTEXT_ANSWER_SUFFIX}"
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(
            [
                {"role": "system", "content": LOCAL_CONTEXT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs_for_generation(disable_thinking=disable_thinking),
        )
    return (
        f"System: {LOCAL_CONTEXT_SYSTEM_PROMPT}\n\n"
        f"User: {user_prompt}\n\n"
        "Assistant:\n"
    )


def _prepare_context_samples_for_local_generation(
    tokenizer,
    samples: List[Dict[str, Any]],
    disable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for sample in samples:
        prepared_sample = dict(sample)
        prepared_sample["raw_prompt"] = sample.get("prompt", "")
        prepared_sample["prompt"] = _render_local_context_prompt(
            tokenizer,
            sample,
            disable_thinking=disable_thinking,
        )
        prepared.append(prepared_sample)
    return prepared


def run_local_context_eval(
    dataset_path: Path,
    model_name_or_path: str,
    output_dir: Path,
    adapter_path: Optional[str] = None,
    batch_size: int = 1,
    max_input_tokens: int = DEFAULT_CONTEXT_MAX_INPUT_TOKENS,
    max_new_tokens: int = DEFAULT_CONTEXT_MAX_NEW_TOKENS,
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
    max_dialogues: Optional[int] = None,
    max_samples: Optional[int] = None,
    eval_profile: str = "umb_tasklight_v1",
    response_format: str = "json",
    prompt_config_path: Optional[str] = None,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "detailed_results.jsonl"
    existing = {} if force_new else _load_existing_results(results_path)

    all_samples = load_context_eval_samples(
        dataset_path=dataset_path,
        max_dialogues=max_dialogues,
        max_samples=max_samples,
        eval_profile=eval_profile,
        response_format=response_format,
        prompt_config_path=prompt_config_path,
    )
    samples = _filter_samples_for_shard(all_samples, num_shards=num_shards, shard_index=shard_index)
    pending = [sample for sample in samples if sample["id"] not in existing]

    generation_backend = build_generation_backend(
        backend=backend,
        model_name_or_path=model_name_or_path,
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
    pending = _prepare_context_samples_for_local_generation(
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
        f"Loaded {len(samples)} context-eval samples from {dataset_path}. "
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
        task_type_budgets=DEFAULT_CONTEXT_TASK_TYPE_MAX_NEW_TOKENS,
    )

    mode = "w" if force_new else "a"
    with results_path.open(mode, encoding="utf-8") as fh:
        progress = tqdm(total=len(pending), desc="Generating context-eval responses", unit="sample") if (tqdm and pending) else None
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
                task_type_budgets=DEFAULT_CONTEXT_TASK_TYPE_MAX_NEW_TOKENS,
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
        description="Run full-dialogue context QA evaluation on a local checkpoint with an hf or vllm backend."
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_input_tokens", type=int, default=DEFAULT_CONTEXT_MAX_INPUT_TOKENS)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_CONTEXT_MAX_NEW_TOKENS)
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
    parser.add_argument("--max_dialogues", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_profile", type=str, default="umb_tasklight_v1")
    parser.add_argument("--response_format", type=str, default="json")
    parser.add_argument("--prompt_config_path", type=str, default=None)
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

    aggregate = run_local_context_eval(
        dataset_path=args.dataset_path,
        model_name_or_path=args.model_name_or_path,
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
        max_dialogues=args.max_dialogues,
        max_samples=args.max_samples,
        eval_profile=args.eval_profile,
        response_format=args.response_format,
        prompt_config_path=args.prompt_config_path,
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
    print(
        json.dumps(
            {"overall_equal_weighted_score": aggregate.get("overall_equal_weighted_score")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
