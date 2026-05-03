#!/usr/bin/env python3
"""
VerbatimEval Runner - Run evaluation on VerbatimEval datasets with checkpoint/resume support.

This script:
1. Loads a VerbatimEval dataset (JSONL format)
2. Checks for existing progress and skips completed samples (checkpoint resume)
3. Calls LLM API to generate responses for each sample with high retry reliability
4. Evaluates responses using VerbatimEval metrics
5. Saves results incrementally (streaming save) to prevent data loss
6. Supports interruption (Ctrl+C) and seamless resume

Usage:
    python run_eval.py \\
        --dataset_path ../output/VerbatimEval/verbatim_eval_4k.jsonl \\
        --model_name Qwen/Qwen3-8B \\
        --output_dir results
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = SCRIPT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables only.")

from llm import get_llm, LLM
from metrics import ensure_nltk_data, evaluate_batch


# ============================================================================
# Task-specific prompt instructions
# ============================================================================

_TASK_INSTRUCTIONS = {
    "Temporal Reasoning": (
        "Read the following dialogue and answer the question.\n"
        "The question asks about a time interval between two events.\n"
        "You must provide the exact number of days as an integer. "
        "Also state the two dates involved in YYYY-MM-DD format.\n"
        "Answer format: X days (from YYYY-MM-DD to YYYY-MM-DD)."
    ),
    "Information Extraction": (
        "Read the following dialogue and answer the question.\n"
        "Extract the specific factual details asked for. "
        "Be concise and direct. List the key facts without unnecessary explanation."
    ),
    "Multi-session Reasoning": (
        "Read the following dialogue and answer the question.\n"
        "The question asks you to trace a causal chain of events. "
        "List each key event in chronological order with its date, "
        "and explain how each event led to the next.\n"
        "Answer format: Step-by-step chain with dates."
    ),
    "Event Summarization": (
        "Read the following dialogue and answer the question.\n"
        "The question asks you to summarize events or psychological states over a time period. "
        "Cover the major developments and turning points in chronological order. "
        "Mention specific dates or time periods where possible."
    ),
    "Memory Arbitration": (
        "Read the following dialogue and answer the question.\n"
        "The question may contain an incorrect or misleading premise based on the dialogue content. "
        "If the premise is wrong, you MUST explicitly point out what is incorrect and provide "
        "the correct facts with specific dates from the dialogue.\n"
        "If the premise is correct, answer the question directly."
    ),
    "Knowledge Updating": (
        "Read the following dialogue and answer the question.\n"
        "The question asks about the CURRENT or LATEST state of something. "
        "Focus on the most recent information from the dialogue. "
        "If earlier information conflicts with later information, use the latest version.\n"
        "Be specific about what the current state is."
    ),
}

BINARY_TASKS = {
    "Information Extraction",
    "Temporal Reasoning",
    "Knowledge Updating",
}
THREE_LEVEL_TASKS = {
    "Multi-session Reasoning",
    "Event Summarization",
    "Memory Arbitration",
}


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value or "").strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@lru_cache(maxsize=4)
def _load_prompt_catalog(prompt_config_path: Optional[str]) -> Dict[str, Any]:
    path = (
        Path(prompt_config_path)
        if prompt_config_path
        else Path(__file__).parent / "task_prompts_v2.json"
    )
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return {}


def _json_instruction_from_catalog(task_type: str, prompt_catalog: Dict[str, Any]) -> str:
    task_item = (prompt_catalog.get("tasks") or {}).get(task_type, {})
    goal = str(task_item.get("goal", "Answer the question based on dialogue evidence.")).strip()
    instructions = task_item.get("instructions") or []
    required_fields = task_item.get("required_fields") or []
    example_output = task_item.get("example_output") or {}

    lines = [
        "Read the following dialogue and answer the question.",
        goal,
    ]
    for item in instructions:
        lines.append(f"- {item}")
    lines.append("Return ONLY valid JSON. No markdown, no extra text.")
    if required_fields:
        lines.append(f"Required JSON fields: {', '.join(str(x) for x in required_fields)}")
    if example_output:
        lines.append("Minimal valid JSON example:")
        lines.append(json.dumps(example_output, ensure_ascii=False))
    return "\n".join(lines)


def _get_task_instruction(
    task_type: str,
    eval_profile: str = "umb_tasklight_v1",
    response_format: str = "json",
    prompt_catalog: Optional[Dict[str, Any]] = None,
) -> str:
    """Get task-specific prompt instruction."""
    if eval_profile == "umb_tasklight_v1" and str(response_format).lower() == "json":
        return _json_instruction_from_catalog(task_type, prompt_catalog or {})
    return _TASK_INSTRUCTIONS.get(
        task_type,
        (
            "Read the following dialogue and answer the question.\n"
            "Provide a concise and accurate answer based on the dialogue content."
        ),
    )


def load_dataset(
    dataset_path: str,
    max_dialogues: Optional[int] = None,
    eval_profile: str = "umb_tasklight_v1",
    response_format: str = "json",
    prompt_catalog: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """
    Load dataset from file. Supports both:
    1. Standard JSONL format (one sample per line)
    2. UMB Hierarchical JSON list format (list of dialogues with tasks)
    
    Args:
        dataset_path: Path to dataset file
        max_dialogues: Max number of dialogues to process (only for hierarchical format)
    """
    samples = []
    
    # Read first character to detect format
    with open(dataset_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # Load hierarchical JSON list
            print("Detected hierarchical JSON list format. Flattening...")
            dialogues = json.load(f)
            
            if max_dialogues is not None:
                print(f"Limiting to first {max_dialogues} dialogues.")
                dialogues = dialogues[:max_dialogues]
            
            for d_idx, dialog in enumerate(dialogues):
                dialogue_text = dialog.get("dialogue", "")
                tasks = dialog.get("tasks_covered", [])
                
                # Format full dialogue context
                # Dialogue is a list of turns: {"role": "user"/"assistant", "content": "..."}
                # We format it as a string for the prompt
                context_str = ""
                if isinstance(dialogue_text, list):
                    for turn in dialogue_text:
                        speaker = turn.get("role", turn.get("speaker", "Unknown"))
                        content = turn.get("content", "")
                        context_str += f"{speaker}: {content}\n"
                else:
                    context_str = str(dialogue_text)
                
                for t_idx, task in enumerate(tasks):
                    # Construct flattened sample
                    # ID format: {dialogue_id}::{task_type}::{index}
                    task_type = task.get("task_type", "Unknown")
                    unique_id = f"{dialog['id']}::{task_type}::{t_idx}"
                    
                    query = task.get("query", "")
                    
                    # Construct Prompt with task-specific instructions
                    instruction = _get_task_instruction(
                        task_type,
                        eval_profile=eval_profile,
                        response_format=response_format,
                        prompt_catalog=prompt_catalog,
                    )
                    prompt = (
                        f"{instruction}\n\n"
                        f"[Question]\n{query}\n\n"
                        f"[Dialogue]\n{context_str}"
                    )
                    
                    sample = {
                        "id": unique_id,
                        "prompt": prompt,
                        "reference": task.get("gold_answer", ""),
                        "metadata": {
                            "task_type": task_type,
                            "original_dialogue_id": dialog["id"],
                            "answer_components": task.get("answer_components", []),
                            "source_event_ids": task.get("source_event_ids", []),
                            "query": query  # Keep query in metadata for LLM Judge
                        }
                    }
                    samples.append(sample)
                    
        else:
            # Load standard JSONL
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
                    
    return samples


def load_completed_ids(results_path: str) -> Set[str]:
    """Load IDs of already completed samples from existing results file."""
    completed_ids = set()
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            result = json.loads(line)
                            if "id" in result and result.get("response"):
                                completed_ids.add(result["id"])
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
        except Exception as e:
            print(f"Warning: Could not read existing results: {e}")
    return completed_ids


def is_valid_response(r) -> bool:
    """Check if a response is valid (not None, not empty)."""
    if r is None:
        return False
    if isinstance(r, str) and r.strip() == "":
        return False
    return True


async def run_single_inference(
    llm: LLM,
    sample: Dict,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 32768
) -> Optional[Dict]:
    """
    Run inference for a single sample with semaphore control.
    Returns a result dict if successful, None if failed.
    """
    async with semaphore:
        try:
            completion = await llm.inference(sample["prompt"], max_tokens=max_tokens)
            response = llm.decode(completion)
            
            if not is_valid_response(response):
                print(f"\n⚠️ Empty response for sample {sample['id']}")
                return None
            
            return {
                "id": sample["id"],
                "response": response,
                "reference": sample["reference"],
                "metadata": sample.get("metadata", {})
            }
        except Exception as e:
            print(f"\n⚠️ Inference failed for sample {sample['id']}: {e}")
            return None


def run_evaluation(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    concurrency: int = 10,
    max_samples: Optional[int] = None,
    compute_semantic: bool = True,
    api_key: Optional[str] = None,
    batch_save_size: int = 10,
    max_tokens: int = 32768,
    force_new: bool = False,
    judge_model_name: Optional[str] = "mimo-v2-flash",
    max_dialogues: Optional[int] = None,
    eval_profile: str = "umb_tasklight_v1",
    response_format: str = "json",
    task_scoring_scheme: str = "A",
    judge_votes: int = 2,
    judge_tiebreak: bool = True,
    binary_fallback_judge: bool = True,
    prompt_config_path: Optional[str] = None,
    scoring_config_path: Optional[str] = None,
):
    """
    Run full evaluation pipeline with checkpoint/resume support.
    
    Args:
        dataset_path: Path to the VerbatimEval JSONL dataset
        model_name: Name of the model to use
        output_dir: Directory to save results
        concurrency: Max concurrent API calls (default: 10)
        max_samples: Maximum number of samples to evaluate (None for all)
        compute_semantic: Whether to compute semantic similarity scores
        api_key: Optional API key for the model
        batch_save_size: Number of samples to process before flushing to disk
        max_tokens: Maximum tokens for model output
        force_new: If True, ignore existing checkpoints
        judge_model_name: Name of the judge model (default: mimo-v2-flash)
        max_dialogues: Maximum number of dialogues to process (only for hierarchical format)
    """
    ensure_nltk_data()
    
    # Create output directory
    dataset_name = os.path.basename(dataset_path).replace('.jsonl', '')
    
    if "/" in model_name:
        provider = model_name.split("/")[0]
        short_model_name = model_name.split("/")[-1]
    else:
        provider = model_name.split("-")[0] if "-" in model_name else model_name
        short_model_name = model_name
    
    parent_dir = os.path.join(output_dir, f"{dataset_name}_{provider}")
    os.makedirs(parent_dir, exist_ok=True)
    
    # Find existing run or create new one
    # Look for existing incomplete run to resume
    existing_runs = []
    if not force_new and os.path.exists(parent_dir):
        for d in os.listdir(parent_dir):
            if d.startswith(short_model_name + "_"):
                existing_runs.append(d)
    
    # Check if there's an existing incomplete run
    run_dir = None
    results_path = None
    completed_ids = set()
    
    for existing_run in sorted(existing_runs, reverse=True):  # Most recent first
        candidate_dir = os.path.join(parent_dir, existing_run)
        candidate_results = os.path.join(candidate_dir, "detailed_results.jsonl")
        if os.path.exists(candidate_results):
            temp_completed = load_completed_ids(candidate_results)
            if temp_completed:
                # Found an existing run with some progress
                run_dir = candidate_dir
                results_path = candidate_results
                completed_ids = temp_completed
                print(f"📂 Resuming from existing run: {run_dir}")
                print(f"✅ Found {len(completed_ids)} completed samples")
                break
    
    # Create new run if no resumable run found
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(parent_dir, f"{short_model_name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        results_path = os.path.join(run_dir, "detailed_results.jsonl")
    
    print(f"Output directory: {run_dir}")
    
    # Load prompt catalog for tasklight JSON prompts
    prompt_catalog = _load_prompt_catalog(prompt_config_path)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    all_samples = load_dataset(
        dataset_path,
        max_dialogues=max_dialogues,
        eval_profile=eval_profile,
        response_format=response_format,
        prompt_catalog=prompt_catalog,
    )
    
    if max_samples:
        all_samples = all_samples[:max_samples]
    
    # Filter to remaining samples (not yet completed)
    remaining_samples = [s for s in all_samples if s["id"] not in completed_ids]
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining to process: {len(remaining_samples)}")
    
    if not remaining_samples:
        print("🎉 All samples already completed!")
    else:
        # Initialize LLM
        print(f"Initializing model: {model_name} (max_workers={concurrency}, max_tokens={max_tokens})")
        llm = get_llm(model_name, api_key=api_key, max_workers=concurrency)
        
        # Run inference with streaming save
        asyncio.run(run_inference_with_streaming_save(
            llm=llm,
            samples=remaining_samples,
            results_path=results_path,
            concurrency=concurrency,
            batch_save_size=batch_save_size,
            max_tokens=max_tokens,
        ))
        
        # Save LLM usage
        llm.save_usage(run_dir)
    
    # Final: Compute aggregate scores from all results
    print("\nComputing final aggregate scores...")
    final_results = load_all_results(results_path)
    
    # Evaluate all results
    if final_results:
        responses = [r["response"] for r in final_results]
        references = [r["reference"] for r in final_results]
        
        # Extract metadata for task-specific scoring
        metadatas = [r.get("metadata", {}) for r in final_results]
        
        # Initialize Judge LLM if needed
        judge_llm = None
        if judge_model_name:
            print(f"Initializing judge model: {judge_model_name}")
            judge_llm = get_llm(judge_model_name, api_key=api_key, max_workers=concurrency)
        
        evaluated_rows, _ = evaluate_batch(
            responses, references,
            metadatas=metadatas,
            judge_llm=judge_llm,
            compute_semantic=compute_semantic,
            semantic_workers=concurrency,
            eval_profile=eval_profile,
            task_scoring_scheme=task_scoring_scheme,
            judge_votes=judge_votes,
            judge_tiebreak=judge_tiebreak,
            binary_fallback_judge=binary_fallback_judge,
            scoring_config_path=scoring_config_path,
        )
        
        # Update results with scores
        for i, result in enumerate(final_results):
            if i < len(evaluated_rows):
                row = evaluated_rows[i]
                if isinstance(row, dict) and isinstance(row.get("scores"), dict):
                    result["scores"] = row["scores"]
                    result["parsed_output"] = row.get("parsed_output")
                    result["parse_ok"] = row.get("parse_ok")
                    result["rule_signals"] = row.get("rule_signals")
                    result["judge_band"] = row.get("judge_band")
                    result["judge_score"] = row.get("judge_score")
                    result["final_score"] = row.get("final_score")
                    result["score_source"] = row.get("score_source")
                    result["judge_meta"] = row.get("judge_meta", {})
                elif isinstance(row, dict):
                    result["scores"] = row
                    result["final_score"] = row.get("final_score")
                else:
                    result["scores"] = {}
            else:
                result["scores"] = {}
    
        aggregate_scores = compute_aggregate_scores(final_results)
    else:
        aggregate_scores = {}
    
    quality_panel = {
        "overall_equal_weighted_score": aggregate_scores.get("overall_equal_weighted_score"),
        "scores_by_task_type": aggregate_scores.get("scores_by_task_type", {}),
    }
    efficiency_panel = compute_efficiency_panel(
        run_dir=run_dir,
        successful_samples=len(final_results),
    )

    # Save final scores
    scores_path = os.path.join(run_dir, "scores.json")
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": model_name,
            "dataset": dataset_path,
            "num_samples": len(all_samples),
            "successful_inferences": len(final_results),
            "inference_failures": len(all_samples) - len(final_results),
            "aggregate_scores": aggregate_scores,
            "quality_panel": quality_panel,
            "efficiency_panel": efficiency_panel,
            "stability_panel": aggregate_scores.get("stability_panel", {}),
        }, f, indent=2)
    
    # Rewrite detailed_results with scores
    with open(results_path, 'w', encoding='utf-8') as f:
        for result in final_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Successful: {len(final_results)}")
    if len(all_samples) > len(final_results):
        print(f"⚠️  Failed: {len(all_samples) - len(final_results)}")
    print()
    print("Global Aggregate Scores:")
    for key, value in aggregate_scores.items():
        if key.startswith("avg_") and not isinstance(value, dict):
            metric_name = key.replace("avg_", "")
            print(f"  {metric_name}: {value:.4f}")
    if aggregate_scores.get("overall_equal_weighted_score") is not None:
        print(f"  overall_equal_weighted_score: {aggregate_scores['overall_equal_weighted_score']:.4f}")
    
    # Per-task-type breakdown
    task_scores = aggregate_scores.get("scores_by_task_type", {})
    if task_scores:
        print()
        print("Scores by Task Type:")
        for task_type, stats in sorted(task_scores.items()):
            count = stats.get("count", 0)
            print(f"  {task_type} (n={count}):")
            for key, value in sorted(stats.items()):
                if key.startswith("avg_"):
                    metric = key.replace("avg_", "")
                    print(f"    {metric}: {value:.4f}")
    print()
    print(f"Results saved to: {run_dir}")
    
    return final_results, aggregate_scores


async def run_inference_with_streaming_save(
    llm: LLM,
    samples: List[Dict],
    results_path: str,
    concurrency: int = 10,
    batch_save_size: int = 10,
    max_tokens: int = 32768,
):
    """
    Run inference on samples with streaming save to disk.
    Failed samples are NOT saved (will be retried on next run).
    """
    from tqdm.asyncio import tqdm as async_tqdm
    
    semaphore = asyncio.Semaphore(concurrency)
    
    # Buffer for batch saving
    pending_results = []
    total_saved = 0
    
    async def process_and_save(sample: Dict, file_handle):
        """Process single sample and save if successful."""
        nonlocal pending_results, total_saved
        
        result = await run_single_inference(llm, sample, semaphore, max_tokens=max_tokens)
        
        if result is not None:
            pending_results.append(result)
            
            # Batch save when we have enough results
            if len(pending_results) >= batch_save_size:
                for r in pending_results:
                    file_handle.write(json.dumps(r, ensure_ascii=False) + '\n')
                file_handle.flush()
                total_saved += len(pending_results)
                pending_results = []
    
    # Open file in append mode for streaming save
    with open(results_path, 'a', encoding='utf-8') as f:
        # Create tasks
        tasks = [process_and_save(sample, f) for sample in samples]
        
        # Run with progress bar
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Inference"):
            await coro
        
        # Flush remaining results
        if pending_results:
            for r in pending_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            total_saved += len(pending_results)
    
    print(f"\n💾 Saved {total_saved} new results")


def load_all_results(results_path: str) -> List[Dict]:
    """Load all results from the results file."""
    results = []
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return results


def compute_efficiency_panel(run_dir: str, successful_samples: int) -> Dict[str, Any]:
    usage_path = os.path.join(run_dir, "usage.json")
    usage: Dict[str, Any] = {}
    if os.path.exists(usage_path):
        try:
            with open(usage_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                usage = loaded
        except Exception:
            usage = {}

    avg_input = usage.get("avg_input")
    avg_output = usage.get("avg_output")
    avg_total = usage.get("avg_total")
    return {
        "api_calls": usage.get("api_calls"),
        "total_input_tokens": usage.get("input"),
        "total_output_tokens": usage.get("output"),
        "total_tokens": usage.get("total"),
        "avg_input_tokens_per_sample": avg_input,
        "avg_output_tokens_per_sample": avg_output,
        "avg_total_tokens_per_sample": avg_total,
        "cost_per_100_samples": None,
        "avg_latency_seconds": None,
        "successful_samples": successful_samples,
    }


def compute_aggregate_scores(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute aggregate statistics from individual results.
    Includes both global aggregates and per-task-type breakdowns.
    """
    import statistics
    
    if not results:
        return {}
    
    aggregate: Dict[str, Any] = {}

    # --- Global aggregates for numeric score keys ---
    score_keys = set()
    for row in results:
        if isinstance(row.get("scores"), dict):
            score_keys.update(row["scores"].keys())

    for key in score_keys:
        values = [
            row["scores"][key]
            for row in results
            if isinstance(row.get("scores"), dict)
            and key in row["scores"]
            and row["scores"][key] is not None
        ]
        if values:
            aggregate[f"avg_{key}"] = float(statistics.fmean(values))
            aggregate[f"std_{key}"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
            aggregate[f"min_{key}"] = float(min(values))
            aggregate[f"max_{key}"] = float(max(values))
            aggregate[f"valid_count_{key}"] = len(values)
            aggregate[f"failed_count_{key}"] = len(results) - len(values)

    # --- Group by task type ---
    by_task_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        task_type = row.get("metadata", {}).get("task_type", "Unknown")
        by_task_type.setdefault(task_type, []).append(row)

    scores_by_task_type: Dict[str, Dict[str, Any]] = {}
    task_final_score_avgs: List[float] = []
    for task_type, task_rows in sorted(by_task_type.items()):
        task_agg: Dict[str, Any] = {"count": len(task_rows)}
        task_score_keys = set()
        for row in task_rows:
            if isinstance(row.get("scores"), dict):
                task_score_keys.update(row["scores"].keys())

        for key in task_score_keys:
            values = [
                row["scores"][key]
                for row in task_rows
                if isinstance(row.get("scores"), dict)
                and key in row["scores"]
                and row["scores"][key] is not None
            ]
            if values:
                task_agg[f"avg_{key}"] = float(statistics.fmean(values))
                task_agg[f"valid_count_{key}"] = len(values)

        final_values = [
            row["scores"]["final_score"]
            for row in task_rows
            if isinstance(row.get("scores"), dict)
            and row["scores"].get("final_score") is not None
        ]
        if final_values:
            avg_final = float(statistics.fmean(final_values))
            task_agg["avg_final_score"] = avg_final
            task_final_score_avgs.append(avg_final)

        if task_type in BINARY_TASKS and final_values:
            pass_count = sum(1 for value in final_values if float(value) >= 1.0)
            task_agg["binary_pass_rate"] = float(pass_count / len(final_values))

        if task_type in THREE_LEVEL_TASKS:
            judge_rows = [
                row for row in task_rows
                if str(row.get("judge_band", "")).strip().lower() in {"correct", "partial", "wrong"}
            ]
            if judge_rows:
                partial = sum(
                    1
                    for row in judge_rows
                    if str(row.get("judge_band", "")).strip().lower() == "partial"
                )
                task_agg["judge_partial_rate"] = float(partial / len(judge_rows))

        scores_by_task_type[task_type] = task_agg

    aggregate["scores_by_task_type"] = scores_by_task_type
    if task_final_score_avgs:
        aggregate["overall_equal_weighted_score"] = float(statistics.fmean(task_final_score_avgs))

    disagreement = 0
    tiebreak = 0
    judged_total = 0
    judge_attempts = 0
    judge_successes = 0
    judge_failed_samples = 0
    judge_failure_counts: Dict[str, int] = {}
    for row in results:
        meta = row.get("judge_meta", {})
        band = str(row.get("judge_band", "")).strip().lower()
        if band in {"correct", "partial", "wrong"}:
            judged_total += 1
        if isinstance(meta, dict):
            if bool(meta.get("disagreement")):
                disagreement += 1
            if bool(meta.get("used_tiebreak")):
                tiebreak += 1
            judge_attempts += int(meta.get("attempted_votes") or 0)
            judge_successes += int(meta.get("successful_votes") or 0)
            failure_counts = meta.get("failure_counts", {})
            if isinstance(failure_counts, dict) and failure_counts:
                judge_failed_samples += 1
                for failure_type, count in failure_counts.items():
                    key = str(failure_type or "other_exception")
                    judge_failure_counts[key] = judge_failure_counts.get(key, 0) + int(count or 0)

    aggregate["stability_panel"] = {
        "judge_disagreement_rate": float(disagreement / max(judged_total, 1)),
        "judge_tiebreak_rate": float(tiebreak / max(judged_total, 1)),
        "samples_with_disagreement": disagreement,
        "samples_with_tiebreak": tiebreak,
    }
    aggregate["judge_failure_panel"] = {
        "judge_attempts": judge_attempts,
        "judge_successful_votes": judge_successes,
        "judge_failed_votes": int(sum(judge_failure_counts.values())),
        "judge_success_rate": float(judge_successes / max(judge_attempts, 1)),
        "samples_with_judge_failures": judge_failed_samples,
        "failure_counts": judge_failure_counts,
    }

    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description="Run VerbatimEval evaluation on a dataset with checkpoint/resume support."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(REPO_ROOT / "dialogue_gen_api/output/final/clean_v8_budget_direct/UMB_dialogue_benchmark.json"),
        help="Path to the dialogue dataset (default: clean benchmark dataset)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="Qwen/Qwen3-8B",
        default="mimo-v2-flash",
        help="Name of the model to use (default: Qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "dialogue_gen_api/evaluation/results"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API calls (default: 10)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--no_semantic",
        action="store_true",
        help="Skip semantic similarity computation (faster, no OpenAI API calls for embeddings)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the model (optional, uses env var if not provided)"
    )
    parser.add_argument(
        "--batch_save_size",
        type=int,
        default=10,
        help="Number of samples to process before saving to disk (default: 10)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Maximum output tokens for model response (default: 32768, increase for 128K+ tasks)"
    )
    parser.add_argument(
        "--force_new",
        action="store_true",
        help="Force a new run, ignoring existing checkpoints"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="mimo-v2-flash",
        help="Name of the model to use for judging (default: mimo-v2-flash)"
    )
    parser.add_argument(
        "--max_dialogues",
        type=int,
        default=None,
        help="Maximum number of dialogues to process (useful for quick testing)"
    )
    parser.add_argument(
        "--eval_profile",
        type=str,
        default="umb_tasklight_v1",
        help="Evaluation profile. Use umb_tasklight_v1 for lightweight task scoring."
    )
    parser.add_argument(
        "--response_format",
        type=str,
        default="json",
        choices=["json", "text"],
        help="Expected response format from tested model."
    )
    parser.add_argument(
        "--task_scoring_scheme",
        type=str,
        default="A",
        help="Task scoring scheme identifier (default: A)."
    )
    parser.add_argument(
        "--judge_votes",
        type=int,
        default=2,
        help="Number of base judge votes per sample (default: 2)."
    )
    parser.add_argument(
        "--judge_tiebreak",
        type=_str2bool,
        default=True,
        help="Whether to add one tiebreak judge vote when first two votes disagree."
    )
    parser.add_argument(
        "--binary_fallback_judge",
        type=_str2bool,
        default=True,
        help="For binary tasks, use judge fallback when rule score is uncertain."
    )
    parser.add_argument(
        "--prompt_config_path",
        type=str,
        default=str(Path(__file__).parent / "task_prompts_v2.json"),
        help="Path to task prompt config JSON."
    )
    parser.add_argument(
        "--scoring_config_path",
        type=str,
        default=str(Path(__file__).parent / "task_scoring_v1.json"),
        help="Path to task scoring config JSON."
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        max_samples=args.max_samples,
        compute_semantic=not args.no_semantic,
        api_key=args.api_key,
        batch_save_size=args.batch_save_size,
        max_tokens=args.max_tokens,
        force_new=args.force_new,
        judge_model_name=args.judge_model,
        max_dialogues=args.max_dialogues,
        eval_profile=args.eval_profile,
        response_format=args.response_format,
        task_scoring_scheme=args.task_scoring_scheme,
        judge_votes=args.judge_votes,
        judge_tiebreak=args.judge_tiebreak,
        binary_fallback_judge=args.binary_fallback_judge,
        prompt_config_path=args.prompt_config_path,
        scoring_config_path=args.scoring_config_path,
    )


if __name__ == "__main__":
    main()
