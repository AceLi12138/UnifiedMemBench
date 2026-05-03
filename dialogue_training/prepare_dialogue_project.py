from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dialogue_training.core import (  # type: ignore
        assign_dialogues_to_stages,
        build_memory_eval_samples,
        build_memory_eval_samples_from_qa_records,
        build_pretrain_records,
        build_qa_sft_records,
        build_task_balanced_qa_holdout_splits,
        build_training_records,
        DEFAULT_QA_TASK_WEIGHTS,
        load_benchmark,
        rebalance_qa_sft_records,
        split_stage_dialogues_by_entity,
        write_jsonl,
    )
else:
    from dialogue_training.core import (
        assign_dialogues_to_stages,
        build_memory_eval_samples,
        build_memory_eval_samples_from_qa_records,
        build_pretrain_records,
        build_qa_sft_records,
        build_task_balanced_qa_holdout_splits,
        build_training_records,
        DEFAULT_QA_TASK_WEIGHTS,
        load_benchmark,
        rebalance_qa_sft_records,
        split_stage_dialogues_by_entity,
        write_jsonl,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stage_file_name(stage_id: int) -> str:
    return f"stage_{stage_id:02d}.jsonl"


def _stage_pt_file_name(stage_id: int) -> str:
    return f"stage_{stage_id:02d}_pt.jsonl"


def _stage_seen_eval_file_name(stage_id: int) -> str:
    return f"memory_eval_stage_{stage_id:02d}_seen.jsonl"


def _stage_holdout_eval_file_name(stage_id: int) -> str:
    return f"memory_eval_stage_{stage_id:02d}_holdout.jsonl"


def _manifest_dataset_path(dataset_path: Path) -> str:
    resolved_path = Path(dataset_path).resolve()
    try:
        relative_path = resolved_path.relative_to(REPO_ROOT)
    except ValueError:
        return str(resolved_path)
    return "${UMB_ROOT}/" + relative_path.as_posix()


def _default_training_plan(num_stages: int, output_dir: Path) -> Dict[str, Any]:
    return {
        "paper_reference": {
            "framework": "LLaMA-Factory",
            "finetuning_type": "LoRA",
            "per_device_train_batch_size": 1,
            "lora_rank": 128,
            "lora_alpha": 256,
            "learning_rate": 1.0e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "num_train_epochs": 3.0,
        },
        "recommended_for_umb_dialogue_level": {
            "num_stages": num_stages,
            "curriculum_unit": "dialogue",
            "training_unit": "chunked_dialogue_segment",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "num_train_epochs": 1.0,
            "bf16": True,
            "cutoff_len": 4096,
            "output_root": str(output_dir),
        },
        "recommended_for_umb_dialogue_level_pt": {
            "num_stages": num_stages,
            "curriculum_unit": "dialogue",
            "training_unit": "chunked_dialogue_segment_text",
            "training_stage": "pt",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "num_train_epochs": 1.0,
            "bf16": True,
            "cutoff_len": 4096,
            "output_root": str(output_dir),
        },
    }


def _build_pt_columns(pt_record_style: str) -> Dict[str, Any]:
    if pt_record_style == "alpaca":
        return {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        }
    if pt_record_style == "text":
        return {
            "prompt": "text",
            "query": None,
            "response": None,
        }

    raise ValueError(f"Unsupported pt_record_style: {pt_record_style}")


def _build_dataset_info(num_stages: int, pt_record_style: str = "alpaca") -> Dict[str, Any]:
    dataset_info: Dict[str, Any] = {}
    for stage_id in range(1, num_stages + 1):
        dataset_info[f"umb_dialogue_stage_{stage_id:02d}"] = {
            "file_name": f"../train/{_stage_file_name(stage_id)}",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
        dataset_info[f"umb_dialogue_stage_{stage_id:02d}_pt"] = {
            "file_name": f"../train_pt/{_stage_pt_file_name(stage_id)}",
            "formatting": "alpaca",
            "columns": _build_pt_columns(pt_record_style),
        }
    return dataset_info


def _build_entity_split_dataset_info(num_stages: int, pt_record_style: str = "alpaca") -> Dict[str, Any]:
    dataset_info: Dict[str, Any] = {}
    for stage_id in range(1, num_stages + 1):
        dataset_info[f"umb_entitysplit_stage_{stage_id:02d}_pt"] = {
            "file_name": f"../train_pt/{_stage_pt_file_name(stage_id)}",
            "formatting": "alpaca",
            "columns": _build_pt_columns(pt_record_style),
        }
        dataset_info[f"umb_entitysplit_stage_{stage_id:02d}_qa"] = {
            "file_name": f"../train_qa_sft/{_stage_file_name(stage_id)}",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    return dataset_info


def _build_stage_runbook(
    output_dir: Path,
    num_stages: int,
) -> str:
    lines = [
        "# Dialogue Training Runbook",
        "",
        "## Workflow",
        "",
        "1. `prepare_dialogue_project.py` prepares train / eval / config artifacts.",
        "2. `train/` keeps the original SFT data; `train_pt/` provides parallel PT data.",
        "3. Choose a training route and train stage by stage: use `--stage sft` for SFT and `--stage pt` for PT.",
        "4. After each stage, run a quick eval on the current stage file `eval/memory_eval_stage_XX.jsonl`.",
        "5. For the formal forgetting curve, evaluate the full `eval/memory_eval_all.jsonl` memory-QA file.",
        "6. Store each evaluation round under `outputs/eval*/checkpoint_stage_XX/`.",
        "7. Finally run `plot_forgetting_curve.py` to export QA curves and forgetting heatmaps.",
        "",
        "## Prompt and Evaluation Protocol",
        "",
        "The memory-QA prompt used by `dialogue_training` follows:",
        "- `dialogue_gen_api/evaluation/task_prompts_v2.json`",
        "",
        "The local memory-evaluation scorer reuses:",
        "- `dialogue_gen_api/evaluation/metrics.py`",
        "",
        "In other words, post-training local evaluation uses:",
        "- a memory-only prompt without the original dialogue transcript",
        "- the output-field requirements and examples from `task_prompts_v2.json`",
        "- TaskLight scoring from `metrics.py`",
        "",
        "## Pre-training Environment Checks",
        "",
        "Check core training libraries first:",
        "",
        "```bash",
        "python - <<'PY'",
        "modules = [",
        "    'torch', 'transformers', 'peft', 'datasets', 'accelerate',",
        "    'trl', 'deepspeed', 'yaml', 'requests', 'dotenv',",
        "    'rapidfuzz', 'matplotlib', 'nltk'",
        "]",
        "for name in modules:",
        "    try:",
        "        mod = __import__(name)",
        "        print(f'OK   {name:<12} {getattr(mod, \"__version__\", \"\") or \"\"}')",
        "    except Exception as e:",
        "        print(f'FAIL {name:<12} {e}')",
        "PY",
        "```",
        "",
        "Check GPU availability, CUDA, and device count:",
        "",
        "```bash",
        "python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)\"",
        "```",
        "",
        "If `matplotlib` reports an unwritable cache directory, set:",
        "",
        "```bash",
        "export MPLCONFIGDIR=/tmp/matplotlib",
        "mkdir -p \"$MPLCONFIGDIR\"",
        "```",
        "",
        "Check the LLaMA-Factory CLI:",
        "",
        "```bash",
        "llamafactory-cli version",
        "```",
        "",
        "Check this repository's training and evaluation entry points:",
        "",
        "```bash",
        "python dialogue_training/prepare_dialogue_project.py --help",
        "python dialogue_training/run_local_memory_eval.py --help",
        "python dialogue_training/plot_forgetting_curve.py --help",
        "python dialogue_gen_api/evaluation/run_eval.py --help",
        "```",
        "",
        "Check whether local evaluation modules can be imported in the current environment:",
        "",
        "```bash",
        "python - <<'PY'",
        "import os",
        "import sys",
        "from pathlib import Path",
        "repo = Path(os.environ.get('UMB_ROOT', '.')).resolve()",
        "sys.path.insert(0, str(repo / 'dialogue_gen_api' / 'evaluation'))",
        "import llm",
        "import metrics",
        "print('OK local eval modules')",
        "PY",
        "```",
        "",
        "If NLTK resources are needed in advance, run:",
        "",
        "```bash",
        "python - <<'PY'",
        "import nltk",
        "nltk.download('punkt')",
        "nltk.download('punkt_tab')",
        "PY",
        "```",
        "",
        "If a judge model is used, check API keys:",
        "",
        "```bash",
        "python - <<'PY'",
        "import os",
        "for key in ['MIMO_API_KEY', 'OPENAI_API_KEY', 'KIMI_API_KEY', 'MOONSHOT_API_KEY', 'SILICONFLOW_API_KEY', 'GEMINI_API_KEY']:",
        "    print(key, 'SET' if os.getenv(key) else 'MISSING')",
        "PY",
        "```",
        "",
        "## LLaMA-Factory Training Command Template",
        "",
        "```bash",
        "llamafactory-cli train \\",
        "  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\",
        "  --stage sft \\",
        "  --finetuning_type lora \\",
        "  --dataset umb_dialogue_stage_01 \\",
        "  --dataset_dir "
        + str((output_dir / "configs").resolve())
        + " \\",
        "  --template qwen \\",
        "  --cutoff_len 4096 \\",
        "  --learning_rate 5e-5 \\",
        "  --num_train_epochs 3.0 \\",
        "  --per_device_train_batch_size 1 \\",
        "  --gradient_accumulation_steps 1 \\",
        "  --lr_scheduler_type cosine \\",
        "  --warmup_steps 550 \\",
        "  --lora_rank 128 \\",
        "  --lora_alpha 256 \\",
        "  --bf16 true",
        "```",
        "",
        "## Local Checkpoint Evaluation Command Template",
        "",
        "```bash",
        "python dialogue_training/run_local_memory_eval.py \\",
        "  --dataset_path "
        + str((output_dir / "eval" / "memory_eval_all.jsonl").resolve())
        + " \\",
        "  --model_name_or_path /path/to/base-or-merged-checkpoint \\",
        "  --adapter_path /path/to/lora_adapter_optional \\",
        "  --output_dir "
        + str((output_dir / "outputs" / "eval" / "checkpoint_stage_01").resolve())
        + " \\",
        "  --batch_size 4 \\",
        "  --max_input_tokens 1024 \\",
        "  --max_new_tokens 512 \\",
        "  --torch_dtype bfloat16 \\",
        "  --judge_model ''",
        "```",
        "",
        "Note: local evaluation defaults to the mainline task-type output-budget policy rather than per-example answer-adaptive budgets.",
        "Default budgets:",
        "- Information Extraction / Temporal Reasoning / Knowledge Updating: 128",
        "- Memory Arbitration: 192",
        "- Multi-session Reasoning: 384",
        "- Event Summarization: 448",
        "To revert to the older fixed/adaptive budget mode, add `--disable_task_type_budgets`.",
        "",
        "## Multi-GPU Parallel Evaluation Template",
        "",
        "```bash",
        "python dialogue_training/run_parallel_memory_eval.py \\",
        "  --dataset_path "
        + str((output_dir / "eval" / "memory_eval_all.jsonl").resolve())
        + " \\",
        "  --model_name_or_path /path/to/base-or-merged-checkpoint \\",
        "  --adapter_path /path/to/lora_adapter_optional \\",
        "  --output_dir "
        + str((output_dir / "outputs" / "eval" / "checkpoint_stage_01").resolve())
        + " \\",
        "  --gpu_ids 0,1,2,3 \\",
        "  --batch_size 8 \\",
        "  --max_input_tokens 1024 \\",
        "  --max_new_tokens 512 \\",
        "  --torch_dtype bfloat16 \\",
        "  --judge_model '' \\",
        "  --force_new",
        "```",
        "",
        "Note: this parallel runner shards examples automatically, buckets them by `task_type + prompt length`, and merges shard outputs at the end.",
        "",
        "## API Evaluation Fallback Template",
        "",
        "```bash",
        "python dialogue_gen_api/evaluation/run_eval.py \\",
        "  --dataset_path "
        + str((output_dir / "eval" / "memory_eval_all.jsonl").resolve())
        + " \\",
        "  --model_name your-model-api-name \\",
        "  --output_dir "
        + str((output_dir / "outputs" / "eval" / "checkpoint_stage_01").resolve())
        + " \\",
        "  --judge_model mimo-v2-flash",
        "```",
        "",
        "## Curve Export",
        "",
        "```bash",
        "python dialogue_training/plot_forgetting_curve.py \\",
        "  --results_root "
        + str((output_dir / 'outputs' / 'eval').resolve())
        + " \\",
        "  --output_dir "
        + str((output_dir / 'outputs' / 'curves').resolve()),
        "```",
        "",
        "## Current Stage Files",
        "",
    ]

    for stage_id in range(1, num_stages + 1):
        lines.append(f"- Stage {stage_id:02d} training file: `train/{_stage_file_name(stage_id)}`")
        lines.append(f"- Stage {stage_id:02d} evaluation file: `eval/memory_eval_stage_{stage_id:02d}.jsonl`")
    lines.append("")
    return "\n".join(lines)


def _default_entity_split_training_plan(
    num_stages: int,
    output_dir: Path,
    train_ratio: float,
) -> Dict[str, Any]:
    return {
        "workflow": {
            "num_stages": num_stages,
            "project_mode": "entity_split",
            "entity_train_ratio": train_ratio,
            "entity_test_ratio": 1.0 - train_ratio,
            "primary_eval": "strict_json_v1_unseen",
            "auxiliary_eval": [
                "strict_json_v1_seen",
                "relaxed_content_v2_unseen",
                "relaxed_content_v2_seen",
            ],
            "output_root": str(output_dir),
        },
        "branch_base": {
            "description": "Base model directly evaluated with vLLM on seen/unseen eval sets.",
        },
        "branch_pt_only": {
            "training_stage": "pt",
            "dataset_pattern": "umb_entitysplit_stage_XX_pt",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "num_train_epochs": 1.0,
            "bf16": True,
            "cutoff_len": 4096,
            "evaluation_backend": "vllm",
            "requires_merge_export": True,
        },
        "branch_pt_plus_qa_sft": {
            "base_model_source": "merged_pt_checkpoint",
            "training_stage": "sft",
            "dataset_pattern": "umb_entitysplit_stage_XX_qa",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5.0e-6,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "num_train_epochs": 1.0,
            "bf16": True,
            "cutoff_len": 2048,
            "evaluation_backend": "vllm",
            "requires_merge_export": True,
        },
    }


def _build_entity_split_runbook(
    output_dir: Path,
    num_stages: int,
) -> str:
    lines = [
        "# Entity-disjoint Parametric Memory Runbook",
        "",
        "## Directory Roles",
        "",
        "- `train_pt/`: PT data for all 100 characters in each stage.",
        "- `train_qa_sft/`: QA-SFT data for the 80 train entities in each stage.",
        "- `eval_unseen/`: the main evaluation split with 20 test entities per stage.",
        "- `eval_seen/`: the auxiliary evaluation split with 80 train entities per stage.",
        "- `manifests/`: fixed stage assignments and entity splits.",
        "",
        "## Three Experimental Branches",
        "",
        "1. Base: evaluate the original base model directly with vLLM.",
        "2. PT-only: train a PT LoRA, merge/export it, and evaluate with vLLM.",
        "3. PT + QA-SFT: start from the merged PT checkpoint, train a lightweight QA-SFT LoRA, merge/export it, and evaluate with vLLM.",
        "",
        "## Evaluation Constraints",
        "",
        "- Formal evaluation uses vLLM and does not evaluate `adapter_path` directly.",
        "- Main results use strict JSON v1 on the unseen split.",
        "- Auxiliary results include strict seen and relaxed v2 seen/unseen runs.",
        "",
        "## Default Training Recommendations",
        "",
        "- PT: `--stage pt`, `lr=1e-5`, `epoch=1`.",
        "- QA-SFT: `--stage sft`, `lr=5e-6`, `epoch=1`, using `Merged_PT_Checkpoint` as the base.",
        "",
        "## Current Stage Files",
        "",
    ]
    for stage_id in range(1, num_stages + 1):
        lines.append(f"- Stage {stage_id:02d} PT: `train_pt/{_stage_pt_file_name(stage_id)}`")
        lines.append(f"- Stage {stage_id:02d} QA-SFT: `train_qa_sft/{_stage_file_name(stage_id)}`")
        lines.append(f"- Stage {stage_id:02d} unseen eval: `eval_unseen/memory_eval_stage_{stage_id:02d}.jsonl`")
        lines.append(f"- Stage {stage_id:02d} seen eval: `eval_seen/{_stage_seen_eval_file_name(stage_id)}`")
    lines.append("")
    return "\n".join(lines)


def build_project(
    dataset_path: Path,
    output_dir: Path,
    num_stages: int = 10,
    max_chunk_tokens: int = 4096,
    overlap_turns: int = 1,
    chunking_mode: str = "turn_overlap",
    sliding_window_overlap_tokens: int = 0,
    pt_header_style: str = "structured",
    pt_record_style: str = "alpaca",
    seed: int = 42,
) -> Dict[str, Any]:
    dialogues = load_benchmark(dataset_path)
    assignments = assign_dialogues_to_stages(dialogues, num_stages=num_stages, seed=seed)
    stage_map = {item["dialogue_id"]: item["stage_id"] for item in assignments}
    dialogue_map = {dialogue["id"]: dialogue for dialogue in dialogues}

    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    train_pt_dir = output_dir / "train_pt"
    eval_dir = output_dir / "eval"
    config_dir = output_dir / "configs"
    manifests_dir = output_dir / "manifests"
    outputs_dir = output_dir / "outputs" / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_pt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    stage_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    stage_pt_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    stage_dialogues: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for assignment in assignments:
        dialogue = dialogue_map[assignment["dialogue_id"]]
        stage_id = assignment["stage_id"]
        stage_dialogues[stage_id].append(dialogue)
        stage_records[stage_id].extend(
            build_training_records(
                dialogue,
                stage_id=stage_id,
                max_chunk_tokens=max_chunk_tokens,
                overlap_turns=overlap_turns,
                chunking_mode=chunking_mode,
                sliding_window_overlap_tokens=sliding_window_overlap_tokens,
            )
        )
        stage_pt_records[stage_id].extend(
            build_pretrain_records(
                dialogue,
                stage_id=stage_id,
                max_chunk_tokens=max_chunk_tokens,
                overlap_turns=overlap_turns,
                chunking_mode=chunking_mode,
                sliding_window_overlap_tokens=sliding_window_overlap_tokens,
                pt_header_style=pt_header_style,
                pt_record_style=pt_record_style,
            )
        )

    for stage_id in range(1, num_stages + 1):
        write_jsonl(stage_records[stage_id], train_dir / _stage_file_name(stage_id))
        write_jsonl(stage_pt_records[stage_id], train_pt_dir / _stage_pt_file_name(stage_id))
        write_jsonl(
            build_memory_eval_samples(stage_dialogues[stage_id], stage_map=stage_map),
            eval_dir / f"memory_eval_stage_{stage_id:02d}.jsonl",
        )

    all_eval_samples = build_memory_eval_samples(dialogues, stage_map=stage_map)
    write_jsonl(all_eval_samples, eval_dir / "memory_eval_all.jsonl")

    (manifests_dir / "dialogue_assignments.json").write_text(
        json.dumps(
            {
                "dataset_path": _manifest_dataset_path(dataset_path),
                "num_dialogues": len(dialogues),
                "num_stages": num_stages,
                "max_chunk_tokens": max_chunk_tokens,
                "overlap_turns": overlap_turns,
                "chunking_mode": chunking_mode,
                "sliding_window_overlap_tokens": sliding_window_overlap_tokens,
                "pt_header_style": pt_header_style,
                "pt_record_style": pt_record_style,
                "seed": seed,
                "assignments": assignments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset_info.json").write_text(
        json.dumps(_build_dataset_info(num_stages, pt_record_style=pt_record_style), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config_dir / "training_plan.yaml").write_text(
        yaml.safe_dump(
            _default_training_plan(num_stages, output_dir),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _build_stage_runbook(output_dir, num_stages),
        encoding="utf-8",
    )

    return {
        "num_dialogues": len(dialogues),
        "num_stages": num_stages,
        "train_dir": str(train_dir),
        "train_pt_dir": str(train_pt_dir),
        "eval_dir": str(eval_dir),
        "config_dir": str(config_dir),
    }


def build_entity_split_project(
    dataset_path: Path,
    output_dir: Path,
    num_stages: int = 10,
    max_chunk_tokens: int = 4096,
    overlap_turns: int = 1,
    chunking_mode: str = "turn_overlap",
    sliding_window_overlap_tokens: int = 0,
    pt_header_style: str = "structured",
    pt_record_style: str = "alpaca",
    qa_sampling_mode: str = "original",
    qa_max_samples_per_character: int = 0,
    qa_task_balance_mode: str = "none",
    qa_holdout_ratio: float = 0.2,
    qa_holdout_seed: int = 27182,
    qa_balanced_train_total_target: int = 0,
    seed: int = 42,
    entity_split_seed: int = 31415,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    if qa_task_balance_mode not in {"none", "task_balanced_holdout", "task_balanced_holdout_upsampled"}:
        raise ValueError(f"Unsupported qa_task_balance_mode: {qa_task_balance_mode}")
    if qa_task_balance_mode != "none" and qa_sampling_mode != "original":
        raise ValueError("qa_task_balance_mode currently requires qa_sampling_mode='original'")
    if qa_task_balance_mode == "task_balanced_holdout_upsampled" and qa_balanced_train_total_target <= 0:
        raise ValueError(
            "qa_balanced_train_total_target must be positive when qa_task_balance_mode='task_balanced_holdout_upsampled'"
        )

    dialogues = load_benchmark(dataset_path)
    assignments = assign_dialogues_to_stages(dialogues, num_stages=num_stages, seed=seed)
    stage_map = {item["dialogue_id"]: item["stage_id"] for item in assignments}
    dialogue_map = {dialogue["id"]: dialogue for dialogue in dialogues}

    output_dir = Path(output_dir)
    train_pt_dir = output_dir / "train_pt"
    train_qa_dir = output_dir / "train_qa_sft"
    eval_unseen_dir = output_dir / "eval_unseen"
    eval_seen_dir = output_dir / "eval_seen"
    eval_holdout_dir = output_dir / "eval_qa_holdout"
    config_dir = output_dir / "configs"
    manifests_dir = output_dir / "manifests"
    outputs_dir = output_dir / "outputs" / "eval"
    for path in [
        train_pt_dir,
        train_qa_dir,
        eval_unseen_dir,
        eval_seen_dir,
        eval_holdout_dir,
        config_dir,
        manifests_dir,
        outputs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    stage_dialogues: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    stage_pt_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    stage_qa_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    stage_seen_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    stage_unseen_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    stage_holdout_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    stage_raw_qa_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    split_manifest: Dict[str, Any] = {
        "entity_split_seed": entity_split_seed,
        "train_ratio": train_ratio,
        "test_ratio": 1.0 - train_ratio,
        "qa_task_balance_mode": qa_task_balance_mode,
        "qa_holdout_ratio": qa_holdout_ratio if qa_task_balance_mode != "none" else 0.0,
        "qa_holdout_seed": qa_holdout_seed if qa_task_balance_mode != "none" else None,
        "qa_balanced_train_total_target": (
            qa_balanced_train_total_target if qa_task_balance_mode == "task_balanced_holdout_upsampled" else None
        ),
        "stages": {},
    }

    for assignment in assignments:
        stage_dialogues[assignment["stage_id"]].append(dialogue_map[assignment["dialogue_id"]])

    for stage_id in range(1, num_stages + 1):
        split = split_stage_dialogues_by_entity(
            stage_dialogues[stage_id],
            train_ratio=train_ratio,
            seed=entity_split_seed + stage_id,
        )
        train_dialogues = split["train_dialogues"]
        test_dialogues = split["test_dialogues"]
        split_manifest["stages"][f"{stage_id:02d}"] = {
            "seed": entity_split_seed + stage_id,
            "train": split["train"],
            "test": split["test"],
        }

        for dialogue in stage_dialogues[stage_id]:
            stage_pt_records[stage_id].extend(
                build_pretrain_records(
                    dialogue,
                    stage_id=stage_id,
                    max_chunk_tokens=max_chunk_tokens,
                    overlap_turns=overlap_turns,
                    chunking_mode=chunking_mode,
                    sliding_window_overlap_tokens=sliding_window_overlap_tokens,
                    pt_header_style=pt_header_style,
                    pt_record_style=pt_record_style,
                )
            )

        raw_qa_records = rebalance_qa_sft_records(
            build_qa_sft_records(train_dialogues, stage_map=stage_map),
            qa_sampling_mode=qa_sampling_mode,
            qa_max_samples_per_character=qa_max_samples_per_character,
        )
        if qa_task_balance_mode == "none":
            stage_qa_records[stage_id].extend(raw_qa_records)
        else:
            stage_raw_qa_records[stage_id].extend(raw_qa_records)
        stage_seen_samples[stage_id].extend(
            build_memory_eval_samples(train_dialogues, stage_map=stage_map)
        )
        stage_unseen_samples[stage_id].extend(
            build_memory_eval_samples(test_dialogues, stage_map=stage_map)
        )

    qa_task_balance_manifest = None
    if qa_task_balance_mode in {"task_balanced_holdout", "task_balanced_holdout_upsampled"}:
        qa_task_balance_result = build_task_balanced_qa_holdout_splits(
            dict(stage_raw_qa_records),
            holdout_ratio=qa_holdout_ratio,
            seed=qa_holdout_seed,
            train_total_target=(
                qa_balanced_train_total_target if qa_task_balance_mode == "task_balanced_holdout_upsampled" else None
            ),
        )
        qa_task_balance_manifest = qa_task_balance_result["manifest"]
        for stage_id in range(1, num_stages + 1):
            stage_qa_records[stage_id].extend(qa_task_balance_result["train_records_by_stage"].get(stage_id, []))
            stage_holdout_samples[stage_id].extend(
                build_memory_eval_samples_from_qa_records(
                    qa_task_balance_result["holdout_records_by_stage"].get(stage_id, [])
                )
            )

    for stage_id in range(1, num_stages + 1):
        write_jsonl(stage_pt_records[stage_id], train_pt_dir / _stage_pt_file_name(stage_id))
        write_jsonl(stage_qa_records[stage_id], train_qa_dir / _stage_file_name(stage_id))
        write_jsonl(
            stage_unseen_samples[stage_id],
            eval_unseen_dir / f"memory_eval_stage_{stage_id:02d}.jsonl",
        )
        write_jsonl(
            stage_seen_samples[stage_id],
            eval_seen_dir / _stage_seen_eval_file_name(stage_id),
        )
        if stage_holdout_samples[stage_id]:
            write_jsonl(
                stage_holdout_samples[stage_id],
                eval_holdout_dir / _stage_holdout_eval_file_name(stage_id),
            )

    write_jsonl(
        [row for stage_id in range(1, num_stages + 1) for row in stage_unseen_samples[stage_id]],
        eval_unseen_dir / "memory_eval_all.jsonl",
    )
    write_jsonl(
        [row for stage_id in range(1, num_stages + 1) for row in stage_seen_samples[stage_id]],
        eval_seen_dir / "memory_eval_all_seen.jsonl",
    )
    if any(stage_holdout_samples.values()):
        write_jsonl(
            [row for stage_id in range(1, num_stages + 1) for row in stage_holdout_samples[stage_id]],
            eval_holdout_dir / "memory_eval_all_holdout.jsonl",
        )

    (manifests_dir / "dialogue_assignments.json").write_text(
        json.dumps(
            {
                "dataset_path": _manifest_dataset_path(dataset_path),
                "num_dialogues": len(dialogues),
                "num_stages": num_stages,
                "max_chunk_tokens": max_chunk_tokens,
                "overlap_turns": overlap_turns,
                "chunking_mode": chunking_mode,
                "sliding_window_overlap_tokens": sliding_window_overlap_tokens,
                "pt_header_style": pt_header_style,
                "pt_record_style": pt_record_style,
                "qa_sampling_mode": qa_sampling_mode,
                "qa_max_samples_per_character": qa_max_samples_per_character,
                "qa_task_weights": DEFAULT_QA_TASK_WEIGHTS if qa_sampling_mode == "role_balanced_upweight" else None,
                "qa_task_balance_mode": qa_task_balance_mode,
                "qa_holdout_ratio": qa_holdout_ratio if qa_task_balance_mode != "none" else 0.0,
                "qa_holdout_seed": qa_holdout_seed if qa_task_balance_mode != "none" else None,
                "qa_balanced_train_total_target": (
                    qa_balanced_train_total_target
                    if qa_task_balance_mode == "task_balanced_holdout_upsampled"
                    else None
                ),
                "seed": seed,
                "assignments": assignments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (manifests_dir / "entity_split_assignments.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if qa_task_balance_manifest is not None:
        (manifests_dir / "qa_task_balance_manifest.json").write_text(
            json.dumps(qa_task_balance_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (config_dir / "dataset_info.json").write_text(
        json.dumps(
            _build_entity_split_dataset_info(num_stages, pt_record_style=pt_record_style),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (config_dir / "training_plan.yaml").write_text(
        yaml.safe_dump(
            _default_entity_split_training_plan(num_stages, output_dir, train_ratio),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _build_entity_split_runbook(output_dir, num_stages),
        encoding="utf-8",
    )

    return {
        "num_dialogues": len(dialogues),
        "num_stages": num_stages,
        "train_pt_dir": str(train_pt_dir),
        "train_qa_dir": str(train_qa_dir),
        "eval_unseen_dir": str(eval_unseen_dir),
        "eval_seen_dir": str(eval_seen_dir),
        "eval_holdout_dir": str(eval_holdout_dir),
        "config_dir": str(config_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a dialogue-level continual-memory training project."
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("dialogue_gen_api/output/final/clean_v8_budget_direct/UMB_dialogue_benchmark.json"),
        help="Path to the clean UMB dialogue benchmark JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dialogue_training/project"),
        help="Directory to write the training scaffold.",
    )
    parser.add_argument(
        "--project_mode",
        choices=["standard", "entity_split"],
        default="standard",
        help="Which project scaffold to generate.",
    )
    parser.add_argument("--num_stages", type=int, default=10)
    parser.add_argument("--max_chunk_tokens", type=int, default=4096)
    parser.add_argument("--overlap_turns", type=int, default=1)
    parser.add_argument(
        "--chunking_mode",
        choices=["turn_overlap", "sliding_window"],
        default="turn_overlap",
    )
    parser.add_argument("--sliding_window_overlap_tokens", type=int, default=0)
    parser.add_argument(
        "--pt_header_style",
        choices=["structured", "natural"],
        default="structured",
    )
    parser.add_argument(
        "--pt_record_style",
        choices=["alpaca", "text"],
        default="alpaca",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entity_split_seed", type=int, default=31415)
    parser.add_argument(
        "--qa_sampling_mode",
        choices=["original", "role_balanced_upweight"],
        default="original",
    )
    parser.add_argument("--qa_max_samples_per_character", type=int, default=0)
    parser.add_argument(
        "--qa_task_balance_mode",
        choices=["none", "task_balanced_holdout", "task_balanced_holdout_upsampled"],
        default="none",
    )
    parser.add_argument("--qa_holdout_ratio", type=float, default=0.2)
    parser.add_argument("--qa_holdout_seed", type=int, default=27182)
    parser.add_argument("--qa_balanced_train_total_target", type=int, default=0)
    parser.add_argument("--entity_train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    if args.project_mode == "entity_split":
        result = build_entity_split_project(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_stages=args.num_stages,
            max_chunk_tokens=args.max_chunk_tokens,
            overlap_turns=args.overlap_turns,
            chunking_mode=args.chunking_mode,
            sliding_window_overlap_tokens=args.sliding_window_overlap_tokens,
            pt_header_style=args.pt_header_style,
            pt_record_style=args.pt_record_style,
            qa_sampling_mode=args.qa_sampling_mode,
            qa_max_samples_per_character=args.qa_max_samples_per_character,
            qa_task_balance_mode=args.qa_task_balance_mode,
            qa_holdout_ratio=args.qa_holdout_ratio,
            qa_holdout_seed=args.qa_holdout_seed,
            qa_balanced_train_total_target=args.qa_balanced_train_total_target,
            seed=args.seed,
            entity_split_seed=args.entity_split_seed,
            train_ratio=args.entity_train_ratio,
        )
    else:
        result = build_project(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_stages=args.num_stages,
            max_chunk_tokens=args.max_chunk_tokens,
            overlap_turns=args.overlap_turns,
            chunking_mode=args.chunking_mode,
            sliding_window_overlap_tokens=args.sliding_window_overlap_tokens,
            pt_header_style=args.pt_header_style,
            pt_record_style=args.pt_record_style,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
