from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dialogue_training.run_parallel_memory_eval import _parse_gpu_ids, run_parallel_memory_eval  # type: ignore
else:
    from dialogue_training.run_parallel_memory_eval import _parse_gpu_ids, run_parallel_memory_eval


@dataclass(frozen=True)
class StagewiseEvalJob:
    checkpoint_stage: int
    eval_stage: int
    split: str
    dataset_path: Path
    output_dir: Path


def _sanitize_label(text: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return label or "run"


def _default_name_suffix(judge_model: str) -> str:
    raw = str(judge_model or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if "mimo" in lowered:
        return "mimo"
    return _sanitize_label(raw)


def _stagewise_output_dir_name(eval_stage: int, split: str, judge_model: str, name_suffix: Optional[str] = None) -> str:
    suffix = str(name_suffix or "").strip() or _default_name_suffix(judge_model)
    base = f"stage_{eval_stage:02d}_{split}"
    return f"{base}_{suffix}" if suffix else base


def _resolve_dataset_path(project_dir: Path, eval_stage: int, split: str) -> Path:
    if split == "seen":
        return project_dir / "eval_seen" / f"memory_eval_stage_{eval_stage:02d}_seen.jsonl"
    if split == "unseen":
        return project_dir / "eval_unseen" / f"memory_eval_stage_{eval_stage:02d}.jsonl"
    raise ValueError(f"Unsupported split: {split}")


def build_stagewise_eval_jobs(
    project_dir: Path,
    checkpoint_stage: int,
    output_root: Path,
    *,
    stage_from: int = 1,
    stage_to: Optional[int] = None,
    splits: Sequence[str] = ("seen", "unseen"),
    judge_model: str = "",
    name_suffix: Optional[str] = None,
    skip_existing: bool = True,
) -> List[StagewiseEvalJob]:
    if checkpoint_stage <= 0:
        raise ValueError("checkpoint_stage must be positive")

    end_stage = checkpoint_stage if stage_to is None else int(stage_to)
    if stage_from <= 0 or end_stage <= 0:
        raise ValueError("stage_from and stage_to must be positive")
    if stage_from > end_stage:
        raise ValueError("stage_from cannot be greater than stage_to")

    checkpoint_dir = Path(output_root) / f"checkpoint_stage_{checkpoint_stage:02d}"
    jobs: List[StagewiseEvalJob] = []
    for eval_stage in range(int(stage_from), end_stage + 1):
        for split in splits:
            dataset_path = _resolve_dataset_path(Path(project_dir), eval_stage, split)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Missing dataset for stagewise eval: {dataset_path}")
            output_dir = checkpoint_dir / _stagewise_output_dir_name(
                eval_stage=eval_stage,
                split=split,
                judge_model=judge_model,
                name_suffix=name_suffix,
            )
            if skip_existing and (output_dir / "scores.json").exists():
                continue
            jobs.append(
                StagewiseEvalJob(
                    checkpoint_stage=checkpoint_stage,
                    eval_stage=eval_stage,
                    split=split,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                )
            )
    return jobs


def run_stagewise_memory_eval(
    project_dir: Path,
    model_name_or_path: str,
    checkpoint_stage: int,
    gpu_ids: Sequence[str],
    tokenizer_name_or_path: Optional[str] = None,
    *,
    output_root: Optional[Path] = None,
    stage_from: int = 1,
    stage_to: Optional[int] = None,
    splits: Sequence[str] = ("seen", "unseen"),
    judge_model: str = "",
    name_suffix: Optional[str] = None,
    skip_existing: bool = True,
    adapter_path: Optional[str] = None,
    batch_size: int = 4,
    max_input_tokens: int = 1024,
    max_new_tokens: int = 512,
    torch_dtype: str = "bfloat16",
    force_new: bool = False,
    adaptive_max_new_tokens: bool = False,
    adaptive_max_new_tokens_multiplier: float = 2.0,
    adaptive_max_new_tokens_min: int = 64,
    adaptive_max_new_tokens_cap: Optional[int] = 256,
    disable_task_type_budgets: bool = False,
    python_executable: Optional[str] = None,
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
    eval_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    output_root = Path(output_root or (Path(project_dir) / "outputs" / "eval_stagewise"))
    jobs = build_stagewise_eval_jobs(
        project_dir=Path(project_dir),
        checkpoint_stage=checkpoint_stage,
        output_root=output_root,
        stage_from=stage_from,
        stage_to=stage_to,
        splits=splits,
        judge_model=judge_model,
        name_suffix=name_suffix,
        skip_existing=skip_existing,
    )

    runner = eval_runner or run_parallel_memory_eval
    results: List[Dict[str, Any]] = []
    checkpoint_dir = output_root / f"checkpoint_stage_{checkpoint_stage:02d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        merged = runner(
            dataset_path=job.dataset_path,
            model_name_or_path=model_name_or_path,
            output_dir=job.output_dir,
            gpu_ids=list(gpu_ids),
            tokenizer_name_or_path=tokenizer_name_or_path,
            adapter_path=adapter_path,
            batch_size=batch_size,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            torch_dtype=torch_dtype,
            judge_model=judge_model,
            force_new=force_new,
            adaptive_max_new_tokens=adaptive_max_new_tokens,
            adaptive_max_new_tokens_multiplier=adaptive_max_new_tokens_multiplier,
            adaptive_max_new_tokens_min=adaptive_max_new_tokens_min,
            adaptive_max_new_tokens_cap=adaptive_max_new_tokens_cap,
            disable_task_type_budgets=disable_task_type_budgets,
            python_executable=python_executable,
            backend=backend,
            vllm_tokenizer_mode=vllm_tokenizer_mode,
            vllm_trust_remote_code=vllm_trust_remote_code,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
            vllm_max_num_seqs=vllm_max_num_seqs,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_seed=vllm_seed,
            vllm_disable_eager=vllm_disable_eager,
            disable_thinking=disable_thinking,
        )
        results.append(
            {
                **asdict(job),
                "dataset_path": str(job.dataset_path),
                "output_dir": str(job.output_dir),
                "overall_equal_weighted_score": merged.get("aggregate", {}).get("overall_equal_weighted_score"),
            }
        )

    manifest = {
        "checkpoint_stage": checkpoint_stage,
        "model_name_or_path": model_name_or_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "output_root": str(output_root),
        "jobs": results,
    }
    (checkpoint_dir / "stagewise_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _parse_splits(raw: str) -> Tuple[str, ...]:
    values = tuple(item.strip() for item in str(raw or "").split(",") if item.strip())
    if not values:
        raise ValueError("splits must contain at least one of seen/unseen")
    invalid = [item for item in values if item not in {"seen", "unseen"}]
    if invalid:
        raise ValueError(f"Unsupported splits: {invalid}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stage-wise memory evaluation for all learned stages.")
    parser.add_argument("--project_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--checkpoint_stage", type=int, required=True)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--stage_from", type=int, default=1)
    parser.add_argument("--stage_to", type=int, default=None)
    parser.add_argument("--splits", type=str, default="seen,unseen")
    parser.add_argument("--name_suffix", type=str, default=None)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_ids", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--judge_model", type=str, default="")
    parser.add_argument("--force_new", action="store_true")
    parser.add_argument("--adaptive_max_new_tokens", action="store_true")
    parser.add_argument("--adaptive_max_new_tokens_multiplier", type=float, default=2.0)
    parser.add_argument("--adaptive_max_new_tokens_min", type=int, default=64)
    parser.add_argument("--adaptive_max_new_tokens_cap", type=int, default=256)
    parser.add_argument("--disable_task_type_budgets", action="store_true")
    parser.add_argument("--python_executable", type=str, default=None)
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

    result = run_stagewise_memory_eval(
        project_dir=args.project_dir,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        checkpoint_stage=args.checkpoint_stage,
        output_root=args.output_root,
        stage_from=args.stage_from,
        stage_to=args.stage_to,
        splits=_parse_splits(args.splits),
        name_suffix=args.name_suffix,
        skip_existing=args.skip_existing,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
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
        disable_task_type_budgets=args.disable_task_type_budgets,
        python_executable=args.python_executable,
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
    print(json.dumps({"checkpoint_stage": result["checkpoint_stage"], "num_jobs": len(result["jobs"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
