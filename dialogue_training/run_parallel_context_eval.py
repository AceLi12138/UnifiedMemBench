from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dialogue_training.run_local_context_eval import (
    DEFAULT_CONTEXT_MAX_INPUT_TOKENS,
    DEFAULT_CONTEXT_MAX_NEW_TOKENS,
    load_context_eval_samples,
)
from dialogue_training.run_parallel_memory_eval import (
    _load_compute_aggregate_scores,
    _parse_gpu_ids,
)


def _group_gpu_ids_for_shards(gpu_ids: List[str], gpus_per_shard: int) -> List[List[str]]:
    group_size = int(gpus_per_shard)
    if group_size <= 0:
        raise ValueError("gpus_per_shard must be >= 1.")
    if not gpu_ids:
        return []
    if len(gpu_ids) % group_size != 0:
        raise ValueError(
            f"gpu_ids count ({len(gpu_ids)}) must be divisible by gpus_per_shard ({group_size})."
        )
    return [gpu_ids[idx : idx + group_size] for idx in range(0, len(gpu_ids), group_size)]


def _load_context_sample_ids(
    dataset_path: Path,
    max_dialogues: Optional[int] = None,
    max_samples: Optional[int] = None,
    eval_profile: str = "umb_tasklight_v1",
    response_format: str = "json",
    prompt_config_path: Optional[str] = None,
) -> List[str]:
    return [
        row["id"]
        for row in load_context_eval_samples(
            dataset_path=dataset_path,
            max_dialogues=max_dialogues,
            max_samples=max_samples,
            eval_profile=eval_profile,
            response_format=response_format,
            prompt_config_path=prompt_config_path,
        )
    ]


def _port_is_bindable(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", int(port)))
        return True
    except OSError:
        return False


def _reserve_vllm_port_bases(
    num_shards: int,
    *,
    block_size: int = 32,
    start_port: int = 41000,
    max_port: int = 65000,
) -> List[int]:
    if num_shards <= 0:
        return []

    port_bases: List[int] = []
    candidate = int(start_port)
    while len(port_bases) < num_shards:
        block_end = candidate + block_size - 1
        if block_end > max_port:
            raise RuntimeError(
                f"Unable to reserve {num_shards} non-overlapping vLLM port blocks starting at {start_port}."
            )
        if all(_port_is_bindable(port) for port in range(candidate, block_end + 1)):
            port_bases.append(candidate)
            candidate = block_end + 1
            continue
        candidate = block_end + 1

    return port_bases


def _build_child_command(
    python_executable: str,
    script_path: Path,
    dataset_path: Path,
    model_name_or_path: str,
    output_dir: Path,
    shard_index: int,
    num_shards: int,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    torch_dtype: str,
    judge_model: str,
    adapter_path: Optional[str] = None,
    force_new: bool = False,
    adaptive_max_new_tokens: bool = False,
    adaptive_max_new_tokens_multiplier: float = 2.0,
    adaptive_max_new_tokens_min: int = 64,
    adaptive_max_new_tokens_cap: Optional[int] = 256,
    disable_task_type_budgets: bool = False,
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
) -> List[str]:
    cmd = [
        python_executable,
        str(script_path),
        "--dataset_path",
        str(dataset_path),
        "--model_name_or_path",
        model_name_or_path,
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(batch_size),
        "--max_input_tokens",
        str(max_input_tokens),
        "--max_new_tokens",
        str(max_new_tokens),
        "--torch_dtype",
        torch_dtype,
        "--judge_model",
        judge_model,
        "--num_shards",
        str(num_shards),
        "--shard_index",
        str(shard_index),
        "--eval_profile",
        eval_profile,
        "--response_format",
        response_format,
        "--backend",
        backend,
    ]
    if adapter_path:
        cmd.extend(["--adapter_path", adapter_path])
    if force_new:
        cmd.append("--force_new")
    if adaptive_max_new_tokens:
        cmd.append("--adaptive_max_new_tokens")
        cmd.extend(
            [
                "--adaptive_max_new_tokens_multiplier",
                str(adaptive_max_new_tokens_multiplier),
                "--adaptive_max_new_tokens_min",
                str(adaptive_max_new_tokens_min),
            ]
        )
        if adaptive_max_new_tokens_cap is not None:
            cmd.extend(
                [
                    "--adaptive_max_new_tokens_cap",
                    str(adaptive_max_new_tokens_cap),
                ]
            )
    if disable_task_type_budgets:
        cmd.append("--disable_task_type_budgets")
    if disable_thinking:
        cmd.append("--disable_thinking")
    if max_dialogues is not None:
        cmd.extend(["--max_dialogues", str(max_dialogues)])
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    if prompt_config_path:
        cmd.extend(["--prompt_config_path", prompt_config_path])
    if backend == "vllm":
        cmd.extend(["--vllm_tokenizer_mode", vllm_tokenizer_mode])
        if vllm_trust_remote_code is True:
            cmd.append("--vllm_trust_remote_code")
        elif vllm_trust_remote_code is False:
            cmd.append("--no-vllm_trust_remote_code")
        cmd.extend([
            "--vllm_gpu_memory_utilization",
            str(vllm_gpu_memory_utilization),
        ])
        if vllm_max_model_len is not None:
            cmd.extend(["--vllm_max_model_len", str(vllm_max_model_len)])
        if vllm_max_num_seqs is not None:
            cmd.extend(["--vllm_max_num_seqs", str(vllm_max_num_seqs)])
        cmd.extend([
            "--vllm_tensor_parallel_size",
            str(vllm_tensor_parallel_size),
            "--vllm_seed",
            str(vllm_seed),
        ])
        if vllm_disable_eager:
            cmd.append("--vllm_disable_eager")
    return cmd


def _merge_shard_results(
    dataset_path: Path,
    shard_output_dirs: List[Path],
    output_dir: Path,
    max_dialogues: Optional[int] = None,
    max_samples: Optional[int] = None,
    eval_profile: str = "umb_tasklight_v1",
    response_format: str = "json",
    prompt_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    ordered_ids = _load_context_sample_ids(
        dataset_path=dataset_path,
        max_dialogues=max_dialogues,
        max_samples=max_samples,
        eval_profile=eval_profile,
        response_format=response_format,
        prompt_config_path=prompt_config_path,
    )
    order_map = {sample_id: idx for idx, sample_id in enumerate(ordered_ids)}

    merged: Dict[str, Dict[str, Any]] = {}
    for shard_dir in shard_output_dirs:
        results_path = shard_dir / "detailed_results.jsonl"
        if not results_path.exists():
            continue
        with results_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                merged[row["id"]] = row

    merged_rows = sorted(
        merged.values(),
        key=lambda row: order_map.get(row["id"], 10**18),
    )

    detailed_results_path = output_dir / "detailed_results.jsonl"
    with detailed_results_path.open("w", encoding="utf-8") as fh:
        for row in merged_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    compute_aggregate_scores = _load_compute_aggregate_scores()
    aggregate = compute_aggregate_scores(merged_rows)
    (output_dir / "scores.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "num_rows": len(merged_rows),
        "detailed_results_path": str(detailed_results_path),
        "scores_path": str(output_dir / "scores.json"),
        "aggregate": aggregate,
    }


def run_parallel_context_eval(
    dataset_path: Path,
    model_name_or_path: str,
    output_dir: Path,
    gpu_ids: List[str],
    gpus_per_shard: int = 1,
    adapter_path: Optional[str] = None,
    batch_size: int = 1,
    max_input_tokens: int = DEFAULT_CONTEXT_MAX_INPUT_TOKENS,
    max_new_tokens: int = DEFAULT_CONTEXT_MAX_NEW_TOKENS,
    torch_dtype: str = "bfloat16",
    judge_model: str = "",
    force_new: bool = False,
    adaptive_max_new_tokens: bool = False,
    adaptive_max_new_tokens_multiplier: float = 2.0,
    adaptive_max_new_tokens_min: int = 64,
    adaptive_max_new_tokens_cap: Optional[int] = 256,
    disable_task_type_budgets: bool = False,
    python_executable: Optional[str] = None,
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
    shards_root = output_dir / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    gpu_groups = _group_gpu_ids_for_shards(gpu_ids, gpus_per_shard)
    if backend != "vllm" and int(gpus_per_shard) != 1:
        raise ValueError("gpus_per_shard > 1 is only supported with backend=vllm.")

    effective_vllm_tensor_parallel_size = int(vllm_tensor_parallel_size)
    if backend == "vllm" and int(gpus_per_shard) > 1:
        if effective_vllm_tensor_parallel_size not in (1, int(gpus_per_shard)):
            raise ValueError(
                "gpus_per_shard must match vllm_tensor_parallel_size when grouping GPUs "
                f"(got gpus_per_shard={gpus_per_shard}, "
                f"vllm_tensor_parallel_size={vllm_tensor_parallel_size})."
            )
        effective_vllm_tensor_parallel_size = int(gpus_per_shard)

    python_exec = python_executable or sys.executable
    local_eval_script = Path(__file__).resolve().parent / "run_local_context_eval.py"

    processes: List[subprocess.Popen] = []
    log_paths: List[Path] = []
    shard_output_dirs: List[Path] = []
    vllm_port_bases = (
        _reserve_vllm_port_bases(len(gpu_groups))
        if backend == "vllm"
        else [None] * len(gpu_groups)
    )

    for shard_index, gpu_group in enumerate(gpu_groups):
        shard_output_dir = shards_root / f"shard_{shard_index:02d}"
        shard_output_dir.mkdir(parents=True, exist_ok=True)
        shard_output_dirs.append(shard_output_dir)
        log_path = shards_root / f"shard_{shard_index:02d}.log"
        log_paths.append(log_path)

        cmd = _build_child_command(
            python_executable=python_exec,
            script_path=local_eval_script,
            dataset_path=dataset_path,
            model_name_or_path=model_name_or_path,
            output_dir=shard_output_dir,
            shard_index=shard_index,
            num_shards=len(gpu_groups),
            batch_size=batch_size,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            torch_dtype=torch_dtype,
            judge_model=judge_model,
            adapter_path=adapter_path,
            force_new=force_new,
            adaptive_max_new_tokens=adaptive_max_new_tokens,
            adaptive_max_new_tokens_multiplier=adaptive_max_new_tokens_multiplier,
            adaptive_max_new_tokens_min=adaptive_max_new_tokens_min,
            adaptive_max_new_tokens_cap=adaptive_max_new_tokens_cap,
            disable_task_type_budgets=disable_task_type_budgets,
            max_dialogues=max_dialogues,
            max_samples=max_samples,
            eval_profile=eval_profile,
            response_format=response_format,
            prompt_config_path=prompt_config_path,
            backend=backend,
            vllm_tokenizer_mode=vllm_tokenizer_mode,
            vllm_trust_remote_code=vllm_trust_remote_code,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
            vllm_max_num_seqs=vllm_max_num_seqs,
            vllm_tensor_parallel_size=effective_vllm_tensor_parallel_size,
            vllm_seed=vllm_seed,
            vllm_disable_eager=vllm_disable_eager,
            disable_thinking=disable_thinking,
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
        env["PYTHONUNBUFFERED"] = "1"
        if backend == "vllm":
            env["VLLM_PORT"] = str(vllm_port_bases[shard_index])
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
        )
        process._codex_log_file = log_file  # type: ignore[attr-defined]
        processes.append(process)

    failures: List[Dict[str, Any]] = []
    for shard_index, process in enumerate(processes):
        return_code = process.wait()
        log_file = getattr(process, "_codex_log_file", None)
        if log_file is not None:
            log_file.close()
        if return_code != 0:
            failures.append(
                {
                    "shard_index": shard_index,
                    "return_code": return_code,
                    "log_path": str(log_paths[shard_index]),
                }
            )

    if failures:
        raise RuntimeError(
            "One or more context-eval shards failed: "
            + json.dumps(failures, ensure_ascii=False)
        )

    merged = _merge_shard_results(
        dataset_path=dataset_path,
        shard_output_dirs=shard_output_dirs,
        output_dir=output_dir,
        max_dialogues=max_dialogues,
        max_samples=max_samples,
        eval_profile=eval_profile,
        response_format=response_format,
        prompt_config_path=prompt_config_path,
    )
    (output_dir / "shard_manifest.json").write_text(
        json.dumps(
            {
                "gpu_ids": gpu_ids,
                "gpus_per_shard": int(gpus_per_shard),
                "gpu_groups": gpu_groups,
                "logs": [str(path) for path in log_paths],
                "shard_output_dirs": [str(path) for path in shard_output_dirs],
                "merged": merged,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full-dialogue context QA evaluation in parallel across multiple GPUs."
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gpu_ids", type=str, required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--gpus_per_shard", type=int, default=1)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_input_tokens", type=int, default=DEFAULT_CONTEXT_MAX_INPUT_TOKENS)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_CONTEXT_MAX_NEW_TOKENS)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--judge_model", type=str, default="")
    parser.add_argument("--force_new", action="store_true")
    parser.add_argument("--adaptive_max_new_tokens", action="store_true")
    parser.add_argument("--adaptive_max_new_tokens_multiplier", type=float, default=2.0)
    parser.add_argument("--adaptive_max_new_tokens_min", type=int, default=64)
    parser.add_argument("--adaptive_max_new_tokens_cap", type=int, default=256)
    parser.add_argument("--disable_task_type_budgets", action="store_true")
    parser.add_argument("--python_executable", type=str, default=None)
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

    result = run_parallel_context_eval(
        dataset_path=args.dataset_path,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        gpus_per_shard=args.gpus_per_shard,
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
            {"overall_equal_weighted_score": result["aggregate"].get("overall_equal_weighted_score")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
