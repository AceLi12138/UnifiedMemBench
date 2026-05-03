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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dialogue_training.memory_eval_utils import load_memory_eval_jsonl  # type: ignore
else:
    from dialogue_training.memory_eval_utils import load_memory_eval_jsonl


def _parse_gpu_ids(raw: str) -> List[str]:
    items = [item.strip() for item in str(raw or "").split(",")]
    gpu_ids = [item for item in items if item]
    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one GPU id")
    return gpu_ids


def _load_compute_aggregate_scores():
    eval_dir = Path(__file__).resolve().parent.parent / "dialogue_gen_api" / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import run_eval  # type: ignore

    return run_eval.compute_aggregate_scores


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
    tokenizer_name_or_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    force_new: bool = False,
    adaptive_max_new_tokens: bool = False,
    adaptive_max_new_tokens_multiplier: float = 2.0,
    adaptive_max_new_tokens_min: int = 64,
    adaptive_max_new_tokens_cap: Optional[int] = 256,
    disable_task_type_budgets: bool = False,
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
        "--backend",
        backend,
        "--num_shards",
        str(num_shards),
        "--shard_index",
        str(shard_index),
    ]
    if tokenizer_name_or_path:
        cmd.extend(["--tokenizer_name_or_path", tokenizer_name_or_path])
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
    if backend == "vllm":
        cmd.extend(["--vllm_tokenizer_mode", vllm_tokenizer_mode])
        if vllm_trust_remote_code is not None:
            cmd.append("--vllm_trust_remote_code" if vllm_trust_remote_code else "--no-vllm_trust_remote_code")
        cmd.extend(["--vllm_gpu_memory_utilization", str(vllm_gpu_memory_utilization)])
        if vllm_max_model_len is not None:
            cmd.extend(["--vllm_max_model_len", str(vllm_max_model_len)])
        if vllm_max_num_seqs is not None:
            cmd.extend(["--vllm_max_num_seqs", str(vllm_max_num_seqs)])
        cmd.extend(["--vllm_tensor_parallel_size", str(vllm_tensor_parallel_size)])
        cmd.extend(["--vllm_seed", str(vllm_seed)])
        if vllm_disable_eager:
            cmd.append("--vllm_disable_eager")
    return cmd


def _merge_shard_results(
    dataset_path: Path,
    shard_output_dirs: List[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    ordered_ids = [row["id"] for row in load_memory_eval_jsonl(dataset_path)]
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


def run_parallel_memory_eval(
    dataset_path: Path,
    model_name_or_path: str,
    output_dir: Path,
    gpu_ids: List[str],
    tokenizer_name_or_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    batch_size: int = 4,
    max_input_tokens: int = 1024,
    max_new_tokens: int = 512,
    torch_dtype: str = "bfloat16",
    judge_model: str = "",
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
    vllm_port_start: int = 41000,
    disable_thinking: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_root = output_dir / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    python_exec = python_executable or sys.executable
    local_eval_script = Path(__file__).resolve().parent / "run_local_memory_eval.py"

    processes: List[subprocess.Popen] = []
    log_paths: List[Path] = []
    shard_output_dirs: List[Path] = []
    vllm_port_bases = (
        _reserve_vllm_port_bases(len(gpu_ids), start_port=int(vllm_port_start))
        if backend == "vllm"
        else [None] * len(gpu_ids)
    )

    for shard_index, gpu_id in enumerate(gpu_ids):
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
            tokenizer_name_or_path=tokenizer_name_or_path,
            output_dir=shard_output_dir,
            shard_index=shard_index,
            num_shards=len(gpu_ids),
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

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
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
            "One or more eval shards failed: "
            + json.dumps(failures, ensure_ascii=False)
        )

    merged = _merge_shard_results(
        dataset_path=dataset_path,
        shard_output_dirs=shard_output_dirs,
        output_dir=output_dir,
    )
    (output_dir / "shard_manifest.json").write_text(
        json.dumps(
            {
                "gpu_ids": gpu_ids,
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
        description="Run memory QA evaluation in parallel across multiple GPUs."
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gpu_ids", type=str, required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
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
    parser.add_argument("--vllm_port_start", type=int, default=41000)
    parser.add_argument("--disable_thinking", action="store_true")
    args = parser.parse_args()

    result = run_parallel_memory_eval(
        dataset_path=args.dataset_path,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        tokenizer_name_or_path=args.tokenizer_name_or_path,
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
        vllm_port_start=args.vllm_port_start,
        disable_thinking=args.disable_thinking,
    )
    print(json.dumps({"overall_equal_weighted_score": result["aggregate"].get("overall_equal_weighted_score")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
