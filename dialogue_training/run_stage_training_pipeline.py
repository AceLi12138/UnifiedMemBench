from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


def _format_stage(stage: int) -> str:
    return f"stage_{int(stage):02d}"


def _format_lr_label(lr: float) -> str:
    text = f"{lr:.0e}".lower()
    return text.replace("-", "").replace("+", "")


def _format_epoch_label(epoch: float) -> str:
    if float(epoch).is_integer():
        return str(int(epoch))
    return str(epoch).replace(".", "p")


@dataclass(frozen=True)
class StageTrainingJob:
    stage: int
    initial_model_path: str
    pt_dataset: str
    qa_dataset: str
    pt_output_dir: Path
    pt_export_dir: Path
    sft_output_dir: Path
    final_export_dir: Path


def _json_ready_mapping(mapping: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in mapping.items():
        normalized[key] = str(value) if isinstance(value, Path) else value
    return normalized


def _read_json_file(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sync_model_config_fields(reference_model_path: str | Path, target_model_path: str | Path) -> List[str]:
    """Restore critical loader metadata that some export paths may drop."""
    reference_config_path = Path(reference_model_path) / "config.json"
    target_config_path = Path(target_model_path) / "config.json"
    reference_config = _read_json_file(reference_config_path)
    target_config = _read_json_file(target_config_path)
    if reference_config is None or target_config is None:
        return []

    if str(reference_config.get("model_type", "")).lower() != "qwen2":
        return []
    if str(target_config.get("model_type", "")).lower() != "qwen2":
        return []

    fields_to_preserve = ("rope_theta", "sliding_window", "torch_dtype")
    changed: List[str] = []
    for field in fields_to_preserve:
        value = reference_config.get(field)
        if value is None and field == "rope_theta":
            rope_parameters = reference_config.get("rope_parameters")
            if isinstance(rope_parameters, dict):
                value = rope_parameters.get("rope_theta")

        if value is None:
            continue
        if target_config.get(field) != value:
            target_config[field] = value
            changed.append(field)

    if changed:
        target_config_path.write_text(
            json.dumps(target_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for metadata_file in ("sparse_attention_config.json",):
        reference_metadata_path = Path(reference_model_path) / metadata_file
        target_metadata_path = Path(target_model_path) / metadata_file
        if not reference_metadata_path.exists():
            continue
        reference_bytes = reference_metadata_path.read_bytes()
        if target_metadata_path.exists() and target_metadata_path.read_bytes() == reference_bytes:
            continue
        target_metadata_path.write_bytes(reference_bytes)
        changed.append(metadata_file)

    return changed


def build_stage_training_jobs(
    *,
    project_dir: Path,
    merged_root: Path,
    initial_model_path: str,
    start_stage: int,
    end_stage: int,
    run_prefix: str,
    pt_epochs: float,
    pt_lr: float,
    sft_epochs: float,
    sft_lr: float,
    pt_name_suffix: str,
    sft_name_suffix: str,
) -> List[StageTrainingJob]:
    if start_stage <= 0 or end_stage <= 0:
        raise ValueError("start_stage and end_stage must be positive")
    if start_stage > end_stage:
        raise ValueError("start_stage cannot be greater than end_stage")

    jobs: List[StageTrainingJob] = []
    current_model_path = str(initial_model_path)
    train_root = Path(project_dir) / "outputs" / "train"
    merged_root = Path(merged_root)

    for stage in range(int(start_stage), int(end_stage) + 1):
        stage_tag = _format_stage(stage)
        pt_base = f"{run_prefix}_{stage_tag}_fullpt_e{_format_epoch_label(pt_epochs)}_lr{_format_lr_label(pt_lr)}_{pt_name_suffix}"
        sft_base = f"{run_prefix}_{stage_tag}_fullptqa_e{_format_epoch_label(sft_epochs)}_lr{_format_lr_label(sft_lr)}_{sft_name_suffix}"

        job = StageTrainingJob(
            stage=stage,
            initial_model_path=current_model_path,
            pt_dataset=f"umb_entitysplit_{stage_tag}_pt",
            qa_dataset=f"umb_entitysplit_{stage_tag}_qa",
            pt_output_dir=train_root / pt_base,
            pt_export_dir=merged_root / pt_base,
            sft_output_dir=train_root / sft_base,
            final_export_dir=merged_root / sft_base,
        )
        jobs.append(job)
        current_model_path = str(job.final_export_dir)

    return jobs


def _build_env(cuda_visible_devices: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env["FORCE_TORCHRUN"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _build_pt_train_cmd(
    *,
    job: StageTrainingJob,
    dataset_dir: Path,
    template: str,
    trust_remote_code: bool,
    deepspeed_config: Path,
    cutoff_len: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    bf16: bool,
    logging_steps: int,
    save_strategy: str,
    save_only_model: bool,
    report_to: str,
    overwrite_cache: bool,
    overwrite_output_dir: bool,
) -> List[str]:
    return [
        "llamafactory-cli",
        "train",
        "--model_name_or_path",
        job.initial_model_path,
        "--trust_remote_code",
        str(trust_remote_code).lower(),
        "--stage",
        "pt",
        "--do_train",
        "true",
        "--finetuning_type",
        "full",
        "--deepspeed",
        str(deepspeed_config),
        "--dataset",
        job.pt_dataset,
        "--dataset_dir",
        str(dataset_dir),
        "--template",
        template,
        "--cutoff_len",
        str(cutoff_len),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--learning_rate",
        str(learning_rate),
        "--num_train_epochs",
        str(num_train_epochs),
        "--lr_scheduler_type",
        lr_scheduler_type,
        "--warmup_ratio",
        str(warmup_ratio),
        "--bf16",
        str(bf16).lower(),
        "--logging_steps",
        str(logging_steps),
        "--save_strategy",
        save_strategy,
        "--save_only_model",
        str(save_only_model).lower(),
        "--report_to",
        report_to,
        "--overwrite_cache",
        str(overwrite_cache).lower(),
        "--overwrite_output_dir",
        str(overwrite_output_dir).lower(),
        "--output_dir",
        str(job.pt_output_dir),
    ]


def _build_export_cmd(
    *,
    model_name_or_path: str,
    export_dir: Path,
    template: str,
    trust_remote_code: bool,
    export_size: int,
    export_device: str,
    export_legacy_format: bool,
    adapter_name_or_path: Optional[str] = None,
    finetuning_type: Optional[str] = None,
) -> List[str]:
    cmd = [
        "llamafactory-cli",
        "export",
        "--model_name_or_path",
        str(model_name_or_path),
    ]
    if adapter_name_or_path:
        cmd.extend(["--adapter_name_or_path", str(adapter_name_or_path)])
    cmd.extend(
        [
            "--template",
            template,
            "--trust_remote_code",
            str(trust_remote_code).lower(),
        ]
    )
    if finetuning_type:
        cmd.extend(["--finetuning_type", finetuning_type])
    cmd.extend(
        [
            "--export_dir",
            str(export_dir),
            "--export_size",
            str(export_size),
            "--export_device",
            export_device,
            "--export_legacy_format",
            str(export_legacy_format).lower(),
        ]
    )
    return cmd


def _build_sft_train_cmd(
    *,
    job: StageTrainingJob,
    dataset_dir: Path,
    template: str,
    trust_remote_code: bool,
    cutoff_len: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    lora_rank: int,
    lora_alpha: int,
    bf16: bool,
    logging_steps: int,
    save_strategy: str,
    save_total_limit: int,
    report_to: str,
    overwrite_cache: bool,
    overwrite_output_dir: bool,
) -> List[str]:
    return [
        "llamafactory-cli",
        "train",
        "--model_name_or_path",
        str(job.pt_export_dir),
        "--trust_remote_code",
        str(trust_remote_code).lower(),
        "--stage",
        "sft",
        "--do_train",
        "true",
        "--finetuning_type",
        "lora",
        "--lora_target",
        "all",
        "--dataset",
        job.qa_dataset,
        "--dataset_dir",
        str(dataset_dir),
        "--template",
        template,
        "--cutoff_len",
        str(cutoff_len),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--learning_rate",
        str(learning_rate),
        "--num_train_epochs",
        str(num_train_epochs),
        "--lr_scheduler_type",
        lr_scheduler_type,
        "--warmup_ratio",
        str(warmup_ratio),
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
        "--bf16",
        str(bf16).lower(),
        "--logging_steps",
        str(logging_steps),
        "--save_strategy",
        save_strategy,
        "--save_total_limit",
        str(save_total_limit),
        "--report_to",
        report_to,
        "--overwrite_cache",
        str(overwrite_cache).lower(),
        "--overwrite_output_dir",
        str(overwrite_output_dir).lower(),
        "--output_dir",
        str(job.sft_output_dir),
    ]


def _default_runner(cmd: Sequence[str], *, cwd: Path, env: Dict[str, str]) -> None:
    subprocess.run(list(cmd), cwd=str(cwd), env=env, check=True)


def run_stage_training_pipeline(
    *,
    llamafactory_root: Path,
    project_dir: Path,
    merged_root: Path,
    initial_model_path: str,
    start_stage: int,
    end_stage: int,
    run_prefix: str,
    pt_cuda_visible_devices: str,
    sft_cuda_visible_devices: str,
    template: str = "glm4",
    trust_remote_code: bool = True,
    deepspeed_config: Optional[Path] = None,
    pt_cutoff_len: int = 2048,
    pt_per_device_train_batch_size: int = 1,
    pt_gradient_accumulation_steps: int = 2,
    pt_learning_rate: float = 1e-5,
    pt_num_train_epochs: float = 3.0,
    pt_lr_scheduler_type: str = "cosine",
    pt_warmup_ratio: float = 0.03,
    pt_bf16: bool = True,
    pt_logging_steps: int = 10,
    pt_save_strategy: str = "no",
    pt_save_only_model: bool = True,
    sft_cutoff_len: int = 2048,
    sft_per_device_train_batch_size: int = 1,
    sft_gradient_accumulation_steps: int = 1,
    sft_learning_rate: float = 5e-6,
    sft_num_train_epochs: float = 1.0,
    sft_lr_scheduler_type: str = "cosine",
    sft_warmup_ratio: float = 0.03,
    sft_lora_rank: int = 64,
    sft_lora_alpha: int = 128,
    sft_bf16: bool = True,
    sft_logging_steps: int = 10,
    sft_save_strategy: str = "epoch",
    sft_save_total_limit: int = 2,
    report_to: str = "none",
    overwrite_cache: bool = True,
    overwrite_output_dir: bool = True,
    export_size: int = 5,
    export_device: str = "cpu",
    export_legacy_format: bool = False,
    pt_name_suffix: str = "8gpu",
    sft_name_suffix: str = "8gpupt_4gpusft",
    manifest_root: Optional[Path] = None,
    skip_existing: bool = True,
    command_runner: Optional[Callable[..., None]] = None,
) -> Dict[str, object]:
    deepspeed_config = deepspeed_config or (
        Path(llamafactory_root) / "examples" / "deepspeed" / "ds_z3_config.json"
    )
    dataset_dir = Path(project_dir) / "configs"
    runner = command_runner or _default_runner

    jobs = build_stage_training_jobs(
        project_dir=project_dir,
        merged_root=merged_root,
        initial_model_path=initial_model_path,
        start_stage=start_stage,
        end_stage=end_stage,
        run_prefix=run_prefix,
        pt_epochs=pt_num_train_epochs,
        pt_lr=pt_learning_rate,
        sft_epochs=sft_num_train_epochs,
        sft_lr=sft_learning_rate,
        pt_name_suffix=pt_name_suffix,
        sft_name_suffix=sft_name_suffix,
    )

    manifest_steps: List[Dict[str, object]] = []
    current_model_path = str(initial_model_path)

    for job in jobs:
        effective_job = StageTrainingJob(
            stage=job.stage,
            initial_model_path=current_model_path,
            pt_dataset=job.pt_dataset,
            qa_dataset=job.qa_dataset,
            pt_output_dir=job.pt_output_dir,
            pt_export_dir=job.pt_export_dir,
            sft_output_dir=job.sft_output_dir,
            final_export_dir=job.final_export_dir,
        )

        if skip_existing and effective_job.final_export_dir.exists():
            _sync_model_config_fields(effective_job.initial_model_path, effective_job.final_export_dir)
            manifest_steps.append(
                _json_ready_mapping(
                    {
                    **asdict(effective_job),
                    "initial_model_path": effective_job.initial_model_path,
                    "status": "skipped_existing",
                    }
                )
            )
            current_model_path = str(effective_job.final_export_dir)
            continue

        pt_train_cmd = _build_pt_train_cmd(
            job=effective_job,
            dataset_dir=dataset_dir,
            template=template,
            trust_remote_code=trust_remote_code,
            deepspeed_config=deepspeed_config,
            cutoff_len=pt_cutoff_len,
            per_device_train_batch_size=pt_per_device_train_batch_size,
            gradient_accumulation_steps=pt_gradient_accumulation_steps,
            learning_rate=pt_learning_rate,
            num_train_epochs=pt_num_train_epochs,
            lr_scheduler_type=pt_lr_scheduler_type,
            warmup_ratio=pt_warmup_ratio,
            bf16=pt_bf16,
            logging_steps=pt_logging_steps,
            save_strategy=pt_save_strategy,
            save_only_model=pt_save_only_model,
            report_to=report_to,
            overwrite_cache=overwrite_cache,
            overwrite_output_dir=overwrite_output_dir,
        )
        pt_export_cmd = _build_export_cmd(
            model_name_or_path=str(effective_job.pt_output_dir),
            export_dir=effective_job.pt_export_dir,
            template=template,
            trust_remote_code=trust_remote_code,
            export_size=export_size,
            export_device=export_device,
            export_legacy_format=export_legacy_format,
        )
        sft_train_cmd = _build_sft_train_cmd(
            job=effective_job,
            dataset_dir=dataset_dir,
            template=template,
            trust_remote_code=trust_remote_code,
            cutoff_len=sft_cutoff_len,
            per_device_train_batch_size=sft_per_device_train_batch_size,
            gradient_accumulation_steps=sft_gradient_accumulation_steps,
            learning_rate=sft_learning_rate,
            num_train_epochs=sft_num_train_epochs,
            lr_scheduler_type=sft_lr_scheduler_type,
            warmup_ratio=sft_warmup_ratio,
            lora_rank=sft_lora_rank,
            lora_alpha=sft_lora_alpha,
            bf16=sft_bf16,
            logging_steps=sft_logging_steps,
            save_strategy=sft_save_strategy,
            save_total_limit=sft_save_total_limit,
            report_to=report_to,
            overwrite_cache=overwrite_cache,
            overwrite_output_dir=overwrite_output_dir,
        )
        sft_export_cmd = _build_export_cmd(
            model_name_or_path=str(effective_job.pt_export_dir),
            adapter_name_or_path=str(effective_job.sft_output_dir),
            finetuning_type="lora",
            export_dir=effective_job.final_export_dir,
            template=template,
            trust_remote_code=trust_remote_code,
            export_size=export_size,
            export_device=export_device,
            export_legacy_format=export_legacy_format,
        )

        runner(pt_train_cmd, cwd=Path(llamafactory_root), env=_build_env(pt_cuda_visible_devices))
        _sync_model_config_fields(effective_job.initial_model_path, effective_job.pt_output_dir)
        runner(pt_export_cmd, cwd=Path(llamafactory_root), env=os.environ.copy())
        _sync_model_config_fields(effective_job.initial_model_path, effective_job.pt_export_dir)
        runner(sft_train_cmd, cwd=Path(llamafactory_root), env=_build_env(sft_cuda_visible_devices))
        _sync_model_config_fields(effective_job.initial_model_path, effective_job.sft_output_dir)
        runner(sft_export_cmd, cwd=Path(llamafactory_root), env=os.environ.copy())
        _sync_model_config_fields(effective_job.initial_model_path, effective_job.final_export_dir)

        manifest_steps.append(
            _json_ready_mapping(
                {
                **asdict(effective_job),
                "initial_model_path": effective_job.initial_model_path,
                "status": "completed",
                "pt_train_cmd": pt_train_cmd,
                "pt_export_cmd": pt_export_cmd,
                "sft_train_cmd": sft_train_cmd,
                "sft_export_cmd": sft_export_cmd,
                }
            )
        )
        current_model_path = str(effective_job.final_export_dir)

    manifest_root = Path(manifest_root or (Path(project_dir) / "outputs" / "train_pipeline"))
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"stage_pipeline_{start_stage:02d}_{end_stage:02d}.json"
    manifest = {
        "llamafactory_root": str(llamafactory_root),
        "project_dir": str(project_dir),
        "merged_root": str(merged_root),
        "initial_model_path": initial_model_path,
        "start_stage": start_stage,
        "end_stage": end_stage,
        "steps": manifest_steps,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chained PT/export/SFT/export stage training pipeline.")
    parser.add_argument("--llamafactory_root", type=Path, required=True)
    parser.add_argument("--project_dir", type=Path, required=True)
    parser.add_argument("--merged_root", type=Path, required=True)
    parser.add_argument("--initial_model_path", type=str, required=True)
    parser.add_argument("--start_stage", type=int, required=True)
    parser.add_argument("--end_stage", type=int, required=True)
    parser.add_argument("--run_prefix", type=str, required=True)
    parser.add_argument("--pt_cuda_visible_devices", type=str, required=True)
    parser.add_argument("--sft_cuda_visible_devices", type=str, required=True)
    parser.add_argument("--template", type=str, default="glm4")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deepspeed_config", type=Path, default=None)
    parser.add_argument("--pt_cutoff_len", type=int, default=2048)
    parser.add_argument("--pt_per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--pt_gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--pt_learning_rate", type=float, default=1e-5)
    parser.add_argument("--pt_num_train_epochs", type=float, default=3.0)
    parser.add_argument("--pt_lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--pt_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pt_bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pt_logging_steps", type=int, default=10)
    parser.add_argument("--pt_save_strategy", type=str, default="no")
    parser.add_argument("--pt_save_only_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sft_cutoff_len", type=int, default=2048)
    parser.add_argument("--sft_per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--sft_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--sft_learning_rate", type=float, default=5e-6)
    parser.add_argument("--sft_num_train_epochs", type=float, default=1.0)
    parser.add_argument("--sft_lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--sft_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--sft_lora_rank", type=int, default=64)
    parser.add_argument("--sft_lora_alpha", type=int, default=128)
    parser.add_argument("--sft_bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sft_logging_steps", type=int, default=10)
    parser.add_argument("--sft_save_strategy", type=str, default="epoch")
    parser.add_argument("--sft_save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--overwrite_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite_output_dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export_size", type=int, default=5)
    parser.add_argument("--export_device", type=str, default="cpu")
    parser.add_argument("--export_legacy_format", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pt_name_suffix", type=str, default="8gpu")
    parser.add_argument("--sft_name_suffix", type=str, default="8gpupt_4gpusft")
    parser.add_argument("--manifest_root", type=Path, default=None)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    result = run_stage_training_pipeline(
        llamafactory_root=args.llamafactory_root,
        project_dir=args.project_dir,
        merged_root=args.merged_root,
        initial_model_path=args.initial_model_path,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        run_prefix=args.run_prefix,
        pt_cuda_visible_devices=args.pt_cuda_visible_devices,
        sft_cuda_visible_devices=args.sft_cuda_visible_devices,
        template=args.template,
        trust_remote_code=args.trust_remote_code,
        deepspeed_config=args.deepspeed_config,
        pt_cutoff_len=args.pt_cutoff_len,
        pt_per_device_train_batch_size=args.pt_per_device_train_batch_size,
        pt_gradient_accumulation_steps=args.pt_gradient_accumulation_steps,
        pt_learning_rate=args.pt_learning_rate,
        pt_num_train_epochs=args.pt_num_train_epochs,
        pt_lr_scheduler_type=args.pt_lr_scheduler_type,
        pt_warmup_ratio=args.pt_warmup_ratio,
        pt_bf16=args.pt_bf16,
        pt_logging_steps=args.pt_logging_steps,
        pt_save_strategy=args.pt_save_strategy,
        pt_save_only_model=args.pt_save_only_model,
        sft_cutoff_len=args.sft_cutoff_len,
        sft_per_device_train_batch_size=args.sft_per_device_train_batch_size,
        sft_gradient_accumulation_steps=args.sft_gradient_accumulation_steps,
        sft_learning_rate=args.sft_learning_rate,
        sft_num_train_epochs=args.sft_num_train_epochs,
        sft_lr_scheduler_type=args.sft_lr_scheduler_type,
        sft_warmup_ratio=args.sft_warmup_ratio,
        sft_lora_rank=args.sft_lora_rank,
        sft_lora_alpha=args.sft_lora_alpha,
        sft_bf16=args.sft_bf16,
        sft_logging_steps=args.sft_logging_steps,
        sft_save_strategy=args.sft_save_strategy,
        sft_save_total_limit=args.sft_save_total_limit,
        report_to=args.report_to,
        overwrite_cache=args.overwrite_cache,
        overwrite_output_dir=args.overwrite_output_dir,
        export_size=args.export_size,
        export_device=args.export_device,
        export_legacy_format=args.export_legacy_format,
        pt_name_suffix=args.pt_name_suffix,
        sft_name_suffix=args.sft_name_suffix,
        manifest_root=args.manifest_root,
        skip_existing=args.skip_existing,
    )
    print(
        json.dumps(
            {
                "start_stage": result["start_stage"],
                "end_stage": result["end_stage"],
                "num_steps": len(result["steps"]),
                "manifest_path": result["manifest_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
