#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from fact_track.validate import validate_project


HF_DATASET_REPO_ID = "Ace1213812/UnifiedMemBench-StateTracking"
HF_PROMPT_SUBDIR = "state_tracking"
PROMPT_FILE_RE = re.compile(
    r"^schema_longest_ladder_L(?P<level>\d+)_tok(?P<tokens>\d+)_"
    r"c(?P<characters>\d+)_s(?P<schemas>\d+)_f(?P<facts>\d+)_"
    r"n(?P<samples>\d+)_seed(?P<seed>\d+)(?:_hf)?\.json$"
)
SUMMARY_COLUMNS = [
    "level",
    "avg_est_tokens",
    "num_characters",
    "max_schemas_per_character",
    "max_facts_per_schema",
    "num_samples",
    "seed",
    "file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Fact Track Memory release artifacts.")
    parser.add_argument("--stories", default="../data_gen/output/stories_v4.json")
    parser.add_argument("--facts", default="facts.json")
    parser.add_argument("--prompts", default="fact_track_schema_longest")
    parser.add_argument("--skip-prompts", action="store_true")
    parser.add_argument(
        "--no-auto-download-prompts",
        action="store_true",
        help="Do not download fact_track_schema_longest from Hugging Face when it is missing.",
    )
    return parser.parse_args()


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _read_hf_json(url: str) -> list[dict[str, object]]:
    request = Request(url, headers=_hf_headers())
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    request = Request(url, headers=_hf_headers())
    try:
        with urlopen(request, timeout=120) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _local_prompt_name(filename: str) -> str:
    return filename.removesuffix("_hf.json") + ".json" if filename.endswith("_hf.json") else filename


def _prompt_json_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(path.glob("schema_longest_ladder_L*.json"))


def _normalize_hf_prompt_names(path: Path) -> None:
    for hf_path in sorted(path.glob("schema_longest_ladder_L*_hf.json")):
        local_path = hf_path.with_name(_local_prompt_name(hf_path.name))
        if local_path.exists():
            hf_path.unlink()
        else:
            hf_path.rename(local_path)


def _normalize_prompt_file(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)

    changed = False
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            data = sample.get("data")
            if not isinstance(data, list):
                continue
            for task in data:
                if not isinstance(task, dict):
                    continue
                if "GT" not in task and "gt" in task:
                    task["GT"] = task.pop("gt")
                    changed = True
                if "Stats" not in task and "stats" in task:
                    task["Stats"] = task.pop("stats")
                    changed = True

                gt = task.get("GT")
                if isinstance(gt, str):
                    task["GT"] = json.loads(gt)
                    changed = True

    if not changed:
        return

    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(samples, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _normalize_prompt_files(path: Path) -> None:
    for prompt_file in _prompt_json_files(path):
        _normalize_prompt_file(prompt_file)


def _write_prompt_summary(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for prompt_file in _prompt_json_files(path):
        match = PROMPT_FILE_RE.match(prompt_file.name)
        if not match:
            continue
        meta = match.groupdict()
        rows.append(
            {
                "level": meta["level"],
                "avg_est_tokens": int(meta["tokens"]),
                "num_characters": int(meta["characters"]),
                "max_schemas_per_character": int(meta["schemas"]),
                "max_facts_per_schema": int(meta["facts"]),
                "num_samples": int(meta["samples"]),
                "seed": int(meta["seed"]),
                "file": prompt_file.name,
            }
        )

    if not rows:
        return

    summary_path = path / "schema_longest_ladder_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _download_prompt_ladder(output_dir: Path) -> None:
    tree_url = (
        "https://huggingface.co/api/datasets/"
        f"{HF_DATASET_REPO_ID}/tree/main/{quote(HF_PROMPT_SUBDIR, safe='/')}?recursive=1"
    )
    try:
        tree = _read_hf_json(tree_url)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to read Hugging Face dataset tree: {exc}") from exc

    prompt_paths = sorted(
        str(item["path"])
        for item in tree
        if item.get("type") == "file" and PROMPT_FILE_RE.match(Path(str(item["path"])).name)
    )
    if not prompt_paths:
        raise RuntimeError(f"No prompt files found in Hugging Face repo {HF_DATASET_REPO_ID!r}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {len(prompt_paths)} prompt files from "
        f"https://huggingface.co/datasets/{HF_DATASET_REPO_ID} ...",
        file=sys.stderr,
    )
    for index, remote_path in enumerate(prompt_paths, start=1):
        filename = _local_prompt_name(Path(remote_path).name)
        output_path = output_dir / filename
        if output_path.exists():
            continue
        file_url = (
            "https://huggingface.co/datasets/"
            f"{HF_DATASET_REPO_ID}/resolve/main/{quote(remote_path, safe='/')}"
        )
        print(f"[{index:02d}/{len(prompt_paths):02d}] {filename}", file=sys.stderr)
        try:
            _download_file(file_url, output_path)
            _normalize_prompt_file(output_path)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Failed to download {remote_path}: {exc}") from exc

    _normalize_prompt_files(output_dir)
    _write_prompt_summary(output_dir)


def ensure_prompt_ladder(root: Path, prompts: str) -> None:
    prompt_path = root / prompts
    if prompt_path.exists() and not prompt_path.is_dir():
        raise ValueError(f"Prompt path exists but is not a directory: {prompt_path}")

    _normalize_hf_prompt_names(prompt_path)
    if _prompt_json_files(prompt_path):
        _normalize_prompt_files(prompt_path)
        if not (prompt_path / "schema_longest_ladder_summary.csv").exists():
            _write_prompt_summary(prompt_path)
        return

    if prompt_path.exists():
        _download_prompt_ladder(prompt_path)
        return

    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fact_track_schema_longest_", dir=prompt_path.parent) as tmpdir:
        tmp_path = Path(tmpdir) / "prompts"
        _download_prompt_ladder(tmp_path)
        tmp_path.replace(prompt_path)


def main() -> None:
    args = parse_args()
    if not args.skip_prompts and not args.no_auto_download_prompts:
        ensure_prompt_ladder(Path.cwd(), args.prompts)
    report = validate_project(
        Path.cwd(),
        stories=args.stories,
        facts=args.facts,
        prompts=args.prompts,
        include_prompts=not args.skip_prompts,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
