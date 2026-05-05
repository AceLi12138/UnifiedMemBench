from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import DATE_RE, load_stories, read_json


FACT_ID_RE = re.compile(r"^\d{6}$")
PROMPT_FILE_RE = re.compile(
    r"^schema_longest_ladder_L(?P<level>\d+)_tok(?P<tokens>\d+)_"
    r"c(?P<characters>\d+)_s(?P<schemas>\d+)_f(?P<facts>\d+)_n(?P<samples>\d+)_seed(?P<seed>\d+)\.json$"
)


def validate_stories(path: str | Path) -> Dict[str, Any]:
    stories = load_stories(path)
    if not stories:
        raise ValueError("stories_v4.json contains no characters.")

    event_count = 0
    for cidx, character in enumerate(stories):
        if not isinstance(character.get("character_name"), str) or not character["character_name"].strip():
            raise ValueError(f"Story character[{cidx}] has no character_name.")
        chronology = character.get("chronology")
        if not isinstance(chronology, list):
            raise ValueError(f"{character['character_name']} chronology must be a list.")
        for yidx, year_block in enumerate(chronology):
            events = year_block.get("events") if isinstance(year_block, dict) else None
            if not isinstance(events, list):
                raise ValueError(f"{character['character_name']} chronology[{yidx}] has no events list.")
            for eidx, event in enumerate(events):
                if not isinstance(event, dict):
                    raise ValueError(f"{character['character_name']} event[{eidx}] must be an object.")
                if not isinstance(event.get("description"), str) or not event["description"].strip():
                    raise ValueError(f"{character['character_name']} event[{eidx}] has no description.")
                ts = event.get("timestamp")
                if not isinstance(ts, str) or not DATE_RE.match(ts[:10]):
                    raise ValueError(f"{character['character_name']} event[{eidx}] has invalid timestamp: {ts!r}")
                event_count += 1

    return {"characters": len(stories), "events": event_count}


def validate_facts(path: str | Path) -> Dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("facts.json must be a dict.")

    ids: List[str] = []
    total = 0
    for character, records in data.items():
        if not isinstance(character, str) or not isinstance(records, list) or not records:
            raise ValueError(f"Invalid facts entry for character {character!r}.")
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"{character} fact[{idx}] must be an object.")
            expected_keys = ["fact", "fact_id", "timestamp", "state_repr", "schema"]
            if list(record.keys()) != expected_keys:
                raise ValueError(f"{character} fact[{idx}] keys must be {expected_keys}; got {list(record.keys())}.")
            for field in expected_keys:
                if not isinstance(record[field], str):
                    raise ValueError(f"{character} fact[{idx}] {field} must be a string.")
            if not record["fact"].strip() or not record["schema"].strip():
                raise ValueError(f"{character} fact[{idx}] has empty fact/schema.")
            if not FACT_ID_RE.match(record["fact_id"]):
                raise ValueError(f"{character} fact[{idx}] has invalid fact_id: {record['fact_id']!r}")
            if not DATE_RE.match(record["timestamp"]):
                raise ValueError(f"{character} fact[{idx}] has invalid timestamp: {record['timestamp']!r}")
            ids.append(record["fact_id"])
            total += 1

    if len(ids) != len(set(ids)):
        raise ValueError("fact_id values must be globally unique.")
    expected = [f"{idx:06d}" for idx in range(1, len(ids) + 1)]
    if sorted(ids) != expected:
        raise ValueError("fact_id values must form a consecutive 000001..N range.")

    return {"characters": len(data), "facts": total}


def _validate_one_prompt_file(path: Path) -> Dict[str, Any]:
    match = PROMPT_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid prompt filename: {path.name}")
    meta = {k: int(v) for k, v in match.groupdict().items()}

    samples = read_json(path)
    if not isinstance(samples, list) or len(samples) != meta["samples"]:
        raise ValueError(f"{path.name} must contain {meta['samples']} samples.")

    token_values: List[int] = []
    for idx, sample in enumerate(samples):
        if sample.get("sample_id") != f"s{idx + 1:06d}":
            raise ValueError(f"{path.name} sample[{idx}] has invalid sample_id.")
        data = sample.get("data")
        if not isinstance(data, list) or len(data) != 1:
            raise ValueError(f"{path.name} sample[{idx}] data must contain one task.")
        task = data[0]
        if task.get("Task") != "schema_longest":
            raise ValueError(f"{path.name} sample[{idx}] has invalid task.")
        prompt = task.get("Prompt")
        gt = task.get("GT")
        stats = task.get("Stats")
        prompt_markers = (
            "Please perform a strict atomic fact state-tracking task.",
            "Directly output the original `FACT_TEXT` of that event.",
            "Make sure the output is in valid JSON format.",
        )
        if not isinstance(prompt, str) or any(marker not in prompt for marker in prompt_markers):
            raise ValueError(f"{path.name} sample[{idx}] prompt does not match the schema_longest template.")
        if not isinstance(gt, dict) or not isinstance(stats, dict):
            raise ValueError(f"{path.name} sample[{idx}] has invalid GT or Stats.")
        if stats.get("task") != "schema_longest":
            raise ValueError(f"{path.name} sample[{idx}] stats task mismatch.")
        token_values.append(int(stats.get("est_tokens", 0)))

    avg_tokens = round(sum(token_values) / len(token_values))
    if avg_tokens != meta["tokens"]:
        raise ValueError(f"{path.name} token value {meta['tokens']} does not match average {avg_tokens}.")

    return {"file": path.name, "samples": len(samples), "avg_est_tokens": avg_tokens}


def validate_prompts(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    files = sorted(path.glob("schema_longest_ladder_L*.json"))
    if not files:
        raise ValueError("No schema_longest prompt files found.")

    rows = [_validate_one_prompt_file(file) for file in files]
    summary_path = path / "schema_longest_ladder_summary.csv"
    if not summary_path.exists():
        raise ValueError("Missing schema_longest_ladder_summary.csv.")

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        summary_rows = list(csv.DictReader(f))
    if len(summary_rows) != len(files):
        raise ValueError("Prompt summary row count does not match prompt files.")

    summary_files = {row["file"] for row in summary_rows}
    actual_files = {file.name for file in files}
    if summary_files != actual_files:
        raise ValueError("Prompt summary file list does not match output directory.")

    return {
        "files": len(files),
        "samples": sum(row["samples"] for row in rows),
        "min_avg_est_tokens": min(row["avg_est_tokens"] for row in rows),
        "max_avg_est_tokens": max(row["avg_est_tokens"] for row in rows),
    }


def validate_project(
    root: str | Path,
    *,
    stories: str = "stories_v4.json",
    facts: str = "facts.json",
    prompts: str = "fact_track_schema_longest",
    include_prompts: bool = True,
) -> Dict[str, Any]:
    root = Path(root)
    report = {
        "stories": validate_stories(root / stories),
        "facts": validate_facts(root / facts),
    }
    if include_prompts:
        report["prompts"] = validate_prompts(root / prompts)
    return report
