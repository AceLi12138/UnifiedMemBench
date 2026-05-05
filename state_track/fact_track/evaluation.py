from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


TASK_NAME = "schema_longest"
FACT_LINE_RE = re.compile(r"^schema=【(?P<schema>.*?)】 \| day_ts=(?P<day_ts>-?\d+) \| fact=(?P<fact>.*)$")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def parse_model_json(text: str) -> Tuple[Optional[Any], str]:
    if not isinstance(text, str) or not text.strip():
        return None, "empty_output"

    match = _FENCE_RE.search(text)
    source = (match.group(1) if match else text).strip()
    start = source.find("{")
    if start < 0:
        return None, "no_json_object"

    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(source)):
        char = source[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = source[start:pos + 1]
                try:
                    return json.loads(candidate), "ok"
                except json.JSONDecodeError:
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(fixed), "ok_trailing_comma_fix"
                    except json.JSONDecodeError as exc:
                        return None, f"json_error:{exc}"

    return None, "unbalanced_braces"


def extract_level(file_name: str) -> Optional[int]:
    match = re.search(r"_L(\d+)_", file_name)
    return int(match.group(1)) if match else None


def level_sort_key(path: Path) -> Tuple[int, str]:
    level = extract_level(path.name)
    return (level if level is not None else 999999, path.name)


def collect_prompt_files(
    input_path: str | Path,
    *,
    level_start: Optional[int],
    level_end: Optional[int],
    max_files: Optional[int],
) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("schema_longest_ladder_L*.json"), key=level_sort_key)
    else:
        raise FileNotFoundError(f"Prompt input not found: {path}")

    filtered = []
    for file in files:
        level = extract_level(file.name)
        if level_start is not None and level is not None and level < level_start:
            continue
        if level_end is not None and level is not None and level > level_end:
            continue
        filtered.append(file)

    if max_files is not None and max_files > 0:
        filtered = filtered[:max_files]
    if not filtered:
        raise ValueError("No prompt JSON files selected.")
    return filtered


def find_task_entry(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = sample.get("data")
    if not isinstance(data, list):
        return None
    for entry in data:
        if isinstance(entry, dict) and entry.get("Task") == TASK_NAME:
            return entry
    return None


def _unique_values(values: List[Any]) -> List[Any]:
    unique: List[Any] = []
    for value in values:
        if not any(value == item for item in unique):
            unique.append(value)
    return unique


def _prompt_fact_lines(prompt: str) -> List[Dict[str, Any]]:
    _, marker, tail = prompt.partition("Now process the following facts:")
    if not marker:
        return []
    facts_text, _, _ = tail.partition("Make sure the output is in valid JSON format.")

    lines: List[Dict[str, Any]] = []
    for input_order, raw_line in enumerate(facts_text.strip().splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        match = FACT_LINE_RE.match(line)
        if not match:
            continue
        lines.append(
            {
                "schema": match.group("schema"),
                "day_ts": int(match.group("day_ts")),
                "fact": match.group("fact"),
                "input_order": input_order,
            }
        )
    return lines


def _infer_fact_character(fact: str, characters: List[str]) -> Optional[str]:
    matches = [character for character in characters if character.lower() in fact.lower()]
    return matches[0] if len(matches) == 1 else None


def single_event_answer_aliases(
    prompt: str,
    gt_root: Dict[str, Any],
) -> Dict[Tuple[str, str], List[Any]]:
    characters = [character for character, value in gt_root.items() if isinstance(character, str) and isinstance(value, dict)]
    if not characters:
        return {}

    events_by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in _prompt_fact_lines(prompt):
        character = _infer_fact_character(str(item["fact"]), characters)
        if character is None and len(characters) == 1:
            character = characters[0]
        if character is None:
            continue
        events_by_group.setdefault((character, str(item["schema"])), []).append(item)

    aliases: Dict[Tuple[str, str], List[Any]] = {}
    for (character, schema), events in events_by_group.items():
        if len(events) != 1:
            continue
        gt_character = gt_root.get(character)
        gt_schema_map = gt_character.get(TASK_NAME) if isinstance(gt_character, dict) else None
        if not isinstance(gt_schema_map, dict) or schema not in gt_schema_map:
            continue
        aliases[(character, schema)] = _unique_values([gt_schema_map.get(schema), None, events[0]["fact"]])
    return aliases


def load_jobs(file_path: str | Path, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError(f"{path} must contain a JSON list.")
    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]

    jobs: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        entry = find_task_entry(sample)
        if not entry:
            continue
        prompt = entry.get("Prompt")
        gt = entry.get("GT")
        if not isinstance(prompt, str) or not isinstance(gt, dict):
            continue
        jobs.append(
            {
                "idx": len(jobs),
                "sample_id": sample.get("sample_id", f"s{idx + 1:06d}"),
                "prompt": prompt,
                "gt": gt,
                "answer_aliases": single_event_answer_aliases(prompt, gt),
                "meta": sample.get("meta", {}),
            }
        )
    return jobs


@dataclass
class FileResult:
    file_name: str
    level: str
    total_prompts: int = 0
    parsed_ok: int = 0
    parse_fail: int = 0
    call_error: int = 0
    prompt_exact_match: int = 0
    total_characters: int = 0
    char_exact_match: int = 0
    schema_total: int = 0
    schema_correct: int = 0
    coverage_total: int = 0
    coverage_present: int = 0
    avg_latency_s: float = 0.0

    @property
    def parse_rate(self) -> float:
        return self.parsed_ok / self.total_prompts if self.total_prompts else 0.0

    @property
    def prompt_exact_rate(self) -> float:
        return self.prompt_exact_match / self.total_prompts if self.total_prompts else 0.0

    @property
    def char_exact_rate(self) -> float:
        return self.char_exact_match / self.total_characters if self.total_characters else 0.0

    @property
    def schema_accuracy(self) -> float:
        return self.schema_correct / self.schema_total if self.schema_total else 0.0

    @property
    def schema_coverage(self) -> float:
        return self.coverage_present / self.coverage_total if self.coverage_total else 0.0


def compare_prediction(
    gt_root: Dict[str, Any],
    pred_root: Any,
    answer_aliases: Optional[Dict[Tuple[str, str], List[Any]]] = None,
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, int]]:
    answer_aliases = answer_aliases or {}
    prompt_exact = True
    details: List[Dict[str, Any]] = []
    counts = {
        "characters": 0,
        "char_exact": 0,
        "schema_total": 0,
        "schema_correct": 0,
        "coverage_total": 0,
        "coverage_present": 0,
    }

    for character, gt_character in gt_root.items():
        if not isinstance(gt_character, dict):
            continue
        gt_schema_map = gt_character.get(TASK_NAME)
        if not isinstance(gt_schema_map, dict):
            continue

        counts["characters"] += 1
        char_exact = True
        pred_character = pred_root.get(character) if isinstance(pred_root, dict) else None
        pred_schema_map = pred_character.get(TASK_NAME) if isinstance(pred_character, dict) else None

        for schema, gt_value in gt_schema_map.items():
            counts["schema_total"] += 1
            counts["coverage_total"] += 1
            present = isinstance(pred_schema_map, dict) and schema in pred_schema_map
            pred_value = pred_schema_map.get(schema) if isinstance(pred_schema_map, dict) else None
            accepted_values = answer_aliases.get((character, schema), [gt_value])
            correct = present and any(pred_value == value for value in accepted_values)

            if present:
                counts["coverage_present"] += 1
            if correct:
                counts["schema_correct"] += 1
            else:
                char_exact = False
                prompt_exact = False

            details.append(
                {
                    "character": character,
                    "schema": schema,
                    "present": present,
                    "correct": correct,
                    "gt": gt_value,
                    "accepted": accepted_values,
                    "pred": pred_value,
                }
            )

        if char_exact:
            counts["char_exact"] += 1

    return prompt_exact, details, counts


ModelCall = Callable[[str], str]


def evaluate_file(
    file_path: str | Path,
    *,
    model_call: ModelCall,
    max_samples: Optional[int],
    concurrency: int,
    sleep_s: float,
    include_prompt: bool = False,
) -> Tuple[FileResult, List[Dict[str, Any]]]:
    path = Path(file_path)
    jobs = load_jobs(path, max_samples)
    level_value = extract_level(path.name)
    result = FileResult(file_name=path.name, level=f"L{level_value:02d}" if level_value is not None else path.name)

    def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        raw = ""
        err = None
        try:
            raw = model_call(job["prompt"])
        except Exception as exc:
            err = str(exc)
        finally:
            if sleep_s > 0:
                time.sleep(sleep_s)
        return {
            "idx": job["idx"],
            "raw": raw,
            "error": err,
            "latency_s": time.time() - started,
        }

    outputs: Dict[int, Dict[str, Any]] = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                item = future.result()
                outputs[item["idx"]] = item

    records: List[Dict[str, Any]] = []
    total_latency = 0.0
    for job in jobs:
        output = outputs.get(job["idx"], {"raw": "", "error": "missing_result", "latency_s": 0.0})
        latency_s = float(output.get("latency_s", 0.0))
        total_latency += latency_s

        result.total_prompts += 1
        record: Dict[str, Any] = {
            "file": path.name,
            "level": result.level,
            "sample_id": job["sample_id"],
            "latency_s": round(latency_s, 3),
            "error": output.get("error"),
            "raw_output": output.get("raw", ""),
            "parse_ok": False,
            "parse_reason": None,
            "parsed_prediction": None,
            "prompt_exact": False,
            "details": [],
        }
        if include_prompt:
            record["prompt"] = job["prompt"]

        if output.get("error"):
            result.call_error += 1
            result.parse_fail += 1
            records.append(record)
            continue

        parsed, parse_reason = parse_model_json(str(output.get("raw", "")))
        record["parse_reason"] = parse_reason
        record["parsed_prediction"] = parsed
        if parsed is None:
            result.parse_fail += 1
            records.append(record)
            continue

        result.parsed_ok += 1
        record["parse_ok"] = True

        prompt_exact, details, counts = compare_prediction(job["gt"], parsed, job.get("answer_aliases"))
        result.total_characters += counts["characters"]
        result.char_exact_match += counts["char_exact"]
        result.schema_total += counts["schema_total"]
        result.schema_correct += counts["schema_correct"]
        result.coverage_total += counts["coverage_total"]
        result.coverage_present += counts["coverage_present"]
        if prompt_exact:
            result.prompt_exact_match += 1
        record["prompt_exact"] = prompt_exact
        record["details"] = details
        records.append(record)

    result.avg_latency_s = round(total_latency / max(1, result.total_prompts), 3)
    return result, records


CSV_COLUMNS = [
    "runner",
    "provider",
    "model",
    "level",
    "file_name",
    "total_prompts",
    "parsed_ok",
    "parse_fail",
    "call_error",
    "prompt_exact_match",
    "prompt_exact_rate",
    "total_characters",
    "char_exact_match",
    "char_exact_rate",
    "schema_total",
    "schema_correct",
    "schema_accuracy",
    "coverage_total",
    "coverage_present",
    "schema_coverage",
    "avg_latency_s",
]


def file_result_row(result: FileResult, *, runner: str, provider: str, model: str) -> Dict[str, Any]:
    return {
        "runner": runner,
        "provider": provider,
        "model": model,
        "level": result.level,
        "file_name": result.file_name,
        "total_prompts": result.total_prompts,
        "parsed_ok": result.parsed_ok,
        "parse_fail": result.parse_fail,
        "call_error": result.call_error,
        "prompt_exact_match": result.prompt_exact_match,
        "prompt_exact_rate": f"{result.prompt_exact_rate:.4f}",
        "total_characters": result.total_characters,
        "char_exact_match": result.char_exact_match,
        "char_exact_rate": f"{result.char_exact_rate:.4f}",
        "schema_total": result.schema_total,
        "schema_correct": result.schema_correct,
        "schema_accuracy": f"{result.schema_accuracy:.4f}",
        "coverage_total": result.coverage_total,
        "coverage_present": result.coverage_present,
        "schema_coverage": f"{result.schema_coverage:.4f}",
        "avg_latency_s": result.avg_latency_s,
    }


class IncrementalWriter:
    def __init__(
        self,
        *,
        csv_path: str | Path,
        jsonl_path: str | Path,
        runner: str,
        provider: str,
        model: str,
        resume: bool,
        overwrite: bool,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        self.runner = runner
        self.provider = provider
        self.model = model
        self.completed_files = set()

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for path in (self.csv_path, self.jsonl_path):
                if path.exists():
                    path.unlink()

        if resume and self.csv_path.exists():
            self.completed_files = self._load_completed_files()

        if not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()
        if not self.jsonl_path.exists():
            self.jsonl_path.write_text("", encoding="utf-8")

    def _load_completed_files(self) -> set[str]:
        done: set[str] = set()
        try:
            with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    file_name = row.get("file_name")
                    if file_name:
                        done.add(file_name)
        except Exception:
            return set()
        return done

    def is_done(self, file_name: str) -> bool:
        return file_name in self.completed_files

    def save(self, result: FileResult, records: List[Dict[str, Any]]) -> None:
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writerow(file_result_row(result, runner=self.runner, provider=self.provider, model=self.model))

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.completed_files.add(result.file_name)


def write_run_config(path: str | Path, data: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def print_summary(results: List[FileResult], *, provider: str, model: str) -> None:
    if not results:
        return
    print("\n" + "=" * 116)
    print(f"SUMMARY | provider={provider} | model={model} | files={len(results)}")
    print("-" * 116)
    print(
        f"{'Level':<8} {'Prompts':>7} {'Parse%':>8} {'PromptEM%':>10} "
        f"{'CharEM%':>9} {'SchemaAcc':>10} {'Coverage':>10} {'Latency':>9}"
    )
    print("-" * 116)

    totals = {
        "prompts": 0,
        "prompt_exact": 0,
        "characters": 0,
        "char_exact": 0,
        "schemas": 0,
        "schema_correct": 0,
        "coverage_total": 0,
        "coverage_present": 0,
        "parsed": 0,
    }
    for item in results:
        print(
            f"{item.level:<8} {item.total_prompts:>7} {item.parse_rate:>8.4f} "
            f"{item.prompt_exact_rate:>10.4f} {item.char_exact_rate:>9.4f} "
            f"{item.schema_accuracy:>10.4f} {item.schema_coverage:>10.4f} "
            f"{item.avg_latency_s:>8.2f}s"
        )
        totals["prompts"] += item.total_prompts
        totals["prompt_exact"] += item.prompt_exact_match
        totals["characters"] += item.total_characters
        totals["char_exact"] += item.char_exact_match
        totals["schemas"] += item.schema_total
        totals["schema_correct"] += item.schema_correct
        totals["coverage_total"] += item.coverage_total
        totals["coverage_present"] += item.coverage_present
        totals["parsed"] += item.parsed_ok

    prompts = max(1, totals["prompts"])
    characters = max(1, totals["characters"])
    schemas = max(1, totals["schemas"])
    coverage_total = max(1, totals["coverage_total"])
    print("-" * 116)
    print(
        f"{'TOTAL':<8} {totals['prompts']:>7} {totals['parsed'] / prompts:>8.4f} "
        f"{totals['prompt_exact'] / prompts:>10.4f} {totals['char_exact'] / characters:>9.4f} "
        f"{totals['schema_correct'] / schemas:>10.4f} "
        f"{totals['coverage_present'] / coverage_total:>10.4f}"
    )
    print("=" * 116 + "\n")
