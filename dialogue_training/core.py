from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PROMPT_CATALOG = (
    Path(__file__).resolve().parent.parent
    / "dialogue_gen_api"
    / "evaluation"
    / "task_prompts_v2.json"
)

DEFAULT_QA_TASK_WEIGHTS: Dict[str, float] = {
    "Information Extraction": 1.0,
    "Knowledge Updating": 1.25,
    "Memory Arbitration": 1.25,
    "Temporal Reasoning": 2.0,
    "Multi-session Reasoning": 1.75,
    "Event Summarization": 1.75,
}

DEFAULT_TASK_BALANCE_ORDER: List[str] = list(DEFAULT_QA_TASK_WEIGHTS.keys())

STAGEWISE_RESULTS_DIR_RE = re.compile(r"^stage_(\d{2})_(seen|unseen)(?:_(.+))?$")


def load_benchmark(dataset_path: Path) -> List[Dict[str, Any]]:
    with Path(dataset_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected benchmark JSON list, got {type(data).__name__}")
    return data


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def estimate_tokens(text: str) -> int:
    raw = str(text or "").strip()
    if not raw:
        return 0
    # A conservative heuristic that works reasonably for English dialogue.
    return max(1, math.ceil(len(raw) / 4.0))


def format_turn(turn: Dict[str, Any]) -> str:
    role = turn.get("role") or turn.get("speaker") or "speaker"
    content = str(turn.get("content") or turn.get("message") or "").strip()
    return f"{role}: {content}"


def format_dialogue(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(format_turn(turn) for turn in turns if str(turn.get("content") or turn.get("message") or "").strip())


def _normalize_dialogue_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [turn for turn in turns if str(turn.get("content") or turn.get("message") or "").strip()]


def chunk_dialogue_turns(
    turns: List[Dict[str, Any]],
    max_chunk_tokens: int,
    overlap_turns: int = 1,
) -> List[Dict[str, Any]]:
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")
    if overlap_turns < 0:
        raise ValueError("overlap_turns must be non-negative")

    normalized_turns = _normalize_dialogue_turns(turns)
    if not normalized_turns:
        return []

    chunks: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(normalized_turns):
        start_idx = idx
        current_turns: List[Dict[str, Any]] = []
        current_tokens = 0

        while idx < len(normalized_turns):
            turn = normalized_turns[idx]
            turn_text = format_turn(turn)
            turn_tokens = estimate_tokens(turn_text)

            if current_turns and current_tokens + turn_tokens > max_chunk_tokens:
                break

            current_turns.append(turn)
            current_tokens += turn_tokens
            idx += 1

            if not current_turns and turn_tokens > max_chunk_tokens:
                break

        if not current_turns:
            current_turns = [normalized_turns[idx]]
            current_tokens = estimate_tokens(format_turn(normalized_turns[idx]))
            idx += 1

        chunks.append(
            {
                "start_turn": start_idx,
                "end_turn": start_idx + len(current_turns) - 1,
                "turns": current_turns,
                "approx_tokens": current_tokens,
            }
        )

        if idx >= len(normalized_turns):
            break

        if overlap_turns > 0:
            idx = max(start_idx + 1, idx - overlap_turns)

    return chunks


def chunk_dialogue_turns_sliding_window(
    turns: List[Dict[str, Any]],
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> List[Dict[str, Any]]:
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= max_chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than max_chunk_tokens")

    normalized_turns = _normalize_dialogue_turns(turns)
    if not normalized_turns:
        return []

    turn_token_counts = [estimate_tokens(format_turn(turn)) for turn in normalized_turns]
    turn_end_tokens: List[int] = []
    total_tokens = 0
    for count in turn_token_counts:
        total_tokens += count
        turn_end_tokens.append(total_tokens)
    turn_start_tokens = [0] + turn_end_tokens[:-1]

    stride_tokens = max_chunk_tokens - overlap_tokens
    chunks: List[Dict[str, Any]] = []
    seen_spans = set()
    window_start_token = 0

    while window_start_token < total_tokens:
        start_idx = bisect_right(turn_end_tokens, window_start_token)
        if start_idx >= len(normalized_turns):
            break

        window_end_token = window_start_token + max_chunk_tokens
        end_idx = start_idx
        while end_idx + 1 < len(normalized_turns) and turn_start_tokens[end_idx + 1] < window_end_token:
            end_idx += 1

        span = (start_idx, end_idx)
        if span not in seen_spans:
            current_turns = normalized_turns[start_idx:end_idx + 1]
            current_tokens = sum(turn_token_counts[start_idx:end_idx + 1])
            chunks.append(
                {
                    "start_turn": start_idx,
                    "end_turn": end_idx,
                    "turns": current_turns,
                    "approx_tokens": current_tokens,
                }
            )
            seen_spans.add(span)

        if window_end_token >= total_tokens and end_idx == len(normalized_turns) - 1:
            break

        window_start_token += stride_tokens

    return chunks


def build_dialogue_chunks(
    turns: List[Dict[str, Any]],
    max_chunk_tokens: int,
    overlap_turns: int = 1,
    chunking_mode: str = "turn_overlap",
    sliding_window_overlap_tokens: int = 0,
) -> List[Dict[str, Any]]:
    if chunking_mode == "turn_overlap":
        return chunk_dialogue_turns(
            turns,
            max_chunk_tokens=max_chunk_tokens,
            overlap_turns=overlap_turns,
        )
    if chunking_mode == "sliding_window":
        return chunk_dialogue_turns_sliding_window(
            turns,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=sliding_window_overlap_tokens,
        )
    raise ValueError(f"Unsupported chunking_mode: {chunking_mode}")


def assign_dialogues_to_stages(
    dialogues: List[Dict[str, Any]],
    num_stages: int,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Dict[str, Any]]:
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    ordered = list(dialogues)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    base = len(ordered) // num_stages
    remainder = len(ordered) % num_stages

    assignments: List[Dict[str, Any]] = []
    cursor = 0
    for stage_id in range(1, num_stages + 1):
        stage_size = base + (1 if stage_id <= remainder else 0)
        stage_dialogues = ordered[cursor: cursor + stage_size]
        cursor += stage_size
        for dialogue in stage_dialogues:
            assignments.append(
                {
                    "dialogue_id": dialogue["id"],
                    "character": dialogue.get("character", ""),
                    "stage_id": stage_id,
                    "qa_count": len(dialogue.get("tasks_covered", [])),
                    "turn_count": len(dialogue.get("dialogue", [])),
                }
            )
    return assignments


def split_stage_dialogues_by_entity(
    dialogues: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    ordered = sorted(
        list(dialogues),
        key=lambda item: (
            str(item.get("character", "")),
            str(item.get("id", "")),
        ),
    )
    rng = random.Random(seed)
    rng.shuffle(ordered)

    if len(ordered) <= 1:
        train_count = len(ordered)
    else:
        train_count = int(round(len(ordered) * train_ratio))
        train_count = max(1, min(len(ordered) - 1, train_count))

    train_dialogues = ordered[:train_count]
    test_dialogues = ordered[train_count:]
    return {
        "train_dialogues": train_dialogues,
        "test_dialogues": test_dialogues,
        "train": [
            {"dialogue_id": row["id"], "character": row.get("character", "")}
            for row in train_dialogues
        ],
        "test": [
            {"dialogue_id": row["id"], "character": row.get("character", "")}
            for row in test_dialogues
        ],
    }


def _load_prompt_catalog(catalog_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(catalog_path) if catalog_path else DEFAULT_PROMPT_CATALOG
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _json_instruction_from_catalog(task_type: str, prompt_catalog: Dict[str, Any]) -> str:
    task_item = (prompt_catalog.get("tasks") or {}).get(task_type, {})
    goal = str(task_item.get("goal", "Answer the question based on prior memory.")).strip()
    instructions = task_item.get("instructions") or []
    required_fields = task_item.get("required_fields") or []
    example_output = task_item.get("example_output") or {}

    lines = [
        "Answer from memory only. Do not assume you can re-read the original dialogue.",
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


_ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DAY_COUNT_PATTERN = re.compile(r"\b(\d+)\s+days?\b", flags=re.IGNORECASE)
_AS_OF_PATTERN = re.compile(r"^As of\s+(.+?),(?:\s|$)", flags=re.IGNORECASE)
_FROM_TO_PATTERN = re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[?.]|$)", flags=re.IGNORECASE)
_SINGLE_YEAR_PATTERN = re.compile(r"\((\d{4})\)")


def _extract_iso_dates(answer_components: List[Any]) -> List[str]:
    dates: List[str] = []
    for component in answer_components:
        for match in _ISO_DATE_PATTERN.findall(str(component or "")):
            if match not in dates:
                dates.append(match)
    return dates


def _extract_day_count(answer_components: List[Any]) -> Optional[int]:
    for component in answer_components:
        match = _DAY_COUNT_PATTERN.search(str(component or ""))
        if match:
            return int(match.group(1))
    return None


def _extract_as_of_time(query: str) -> str:
    match = _AS_OF_PATTERN.search(str(query or "").strip())
    if match:
        return match.group(1).strip()
    return ""


def _extract_time_span(query: str) -> str:
    text = str(query or "").strip()
    from_to = _FROM_TO_PATTERN.search(text)
    if from_to:
        return f"{from_to.group(1).strip()} to {from_to.group(2).strip()}"

    single_year = _SINGLE_YEAR_PATTERN.search(text)
    if single_year:
        return single_year.group(1)

    year_matches = re.findall(r"\b(?:19|20)\d{2}\b", text)
    if len(year_matches) == 1:
        return year_matches[0]
    if len(year_matches) >= 2:
        return f"{year_matches[0]} to {year_matches[-1]}"
    return ""


def _infer_premise_verdict(gold_answer: str) -> str:
    text = str(gold_answer or "").strip().lower()
    if "premise is incorrect" in text:
        return "incorrect"
    if "premise is correct" in text:
        return "correct"
    return ""


def _build_qa_sft_output(task: Dict[str, Any]) -> Dict[str, Any]:
    task_type = str(task.get("task_type") or "Unknown")
    gold_answer = str(task.get("gold_answer") or "")
    answer_components = list(task.get("answer_components") or [])

    if task_type == "Information Extraction":
        return {
            "answer": gold_answer,
            "evidence_snippets": [],
        }

    if task_type == "Temporal Reasoning":
        dates = _extract_iso_dates(answer_components)
        day_count = _extract_day_count(answer_components)
        start_date = dates[0] if len(dates) >= 1 else ""
        end_date = dates[1] if len(dates) >= 2 else ""
        return {
            "final_answer": gold_answer,
            "days": day_count if day_count is not None else "",
            "start_date": start_date,
            "end_date": end_date,
        }

    if task_type == "Knowledge Updating":
        return {
            "latest_state": gold_answer,
            "as_of_time": "",
            "deprecated_state": str(task.get("old_state_value") or ""),
        }

    if task_type == "Multi-session Reasoning":
        return {
            "event_chain": answer_components,
            "final_outcome": gold_answer,
        }

    if task_type == "Event Summarization":
        return {
            "time_span": _extract_time_span(str(task.get("query") or "")),
            "key_turning_points": answer_components,
            "summary": gold_answer,
        }

    if task_type == "Memory Arbitration":
        return {
            "premise_verdict": _infer_premise_verdict(gold_answer),
            "premise_error": "",
            "corrected_facts": answer_components,
            "final_answer": gold_answer,
        }

    return {
        "answer": gold_answer,
    }


def build_training_records(
    dialogue: Dict[str, Any],
    stage_id: int,
    max_chunk_tokens: int = 4096,
    overlap_turns: int = 1,
    chunking_mode: str = "turn_overlap",
    sliding_window_overlap_tokens: int = 0,
) -> List[Dict[str, Any]]:
    chunks = build_dialogue_chunks(
        dialogue.get("dialogue", []),
        max_chunk_tokens=max_chunk_tokens,
        overlap_turns=overlap_turns,
        chunking_mode=chunking_mode,
        sliding_window_overlap_tokens=sliding_window_overlap_tokens,
    )
    total_chunks = len(chunks)
    records: List[Dict[str, Any]] = []

    for chunk_idx, chunk in enumerate(chunks, start=1):
        records.append(
            {
                "instruction": (
                    f"Memorize this dialogue segment from an ongoing long conversation with "
                    f"{dialogue.get('character', 'the user')}."
                ),
                "input": f"Segment {chunk_idx}/{total_chunks}",
                "output": format_dialogue(chunk["turns"]),
                "metadata": {
                    "dialogue_id": dialogue["id"],
                    "dialogue_stage": stage_id,
                    "character": dialogue.get("character", ""),
                    "chunk_id": chunk_idx,
                    "num_chunks": total_chunks,
                    "chunk_start_turn": chunk["start_turn"],
                    "chunk_end_turn": chunk["end_turn"],
                    "approx_tokens": chunk["approx_tokens"],
                    "qa_count": len(dialogue.get("tasks_covered", [])),
                },
            }
        )

    return records


def build_pretrain_records(
    dialogue: Dict[str, Any],
    stage_id: int,
    max_chunk_tokens: int = 4096,
    overlap_turns: int = 1,
    chunking_mode: str = "turn_overlap",
    sliding_window_overlap_tokens: int = 0,
    pt_header_style: str = "structured",
    pt_record_style: str = "alpaca",
) -> List[Dict[str, Any]]:
    chunks = build_dialogue_chunks(
        dialogue.get("dialogue", []),
        max_chunk_tokens=max_chunk_tokens,
        overlap_turns=overlap_turns,
        chunking_mode=chunking_mode,
        sliding_window_overlap_tokens=sliding_window_overlap_tokens,
    )
    total_chunks = len(chunks)
    records: List[Dict[str, Any]] = []
    character = dialogue.get("character", "the user")

    if pt_header_style == "structured":
        def build_instruction(chunk_idx: int) -> str:
            return "\n".join(
                [
                    f"[Character] {character}",
                    f"[Dialogue Stage] {stage_id}",
                    f"[Segment] {chunk_idx}/{total_chunks}",
                    f"[Chronology] This segment is part {chunk_idx} of {total_chunks} in chronological order.",
                ]
            )
    elif pt_header_style == "natural":
        def build_instruction(chunk_idx: int) -> str:  # noqa: ARG001
            return "\n".join(
                [
                    f"The following dialogue details concern {character}.",
                    f"These conversation notes involve {character} and preserve facts from earlier interactions.",
                ]
            )
    else:
        raise ValueError(f"Unsupported pt_header_style: {pt_header_style}")

    for chunk_idx, chunk in enumerate(chunks, start=1):
        metadata = {
            "dialogue_id": dialogue["id"],
            "dialogue_stage": stage_id,
            "character": dialogue.get("character", ""),
            "chunk_id": chunk_idx,
            "num_chunks": total_chunks,
            "chunk_start_turn": chunk["start_turn"],
            "chunk_end_turn": chunk["end_turn"],
            "approx_tokens": chunk["approx_tokens"],
            "qa_count": len(dialogue.get("tasks_covered", [])),
            "training_mode": "pt",
            "pt_header_style": pt_header_style,
            "pt_record_style": pt_record_style,
        }
        instruction = build_instruction(chunk_idx)
        dialogue_text = format_dialogue(chunk["turns"])

        if pt_record_style == "alpaca":
            records.append(
                {
                    "instruction": instruction,
                    "input": dialogue_text,
                    "output": "",
                    "metadata": metadata,
                }
            )
        elif pt_record_style == "text":
            records.append(
                {
                    "text": f"{instruction}\n{dialogue_text}",
                    "metadata": metadata,
                }
            )
        else:
            raise ValueError(f"Unsupported pt_record_style: {pt_record_style}")

    return records


def build_qa_sft_records(
    dialogues: List[Dict[str, Any]],
    stage_map: Dict[str, int],
    prompt_catalog_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    prompt_catalog = _load_prompt_catalog(prompt_catalog_path)
    records: List[Dict[str, Any]] = []

    for dialogue in dialogues:
        dialogue_stage = stage_map[dialogue["id"]]
        character = dialogue.get("character", "")
        for task_idx, task in enumerate(dialogue.get("tasks_covered", [])):
            task_type = str(task.get("task_type") or "Unknown")
            instruction = _json_instruction_from_catalog(task_type, prompt_catalog)
            records.append(
                {
                    "instruction": (
                        "You previously learned a long conversation and must answer from memory.\n"
                        f"Character: {character}\n"
                        f"{instruction}"
                    ),
                    "input": str(task.get("query") or ""),
                    "output": json.dumps(_build_qa_sft_output(task), ensure_ascii=False),
                    "metadata": {
                        "task_type": task_type,
                        "dialogue_id": dialogue["id"],
                        "dialogue_stage": dialogue_stage,
                        "character": character,
                        "query": str(task.get("query") or ""),
                        "gold_answer": str(task.get("gold_answer") or ""),
                        "answer_components": list(task.get("answer_components") or []),
                        "source_event_ids": list(task.get("source_event_ids") or []),
                        "training_mode": "qa_sft",
                    },
                }
            )

    return records


def build_memory_eval_samples_from_qa_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for task_idx, row in enumerate(records):
        metadata = deepcopy(row.get("metadata", {}))
        dialogue_id = str(metadata.get("dialogue_id", ""))
        task_type = str(metadata.get("task_type", "Unknown"))
        query = str(row.get("input") or metadata.get("query") or "")
        prompt = (
            f"{str(row.get('instruction') or '').strip()}\n\n"
            f"[Question]\n{query}"
        )
        samples.append(
            {
                "id": f"{dialogue_id}::{task_type}::{task_idx}",
                "prompt": prompt,
                "reference": str(metadata.get("gold_answer") or ""),
                "metadata": metadata,
            }
        )
    return samples


def _qa_record_uid(row: Dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata", {})
    return (
        str(metadata.get("dialogue_id", "")),
        str(metadata.get("task_type", "")),
        str(metadata.get("query", row.get("input", ""))),
    )


def _qa_record_group_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    metadata = row.get("metadata", {})
    source_event_ids = [str(item) for item in list(metadata.get("source_event_ids") or []) if str(item)]
    if source_event_ids:
        return ("source_event_ids", tuple(sorted(source_event_ids)))
    return ("dialogue_id", str(metadata.get("dialogue_id", "")))


def _select_records_with_group_preference(
    records: List[Dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if target_count <= 0:
        return [], [deepcopy(row) for row in records]
    if target_count >= len(records):
        return [deepcopy(row) for row in records], []

    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(_qa_record_group_key(row), []).append(row)

    group_items = list(grouped.items())
    rng.shuffle(group_items)

    selected_ids: set[tuple[str, str, str]] = set()
    selected: List[Dict[str, Any]] = []

    for _, group_rows in group_items:
        ordered_group_rows = sorted(group_rows, key=_qa_record_uid)
        remaining = target_count - len(selected)
        if remaining <= 0:
            break
        if len(ordered_group_rows) <= remaining:
            selected.extend(deepcopy(row) for row in ordered_group_rows)
            selected_ids.update(_qa_record_uid(row) for row in ordered_group_rows)
            continue

        selected.extend(deepcopy(row) for row in ordered_group_rows[:remaining])
        selected_ids.update(_qa_record_uid(row) for row in ordered_group_rows[:remaining])
        break

    remaining_rows = [deepcopy(row) for row in records if _qa_record_uid(row) not in selected_ids]
    selected = selected[:target_count]
    return selected, remaining_rows


def _scale_task_targets_to_total(
    task_targets: Dict[str, int],
    target_total: int,
) -> Dict[str, int]:
    if target_total < 0:
        raise ValueError("target_total must be non-negative")

    base_total = sum(int(count) for count in task_targets.values())
    if base_total == 0:
        return {task_type: 0 for task_type in task_targets}
    if target_total < base_total:
        raise ValueError("target_total must be greater than or equal to the balanced base total")
    if target_total == base_total:
        return {task_type: int(count) for task_type, count in task_targets.items()}

    scaled: Dict[str, int] = {}
    remainders: List[tuple[float, int, str]] = []
    assigned = 0
    for task_type, count in task_targets.items():
        raw = float(target_total) * float(count) / float(base_total)
        floor_count = int(math.floor(raw))
        scaled[task_type] = floor_count
        assigned += floor_count
        remainders.append(
            (
                raw - floor_count,
                DEFAULT_TASK_BALANCE_ORDER.index(task_type)
                if task_type in DEFAULT_TASK_BALANCE_ORDER
                else len(DEFAULT_TASK_BALANCE_ORDER),
                task_type,
            )
        )

    remainders.sort(key=lambda item: (-item[0], item[1], item[2]))
    remaining = target_total - assigned
    for _, _, task_type in remainders[:remaining]:
        scaled[task_type] += 1

    return scaled


def _select_train_records_with_optional_upsampling(
    records: List[Dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if target_count <= 0 or not records:
        return []
    if target_count <= len(records):
        selected, _ = _select_records_with_group_preference(records, target_count, rng)
        return selected

    selected, _ = _select_records_with_group_preference(records, len(records), rng)
    source_rows = sorted(records, key=_qa_record_uid)
    rng.shuffle(source_rows)

    next_copy_index = 1
    while len(selected) < target_count:
        source_row = source_rows[(len(selected) - len(records)) % len(source_rows)]
        duplicated = deepcopy(source_row)
        duplicated.setdefault("metadata", {})
        duplicated["metadata"]["resampled_copy_index"] = next_copy_index
        selected.append(duplicated)
        next_copy_index += 1

    return selected


def build_task_balanced_qa_holdout_splits(
    records_by_stage: Dict[int, List[Dict[str, Any]]],
    holdout_ratio: float = 0.2,
    seed: int = 42,
    min_holdout_per_task: int = 1,
    train_total_target: Optional[int] = None,
) -> Dict[str, Any]:
    if not 0.0 <= holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in [0, 1)")

    ordered_stage_ids = sorted(records_by_stage)
    task_types = sorted(
        {
            str(row.get("metadata", {}).get("task_type", "Unknown"))
            for rows in records_by_stage.values()
            for row in rows
        },
        key=lambda task: (
            DEFAULT_TASK_BALANCE_ORDER.index(task)
            if task in DEFAULT_TASK_BALANCE_ORDER
            else len(DEFAULT_TASK_BALANCE_ORDER),
            task,
        ),
    )

    rows_by_stage_and_task: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for stage_id in ordered_stage_ids:
        task_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in records_by_stage[stage_id]:
            task_type = str(row.get("metadata", {}).get("task_type", "Unknown"))
            task_map.setdefault(task_type, []).append(row)
        rows_by_stage_and_task[stage_id] = task_map

    holdout_task_targets: Dict[str, int] = {}
    for task_type in task_types:
        available_counts = [
            len(rows_by_stage_and_task[stage_id].get(task_type, []))
            for stage_id in ordered_stage_ids
        ]
        min_count = min(available_counts) if available_counts else 0
        if min_count <= 1 or holdout_ratio <= 0:
            holdout_task_targets[task_type] = 0
            continue
        target = int(math.floor(min_count * holdout_ratio))
        target = max(min_holdout_per_task, target)
        target = min(target, min_count - 1)
        holdout_task_targets[task_type] = target

    holdout_records_by_stage: Dict[int, List[Dict[str, Any]]] = {stage_id: [] for stage_id in ordered_stage_ids}
    remaining_records_by_stage_and_task: Dict[int, Dict[str, List[Dict[str, Any]]]] = {
        stage_id: {} for stage_id in ordered_stage_ids
    }

    for stage_offset, stage_id in enumerate(ordered_stage_ids):
        for task_offset, task_type in enumerate(task_types):
            rows = rows_by_stage_and_task[stage_id].get(task_type, [])
            target = holdout_task_targets.get(task_type, 0)
            rng = random.Random(seed + stage_offset * 1000 + task_offset)
            selected, remaining = _select_records_with_group_preference(rows, target, rng)
            holdout_records_by_stage[stage_id].extend(selected)
            remaining_records_by_stage_and_task[stage_id][task_type] = remaining

    base_train_task_targets: Dict[str, int] = {}
    for task_type in task_types:
        remaining_counts = [
            len(remaining_records_by_stage_and_task[stage_id].get(task_type, []))
            for stage_id in ordered_stage_ids
        ]
        base_train_task_targets[task_type] = min(remaining_counts) if remaining_counts else 0

    base_train_total = sum(base_train_task_targets.values())
    if train_total_target is None:
        train_task_targets = dict(base_train_task_targets)
    else:
        train_task_targets = _scale_task_targets_to_total(
            base_train_task_targets,
            train_total_target,
        )

    train_records_by_stage: Dict[int, List[Dict[str, Any]]] = {stage_id: [] for stage_id in ordered_stage_ids}
    for stage_offset, stage_id in enumerate(ordered_stage_ids):
        for task_offset, task_type in enumerate(task_types):
            rows = remaining_records_by_stage_and_task[stage_id].get(task_type, [])
            target = train_task_targets.get(task_type, 0)
            rng = random.Random(seed + 100000 + stage_offset * 1000 + task_offset)
            selected = _select_train_records_with_optional_upsampling(rows, target, rng)
            train_records_by_stage[stage_id].extend(selected)

    manifest = {
        "holdout_ratio": holdout_ratio,
        "seed": seed,
        "min_holdout_per_task": min_holdout_per_task,
        "base_train_task_targets": base_train_task_targets,
        "base_train_total": base_train_total,
        "train_total_target": sum(train_task_targets.values()),
        "train_task_targets": train_task_targets,
        "holdout_task_targets": holdout_task_targets,
        "stages": {},
    }
    for stage_id in ordered_stage_ids:
        train_counts: Dict[str, int] = {}
        holdout_counts: Dict[str, int] = {}
        for row in train_records_by_stage[stage_id]:
            task_type = str(row.get("metadata", {}).get("task_type", "Unknown"))
            train_counts[task_type] = train_counts.get(task_type, 0) + 1
        for row in holdout_records_by_stage[stage_id]:
            task_type = str(row.get("metadata", {}).get("task_type", "Unknown"))
            holdout_counts[task_type] = holdout_counts.get(task_type, 0) + 1
        manifest["stages"][f"{stage_id:02d}"] = {
            "train_task_counts": train_counts,
            "holdout_task_counts": holdout_counts,
        }

    return {
        "train_records_by_stage": train_records_by_stage,
        "holdout_records_by_stage": holdout_records_by_stage,
        "base_train_task_targets": base_train_task_targets,
        "train_task_targets": train_task_targets,
        "holdout_task_targets": holdout_task_targets,
        "manifest": manifest,
    }


def rebalance_qa_sft_records(
    records: List[Dict[str, Any]],
    qa_sampling_mode: str = "original",
    qa_max_samples_per_character: int = 0,
    qa_task_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if qa_sampling_mode == "original" or qa_max_samples_per_character <= 0:
        return [deepcopy(row) for row in records]
    if qa_sampling_mode != "role_balanced_upweight":
        raise ValueError(f"Unsupported qa_sampling_mode: {qa_sampling_mode}")

    task_weights = dict(DEFAULT_QA_TASK_WEIGHTS)
    if qa_task_weights:
        task_weights.update(qa_task_weights)

    rows_by_character: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        character = str(row.get("metadata", {}).get("character", ""))
        rows_by_character.setdefault(character, []).append(row)

    rebalanced: List[Dict[str, Any]] = []
    for character in sorted(rows_by_character):
        rows = rows_by_character[character]
        weighted_rows = []
        for idx, row in enumerate(rows):
            task_type = str(row.get("metadata", {}).get("task_type", "Unknown"))
            weight = float(task_weights.get(task_type, 1.0))
            weighted_rows.append((idx, weight, row))

        weighted_rows.sort(key=lambda item: (-item[1], item[0]))

        if len(rows) >= qa_max_samples_per_character:
            selected_rows = [deepcopy(item[2]) for item in weighted_rows[:qa_max_samples_per_character]]
            selected_rows.sort(
                key=lambda row: (
                    str(row.get("metadata", {}).get("dialogue_id", "")),
                    str(row.get("metadata", {}).get("query", "")),
                    str(row.get("metadata", {}).get("task_type", "")),
                )
            )
            rebalanced.extend(selected_rows)
            continue

        selected_rows = [deepcopy(row) for row in rows]
        next_copy_index = 1
        while len(selected_rows) < qa_max_samples_per_character:
            source_row = weighted_rows[(len(selected_rows) - len(rows)) % len(weighted_rows)][2]
            duplicated = deepcopy(source_row)
            duplicated.setdefault("metadata", {})
            duplicated["metadata"]["resampled_copy_index"] = next_copy_index
            selected_rows.append(duplicated)
            next_copy_index += 1

        rebalanced.extend(selected_rows)

    return rebalanced


def build_memory_eval_samples(
    dialogues: List[Dict[str, Any]],
    stage_map: Dict[str, int],
    prompt_catalog_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    prompt_catalog = _load_prompt_catalog(prompt_catalog_path)
    samples: List[Dict[str, Any]] = []

    for dialogue in dialogues:
        dialogue_stage = stage_map[dialogue["id"]]
        character = dialogue.get("character", "")
        for task_idx, task in enumerate(dialogue.get("tasks_covered", [])):
            task_type = task.get("task_type", "Unknown")
            instruction = _json_instruction_from_catalog(task_type, prompt_catalog)
            prompt = (
                "You previously learned a long conversation and must answer from memory.\n"
                f"Character: {character}\n"
                f"{instruction}\n\n"
                f"[Question]\n{task.get('query', '')}"
            )
            samples.append(
                {
                    "id": f"{dialogue['id']}::{task_type}::{task_idx}",
                    "prompt": prompt,
                    "reference": task.get("gold_answer", ""),
                    "metadata": {
                        "task_type": task_type,
                        "dialogue_id": dialogue["id"],
                        "dialogue_stage": dialogue_stage,
                        "character": character,
                        "query": task.get("query", ""),
                        "answer_components": task.get("answer_components", []),
                        "source_event_ids": task.get("source_event_ids", []),
                    },
                }
            )

    return samples


def load_results_rows(results_file: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(results_file).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_latest_results_file(stage_dir: Path) -> Optional[Path]:
    candidates = sorted(stage_dir.rglob("detailed_results.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _find_stagewise_results_files(checkpoint_dir: Path, split: str) -> List[tuple[int, Path]]:
    if split not in {"seen", "unseen"}:
        raise ValueError(f"Unsupported split: {split}")

    results: List[tuple[int, Path]] = []
    for child in sorted(Path(checkpoint_dir).iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        match = STAGEWISE_RESULTS_DIR_RE.match(child.name)
        if match is None:
            continue
        dialogue_stage = int(match.group(1))
        child_split = match.group(2)
        if child_split != split:
            continue
        results_file = child / "detailed_results.jsonl"
        if results_file.exists():
            results.append((dialogue_stage, results_file))
    return results


def build_forgetting_summary(results_root: Path) -> Dict[str, Any]:
    checkpoint_dirs = sorted(
        [p for p in Path(results_root).iterdir() if p.is_dir() and p.name.startswith("checkpoint_stage_")],
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    checkpoint_stages: List[int] = []
    rows_by_checkpoint: Dict[int, List[Dict[str, Any]]] = {}
    dialogue_stages = set()

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_stage = int(checkpoint_dir.name.rsplit("_", 1)[-1])
        results_file = _find_latest_results_file(checkpoint_dir)
        if results_file is None:
            continue
        rows = load_results_rows(results_file)
        rows_by_checkpoint[checkpoint_stage] = rows
        checkpoint_stages.append(checkpoint_stage)
        for row in rows:
            metadata = row.get("metadata", {})
            if "dialogue_stage" in metadata:
                dialogue_stages.add(int(metadata["dialogue_stage"]))

    ordered_dialogue_stages = sorted(dialogue_stages)
    matrix: List[List[Optional[float]]] = []
    overall_curve: List[Dict[str, Any]] = []
    per_task_overall_curve: Dict[str, List[Dict[str, Any]]] = {}

    for checkpoint_stage in checkpoint_stages:
        rows = rows_by_checkpoint[checkpoint_stage]
        line: List[Optional[float]] = []
        scores = [float(row.get("final_score", row.get("scores", {}).get("final_score", 0.0))) for row in rows]
        overall_curve.append(
            {
                "checkpoint_stage": checkpoint_stage,
                "overall_avg_final_score": sum(scores) / len(scores) if scores else 0.0,
                "num_samples": len(scores),
            }
        )
        task_groups: Dict[str, List[float]] = {}
        for row in rows:
            task_type = str(row.get("metadata", {}).get("task_type", "Unknown"))
            task_groups.setdefault(task_type, []).append(
                float(row.get("final_score", row.get("scores", {}).get("final_score", 0.0)))
            )
        for task_type, task_scores in task_groups.items():
            per_task_overall_curve.setdefault(task_type, []).append(
                {
                    "checkpoint_stage": checkpoint_stage,
                    "avg_final_score": sum(task_scores) / len(task_scores) if task_scores else 0.0,
                    "num_samples": len(task_scores),
                }
            )

        for dialogue_stage in ordered_dialogue_stages:
            stage_scores = [
                float(row.get("final_score", row.get("scores", {}).get("final_score", 0.0)))
                for row in rows
                if int(row.get("metadata", {}).get("dialogue_stage", -1)) == dialogue_stage
            ]
            line.append(sum(stage_scores) / len(stage_scores) if stage_scores else None)
        matrix.append(line)

    return {
        "checkpoint_stages": checkpoint_stages,
        "dialogue_stages": ordered_dialogue_stages,
        "matrix": matrix,
        "overall_curve": overall_curve,
        "per_task_overall_curve": per_task_overall_curve,
    }


def build_stagewise_forgetting_summary(results_root: Path, split: str) -> Dict[str, Any]:
    if split not in {"seen", "unseen"}:
        raise ValueError(f"Unsupported split: {split}")

    checkpoint_dirs = sorted(
        [p for p in Path(results_root).iterdir() if p.is_dir() and p.name.startswith("checkpoint_stage_")],
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )

    checkpoint_stages: List[int] = []
    rows_by_checkpoint: Dict[int, List[Dict[str, Any]]] = {}
    rows_by_checkpoint_and_stage: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    dialogue_stages = set()

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_stage = int(checkpoint_dir.name.rsplit("_", 1)[-1])
        stage_files = _find_stagewise_results_files(checkpoint_dir, split=split)
        if not stage_files:
            continue

        checkpoint_rows: List[Dict[str, Any]] = []
        checkpoint_stage_rows: Dict[int, List[Dict[str, Any]]] = {}
        for dialogue_stage, results_file in stage_files:
            rows = load_results_rows(results_file)
            checkpoint_rows.extend(rows)
            checkpoint_stage_rows[dialogue_stage] = rows
            dialogue_stages.add(dialogue_stage)

        rows_by_checkpoint[checkpoint_stage] = checkpoint_rows
        rows_by_checkpoint_and_stage[checkpoint_stage] = checkpoint_stage_rows
        checkpoint_stages.append(checkpoint_stage)

    ordered_dialogue_stages = sorted(dialogue_stages)
    matrix: List[List[Optional[float]]] = []
    overall_curve: List[Dict[str, Any]] = []
    per_task_overall_curve: Dict[str, List[Dict[str, Any]]] = {}

    for checkpoint_stage in checkpoint_stages:
        rows = rows_by_checkpoint[checkpoint_stage]
        line: List[Optional[float]] = []
        scores = [float(row.get("final_score", row.get("scores", {}).get("final_score", 0.0))) for row in rows]
        overall_curve.append(
            {
                "checkpoint_stage": checkpoint_stage,
                "overall_avg_final_score": sum(scores) / len(scores) if scores else 0.0,
                "num_samples": len(scores),
            }
        )

        task_groups: Dict[str, List[float]] = {}
        for row in rows:
            task_type = str(row.get("metadata", {}).get("task_type", "Unknown"))
            task_groups.setdefault(task_type, []).append(
                float(row.get("final_score", row.get("scores", {}).get("final_score", 0.0)))
            )
        for task_type, task_scores in task_groups.items():
            per_task_overall_curve.setdefault(task_type, []).append(
                {
                    "checkpoint_stage": checkpoint_stage,
                    "avg_final_score": sum(task_scores) / len(task_scores) if task_scores else 0.0,
                    "num_samples": len(task_scores),
                }
            )

        stage_rows_map = rows_by_checkpoint_and_stage[checkpoint_stage]
        for dialogue_stage in ordered_dialogue_stages:
            stage_rows = stage_rows_map.get(dialogue_stage, [])
            stage_scores = [
                float(row.get("final_score", row.get("scores", {}).get("final_score", 0.0)))
                for row in stage_rows
            ]
            line.append(sum(stage_scores) / len(stage_scores) if stage_scores else None)
        matrix.append(line)

    return {
        "split": split,
        "checkpoint_stages": checkpoint_stages,
        "dialogue_stages": ordered_dialogue_stages,
        "matrix": matrix,
        "overall_curve": overall_curve,
        "per_task_overall_curve": per_task_overall_curve,
    }
