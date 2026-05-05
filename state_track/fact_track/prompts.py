from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io_utils import load_facts, remove_tree_contents, write_json_atomic


TASK_NAME = "schema_longest"

DEFAULT_LADDER_CONFIGS: Tuple[Tuple[str, int, int, int], ...] = (
    ("01", 1, 1, 1),
    ("02", 1, 1, 2),
    ("03", 1, 1, 3),
    ("04", 1, 1, 4),
    ("05", 1, 1, 5),
    ("06", 1, 1, 6),
    ("07", 1, 1, 7),
    ("08", 1, 1, 8),
    ("09", 1, 1, 9),
    ("10", 1, 1, 10),
    ("11", 1, 2, 10),
    ("12", 1, 3, 10),
    ("13", 1, 4, 10),
    ("14", 1, 5, 10),
    ("15", 1, 6, 10),
    ("16", 1, 7, 10),
    ("17", 1, 8, 10),
    ("18", 1, 9, 10),
    ("19", 1, 10, 10),
    ("20", 2, 10, 10),
    ("21", 3, 10, 10),
    ("22", 4, 10, 10),
    ("23", 5, 10, 10),
    ("24", 6, 10, 10),
    ("25", 7, 10, 10),
    ("26", 8, 10, 10),
    ("27", 9, 10, 10),
    ("28", 10, 10, 10),
    ("29", 15, 10, 10),
    ("30", 20, 10, 10),
    ("31", 25, 10, 10),
    ("32", 30, 10, 10),
    ("33", 35, 10, 10),
    ("34", 40, 10, 10),
)


@dataclass(frozen=True)
class LoaderConfig:
    min_schema_types: int = 10
    min_events_per_schema: int = 2


def day_ts(yyyy_mm_dd: str) -> int:
    current = datetime.strptime(yyyy_mm_dd, "%Y-%m-%d").date()
    return (current - date(1970, 1, 1)).days


def estimate_tokens(text: str, token_char_ratio: float) -> int:
    return max(1, int(round(len(text) / token_char_ratio))) if text else 0


def load_prompt_ready_facts(path: str | Path, config: LoaderConfig) -> Dict[str, List[Dict[str, str]]]:
    data = load_facts(path, required_fields=("fact", "fact_id", "timestamp", "schema"))
    kept: Dict[str, List[Dict[str, str]]] = {}

    for character, records in data.items():
        sorted_records = sorted(records, key=lambda x: (x["timestamp"], x["fact_id"]))
        schema_counts: Dict[str, int] = {}
        for record in sorted_records:
            schema_counts[record["schema"]] = schema_counts.get(record["schema"], 0) + 1

        if len(schema_counts) < config.min_schema_types:
            continue
        if any(count < config.min_events_per_schema for count in schema_counts.values()):
            continue
        kept[character] = sorted_records

    if not kept:
        raise ValueError("No characters passed the prompt loader filters.")
    return kept


class PromptBuilder:
    def __init__(self, *, seed: int = 42, token_char_ratio: float = 4.0) -> None:
        self.seed = seed
        self.token_char_ratio = token_char_ratio
        self.rng = random.Random(seed)

    @staticmethod
    def _fact_line(record: Dict[str, str]) -> str:
        return f"schema=【{record['schema']}】 | day_ts={day_ts(record['timestamp'])} | fact={record['fact']}"

    @staticmethod
    def _events_by_character_schema(
        selected: List[Tuple[str, Dict[str, str]]],
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for input_order, (character, record) in enumerate(selected):
            out.setdefault(character, {}).setdefault(record["schema"], []).append(
                {
                    "day_ts": day_ts(record["timestamp"]),
                    "fact": record["fact"],
                    "timestamp": record["timestamp"],
                    "fact_id": record["fact_id"],
                    "input_order": input_order,
                }
            )

        for character in out:
            for schema in out[character]:
                out[character][schema] = sorted(
                    out[character][schema],
                    key=lambda x: (x["day_ts"], x["input_order"]),
                )
        return out

    @staticmethod
    def _longest_gap_start(events: List[Dict[str, Any]]) -> Optional[str]:
        if not events:
            return None
        if len(events) == 1:
            return events[0]["fact"]
        if all(event["day_ts"] == events[0]["day_ts"] for event in events):
            return events[0]["fact"]

        best_gap = -1
        best_fact: Optional[str] = None
        for idx in range(1, len(events)):
            gap = events[idx]["day_ts"] - events[idx - 1]["day_ts"]
            if gap > best_gap:
                best_gap = gap
                best_fact = events[idx - 1]["fact"]
        return best_fact

    def _select_character_facts(
        self,
        character: str,
        records: List[Dict[str, str]],
        max_schemas_per_character: int,
        max_facts_per_schema: Optional[int],
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        by_schema: Dict[str, List[Dict[str, str]]] = {}
        for record in records:
            by_schema.setdefault(record["schema"], []).append(record)

        schemas = sorted(by_schema.keys())
        if max_schemas_per_character > 0:
            if len(schemas) < max_schemas_per_character:
                raise ValueError(
                    f"{character} has {len(schemas)} schemas, fewer than requested {max_schemas_per_character}."
                )
            schemas = sorted(self.rng.sample(schemas, k=max_schemas_per_character))

        selected: List[Dict[str, str]] = []
        for schema in schemas:
            items = sorted(by_schema[schema], key=lambda x: (x["timestamp"], x["fact_id"]))
            if max_facts_per_schema and max_facts_per_schema > 0:
                items = items[:max_facts_per_schema]
            selected.extend(items)

        selected.sort(key=lambda x: (x["timestamp"], x["fact_id"]))
        return selected, schemas

    @staticmethod
    def _schemas_by_character(selected: List[Tuple[str, Dict[str, str]]]) -> Dict[str, List[str]]:
        out: Dict[str, set[str]] = {}
        for character, record in selected:
            out.setdefault(character, set()).add(record["schema"])
        return {character: sorted(schemas) for character, schemas in out.items()}

    def _ground_truth(self, selected: List[Tuple[str, Dict[str, str]]]) -> Dict[str, Any]:
        schemas_by_character = self._schemas_by_character(selected)
        event_map = self._events_by_character_schema(selected)
        out: Dict[str, Any] = {}

        for character, schemas in schemas_by_character.items():
            schema_answers: Dict[str, Optional[str]] = {}
            for schema in schemas:
                schema_answers[schema] = self._longest_gap_start(event_map.get(character, {}).get(schema, []))
            out[character] = {TASK_NAME: schema_answers}
        return out

    @staticmethod
    def _prompt(facts_description: str) -> str:
        return f"""
            Please perform a strict atomic fact state-tracking task. Do not output your reasoning process; only output the final result in JSON.

            The input consists of multiple lines of facts, each with the fixed format:
            schema=【State x】 | day_ts=DAY_TS | fact=FACT_TEXT

            Field descriptions:
            - `schema=【State x】`: the state label, used only for grouping and must not be output;
            - `day_ts=DAY_TS`: the timestamp, used only for sorting and calculating adjacent time gaps and must not be output;
            - `fact=FACT_TEXT`: the original event text; only this `FACT_TEXT` may be output.

            Task objective:
            For each 【Person Name】 and each 【State x】 group under that person:
            1. Sort by `day_ts` in ascending order;
            2. Compute the time gap between each pair of adjacent events after sorting:
            `gap = next.day_ts - current.day_ts`
            3. Find the adjacent pair with the largest `gap`;
            4. Output the original `FACT_TEXT` of the earlier event in that adjacent pair.

            You must follow this order only: “group → sort → compare adjacent gaps → output the earlier one”.
            It is strictly forbidden to use any semantic judgment, such as:
            - best matches the meaning of the state
            - most typical / most important / most critical
            - earliest / latest
            - looks like the starting event
            - any judgment based on common sense, semantics, representativeness, or completeness

            Execution rules:

            I. Grouping
            - First identify the corresponding 【Person Name】 from `FACT_TEXT`;
            - Group by 【Person Name】;
            - Then within each person, group by `schema=【State x】`.

            II. Sorting
            - Within each 【Person Name】【State x】 group, sort by `day_ts` in ascending order;
            - If `day_ts` values are the same, preserve the original input order.

            III. Selection
            For each 【Person Name】【State x】 group:

            1. If there is only 1 event:
            - Directly output the original `FACT_TEXT` of that event.

            2. If there are multiple events and all `day_ts` values are exactly the same:
            - Output the original `FACT_TEXT` of the first event after sorting.

            3. Otherwise:
            - Only compute gaps for adjacent event pairs; do not compute gaps for non-adjacent events;
            - Find the adjacent pair with the largest `gap`;
            - Output the original `FACT_TEXT` of the earlier event in that adjacent pair;
            - If there is a tie for the maximum `gap`, choose the adjacent pair that appears earliest after sorting;
            - Therefore, in case of a tie, output the earlier event corresponding to the earlier pair position.

            IV. Output constraints
            - The output content must be strictly identical to the complete original text after `fact=` on one input line;
            - Only `FACT_TEXT` may be output, not the entire line;
            - No rewriting, no summarizing, no adding, no explaining;
            - The final output must be valid JSON only.

            The output format is fixed as:
            {{
            "Person Name": {{
                "schema_longest": {{
                "State x": "Original FACT_TEXT"
                }}
            }}
            }}

            Below is an algorithm example. Note: first sort, then find the “adjacent event pair”, then output the earlier event in that pair.

            Example 1: State A
            Input:
            schema=【State A】 | day_ts=1 | fact=Xiao Li starts preparing for the exam
            schema=【State A】 | day_ts=3 | fact=Xiao Li keeps practicing questions
            schema=【State A】 | day_ts=10 | fact=Xiao Li takes a mock test

            Step 1: Sort by `day_ts` in ascending order
            1. day_ts=1 → Xiao Li starts preparing for the exam
            2. day_ts=3 → Xiao Li keeps practicing questions
            3. day_ts=10 → Xiao Li takes a mock test

            Step 2: Only compute the time gaps of “adjacent event pairs” after sorting
            - Pair 1: `(1, 3)`, gap = 3 - 1 = 2
            - Earlier event: Xiao Li starts preparing for the exam
            - Pair 2: `(3, 10)`, gap = 10 - 3 = 7
            - Earlier event: Xiao Li keeps practicing questions

            Step 3: Find the adjacent pair corresponding to the maximum gap
            - Maximum gap = 7
            - Corresponding adjacent pair: `(3, 10)`

            Step 4: Output the `FACT_TEXT` of the “earlier event” in that adjacent pair
            - The final selected result should be:
            Xiao Li keeps practicing questions

            The output for this state in the final JSON should be:
            {{
            "Xiao Li": {{
                "schema_longest": {{
                "State A": "Xiao Li keeps practicing questions"
                }}
            }}
            }}

            Now process the following facts:
            {facts_description}

            Make sure the output is in valid JSON format.
            """

    def build_samples(
        self,
        facts_data: Dict[str, List[Dict[str, str]]],
        *,
        num_samples: int,
        num_characters: int,
        max_schemas_per_character: int,
        max_facts_per_schema: Optional[int],
        shuffle_facts: bool,
    ) -> List[Dict[str, Any]]:
        characters = sorted(facts_data.keys())
        if num_characters < 1:
            raise ValueError("num_characters must be positive.")
        if num_characters > len(characters):
            raise ValueError(f"Requested {num_characters} characters, but only {len(characters)} are available.")

        samples: List[Dict[str, Any]] = []
        for sample_idx in range(num_samples):
            chosen_characters = sorted(self.rng.sample(characters, k=num_characters))
            selected_pairs: List[Tuple[str, Dict[str, str]]] = []
            chosen_schemas: Dict[str, List[str]] = {}

            for character in chosen_characters:
                items, schemas = self._select_character_facts(
                    character,
                    facts_data[character],
                    max_schemas_per_character,
                    max_facts_per_schema,
                )
                chosen_schemas[character] = schemas
                selected_pairs.extend((character, item) for item in items)

            if shuffle_facts:
                self.rng.shuffle(selected_pairs)

            facts_description = "\n".join(self._fact_line(record) for _, record in selected_pairs)
            prompt = self._prompt(facts_description)
            facts_lines = len(selected_pairs)
            stats = {
                "chars": len(prompt),
                "est_tokens": estimate_tokens(prompt, self.token_char_ratio),
                "task": TASK_NAME,
                "facts_lines": facts_lines,
            }

            samples.append(
                {
                    "sample_id": f"s{sample_idx + 1:06d}",
                    "meta": {
                        "seed": self.seed,
                        "token_char_ratio": self.token_char_ratio,
                        "num_characters": num_characters,
                        "max_schemas_per_character": max_schemas_per_character,
                        "max_facts_per_schema": max_facts_per_schema,
                        "shuffle_facts": shuffle_facts,
                        "chosen_characters": chosen_characters,
                        "chosen_schemas": chosen_schemas,
                        "stats": {
                            "facts_lines": facts_lines,
                            "prompts": {TASK_NAME: {"chars": stats["chars"], "est_tokens": stats["est_tokens"]}},
                            "sum_chars": stats["chars"],
                            "sum_est_tokens": stats["est_tokens"],
                        },
                    },
                    "data": [
                        {
                            "Task": TASK_NAME,
                            "Prompt": prompt,
                            "GT": self._ground_truth(selected_pairs),
                            "Stats": stats,
                        }
                    ],
                }
            )

        return samples


def write_prompt_ladder(
    facts_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    num_samples: int = 50,
    min_schema_types: int = 10,
    min_events_per_schema: int = 2,
    token_char_ratio: float = 4.0,
    shuffle_facts: bool = False,
    clean_output: bool = False,
    max_characters: Optional[int] = None,
    ladder_configs: Iterable[Tuple[str, int, int, int]] = DEFAULT_LADDER_CONFIGS,
) -> List[Dict[str, Any]]:
    if max_characters is not None and max_characters < 1:
        raise ValueError("max_characters must be positive when provided.")

    output_dir = Path(output_dir)
    if clean_output:
        remove_tree_contents(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    facts_data = load_prompt_ready_facts(
        facts_path,
        LoaderConfig(min_schema_types=min_schema_types, min_events_per_schema=min_events_per_schema),
    )
    available_characters = len(facts_data)
    character_cap = available_characters if max_characters is None else min(max_characters, available_characters)
    effective_configs = [config for config in ladder_configs if config[1] <= character_cap]
    if not effective_configs:
        raise ValueError(
            f"No prompt ladder configs are feasible for {available_characters} prompt-ready characters."
        )

    rows: List[Dict[str, Any]] = []
    for level, num_characters, max_schemas, max_facts in effective_configs:
        builder = PromptBuilder(seed=seed, token_char_ratio=token_char_ratio)
        max_facts_value = max_facts if max_facts > 0 else None
        samples = builder.build_samples(
            facts_data,
            num_samples=num_samples,
            num_characters=num_characters,
            max_schemas_per_character=max_schemas,
            max_facts_per_schema=max_facts_value,
            shuffle_facts=shuffle_facts,
        )
        token_values = [int(sample["data"][0]["Stats"]["est_tokens"]) for sample in samples]
        avg_tokens = round(sum(token_values) / len(token_values))
        filename = (
            f"schema_longest_ladder_L{level}_tok{avg_tokens:04d}_"
            f"c{num_characters}_s{max_schemas}_f{max_facts}_n{num_samples}_seed{seed}.json"
        )
        path = output_dir / filename
        write_json_atomic(samples, path)
        rows.append(
            {
                "level": level,
                "avg_est_tokens": avg_tokens,
                "num_characters": num_characters,
                "max_schemas_per_character": max_schemas,
                "max_facts_per_schema": max_facts,
                "num_samples": num_samples,
                "seed": seed,
                "file": filename,
            }
        )

    summary_path = output_dir / "schema_longest_ladder_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "avg_est_tokens",
                "num_characters",
                "max_schemas_per_character",
                "max_facts_per_schema",
                "num_samples",
                "seed",
                "file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows
