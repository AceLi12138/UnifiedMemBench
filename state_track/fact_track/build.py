from __future__ import annotations

import math
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io_utils import (
    character_events,
    compact_final_facts,
    load_facts,
    load_stories,
    normalize_date,
    normalize_ws,
    read_json,
    write_json_atomic,
)
from .llm import ChatClient, extract_json_value
from .prompts import write_prompt_ladder


def _chunks(items: List[Any], size: int) -> Iterable[List[Any]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def select_character_items(
    characters: List[Dict[str, Any]],
    *,
    character_count: Optional[int] = None,
    character_offset: int = 0,
    character_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if character_offset < 0:
        raise ValueError("character_offset must be non-negative.")
    if character_count is not None and character_count < 1:
        raise ValueError("character_count must be positive when provided.")
    if character_offset > len(characters):
        raise ValueError(
            f"Requested character_offset {character_offset}, but only {len(characters)} characters are available."
        )

    pool = list(characters[character_offset:])
    if character_count is None:
        selected = pool
    elif character_count > len(pool):
        raise ValueError(
            f"Requested {character_count} characters after offset {character_offset}, "
            f"but only {len(pool)} are available."
        )
    elif character_seed is None:
        selected = pool[:character_count]
    else:
        rng = random.Random(character_seed)
        chosen_positions = set(rng.sample(range(len(pool)), k=character_count))
        selected = [item for pos, item in enumerate(pool) if pos in chosen_positions]

    names = [str(item.get("character_name", "")) for item in selected]
    if len(names) != len(set(names)):
        raise ValueError("Selected character names must be unique.")
    return selected


def _filter_by_characters(
    data: Dict[str, List[Dict[str, Any]]],
    character_names: Optional[Iterable[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    if character_names is None:
        return data
    return {name: data[name] for name in character_names if name in data}


def _extract_fact_texts(raw: str) -> List[str]:
    obj = extract_json_value(raw)
    if not isinstance(obj, dict):
        return []
    facts = obj.get("facts")
    if not isinstance(facts, list):
        return []

    out: List[str] = []
    for item in facts:
        if isinstance(item, dict):
            value = item.get("fact")
        else:
            value = item
        if isinstance(value, str) and value.strip():
            out.append(normalize_ws(value))
    return out


class FactExtractor:
    def __init__(self, client: ChatClient, *, history_max_events: int = 5, workers: int = 4) -> None:
        self.client = client
        self.history_max_events = history_max_events
        self.workers = workers

    @staticmethod
    def _messages(character: str, event: Dict[str, Any], history: str, causes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        prompt = f"""Extract high-confidence atomic facts from one story event.

Target character: {character}
Event description: {event.get("description", "")}
Past event history for context only: {history}
Caused-by events for context only: {causes}

Rules:
- Every fact must be directly supported by the event description.
- Every fact must use the full target character name as the subject.
- Do not infer motives, feelings, outcomes, or relationships unless explicitly stated.
- Do not extract facts from the history or caused-by context.
- Split multi-action descriptions into separate atomic facts.
- Write facts in English.

Return JSON only:
{{"facts": [{{"fact": "..."}}]}}
"""
        return [
            {"role": "system", "content": "Return valid JSON only. No markdown and no explanation."},
            {"role": "user", "content": prompt},
        ]

    def process_character(self, character: str, events: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        event_index = {event.get("event_id"): event for event in events if event.get("event_id")}
        history_by_id: Dict[str, str] = {}
        previous_lines: List[str] = []

        for event in events:
            event_id = str(event.get("event_id", ""))
            history = previous_lines[-self.history_max_events:] if self.history_max_events > 0 else previous_lines
            history_by_id[event_id] = "\n".join(history)
            if event.get("timestamp") or event.get("description"):
                previous_lines.append(f"{event.get('timestamp', '')} | {event.get('description', '')}")

        def run_event(pos: int, event: Dict[str, Any]) -> Tuple[int, List[Dict[str, str]]]:
            causes: List[Dict[str, Any]] = []
            for cause_id in event.get("caused_by_event_ids", []) or []:
                source = event_index.get(cause_id)
                if isinstance(source, dict):
                    causes.append(
                        {
                            "timestamp": source.get("timestamp"),
                            "description": source.get("description"),
                        }
                    )
            raw = self.client.chat(
                self._messages(character, event, history_by_id.get(str(event.get("event_id", "")), ""), causes),
                max_completion_tokens=8192,
                temperature=0.7,
            )
            timestamp = normalize_date(event.get("timestamp"))
            records = [
                {
                    "fact": fact,
                    "fact_id": uuid.uuid4().hex,
                    "timestamp": timestamp,
                }
                for fact in _extract_fact_texts(raw)
                if timestamp
            ]
            return pos, records

        out: List[Tuple[int, int, Dict[str, str]]] = []
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as executor:
            futures = [executor.submit(run_event, pos, event) for pos, event in enumerate(events)]
            for future in as_completed(futures):
                pos, records = future.result()
                for local_idx, record in enumerate(records):
                    out.append((pos, local_idx, record))

        return [record for _, _, record in sorted(out, key=lambda x: (x[0], x[1]))]


def extract_facts(
    stories_path: str | Path,
    output_path: str | Path,
    *,
    client: ChatClient,
    character_items: Optional[List[Dict[str, Any]]] = None,
    workers: int = 4,
    character_workers: int = 4,
    history_max_events: int = 5,
    reuse_existing: bool = True,
) -> Dict[str, List[Dict[str, str]]]:
    existing: Dict[str, List[Dict[str, str]]] = {}
    output_path = Path(output_path)
    if reuse_existing and output_path.exists():
        existing = load_facts(output_path, required_fields=("fact", "fact_id", "timestamp"))

    characters = character_items if character_items is not None else character_events(load_stories(stories_path))
    selected_names = [str(item["character_name"]) for item in characters]
    existing = _filter_by_characters(existing, selected_names)
    pending = [item for item in characters if item["character_name"] not in existing]
    results = dict(existing)

    def run_character(item: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        extractor = FactExtractor(client, history_max_events=history_max_events, workers=workers)
        return item["character_name"], extractor.process_character(item["character_name"], item["events"])

    with ThreadPoolExecutor(max_workers=max(1, character_workers)) as executor:
        futures = [executor.submit(run_character, item) for item in pending]
        for done, future in enumerate(as_completed(futures), start=1):
            character, facts = future.result()
            results[character] = facts
            write_json_atomic(results, output_path)
            print(f"[extract] {done}/{len(pending)} {character}: {len(facts)} facts")

    write_json_atomic(results, output_path)
    return results


def _state_cleaning_messages(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    blocks = []
    for item in items:
        blocks.append(
            f"Item ID: {item['item_id']}\n"
            f"Target character: {item['character']}\n"
            f"Fact: {item['fact']}"
        )

    prompt = f"""Decide whether each fact is primarily about the target character.

For each item:
- keep=true only when the target character is the main subject, experiencer, owner, or holder of the state.
- keep=false if the subject is another person, ambiguous, omitted, or uncertain.
- For kept facts, write a concise canonical state_repr for embedding and clustering.
- state_repr should be a short reusable phrase, not a sentence.

Items:
{chr(10).join(chr(10).join(('', block)) for block in blocks)}

Return exactly one JSON array:
[
  {{"item_id": "item_000001", "keep": true, "state_repr": "canonical phrase", "reason": "kept"}},
  {{"item_id": "item_000002", "keep": false, "state_repr": null, "reason": "subject_not_target_or_uncertain"}}
]
"""
    return [
        {"role": "system", "content": "Return valid JSON only. No markdown and no explanation."},
        {"role": "user", "content": prompt},
    ]


def clean_facts_with_state_repr(
    facts_path: str | Path,
    output_path: str | Path,
    *,
    client: ChatClient,
    character_names: Optional[Iterable[str]] = None,
    batch_size: int = 30,
    workers: int = 8,
) -> Dict[str, List[Dict[str, str]]]:
    facts = _filter_by_characters(
        load_facts(facts_path, required_fields=("fact", "fact_id", "timestamp")),
        character_names,
    )
    flat: List[Dict[str, Any]] = []
    for character, records in facts.items():
        for idx, record in enumerate(records):
            flat.append(
                {
                    "item_id": f"item_{len(flat) + 1:09d}",
                    "character": character,
                    "record_idx": idx,
                    "record": record,
                    "fact": record["fact"],
                }
            )

    def run_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        request_items = [
            {"item_id": item["item_id"], "character": item["character"], "fact": item["fact"]}
            for item in batch
        ]
        raw = client.chat(_state_cleaning_messages(request_items), max_completion_tokens=4096, temperature=0.3)
        parsed = extract_json_value(raw)
        by_id: Dict[str, Dict[str, Any]] = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and isinstance(item.get("item_id"), str):
                    by_id[item["item_id"]] = item

        out: List[Dict[str, Any]] = []
        for item in batch:
            result = by_id.get(item["item_id"], {})
            keep = result.get("keep") is True
            state_repr = result.get("state_repr")
            if keep and isinstance(state_repr, str) and state_repr.strip():
                record = dict(item["record"])
                record["state_repr"] = normalize_ws(state_repr)
                out.append({"character": item["character"], "record": record})
        return out

    output: Dict[str, List[Dict[str, str]]] = {character: [] for character in facts}
    batches = list(_chunks(flat, batch_size))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_batch, batch) for batch in batches]
        for done, future in enumerate(as_completed(futures), start=1):
            for item in future.result():
                output[item["character"]].append(item["record"])
            if done % 20 == 0 or done == len(batches):
                write_json_atomic({k: v for k, v in output.items() if v}, output_path)
                print(f"[clean] {done}/{len(batches)} batches")

    output = {k: v for k, v in output.items() if v}
    write_json_atomic(output, output_path)
    return output


def cluster_facts(
    facts_path: str | Path,
    output_path: str | Path,
    *,
    hf_local_dir: str | Path,
    character_names: Optional[Iterable[str]] = None,
    batch_size: int = 32,
    max_length: int = 128,
    min_clusters: int = 5,
    max_clusters: int = 15,
    target_clusters: int = 10,
    min_cluster_size: int = 3,
    max_cluster_size: int = 20,
    local_files_only: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    try:
        import numpy as np
        import transformers.utils.import_utils as import_utils

        if not hasattr(import_utils, "is_torch_fx_available"):
            import_utils.is_torch_fx_available = lambda: False

        from FlagEmbedding import BGEM3FlagModel
        from huggingface_hub import snapshot_download
        from scipy.optimize import linear_sum_assignment
        from sklearn.cluster import AgglomerativeClustering
        import torch
    except Exception as exc:
        raise RuntimeError("Embedding dependencies are missing. Use the py310 Conda environment.") from exc

    def l2(x: Any) -> Any:
        arr = np.asarray(x, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)

    def feasible_k(n: int) -> int:
        low = max(min_clusters, math.ceil(n / max_cluster_size))
        high = min(max_clusters, n // min_cluster_size)
        if low > high:
            raise ValueError(f"Infeasible cluster constraints for {n} facts.")
        return sorted(range(low, high + 1), key=lambda k: (abs(k - target_clusters), k))[0]

    def target_sizes(n: int, k: int) -> List[int]:
        sizes = [min_cluster_size] * k
        remaining = n - k * min_cluster_size
        idx = 0
        while remaining > 0:
            if sizes[idx] < max_cluster_size:
                sizes[idx] += 1
                remaining -= 1
            idx = (idx + 1) % k
        return sizes

    def centroids(x: Any, labels: Any) -> Any:
        rows = []
        for cid in sorted(set(int(v) for v in labels.tolist())):
            rows.append(l2(x[labels == cid].mean(axis=0))[0])
        return np.vstack(rows)

    def constrained_assign(x: Any, c: Any, sizes: List[int]) -> Any:
        slots = np.asarray([cid for cid, size in enumerate(sizes) for _ in range(size)], dtype=np.int32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = l2(x).astype(np.float64) @ l2(c[slots]).astype(np.float64).T
        sims = np.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
        row_ind, col_ind = linear_sum_assignment(-sims)
        labels = np.empty((x.shape[0],), dtype=np.int32)
        labels[row_ind] = slots[col_ind]
        return labels

    def process_embeddings(emb: Any) -> Tuple[Any, int]:
        n = emb.shape[0]
        k = feasible_k(n)
        seed = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(emb)
        sizes = target_sizes(n, k)
        labels = constrained_assign(emb, centroids(emb, seed), sizes)
        for _ in range(2):
            new_labels = constrained_assign(emb, centroids(emb, labels), sizes)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
        return labels + 1, k

    facts = _filter_by_characters(
        load_facts(facts_path, required_fields=("fact", "fact_id", "timestamp", "state_repr")),
        character_names,
    )
    model_dir = snapshot_download(
        repo_id="BAAI/bge-m3",
        local_dir=str(hf_local_dir),
        local_files_only=local_files_only,
        allow_patterns=[
            "pytorch_model.bin",
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "modules.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
            "sparse_linear.pt",
            "colbert_linear.pt",
        ],
    )
    model = BGEM3FlagModel(model_dir, use_fp16=bool(torch.cuda.is_available()))
    output: Dict[str, List[Dict[str, Any]]] = {}

    for done, (character, records) in enumerate(facts.items(), start=1):
        texts = [record["state_repr"] for record in records]
        emb = l2(model.encode(texts, batch_size=batch_size, max_length=max_length)["dense_vecs"])
        if not np.isfinite(emb).all():
            raise ValueError(f"Non-finite embedding values produced for character: {character}")
        labels, final_k = process_embeddings(emb)
        clustered: List[Dict[str, Any]] = []
        for record, label in zip(records, labels.tolist()):
            item = dict(record)
            item["cluster"] = int(label)
            item["final_k"] = int(final_k)
            clustered.append(item)
        output[character] = clustered
        if done % 50 == 0:
            write_json_atomic(output, output_path)
            print(f"[cluster] {done}/{len(facts)} characters")

    write_json_atomic(output, output_path)
    return output


def _schema_messages(facts: List[str]) -> List[Dict[str, str]]:
    facts_block = "\n".join(f"{idx}. {normalize_ws(fact)}" for idx, fact in enumerate(facts, 1))
    prompt = f"""Summarize these clustered character facts into one reusable state schema.

Requirements:
- Output only the schema phrase.
- Use 3 to 7 words.
- Use a noun phrase or gerund phrase, not a full sentence.
- Focus on the shared state or situation, not one-off details.

Facts:
{facts_block}

Schema:"""
    return [
        {"role": "system", "content": "Return one short schema phrase only."},
        {"role": "user", "content": prompt},
    ]


def _clean_schema(text: str) -> str:
    text = text.strip().splitlines()[0].strip()
    text = re.sub(r"^```(?:text|json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(schema|state schema|state|label|summary)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip("\"'“”‘’").replace("_", " ")
    words = normalize_ws(text).split()
    return " ".join(words[:7])


def summarize_schemas(
    clustered_path: str | Path,
    output_path: str | Path,
    *,
    client: ChatClient,
    character_names: Optional[Iterable[str]] = None,
    workers: int = 16,
) -> Dict[str, List[Dict[str, Any]]]:
    data = read_json(clustered_path)
    if not isinstance(data, dict):
        raise ValueError("Clustered facts must be a dict.")
    data = _filter_by_characters(data, character_names)

    tasks: List[Tuple[str, Any, List[int], List[str]]] = []
    for character, records in data.items():
        groups: Dict[Any, Tuple[List[int], List[str]]] = {}
        if not isinstance(records, list):
            continue
        for idx, record in enumerate(records):
            if not isinstance(record, dict) or "cluster" not in record:
                continue
            groups.setdefault(record["cluster"], ([], []))
            groups[record["cluster"]][0].append(idx)
            groups[record["cluster"]][1].append(str(record.get("fact", "")))
        for cluster, (indices, facts) in groups.items():
            tasks.append((character, cluster, indices, facts))

    def run_task(task: Tuple[str, Any, List[int], List[str]]) -> Tuple[str, Any, List[int], str]:
        character, cluster, indices, facts = task
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                raw = client.chat(_schema_messages(facts), max_completion_tokens=256, temperature=0.2)
                schema = _clean_schema(raw)
                if 3 <= len(schema.split()) <= 7:
                    return character, cluster, indices, schema
                last_error = ValueError(f"Invalid schema: {schema!r}")
            except Exception as exc:
                last_error = exc
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Failed to summarize {character}/{cluster}: {last_error}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            character, _, indices, schema = future.result()
            for idx in indices:
                data[character][idx]["schema"] = schema
            if done % 50 == 0 or done == len(tasks):
                write_json_atomic(data, output_path)
                print(f"[schema] {done}/{len(tasks)} groups")

    write_json_atomic(data, output_path)
    return data


def build_dataset(
    *,
    root: str | Path,
    stories: str = "stories_v4.json",
    facts_output: str = "facts.json",
    prompts_output: str = "fact_track_schema_longest",
    cache_dir: str = ".cache/fact_track",
    provider: str = "mimo",
    model: Optional[str] = None,
    reuse_cache: bool = True,
    regenerate_prompts: bool = True,
    local_files_only: bool = False,
    character_count: Optional[int] = None,
    character_offset: int = 0,
    character_seed: Optional[int] = None,
    prompt_max_characters: Optional[int] = None,
) -> None:
    root = Path(root)
    cache = root / cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    client = ChatClient(provider=provider, model=model)
    all_characters = character_events(load_stories(root / stories))
    selected_characters = select_character_items(
        all_characters,
        character_count=character_count,
        character_offset=character_offset,
        character_seed=character_seed,
    )
    selected_names = [str(item["character_name"]) for item in selected_characters]
    print(f"[select] {len(selected_names)}/{len(all_characters)} characters")

    raw_facts = cache / "facts_raw.json"
    facts_with_state = cache / "facts_with_state.json"
    clustered = cache / "facts_clustered.json"
    summarized = cache / "facts_summarized.json"

    if not (reuse_cache and summarized.exists()):
        if not (reuse_cache and clustered.exists()):
            if not (reuse_cache and facts_with_state.exists()):
                if not (reuse_cache and raw_facts.exists()):
                    extract_facts(
                        root / stories,
                        raw_facts,
                        client=client,
                        character_items=selected_characters,
                        reuse_existing=reuse_cache,
                    )
                clean_facts_with_state_repr(raw_facts, facts_with_state, client=client, character_names=selected_names)
            cluster_facts(
                facts_with_state,
                clustered,
                hf_local_dir=root / ".cache/hf_models/bge-m3",
                character_names=selected_names,
                local_files_only=local_files_only,
            )
        summarize_schemas(clustered, summarized, client=client, character_names=selected_names)

    final_data = compact_final_facts(_filter_by_characters(read_json(summarized), selected_names))
    write_json_atomic(final_data, root / facts_output)

    if regenerate_prompts:
        write_prompt_ladder(
            root / facts_output,
            root / prompts_output,
            clean_output=True,
            max_characters=prompt_max_characters,
        )
