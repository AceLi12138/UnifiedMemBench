from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_date(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    s = value.strip()
    if not s:
        return ""
    if DATE_RE.match(s):
        return s

    iso = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        return datetime.fromisoformat(iso).date().isoformat()
    except Exception:
        pass

    match = re.search(r"\d{4}-\d{2}-\d{2}", s)
    return match.group(0) if match else ""


def parse_datetime(value: Any) -> datetime:
    date_value = normalize_date(value)
    if date_value:
        return datetime.fromisoformat(date_value).replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def load_stories(path: str | Path) -> List[Dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Stories JSON must be a list of character objects.")
    return [item for item in data if isinstance(item, dict)]


def character_events(stories: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    characters: List[Dict[str, Any]] = []
    for character in stories:
        events: List[Dict[str, Any]] = []
        chronology = character.get("chronology", [])
        if isinstance(chronology, list):
            for year_block in chronology:
                if not isinstance(year_block, dict):
                    continue
                year = year_block.get("year")
                for event in year_block.get("events", []) or []:
                    if isinstance(event, dict):
                        item = dict(event)
                        item["year"] = year
                        events.append(item)

        characters.append(
            {
                "character_name": character.get("character_name", "Unknown"),
                "events": sorted(
                    events,
                    key=lambda x: (parse_datetime(x.get("timestamp")), str(x.get("event_id", ""))),
                ),
            }
        )
    return characters


def load_facts(path: str | Path, *, required_fields: Tuple[str, ...]) -> Dict[str, List[Dict[str, Any]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("Facts JSON must be a dict: {character_name: [facts...]}.")

    out: Dict[str, List[Dict[str, Any]]] = {}
    for character, records in data.items():
        if not isinstance(character, str) or not isinstance(records, list):
            continue

        cleaned: List[Dict[str, Any]] = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            missing = [field for field in required_fields if field not in record]
            if missing:
                raise ValueError(f"{character} fact[{idx}] missing fields: {missing}")

            item = dict(record)
            for field in required_fields:
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{character} fact[{idx}] has invalid {field!r}: {value!r}")
                item[field] = normalize_ws(value)

            if "timestamp" in item:
                item["timestamp"] = normalize_date(item["timestamp"])
                if not item["timestamp"]:
                    raise ValueError(f"{character} fact[{idx}] has invalid timestamp.")

            cleaned.append(item)

        if cleaned:
            out[character] = cleaned

    if not out:
        raise ValueError("No valid facts found.")
    return out


def compact_final_facts(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    next_id = 1

    for character in sorted(data.keys()):
        records = data[character]
        sortable: List[Tuple[str, str, int, Dict[str, Any]]] = []
        for pos, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            fact = normalize_ws(str(record.get("fact", "")))
            timestamp = normalize_date(record.get("timestamp"))
            schema = normalize_ws(str(record.get("schema", "")))
            state_repr = normalize_ws(str(record.get("state_repr", "")))
            if not fact or not timestamp or not schema:
                continue
            old_id = str(record.get("fact_id", ""))
            sortable.append((timestamp, old_id, pos, {
                "fact": fact,
                "timestamp": timestamp,
                "state_repr": state_repr,
                "schema": schema,
            }))

        character_records: List[Dict[str, str]] = []
        for _, _, _, item in sorted(sortable, key=lambda x: (x[0], x[1], x[2])):
            item["fact_id"] = f"{next_id:06d}"
            ordered = {
                "fact": item["fact"],
                "fact_id": item["fact_id"],
                "timestamp": item["timestamp"],
                "state_repr": item["state_repr"],
                "schema": item["schema"],
            }
            character_records.append(ordered)
            next_id += 1

        if character_records:
            out[character] = character_records

    return out


def remove_tree_contents(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            remove_tree_contents(child)
            child.rmdir()
        else:
            child.unlink()
