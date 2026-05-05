#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from fact_track.build import cluster_facts
from fact_track.io_utils import read_json, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test BGE-M3 encoding and constrained clustering.")
    parser.add_argument("--facts", default="facts.json", help="Source facts used to build a small test batch.")
    parser.add_argument("--hf-local-dir", default=".cache/hf_models/bge-m3", help="BGE-M3 local model directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files.")
    return parser.parse_args()


def sample_records(facts_path: str) -> dict[str, list[dict[str, str]]]:
    data = read_json(facts_path)
    if not isinstance(data, dict):
        raise ValueError("facts input must be a dict.")

    for character, records in data.items():
        if not isinstance(records, list) or len(records) < 6:
            continue
        usable = []
        for record in records:
            if all(isinstance(record.get(field), str) and record[field].strip() for field in ("fact", "fact_id", "timestamp", "state_repr")):
                usable.append({
                    "fact": record["fact"],
                    "fact_id": record["fact_id"],
                    "timestamp": record["timestamp"],
                    "state_repr": record["state_repr"],
                })
            if len(usable) == 6:
                return {character: usable}
    raise ValueError("Could not find a character with at least six usable facts.")


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="fact_track_cluster_check_") as tmpdir:
        input_path = Path(tmpdir) / "sample_facts.json"
        output_path = Path(tmpdir) / "clustered.json"
        write_json_atomic(sample_records(args.facts), input_path)
        output = cluster_facts(
            input_path,
            output_path,
            hf_local_dir=args.hf_local_dir,
            min_clusters=2,
            max_clusters=3,
            target_clusters=2,
            min_cluster_size=2,
            max_cluster_size=4,
            local_files_only=args.local_files_only,
        )

    character, records = next(iter(output.items()))
    clusters = sorted({record["cluster"] for record in records})
    print(json.dumps({
        "status": "ok",
        "character": character,
        "encoded_records": len(records),
        "clusters": clusters,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
