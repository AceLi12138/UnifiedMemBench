#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fact_track.io_utils import compact_final_facts, read_json, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize facts.json and rewrite consecutive fact IDs.")
    parser.add_argument("--input", default="facts.json")
    parser.add_argument("--output", default="facts.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compacted = compact_final_facts(read_json(args.input))
    write_json_atomic(compacted, args.output)
    print(json.dumps({
        "characters": len(compacted),
        "facts": sum(len(records) for records in compacted.values()),
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
