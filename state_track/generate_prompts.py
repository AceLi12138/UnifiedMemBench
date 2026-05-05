#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fact_track.prompts import write_prompt_ladder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fact_track_schema_longest from facts.json.")
    parser.add_argument("--facts", default="facts.json", help="Input facts JSON.")
    parser.add_argument("--output", default="fact_track_schema_longest", help="Prompt ladder output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--min-schema-types", type=int, default=10)
    parser.add_argument("--min-events-per-schema", type=int, default=2)
    parser.add_argument("--token-char-ratio", type=float, default=4.0)
    parser.add_argument("--shuffle-facts", action="store_true")
    parser.add_argument(
        "--max-characters",
        type=int,
        default=None,
        help="Only generate ladder files whose samples use at most this many characters.",
    )
    parser.add_argument("--clean-output", action="store_true", help="Clear the output directory before generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = write_prompt_ladder(
        Path(args.facts),
        Path(args.output),
        seed=args.seed,
        num_samples=args.num_samples,
        min_schema_types=args.min_schema_types,
        min_events_per_schema=args.min_events_per_schema,
        token_char_ratio=args.token_char_ratio,
        shuffle_facts=args.shuffle_facts,
        max_characters=args.max_characters,
        clean_output=args.clean_output,
    )
    print(json.dumps({"generated_levels": len(rows), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
