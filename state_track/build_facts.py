#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fact_track.build import build_dataset


DEFAULT_CACHE_DIR = ".cache/fact_track"


def default_cache_dir(
    character_count: int | None,
    character_offset: int,
    character_seed: int | None,
) -> str:
    default_selection = character_count is None and character_offset == 0 and character_seed is None
    if default_selection:
        return DEFAULT_CACHE_DIR

    parts = [f"c{character_count}" if character_count is not None else "all"]
    if character_offset:
        parts.append(f"o{character_offset}")
    if character_seed is not None:
        parts.append(f"seed{character_seed}")
    return f"{DEFAULT_CACHE_DIR}_{'_'.join(parts)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build facts.json from stories_v4.json and optionally regenerate the prompt ladder."
    )
    parser.add_argument("--stories", default="../data_gen/output/stories_v4.json", help="Input stories JSON.")
    parser.add_argument("--facts-output", default="facts.json", help="Final facts JSON.")
    parser.add_argument("--prompts-output", default="fact_track_schema_longest", help="Prompt ladder output directory.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Resumable intermediate cache directory. Defaults to .cache/fact_track for full builds "
            "and a scoped cache for character subsets."
        ),
    )
    parser.add_argument("--provider", choices=["mimo", "siliconflow", "kimi"], default="mimo", help="Chat API provider.")
    parser.add_argument("--model", default=None, help="Override provider model name.")
    parser.add_argument("--no-reuse-cache", action="store_true", help="Recompute every stage instead of reusing cache files.")
    parser.add_argument("--skip-prompts", action="store_true", help="Only build facts.json.")
    parser.add_argument(
        "--character-count",
        "--max-characters",
        "--num-characters",
        dest="character_count",
        type=int,
        default=None,
        help="Use only this many characters from the stories file.",
    )
    parser.add_argument(
        "--character-offset",
        type=int,
        default=0,
        help="Skip this many story characters before applying --character-count.",
    )
    parser.add_argument(
        "--character-seed",
        type=int,
        default=None,
        help="Randomly sample --character-count characters with this seed instead of taking the first N.",
    )
    parser.add_argument(
        "--prompt-max-characters",
        type=int,
        default=None,
        help="Only generate prompt ladder files whose samples use at most this many characters.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load BGE-M3 from local Hugging Face cache only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir or default_cache_dir(
        args.character_count,
        args.character_offset,
        args.character_seed,
    )
    build_dataset(
        root=Path.cwd(),
        stories=args.stories,
        facts_output=args.facts_output,
        prompts_output=args.prompts_output,
        cache_dir=cache_dir,
        provider=args.provider,
        model=args.model,
        reuse_cache=not args.no_reuse_cache,
        regenerate_prompts=not args.skip_prompts,
        local_files_only=args.local_files_only,
        character_count=args.character_count,
        character_offset=args.character_offset,
        character_seed=args.character_seed,
        prompt_max_characters=args.prompt_max_characters,
    )


if __name__ == "__main__":
    main()
