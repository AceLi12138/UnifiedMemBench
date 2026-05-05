# Fact Track Memory

This folder contains a simple script-based pipeline for the Fact Track Memory dataset.

The repository has one source input and two release artifacts:

- Input: `data_gen/output/stories_v4.json`
- Final facts: `facts.json`
- Final prompt ladder: `fact_track_schema_longest/`

Intermediate files are not part of the release. When the full build is run, resumable stage files are written under `.cache/fact_track/`.


## Current Results

- `stories_v4.json`: 1,000 characters
- `facts.json`: 1,000 characters and 124,169 facts
- `fact_track_schema_longest/`: 34 ladder files, 50 samples per file

## Common Commands

Validate the current release artifacts:

```bash
python validate_dataset.py
```

If `fact_track_schema_longest/` is not available locally, `validate_dataset.py` will automatically download the prompt ladder from the Hugging Face dataset `Ace1213812/UnifiedMemBench-StateTracking` before validation. To disable this behavior, pass `--no-auto-download-prompts`.

Run the full facts build from `stories_v4.json`:

```bash
export MIMO_API_KEY="..."
python build_facts.py 
```

Build a smaller dataset with only the first 25 characters:

```bash
python build_facts.py \
  --character-count 25 \
  --facts-output facts_c25.json \
  --prompts-output fact_track_schema_c25
```

For reproducible random subsets, add `--character-seed`:

```bash
python build_facts.py \
  --character-count 25 \
  --character-seed 7 \
  --facts-output facts_c25_seed7.json \
  --prompts-output fact_track_schema_c25_seed7
```

When `--character-count`, `--character-offset`, or `--character-seed` is used, the default cache directory is automatically scoped, for example `.cache/fact_track_c25_seed7`. Pass `--cache-dir` if you want a custom cache location.

Regenerate the prompt ladder from `facts.json`:

```bash
python generate_prompts.py --clean-output
```

Regenerate only prompt ladder files with up to 10 characters per sample:

```bash
python generate_prompts.py --max-characters 10 --clean-output
```

Validate custom output paths:

```bash
python validate_dataset.py \
  --facts facts_c25.json \
  --prompts fact_track_schema_c25
```

Evaluate prompts with an API model:

```bash
export MIMO_API_KEY="..."
python eval_api.py \
  --provider mimo \
  --input fact_track_schema_longest \
  --num-samples 5 \
  --level-start 1 \
  --level-end 3
```

Kimi evaluation uses `MOONSHOT_API_KEY`, `KIMI_API_KEY`, or `MOONSHOTAI_API_KEY`:

```bash
export MOONSHOT_API_KEY="..."
python eval_api.py \
  --provider kimi \
  --model kimi-k2.5 \
  --input fact_track_schema_longest
```

Inspect the local-model evaluation selection scaffold:

```bash
python eval_local.py --dry-run --input fact_track_schema_longest --level-start 1 --level-end 2
```

## Full Build Path

`build_facts.py` runs this pipeline:

```text
stories_v4.json
  -> .cache/fact_track/facts_raw.json
  -> .cache/fact_track/facts_with_state.json
  -> .cache/fact_track/facts_clustered.json
  -> .cache/fact_track/facts_summarized.json
  -> facts.json
  -> fact_track_schema_longest/
```

Stage details:

1. `stories_v4.json` is loaded, optional character selection is applied, and each selected character's chronology is flattened into sorted events.
2. Each event is sent to the chat model and converted into high-confidence atomic facts.
3. Facts are cleaned by another chat pass; kept facts receive `state_repr`.
4. `state_repr` is encoded with BGE-M3 and clustered per character with hard size constraints.
5. Each character-cluster is summarized into a short `schema`.
6. The final compact `facts.json` keeps only release fields and rewrites `fact_id` as consecutive 6-digit IDs.

## Character Selection

`build_facts.py` now supports selecting a subset of story characters:

- `--character-count N` / `--max-characters N` / `--num-characters N`: use N characters.
- `--character-offset N`: skip the first N story characters before selection.
- `--character-seed N`: randomly sample the requested count with a reproducible seed.
- `--prompt-max-characters N`: cap generated prompt ladder files by per-sample character count.

Prompt ladder generation automatically skips default ladder levels that require more prompt-ready characters than the selected `facts.json` contains, so a 10-character dataset will not try to emit the `c15`, `c20`, ..., `c40` files.

## Dataset Contract

Each `facts.json` record has exactly:

```json
{
  "fact": "Original atomic fact text.",
  "fact_id": "000001",
  "timestamp": "YYYY-MM-DD",
  "state_repr": "canonical embedding phrase",
  "schema": "cluster-level state schema"
}
```

The `schema_longest` task asks models to group facts by person and schema, sort each group by `day_ts`, find the adjacent event pair with the largest time gap, and output the earlier event's exact `fact` text. Groups with one event return that event's exact `fact` text.

## Generated Prompt Structure

Prompt ladder files are written under the prompt output directory. Each filename records the ladder settings:

```text
schema_longest_ladder_L{level}_tok{avg_tokens}_c{num_characters}_s{max_schemas}_f{max_facts}_n{num_samples}_seed{seed}.json
```

Each file contains a JSON list of samples. One sample has this shape:

```json
{
  "sample_id": "s000001",
  "meta": {
    "seed": 42,
    "token_char_ratio": 4.0,
    "num_characters": 1,
    "max_schemas_per_character": 1,
    "max_facts_per_schema": 10,
    "shuffle_facts": false,
    "chosen_characters": ["Person Name"],
    "chosen_schemas": {
      "Person Name": ["State Label"]
    },
    "stats": {
      "facts_lines": 10,
      "prompts": {
        "schema_longest": {
          "chars": 2500,
          "est_tokens": 625
        }
      },
      "sum_chars": 2500,
      "sum_est_tokens": 625
    }
  },
  "data": [
    {
      "Task": "schema_longest",
      "Prompt": "full prompt text",
      "GT": {
        "Person Name": {
          "schema_longest": {
            "State x": "Original FACT_TEXT"
          }
        }
      },
      "Stats": {
        "chars": 2500,
        "est_tokens": 625,
        "task": "schema_longest",
        "facts_lines": 10
      }
    }
  ]
}
```

The `Prompt` field uses the same fixed instruction template as the previous prompt generator, followed by selected fact lines:

```text
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
It is strictly forbidden to use any semantic judgment.

Execution rules:
- First identify the corresponding 【Person Name】 from `FACT_TEXT`;
- Group by 【Person Name】, then by `schema=【State x】`;
- Sort by `day_ts` ascending; if `day_ts` values are the same, preserve the original input order;
- If there is only 1 event, directly output that event's original `FACT_TEXT`;
- If all `day_ts` values are exactly the same, output the first event after sorting;
- Otherwise, compare adjacent gaps only and output the earlier event from the earliest maximum-gap pair.

Output constraints:
- Only `FACT_TEXT` may be output, not the entire input line;
- No rewriting, summarizing, adding, or explaining;
- The final output must be valid JSON only.

The output format is fixed as:
{
  "Person Name": {
    "schema_longest": {
      "State x": "Original FACT_TEXT"
    }
  }
}

Now process the following facts:
schema=【State Label】 | day_ts=12345 | fact=Original FACT_TEXT
schema=【State Label】 | day_ts=12400 | fact=Original FACT_TEXT

Make sure the output is in valid JSON format.
```

`GT` stores the expected JSON answer for that prompt. The answer key is computed from the same selected facts by grouping on `(person, schema)`, sorting by `day_ts` while preserving input order for ties, and choosing the earlier fact in the adjacent pair with the largest timestamp gap. If a selected group has only one fact, `GT` uses that fact directly.

## Evaluation

`eval_api.py` runs exact-match evaluation through OpenAI-compatible chat APIs. It currently exposes the two intended providers for this project:

- `mimo`: default model `mimo-v2-flash`, API key from `MIMO_API_KEY` or `XIAOMI_MIMO_API_KEY`.
- `kimi`: default model `kimi-k2.5`, API key from `MOONSHOT_API_KEY`, `KIMI_API_KEY`, or `MOONSHOTAI_API_KEY`.

Useful API options:

- `--input`: prompt directory or one prompt JSON file.
- `--output-dir`: result directory, default `eval_results/api`.
- `--num-samples`: evaluate only the first N samples per file.
- `--level-start` / `--level-end`: evaluate a ladder level range.
- `--concurrency`: parallel API calls per file.
- `--dry-run`: list selected files and sample counts without requiring an API key.
- `--overwrite`: replace existing result files.
- `--no-resume`: do not skip files already listed in the CSV.
- `--include-prompt`: store full prompt text in the JSONL details.

Outputs are written incrementally:

```text
eval_results/api/
  eval_api_{provider}_{model}.csv
  eval_api_{provider}_{model}.jsonl
  eval_api_{provider}_{model}.config.json
```

The CSV stores one aggregate row per prompt file. The JSONL stores per-sample raw output, parsed JSON, parse status, exact-match status, and per-character/per-schema comparison details. Scoring is strict: the predicted value must exactly match `GT`; old compatibility behavior that accepted any non-empty string for `null` GT is intentionally not used in v2.

Shared evaluation code lives in `fact_track/evaluation.py`. Both runners use that module for prompt-file loading, model-output JSON extraction, strict scoring, per-file aggregation, CSV/JSONL writing, resume handling, and summary printing. `eval_api.py` only owns API provider calls; `eval_local.py` only owns local model calls.

`eval_local.py` supports local vLLM OpenAI-compatible servers. It keeps the benchmark defaults at `--max-input-tokens 131072` and `--max-output-tokens 16384`, and includes built-in model profiles for:

```text
qwen2.5-7b-instruct-1m
glm-4-9b-chat-1m
qwen3.5-9b
hunyuan-a13b-instruct
qwen3.5-27b
gemma-4-31b-it
glm-4.7-flash
qwen3.5-35b-a3b
ministral-3-8b-instruct-2512
gemma-4-26b-a4b-it
```

Inspect the registered profiles:

```bash
python eval_local.py --list-models
```

Print a recommended vLLM launch command for one model:

```bash
python eval_local.py --model-key qwen3.5-9b --print-vllm-command
```

After starting the vLLM server, run evaluation through its local endpoint:

```bash
python eval_local.py \
  --model-key qwen3.5-9b \
  --endpoint http://localhost:8000/v1 \
  --input fact_track_schema_longest \
  --level-start 1 \
  --level-end 3 \
  --concurrency 1
```

For thinking-capable models, the local runner applies the documented non-thinking controls by default: Qwen3.5/Gemma/Hunyuan use `chat_template_kwargs.enable_thinking=false`; Hunyuan also receives the documented `/no_think` prefix; GLM-4.7 uses `thinking.type=disabled`. Use `--disable-built-in-thinking-control` or `--disable-prompt-prefix` only when debugging a server-specific compatibility issue.
