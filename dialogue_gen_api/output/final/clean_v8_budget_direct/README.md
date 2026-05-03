# Clean Long-Dialogue Benchmark

The full clean UnifiedMemBench long-dialogue benchmark is hosted in the
UnifiedMemBench dataset collection:

https://huggingface.co/collections/Ace1213812/unifiedmembench-dataset

Dataset component: `UnifiedMemBench-LongDialogue`.

Expected local file after download:

```text
dialogue_gen_api/output/final/clean_v8_budget_direct/UMB_dialogue_benchmark.json
```

This directory keeps `dataset_summary.json` as a lightweight summary of the
clean benchmark. The full JSON benchmark is not stored directly in this GitHub
release candidate.

The clean benchmark keeps all 1,000 dialogues and removes unsupported QA tasks.
The current summary records:

- raw QA tasks: 12,112
- valid QA tasks: 11,795
- removed QA tasks: 317
- valid rate: 97.4%
