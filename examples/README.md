# Examples

This directory contains small schema examples for understanding UnifiedMemBench
without downloading the full benchmark or running model inference.

Files:

- `sample_seed_bank.json`: schema-only example for an MBTI-style seed bank entry. This is illustrative and is not copied from the CharacterChat raw seed bank.
- `sample_event_source_record.json`: truncated example from the released synthetic event-source artifact.
- `sample_qa_task.json`: one QA task example from the released QA construction artifact.
- `sample_dialogue_record.json`: heavily truncated preview of a clean long-dialogue benchmark record.
- `sample_contextual_eval_record.json`: example contextual-memory evaluation object.
- `sample_parametric_eval_record.json`: example no-context parametric-memory evaluation object from the held-out split.

The full clean long-dialogue benchmark is hosted in the UnifiedMemBench dataset collection:

https://huggingface.co/collections/Ace1213812/unifiedmembench-dataset

The collection includes `UnifiedMemBench-LongDialogue`,
`UnifiedMemBench-ParametricMemory`, and the high-density state-tracking
evaluation dataset.
