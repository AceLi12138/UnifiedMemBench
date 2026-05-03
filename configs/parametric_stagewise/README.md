# Parametric Stagewise Configuration

This directory collects the small configuration and manifest files used by the
main stagewise parametric-memory setup.

Files:

- `training_plan.yaml`: training recipe summary for base, PT-only, and PT + QA-SFT branches.
- `dataset_info.json`: LLaMA-Factory dataset registry entries for stagewise PT and QA-SFT files.
- `dialogue_assignments.json`: dialogue-to-stage assignment manifest.
- `entity_split_assignments.json`: per-stage train/test entity split manifest.

The files are copied from:

```text
dialogue_training/project_entity_split_sw_natural_header_qa_upweight24/
```

Path note:

`dialogue_assignments.json` records the benchmark path with the `${UMB_ROOT}`
placeholder. Set `UMB_ROOT` to the repository root before using this manifest in
local scripts, or replace the placeholder with the local dataset path.
