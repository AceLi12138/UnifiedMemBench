# UnifiedMemBench

UnifiedMemBench (UMB) is an event-centric benchmark for evaluating memory in large language models beyond a single long-context recall score. UMB builds a shared synthetic event universe with evolving characters, timestamped life events, and causal links, then derives long-dialogue QA, contextual evaluation, parametric memory evaluation, and retention / forgetting analysis from the same source.

## Repository Status

This repository contains source code, small benchmark artifacts, small result summaries, and figure data. Large benchmark datasets are hosted separately in the UnifiedMemBench HuggingFace collection:

https://huggingface.co/collections/Ace1213812/unifiedmembench-dataset

The collection contains three dataset components:

- `UnifiedMemBench-LongDialogue`: the long-dialogue dataset generated under the UnifiedMemBench framework.
- `UnifiedMemBench-ParametricMemory`: stagewise PT data, QA-SFT data, and held-out / unseen evaluation data for parametric memory.
- `High-density state-tracking evaluation data`: the dense state-tracking benchmark artifact.

## What UMB Evaluates

UMB separates memory evaluation into different information conditions:

- Contextual memory: the model receives the full long dialogue and answers structured QA tasks.
- Parametric memory: the model is trained on stagewise dialogue-derived data and evaluated without the full dialogue transcript.
- Retention and forgetting: later checkpoints are evaluated on earlier stages to measure whether previously written memories remain accessible.
- Dense state-tracking: the model receives a dense list of atomic facts and outputs the required schema field.

The long-dialogue branch uses six QA families:

- IE: Information Extraction
- MSR: Multi-session Reasoning
- ES: Event Summarization
- TR: Temporal Reasoning
- KU: Knowledge Updating
- MA: Memory Arbitration

## Current Repository Layout

```text
UnifiedMemBench/
  README.md
  LICENSE
  DATA_LICENSE
  CITATION.cff
  requirements.txt
  requirements-local.txt
  .env.example
  .gitignore

  configs/
    parametric_stagewise/

  data_gen/
    generate_evolving_personas.py
    generate_life_stories.py
    generate_timestamps4stories.py
    label_characters_qa_v4_knowledgeupdate.py
    validate_stories_timeissues.py
    output/
      personas_1000_v3.json
      stories_v4.json
      stories_v4_characters_qa.json
      stories_v4_validation_report.json

  dialogue_gen_api/
    gen_event_dialogue_api_v8_natural.py
    natural_dialogue/
    quality_improve/
    evaluation/
    output/final/clean_v8_budget_direct/

  dialogue_training/
    prepare_dialogue_project.py
    run_stage_training_pipeline.py
    run_stage_train_eval_range.sh
    run_local_context_eval.py
    run_parallel_context_eval.py
    run_local_memory_eval.py
    run_parallel_memory_eval.py
    run_stagewise_memory_eval.py
    memory_eval_utils.py
    core.py

  state_track/
    README.md
    build_facts.py
    generate_prompts.py
    validate_dataset.py
    eval_api.py
    eval_local.py

  scripts/
    plotting/
    analysis/

  examples/

  results/
    figures/
    retention/
    appendix/
```

## Installation

For API-based construction, cleaning, scoring, and result aggregation:

```bash
conda create -n umb python=3.11 -y
conda activate umb
pip install -r requirements.txt
cp .env.example .env
```

Alternatively, create the same core environment from the provided conda file:

```bash
conda env create -f environment.yml
conda activate umb
cp .env.example .env
```

Fill in the API keys required by the model provider you use. The API-based contextual evaluator and LLM judges read provider credentials from environment variables.

For local long-context model inference, vLLM-based evaluation, or parametric training, also install the optional dependencies below. These should be customized according to the vLLM configuration required by the evaluated model; the following is only an example:

```bash
conda activate umb
pip install -r requirements-local.txt
```

Parametric training uses LLaMA-Factory in the original experiments. Set `LLAMAFACTORY_ROOT` to your local LLaMA-Factory checkout when running the stagewise training scripts.

## Data Components

Small event-source artifacts are included in `data_gen/output/`:

- `personas_1000_v3.json`: synthetic evolving persona records.
- `stories_v4.json`: event-centric story source with timestamped events.
- `stories_v4_characters_qa.json`: QA tasks generated from the event source.
- `stories_v4_validation_report.json`: validation summary for the story source.

The long-dialogue dataset generated under the UnifiedMemBench framework is not stored directly in this GitHub repository because it is a large dataset artifact. It is hosted in the UnifiedMemBench dataset collection:

https://huggingface.co/collections/Ace1213812/unifiedmembench-dataset

Expected clean benchmark path after download:

```text
dialogue_gen_api/output/final/clean_v8_budget_direct/UMB_dialogue_benchmark.json
```

The initial MBTI-style character seed bank is derived from the CharacterChat MBTI-1024 Bank. The raw seed bank is not redistributed in this repository. To regenerate the persona source from a seed bank, provide your own `mbti_1024_bank.json`-style file to `data_gen/generate_evolving_personas.py`.

CharacterChat:

https://github.com/morecry/CharacterChat

CharacterChat paper:

https://arxiv.org/abs/2308.10278

Small schema examples are provided under `examples/`. These files are intended
for quick inspection and lightweight smoke tests; they are not substitutes for
the full benchmark.

Stagewise parametric-memory configuration files are collected under
`configs/parametric_stagewise/`. They mirror the main setup used to generate the
stagewise training and evaluation scaffold.

Dense state-tracking data, prompt-ladder generation, validation, and evaluation
utilities are collected under `state_track/`. See `state_track/README.md` for
the detailed Fact Track Memory workflow and command reference.

## Main Construction Pipeline

### 1. Persona and Event Source

Generate evolving personas from an MBTI-style seed bank:

```bash
python data_gen/generate_evolving_personas.py \
  --bank_file path/to/mbti_1024_bank.json \
  --output_file data_gen/output/personas_1000_v3.json
```

Generate life stories and event histories:

```bash
python data_gen/generate_life_stories.py \
  --input_file data_gen/output/personas_1000_v3.json \
  --output_file data_gen/output/stories_v4.json
```

Add precise timestamps:

```bash
python data_gen/generate_timestamps4stories.py \
  --input_file data_gen/output/stories_v4.json \
  --output_file data_gen/output/stories_v4.json
```

Generate six-family QA tasks:

```bash
python data_gen/label_characters_qa_v4_knowledgeupdate.py \
  --input_file data_gen/output/stories_v4.json \
  --output_file data_gen/output/stories_v4_characters_qa.json
```

### 2. Long Dialogue Generation

Use the natural long-dialogue generator:

```bash
python dialogue_gen_api/gen_event_dialogue_api_v8_natural.py \
  --stories_file data_gen/output/stories_v4.json \
  --qa_file data_gen/output/stories_v4_characters_qa.json \
  --output_file dialogue_gen_api/output/final_dialogues_v8_seg150_tok3000.json
```

The generator uses scene categories, filler topics, dialogue dynamics, director notes, and persona-conditioned speech profiles under `dialogue_gen_api/natural_dialogue/`.

### 3. Coverage Audit, Repair, and Direct Clean

Run strict QA coverage checking:

```bash
python dialogue_gen_api/quality_improve/strict_eval_qa_coverage.py \
  --input dialogue_gen_api/output/final_dialogues_v8_seg150_tok3000.json \
  --output dialogue_gen_api/output/final/eval_report_v8_full_budget.json \
  --use_llm
```

Repair missing QA evidence when applicable:

```bash
python dialogue_gen_api/quality_improve/repair_missing_qa.py
```

Create the final clean benchmark:

```bash
python dialogue_gen_api/quality_improve/generate_clean_dataset.py
```

The final clean release keeps all 1,000 dialogues and removes unsupported QA tasks, reducing 12,112 raw QA tasks to 11,795 valid tasks according to the stored dataset summary.

## Contextual Memory Evaluation

API-based contextual evaluation uses:

```bash
python dialogue_gen_api/evaluation/run_eval.py \
  --benchmark_file dialogue_gen_api/output/final/clean_v8_budget_direct/UMB_dialogue_benchmark.json \
  --model_name mimo-v2-flash \
  --judge_model mimo-v2-flash \
  --output_dir dialogue_gen_api/evaluation/results/mimo-v2-flash
```

The evaluator uses task prompts from `dialogue_gen_api/evaluation/task_prompts_v2.json` and scoring configuration from `dialogue_gen_api/evaluation/task_scoring_v1.json`. It expects structured JSON outputs and reports task-wise scores, task-balanced overall score, and invalid JSON statistics.

Local long-context evaluation uses:

```bash
python dialogue_training/run_local_context_eval.py --help
python dialogue_training/run_parallel_context_eval.py --help
```

These local scripts require GPU inference dependencies from `requirements-local.txt`.

## Parametric Memory and Retention

Prepare stagewise parametric data from the clean benchmark:

```bash
python dialogue_training/prepare_dialogue_project.py \
  --dataset_path dialogue_gen_api/output/final/clean_v8_budget_direct/UMB_dialogue_benchmark.json \
  --output_dir dialogue_training/project_entity_split_sw_natural_header_qa_upweight24 \
  --project_mode entity_split \
  --num_stages 10 \
  --chunking_mode sliding_window \
  --sliding_window_overlap_tokens 1024 \
  --pt_header_style natural \
  --qa_sampling_mode role_balanced_upweight \
  --qa_max_samples_per_character 24
```

The generated PT, QA-SFT, and held-out / unseen evaluation JSONL files are distributed through `UnifiedMemBench-ParametricMemory`:

https://huggingface.co/datasets/Ace1213812/UnifiedMemBench-ParametricMemory

The public parametric-memory dataset includes PT, SFT, and unseen evaluation files. See `dialogue_training/project_entity_split_sw_natural_header_qa_upweight24/train_pt/README.md`.

Run stagewise training / evaluation:

```bash
bash dialogue_training/run_stage_train_eval_range.sh --help
python dialogue_training/run_stagewise_memory_eval.py --help
python dialogue_training/run_parallel_memory_eval.py --help
```

## Result Figures and Tables

Small result summaries and figure data are stored under `results/`.

```text
results/figures/paper_images/
```

Plotting and analysis scripts are in:

```text
scripts/plotting/
scripts/analysis/
```

## License

Code in this repository is licensed under the Apache License 2.0. Dataset artifacts are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE` and `DATA_LICENSE`.

## Intended Use and Limitations

UMB is intended for controlled research on LLM memory evaluation. It should not be used as a source of real user histories, as a psychological assessment tool, or as evidence that a model is safe for deployment in personalized memory systems. The benchmark uses synthetic characters and generated events; synthetic persona records should not be interpreted as real individuals.

## Acknowledgements

The initial MBTI-style seed source used in persona construction is derived from the CharacterChat MBTI-1024 Bank. We thank the CharacterChat authors for making their data publicly available.
