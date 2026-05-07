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

## Environment Variables

Before running the examples below, set the repository root and the main dataset
paths from the repository root directory:

```bash
export UMB_ROOT=$(pwd)
export UMB_BENCHMARK=${UMB_ROOT}/dialogue_gen_api/output/final/clean_v8_budget_direct/UMB_dialogue_benchmark.json
export UMB_PARAMETRIC_PROJECT=${UMB_ROOT}/dialogue_training/project_entity_split_sw_natural_header_qa_upweight24
```

For local model evaluation or training, also set model-specific paths:

```bash
export MODEL_PATH=/path/to/local_or_hf_model
export TOKENIZER_PATH=/path/to/tokenizer_or_base_model
export LLAMAFACTORY_ROOT=/path/to/LLaMA-Factory
```

Provider API keys should be placed in `.env` or exported in the shell. Do not
commit `.env` or private model paths.

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

For local long-context evaluation, first verify the script entry points:

```bash
python dialogue_training/run_local_context_eval.py --help
python dialogue_training/run_parallel_context_eval.py --help
```

These local scripts require GPU inference dependencies from `requirements-local.txt`.
The exact model path, context length, tensor parallelism, and vLLM settings
should be adjusted for the evaluated model and available GPUs. A minimal
single-process vLLM template is:

```bash
python dialogue_training/run_local_context_eval.py \
  --dataset_path "${UMB_BENCHMARK}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir results/contextual/local_model_smoke \
  --backend vllm \
  --batch_size 1 \
  --max_samples 20 \
  --max_input_tokens 262144 \
  --max_new_tokens 1024 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 262144 \
  --vllm_max_num_seqs 1 \
  --vllm_trust_remote_code
```

For multi-GPU sharded contextual evaluation:

```bash
python dialogue_training/run_parallel_context_eval.py \
  --dataset_path "${UMB_BENCHMARK}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir results/contextual/local_model_parallel \
  --gpu_ids 0,1,2,3 \
  --gpus_per_shard 1 \
  --backend vllm \
  --batch_size 1 \
  --max_samples 100 \
  --max_input_tokens 262144 \
  --max_new_tokens 1024 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 262144 \
  --vllm_max_num_seqs 1 \
  --vllm_trust_remote_code
```

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

For parametric-memory evaluation, first verify the script entry points:

```bash
bash dialogue_training/run_stage_train_eval_range.sh --help
python dialogue_training/run_stagewise_memory_eval.py --help
python dialogue_training/run_parallel_memory_eval.py --help
```

If a trained checkpoint is already available, evaluate one checkpoint against
the learned stages with:

```bash
python dialogue_training/run_stagewise_memory_eval.py \
  --project_dir "${UMB_PARAMETRIC_PROJECT}" \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --checkpoint_stage 10 \
  --stage_from 1 \
  --stage_to 10 \
  --splits unseen \
  --gpu_ids 0,1,2,3 \
  --output_root results/parametric/local_model \
  --backend vllm \
  --batch_size 4 \
  --max_input_tokens 1024 \
  --max_new_tokens 512 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 1536 \
  --vllm_max_num_seqs 1 \
  --vllm_trust_remote_code
```

To run memory-only evaluation on one JSONL split directly:

```bash
python dialogue_training/run_parallel_memory_eval.py \
  --dataset_path "${UMB_PARAMETRIC_PROJECT}/eval_unseen/memory_eval_stage_10.jsonl" \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --output_dir results/parametric/local_model/stage_10_unseen \
  --gpu_ids 0,1,2,3 \
  --backend vllm \
  --batch_size 4 \
  --max_input_tokens 1024 \
  --max_new_tokens 512 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 1536 \
  --vllm_max_num_seqs 1 \
  --vllm_trust_remote_code
```

Full stagewise training depends on LLaMA-Factory and local GPU resources. The
following is a template that should be customized before use:

```bash
export BASE_MODEL=/path/to/base_model

bash dialogue_training/run_stage_train_eval_range.sh \
  --start-stage 1 \
  --end-stage 10 \
  --initial-model-path "${BASE_MODEL}" \
  --tokenizer-name-or-path "${TOKENIZER_PATH}" \
  --project-dir "${UMB_PARAMETRIC_PROJECT}" \
  --train-conda-env umb \
  --eval-conda-env umb \
  --pt-gpus 0,1,2,3,4,5,6,7 \
  --sft-gpus 0,1,2,3 \
  --eval-gpus 0,1,2,3,4,5,6,7 \
  --vllm-tokenizer-mode slow \
  --vllm-trust-remote-code
```

## Dense State-Tracking

The dense state-tracking branch lives under `state_track/`. It builds atomic
facts from the event source, clusters them into state schemas, and generates the
`schema_longest` prompt ladder used for high-density state-tracking evaluation.

The release artifact is distributed as `UnifiedMemBench-StateTracking`:

https://huggingface.co/datasets/Ace1213812/UnifiedMemBench-StateTracking

Typical entry points:

```bash
cd state_track
python validate_dataset.py
python eval_api.py --provider mimo --input fact_track_schema_longest --num-samples 5
python eval_local.py --dry-run --input fact_track_schema_longest
```

See `state_track/README.md` for the full build path, dataset contract, prompt
schema, and evaluation options.

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
