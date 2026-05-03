#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-${ROOT}/../LlamaFactory}"
PROJECT_DIR="${ROOT}/dialogue_training/project_entity_split_sw_natural_header_qa_upweight24"
MERGED_ROOT="${ROOT}/dialogue_training/merged_checkpoints/glm4_9b"

RUN_PREFIX="entitysplit_sw_nh_qa_upweight24"
TEMPLATE="glm4"

PT_GPUS="0,1,2,3,4,5,6,7"
SFT_GPUS="0,1,2,3"
EVAL_GPUS="0,1,2,3,4,5,6,7"
TRAIN_CONDA_ENV="umb_lora"
EVAL_CONDA_ENV="glm_official"

JUDGE_MODEL="mimo-v2-flash"
VLLM_GPU_MEMORY_UTILIZATION="0.75"
VLLM_MAX_MODEL_LEN="1536"
VLLM_MAX_NUM_SEQS="1"
VLLM_TENSOR_PARALLEL_SIZE="1"
VLLM_SEED="0"
VLLM_TOKENIZER_MODE="slow"
VLLM_TRUST_REMOTE_CODE="true"

PT_NAME_SUFFIX="8gpu"
SFT_NAME_SUFFIX="8gpupt_4gpusft"

PT_NUM_TRAIN_EPOCHS="3.0"
PT_LEARNING_RATE="1e-5"
SFT_NUM_TRAIN_EPOCHS="1.0"
SFT_LEARNING_RATE="5e-6"

START_STAGE=""
END_STAGE=""
INITIAL_MODEL_PATH=""
TOKENIZER_NAME_OR_PATH=""
EVAL_STAGE_FROM="1"
EVAL_STAGE_TO=""
EVAL_OUTPUT_ROOT=""
EVAL_NAME_SUFFIX=""
MANIFEST_ROOT=""
DISABLE_THINKING="0"

function usage() {
  cat <<'EOF'
Usage:
  bash dialogue_training/run_stage_train_eval_range.sh \
    --start-stage 4 \
    --end-stage 10 \
    --initial-model-path /abs/path/to/stage_03_final_model

Required:
  --start-stage INT
  --end-stage INT
  --initial-model-path PATH
  --tokenizer-name-or-path PATH

Optional:
  --root PATH
  --llamafactory-root PATH
  --project-dir PATH
  --merged-root PATH
  --run-prefix TEXT
  --template TEXT
  --pt-gpus IDS
  --sft-gpus IDS
  --eval-gpus IDS
  --train-conda-env NAME
  --eval-conda-env NAME
  --eval-output-root PATH
  --eval-name-suffix TEXT
  --manifest-root PATH
  --disable-thinking
  --vllm-tokenizer-mode TEXT
  --vllm-trust-remote-code / --no-vllm-trust-remote-code
  --eval-stage-from INT
    Default: 1
    For each checkpoint_stage=n, evaluation runs stages [eval-stage-from, n].
  --eval-stage-to INT
    Optional hard cap. If set, for checkpoint_stage=n evaluation runs
    stages [eval-stage-from, min(eval-stage-to, n)].
  --pt-name-suffix TEXT
  --sft-name-suffix TEXT
  --pt-num-train-epochs FLOAT
  --pt-learning-rate FLOAT
  --sft-num-train-epochs FLOAT
  --sft-learning-rate FLOAT

Behavior:
  1. Trains each stage sequentially with the previous stage final model as input.
  2. After each stage, runs stagewise evaluation for all learned stages by default.
  3. Training and evaluation both reuse existing outputs when present and skip them.
EOF
}

function ensure_conda() {
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
}

function stage_tag() {
  printf "stage_%02d" "$1"
}

function format_lr_label() {
  printf "%.0e" "$1" | tr -d '+-'
}

function format_epoch_label() {
  local value="$1"
  if [[ "${value}" =~ ^[0-9]+\.0+$ ]]; then
    printf "%s" "${value%%.*}"
  elif [[ "${value}" == *.* ]]; then
    printf "%s" "${value//./p}"
  else
    printf "%s" "${value}"
  fi
}

function final_model_path_for_stage() {
  local stage="$1"
  local tag
  tag="$(stage_tag "${stage}")"
  printf "%s/%s_%s_fullptqa_e%s_lr%s_%s" \
    "${MERGED_ROOT}" \
    "${RUN_PREFIX}" \
    "${tag}" \
    "$(format_epoch_label "${SFT_NUM_TRAIN_EPOCHS}")" \
    "$(format_lr_label "${SFT_LEARNING_RATE}")" \
    "${SFT_NAME_SUFFIX}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-stage) START_STAGE="$2"; shift 2 ;;
    --end-stage) END_STAGE="$2"; shift 2 ;;
    --initial-model-path) INITIAL_MODEL_PATH="$2"; shift 2 ;;
    --tokenizer-name-or-path) TOKENIZER_NAME_OR_PATH="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --llamafactory-root) LLAMAFACTORY_ROOT="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --merged-root) MERGED_ROOT="$2"; shift 2 ;;
    --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
    --template) TEMPLATE="$2"; shift 2 ;;
    --pt-gpus) PT_GPUS="$2"; shift 2 ;;
    --sft-gpus) SFT_GPUS="$2"; shift 2 ;;
    --eval-gpus) EVAL_GPUS="$2"; shift 2 ;;
    --train-conda-env) TRAIN_CONDA_ENV="$2"; shift 2 ;;
    --eval-conda-env) EVAL_CONDA_ENV="$2"; shift 2 ;;
    --eval-output-root) EVAL_OUTPUT_ROOT="$2"; shift 2 ;;
    --eval-name-suffix) EVAL_NAME_SUFFIX="$2"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="$2"; shift 2 ;;
    --disable-thinking) DISABLE_THINKING="1"; shift 1 ;;
    --eval-stage-from) EVAL_STAGE_FROM="$2"; shift 2 ;;
    --eval-stage-to) EVAL_STAGE_TO="$2"; shift 2 ;;
    --pt-name-suffix) PT_NAME_SUFFIX="$2"; shift 2 ;;
    --sft-name-suffix) SFT_NAME_SUFFIX="$2"; shift 2 ;;
    --pt-num-train-epochs) PT_NUM_TRAIN_EPOCHS="$2"; shift 2 ;;
    --pt-learning-rate) PT_LEARNING_RATE="$2"; shift 2 ;;
    --sft-num-train-epochs) SFT_NUM_TRAIN_EPOCHS="$2"; shift 2 ;;
    --sft-learning-rate) SFT_LEARNING_RATE="$2"; shift 2 ;;
    --vllm-tokenizer-mode) VLLM_TOKENIZER_MODE="$2"; shift 2 ;;
    --vllm-trust-remote-code) VLLM_TRUST_REMOTE_CODE="true"; shift 1 ;;
    --no-vllm-trust-remote-code) VLLM_TRUST_REMOTE_CODE="false"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${START_STAGE}" || -z "${END_STAGE}" || -z "${INITIAL_MODEL_PATH}" ]]; then
  usage
  exit 1
fi

if (( START_STAGE <= 0 || END_STAGE <= 0 )); then
  echo "start-stage and end-stage must be positive." >&2
  exit 1
fi

if (( START_STAGE > END_STAGE )); then
  echo "start-stage cannot be greater than end-stage." >&2
  exit 1
fi

if [[ -z "${EVAL_OUTPUT_ROOT}" ]]; then
  EVAL_OUTPUT_ROOT="${PROJECT_DIR}/outputs/eval_stagewise"
fi

if [[ -z "${MANIFEST_ROOT}" ]]; then
  MANIFEST_ROOT="${PROJECT_DIR}/outputs/train_pipeline"
fi

ensure_conda

current_model="${INITIAL_MODEL_PATH}"

for stage in $(seq "${START_STAGE}" "${END_STAGE}"); do
  tag="$(stage_tag "${stage}")"
  echo "===== TRAIN ${tag} ====="
  cd "${ROOT}"
  conda activate "${TRAIN_CONDA_ENV}"

  python dialogue_training/run_stage_training_pipeline.py \
    --llamafactory_root "${LLAMAFACTORY_ROOT}" \
    --project_dir "${PROJECT_DIR}" \
    --merged_root "${MERGED_ROOT}" \
    --initial_model_path "${current_model}" \
    --start_stage "${stage}" \
    --end_stage "${stage}" \
    --run_prefix "${RUN_PREFIX}" \
    --pt_cuda_visible_devices "${PT_GPUS}" \
    --sft_cuda_visible_devices "${SFT_GPUS}" \
    --template "${TEMPLATE}" \
    --pt_num_train_epochs "${PT_NUM_TRAIN_EPOCHS}" \
    --pt_learning_rate "${PT_LEARNING_RATE}" \
    --sft_num_train_epochs "${SFT_NUM_TRAIN_EPOCHS}" \
    --sft_learning_rate "${SFT_LEARNING_RATE}" \
    --pt_name_suffix "${PT_NAME_SUFFIX}" \
    --sft_name_suffix "${SFT_NAME_SUFFIX}" \
    --manifest_root "${MANIFEST_ROOT}"

  current_model="$(final_model_path_for_stage "${stage}")"

  echo "===== EVAL ${tag} ====="
  cd "${ROOT}"
  conda activate "${EVAL_CONDA_ENV}"
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/dialogue_gen_api/evaluation/.env"
  set +a
  unset VLLM_ATTENTION_BACKEND

  eval_stage_to="${stage}"
  if [[ -n "${EVAL_STAGE_TO}" && "${EVAL_STAGE_TO}" -lt "${eval_stage_to}" ]]; then
    eval_stage_to="${EVAL_STAGE_TO}"
  fi

  python dialogue_training/run_stagewise_memory_eval.py \
    --project_dir "${PROJECT_DIR}" \
    --model_name_or_path "${current_model}" \
    $([[ -n "${TOKENIZER_NAME_OR_PATH}" ]] && printf '%s %q' '--tokenizer_name_or_path' "${TOKENIZER_NAME_OR_PATH}") \
    --checkpoint_stage "${stage}" \
    --output_root "${EVAL_OUTPUT_ROOT}" \
    --stage_from "${EVAL_STAGE_FROM}" \
    --stage_to "${eval_stage_to}" \
    --gpu_ids "${EVAL_GPUS}" \
    --judge_model "${JUDGE_MODEL}" \
    --backend vllm \
    --vllm_tokenizer_mode "${VLLM_TOKENIZER_MODE}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --vllm_seed "${VLLM_SEED}" \
    --batch_size 4 \
    --max_input_tokens 1024 \
    --max_new_tokens 512 \
    $([[ "${VLLM_TRUST_REMOTE_CODE}" == "true" ]] && printf '%s' '--vllm_trust_remote_code') \
    $([[ -n "${EVAL_NAME_SUFFIX}" ]] && printf '%s %q' '--name_suffix' "${EVAL_NAME_SUFFIX}") \
    $([[ "${DISABLE_THINKING}" == "1" ]] && printf '%s' '--disable_thinking')

  echo "===== DONE ${tag} ====="
done

echo "All stages completed: ${START_STAGE} -> ${END_STAGE}"
