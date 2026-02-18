#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

AUTOAWQ="${AUTOAWQ:-False}"
PRETRAINED="${PRETRAINED:-./models/OLMoE-1B-7B-0924}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

if [[ ! -d "${PRETRAINED}" ]]; then
  echo "Model path does not exist: ${PRETRAINED}" >&2
  exit 1
fi

OUTPUT_PATH="${OUTPUT_PATH:-${PRETRAINED}}"
mkdir -p "${OUTPUT_PATH}"

echo "PRETRAINED=${PRETRAINED}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "AUTOAWQ=${AUTOAWQ}"

TASKS=(openbookqa piqa rte winogrande boolq arc_challenge hellaswag mmlu gsm8k)
FEWSHOTS=(0 0 0 5 0 25 10 5 5)

run_task() {
  local task="$1"
  local num_fewshot="$2"
  local task_json="${OUTPUT_PATH}/${task}.json"
  local task_log="${OUTPUT_PATH}/${task}.out"

  nohup lm_eval \
    --model hf \
    --model_args "pretrained=${PRETRAINED},parallelize=True,trust_remote_code=True,dtype=bfloat16,autoawq=${AUTOAWQ}" \
    --tasks "$task" \
    --num_fewshot "$num_fewshot" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${task_json}" \
    > "${task_log}" 2>&1
}

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  num_fewshot="${FEWSHOTS[$i]}"
  echo "Running task=${task}, fewshot=${num_fewshot}"
  run_task "${task}" "${num_fewshot}"
done
