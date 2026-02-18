#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

autoawq="${AUTOAWQ:-False}"
pretrained="${PRETRAINED:-./models/OLMoE-1B-7B-0924}"
batch_size="${BATCH_SIZE:-auto}"

if [ ! -d "$pretrained" ]; then
  echo "Model path does not exist: $pretrained"
  exit 1
fi

output_path="${OUTPUT_PATH:-$pretrained}"
mkdir -p "$output_path"

echo "pretrained=$pretrained"

tasks=(openbookqa piqa rte winogrande boolq arc_challenge hellaswag mmlu gsm8k)
fewshots=(0 0 0 5 0 25 10 5 5)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  num_fewshot="${fewshots[$i]}"

  nohup lm_eval \
    --model hf \
    --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype=bfloat16,autoawq=$autoawq \
    --tasks "$task" \
    --num_fewshot "$num_fewshot" \
    --batch_size "$batch_size" \
    --output_path "$output_path/$task.json" \
    > "$output_path/${task}.out" 2>&1

done
