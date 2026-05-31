#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LMEVAL_ROOT="${LMEVAL_ROOT:-$ROOT/lm-evaluation-harness}"
PYTHON="${PYTHON:-python3}"

if [[ -n "${MERA_ROOT:-}" && -r "$MERA_ROOT/scripts/cache_env.sh" ]]; then
  # Optional local cache setup for internal clusters.
  # shellcheck source=/dev/null
  source "$MERA_ROOT/scripts/cache_env.sh"
  configure_mera_cache_env "$MERA_ROOT"
fi

PROJECT_HF_ENV="${PROJECT_HF_ENV:-$ROOT/.secrets/hf_env.sh}"
if [[ -z "${HF_TOKEN:-}" && -r "$PROJECT_HF_ENV" ]]; then
  # shellcheck source=/beacon-projects/traumallm/shwaihe/.secrets/hf_env.sh
  source "$PROJECT_HF_ENV"
fi

export PYTHONPATH="$LMEVAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CAPACITY_AWARE_PATCH_PATH="${CAPACITY_AWARE_PATCH_PATH:-$ROOT/capacity_aware/capacity_patch.py}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
TASKS="${TASKS:-hellaswag}"
LIMIT="${LIMIT-0.01}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
PARALLELIZE="${PARALLELIZE:-False}"
CAPACITY_MODE="${CAPACITY_MODE:-baseline}"
EXPERT_CAPACITY="${EXPERT_CAPACITY:-1.0}"
STRATEGY="${STRATEGY:-score}"
ROUNDS="${ROUNDS:-1}"
CAPACITY_SCOPE="${CAPACITY_SCOPE:-expert}"
CAPACITY_DEVICES="${CAPACITY_DEVICES:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
LOG_SAMPLES="${LOG_SAMPLES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/qwen35_capacity_lmeval}"

safe_model="${MODEL_ID//\//__}"
safe_model="${safe_model//./p}"
run_name="${CAPACITY_MODE}_c${EXPERT_CAPACITY}_${STRATEGY}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$safe_model/$run_name}"
mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=$MODEL_ID,dtype=$MODEL_DTYPE,trust_remote_code=$TRUST_REMOTE_CODE,parallelize=$PARALLELIZE"
if [[ "$CAPACITY_MODE" != "baseline" ]]; then
  MODEL_ARGS="$MODEL_ARGS,expert_capacity=$EXPERT_CAPACITY,strategy=$STRATEGY,rounds=$ROUNDS"
  MODEL_ARGS="$MODEL_ARGS,capacity_scope=$CAPACITY_SCOPE"
  if [[ -n "$CAPACITY_DEVICES" ]]; then
    MODEL_ARGS="$MODEL_ARGS,capacity_devices=$CAPACITY_DEVICES"
  fi
fi

echo "=== Qwen3.5 capacity-aware lm-eval ==="
echo "date=$(date)"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "python=$PYTHON"
echo "lmeval_root=$LMEVAL_ROOT"
echo "capacity_patch=$CAPACITY_AWARE_PATCH_PATH"
echo "model_id=$MODEL_ID"
echo "tasks=$TASKS"
echo "limit=${LIMIT:-none}"
echo "capacity_mode=$CAPACITY_MODE"
echo "expert_capacity=$EXPERT_CAPACITY"
echo "strategy=$STRATEGY"
echo "capacity_scope=$CAPACITY_SCOPE"
echo "capacity_devices=${CAPACITY_DEVICES:-auto}"
echo "output_dir=$OUTPUT_DIR"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

CMD=(
  "$PYTHON" -m lm_eval run
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --num_fewshot "$NUM_FEWSHOT" \
  --batch_size "$BATCH_SIZE" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --device "$DEVICE" \
  --output_path "$OUTPUT_DIR"
)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

"${CMD[@]}"

echo "done=$(date)"
