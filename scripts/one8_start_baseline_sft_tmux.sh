#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/cwu319/OpenOneRec-base}"
RUN_NAME="${RUN_NAME:-baseline_pretrain_sft_8192_filtered_v1}"
SESSION_NAME="${SESSION_NAME:-$RUN_NAME}"
DATASET_CONFIG="${DATASET_CONFIG:-$PROJECT_ROOT/pretrain/examples/dataset_config/sft.one8.filtered8192.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/$RUN_NAME/train}"
MASTER_PORT="${MASTER_PORT:-29514}"
SAVE_CHECKPOINT_PER_STEP="${SAVE_CHECKPOINT_PER_STEP:-500}"
LOGGING_PER_STEP="${LOGGING_PER_STEP:-5}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-500}"
NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-5000}"

cd "$PROJECT_ROOT"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "bash -lc 'cd \"$PROJECT_ROOT\" && env \
RUN_NAME=\"$RUN_NAME\" \
DATASET_CONFIG=\"$DATASET_CONFIG\" \
OUTPUT_DIR=\"$OUTPUT_DIR\" \
SAVE_CHECKPOINT_PER_STEP=\"$SAVE_CHECKPOINT_PER_STEP\" \
LOGGING_PER_STEP=\"$LOGGING_PER_STEP\" \
NUM_WARMUP_STEPS=\"$NUM_WARMUP_STEPS\" \
NUM_TRAINING_STEPS=\"$NUM_TRAINING_STEPS\" \
MASTER_PORT=\"$MASTER_PORT\" \
RESUME_FROM=\"${RESUME_FROM:-}\" \
RESUME_FROM_TAG=\"${RESUME_FROM_TAG:-}\" \
RESUME_TRAINING_STATE=\"${RESUME_TRAINING_STATE:-0}\" \
bash scripts/one8_run_baseline_sft_8npu.sh'"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Train log: $PROJECT_ROOT/pretrain/output_logs/$RUN_NAME/torchrun.log"
echo "Output dir: $OUTPUT_DIR"
