#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/cwu319/OpenOneRec-base}"
RUN_NAME="${RUN_NAME:-baseline_pretrain_sft_full_v1}"
DATASET_CONFIG="${DATASET_CONFIG:-$PROJECT_ROOT/pretrain/examples/dataset_config/sft.one8.base.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/$RUN_NAME/train}"
LAUNCHER_LOG="${LAUNCHER_LOG:-$PROJECT_ROOT/pretrain/output_logs/$RUN_NAME/launcher.log}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$LAUNCHER_LOG")"

nohup env \
    RUN_NAME="$RUN_NAME" \
    DATASET_CONFIG="$DATASET_CONFIG" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    bash "$PROJECT_ROOT/scripts/one8_run_baseline_sft_8npu.sh" \
    > "$LAUNCHER_LOG" 2>&1 < /dev/null &

echo "$!"
