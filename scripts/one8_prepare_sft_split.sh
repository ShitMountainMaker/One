#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/cwu319/OpenOneRec-base}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/data/user/cwu319/OpenOneRec/.env/activate_onerec_npu.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/split_data_sft}"
MAX_ROWS="${MAX_ROWS:-1000}"
ENGINE="${ENGINE:-pyarrow}"
STREAMING_BATCH_SIZE="${STREAMING_BATCH_SIZE:-5000}"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/output_logs"

if [ -d "$OUTPUT_DIR" ]; then
    mv "$OUTPUT_DIR" "${OUTPUT_DIR}.bak.$(date +%Y%m%d%H%M%S)"
fi

source "$ENV_ACTIVATE"

cd "$PROJECT_ROOT/data"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "STREAMING_BATCH_SIZE=$STREAMING_BATCH_SIZE"

python3 scripts/split_data.py \
    --general_text_path ../raw_data/general_text/sft \
    --rec_data_path ../output \
    --rec_glob 'sft_*.parquet' \
    --output_dir "$OUTPUT_DIR" \
    --max_rows "$MAX_ROWS" \
    --engine "$ENGINE" \
    --streaming_batch_size "$STREAMING_BATCH_SIZE"
