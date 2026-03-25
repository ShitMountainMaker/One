#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/cwu319/OpenOneRec-base}"
INPUT_FILE_LIST="${INPUT_FILE_LIST:-$PROJECT_ROOT/output/split_data_sft/file_list.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/split_data_sft_8192}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$PROJECT_ROOT/models/OneRec-1.7B-pretrain}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-1000}"

cd "$PROJECT_ROOT"
source .env/activate_onerec_npu.sh

python3 data/scripts/filter_sft_by_length.py \
  --input-file-list "$INPUT_FILE_LIST" \
  --output-dir "$OUTPUT_DIR" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --max-length "$MAX_LENGTH" \
  --rows-per-shard "$ROWS_PER_SHARD"
