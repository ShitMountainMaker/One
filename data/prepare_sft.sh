#!/bin/bash
# Data splitting script: Merge general text and recommendation data, then split by every 1000 samples

set -e

# Configuration
# General SFT comes from raw_data; recommendation SFT parquet is generated into project output/
GENERAL_TEXT_PATH="../raw_data/general_text/sft"
REC_DATA_PATH="../output"
OUTPUT_DIR="../output/split_data_sft"
MAX_ROWS="${MAX_ROWS:-1000}"
ENGINE="${ENGINE:-pyarrow}"
STREAMING_BATCH_SIZE="${STREAMING_BATCH_SIZE:-5000}"

# Check if paths exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -e "${GENERAL_TEXT_PATH}" ]; then
    echo "Error: General text path does not exist: ${GENERAL_TEXT_PATH}"
    exit 1
fi

if [ ! -e "${REC_DATA_PATH}" ]; then
    echo "Error: Recommendation data path does not exist: ${REC_DATA_PATH}"
    exit 1
fi

# Execute
python3 "${SCRIPT_DIR}/scripts/split_data.py" \
    --general_text_path "${GENERAL_TEXT_PATH}" \
    --rec_data_path "${REC_DATA_PATH}" \
    --rec_glob "sft_*.parquet" \
    --output_dir "${OUTPUT_DIR}" \
    --max_rows "${MAX_ROWS}" \
    --engine "${ENGINE}" \
    --streaming_batch_size "${STREAMING_BATCH_SIZE}"
