#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_METADATA="${INPUT_METADATA:-../../raw_data/onerec_data/onerec_bench_release.parquet}"
PID2SID_MAPPING="${PID2SID_MAPPING:-../../raw_data/onerec_data/video_ad_pid2sid.parquet}"
PRODUCT_PID2SID_MAPPING="${PRODUCT_PID2SID_MAPPING:-../../raw_data/onerec_data/product_pid2sid.parquet}"
CAPTION_INPUT="${CAPTION_INPUT:-../../raw_data/onerec_data/pid2caption.parquet}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-../../output}"
SEED="${SEED:-42}"

run_task_if_missing() {
    local task_name=$1
    local script_path=$2
    shift 2
    local output_file="${OUTPUT_BASE_DIR}/sft_${task_name}.parquet"

    if [ -f "${output_file}" ]; then
        echo "[skip] ${task_name}: ${output_file}"
        return 0
    fi

    local temp_dir
    temp_dir="$(mktemp -d)"
    echo "[run] ${task_name}: ${output_file}"

    python3 "${script_path}" --output_dir "${temp_dir}" "$@"

    if [ ! -f "${temp_dir}/train.parquet" ]; then
        echo "[error] ${task_name}: missing ${temp_dir}/train.parquet" >&2
        rm -rf "${temp_dir}"
        return 1
    fi

    mv "${temp_dir}/train.parquet" "${output_file}"
    rm -rf "${temp_dir}"
    ls -lh "${output_file}"
}

mkdir -p "${OUTPUT_BASE_DIR}"

run_task_if_missing "video_rec" "${SCRIPT_DIR}/sft/video_rec.py" \
    --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed "${SEED}"

run_task_if_missing "interactive_rec" "${SCRIPT_DIR}/sft/interactive_rec.py" \
    --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed "${SEED}"

run_task_if_missing "label_cond_rec" "${SCRIPT_DIR}/sft/label_cond_rec.py" \
    --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed "${SEED}"

run_task_if_missing "label_pred" "${SCRIPT_DIR}/sft/label_pred.py" \
    --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed "${SEED}"

run_task_if_missing "ad_rec" "${SCRIPT_DIR}/sft/ad_rec.py" \
    --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed "${SEED}"

run_task_if_missing "product_rec" "${SCRIPT_DIR}/sft/product_rec.py" \
    --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" \
    --product_pid2sid "${PRODUCT_PID2SID_MAPPING}" --seed "${SEED}"

run_task_if_missing "item_understand" "${SCRIPT_DIR}/sft/item_understand.py" \
    --input "${CAPTION_INPUT}" --pid2sid "${PID2SID_MAPPING}" --seed "${SEED}"

run_task_if_missing "rec_reason" "${SCRIPT_DIR}/sft/rec_reason.py" \
    --input "${INPUT_METADATA}"

echo "[done] missing SFT tasks complete"
