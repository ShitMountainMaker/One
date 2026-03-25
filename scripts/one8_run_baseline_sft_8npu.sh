#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/cwu319/OpenOneRec-base}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/OneRec-1.7B-pretrain}"
DATASET_CONFIG="${DATASET_CONFIG:-$PROJECT_ROOT/pretrain/examples/dataset_config/sft.one8.base.json}"
RUN_NAME="${RUN_NAME:-baseline_pretrain_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/$RUN_NAME/train}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

MAX_LENGTH="${MAX_LENGTH:-8192}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MIN_LR="${MIN_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-100}"
NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-1000}"
SAVE_CHECKPOINT_PER_STEP="${SAVE_CHECKPOINT_PER_STEP:-50}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-2048}"
LOGGING_PER_STEP="${LOGGING_PER_STEP:-5}"
SEED="${SEED:-19260817}"

cd "$PROJECT_ROOT"
source .env/activate_onerec_npu.sh
cd "$PROJECT_ROOT/pretrain"

mkdir -p "$OUTPUT_DIR"
mkdir -p /tmp/_wids_cache

export PYTHONPATH="$PWD:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

LOG_DIR="$PROJECT_ROOT/pretrain/output_logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "MODEL_DIR=$MODEL_DIR"
echo "DATASET_CONFIG=$DATASET_CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_DIR=$LOG_DIR"

python3 -m torch.distributed.run \
  --nnodes "$NNODES" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  recipes/train_qwen3.py \
  --model_dir "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_config "$DATASET_CONFIG" \
  --use_tie_weights \
  --model_class Qwen3ForCausalLM \
  --monitor_datasource_loss \
  --monitor_datasource_cnt \
  --max_length "$MAX_LENGTH" \
  --learning_rate "$LEARNING_RATE" \
  --min_lr "$MIN_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --lr_scheduler_type cosine \
  --num_warmup_steps "$NUM_WARMUP_STEPS" \
  --num_training_steps "$NUM_TRAINING_STEPS" \
  --save_checkpoint_per_step "$SAVE_CHECKPOINT_PER_STEP" \
  --minibatch_size "$MINIBATCH_SIZE" \
  --logging_per_step "$LOGGING_PER_STEP" \
  --use_fp32_weight \
  --seed "$SEED" \
  --enable_gradient_checkpointing \
  --use_chunked_loss_computer \
  2>&1 | tee "$LOG_DIR/torchrun.log"
