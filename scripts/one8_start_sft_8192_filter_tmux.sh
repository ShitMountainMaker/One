#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/cwu319/OpenOneRec-base}"
SESSION_NAME="${SESSION_NAME:-filter-sft-8192}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/pretrain/output_logs/$SESSION_NAME.log}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$LOG_FILE")"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "bash -lc 'cd \"$PROJECT_ROOT\" && bash scripts/one8_prepare_sft_8192_filter.sh 2>&1 | tee \"$LOG_FILE\"'"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Log: $LOG_FILE"
