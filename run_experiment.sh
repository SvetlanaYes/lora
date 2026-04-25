#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash run_experiment.sh [model_key] [task_key] [num_epochs] [analysis_epoch] [split]

Examples:
  bash run_experiment.sh distilbert emotion 8 0 validation
  bash run_experiment.sh bert clinc_oos 8 0 validation
  bash run_experiment.sh roberta ag_news 8 0 validation
EOF
  exit 0
fi

MODEL_KEY="${1:-distilbert}"
TASK_KEY="${2:-emotion}"
NUM_EPOCHS="${3:-8}"
ANALYSIS_EPOCH="${4:-0}"
SPLIT="${5:-validation}"

echo "Running training for model=${MODEL_KEY} task=${TASK_KEY} epochs=${NUM_EPOCHS}"
python3 scripts/train.py \
  --model-key "${MODEL_KEY}" \
  --task-key "${TASK_KEY}" \
  --num-epochs "${NUM_EPOCHS}"

echo "Training completed for model=${MODEL_KEY} task=${TASK_KEY}"

echo "Running checkpoint evaluation on split=${SPLIT}"
python3 scripts/evaluate_checkpoints.py \
  --model-key "${MODEL_KEY}" \
  --task-key "${TASK_KEY}" \
  --num-epochs "${NUM_EPOCHS}" \
  --split "${SPLIT}"

echo "Checkpoint evaluation completed for model=${MODEL_KEY} task=${TASK_KEY}"

if [[ "${ANALYSIS_EPOCH}" == "0" ]]; then
  echo "Running weight-update analysis for best validation checkpoint"
  python3 scripts/analyze_updates.py \
    --model-key "${MODEL_KEY}" \
    --task-key "${TASK_KEY}"
elif [[ "${ANALYSIS_EPOCH}" == "skip" ]]; then
  echo "Skipping weight-update analysis for model=${MODEL_KEY} task=${TASK_KEY}"
else
  echo "Running weight-update analysis for epoch=${ANALYSIS_EPOCH}"
  python3 scripts/analyze_updates.py \
    --model-key "${MODEL_KEY}" \
    --task-key "${TASK_KEY}" \
    --epoch "${ANALYSIS_EPOCH}"
fi

echo "Experiment completed for model=${MODEL_KEY} task=${TASK_KEY}"
