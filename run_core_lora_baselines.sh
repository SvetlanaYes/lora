#!/usr/bin/env bash

set -euo pipefail

ROOT="/Users/admin/Desktop/MASTERINGMASTERS"
NUM_EPOCHS="${1:-8}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash run_core_lora_baselines.sh [num_epochs]

Runs fixed-rank LoRA baselines for core thesis settings only:
  models: distilbert, bert
  tasks: emotion, clinc_oos, ag_news
  ranks: 8, 16

Example:
  bash run_core_lora_baselines.sh 8
EOF
  exit 0
fi

for model in distilbert bert; do
  for task in emotion clinc_oos ag_news; do
    for rank in 8 16; do
      echo "=================================================="
      echo "LoRA fixed baseline: model=${model} task=${task} rank=${rank} epochs=${NUM_EPOCHS}"
      echo "=================================================="
      python3 "${ROOT}/scripts/train_lora.py" \
        --model-key "${model}" \
        --task-key "${task}" \
        --mode fixed \
        --rank "${rank}" \
        --num-epochs "${NUM_EPOCHS}"
    done
  done
done
