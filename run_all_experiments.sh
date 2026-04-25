#!/usr/bin/env bash

set -euo pipefail

ROOT="/Users/admin/Desktop/MASTERINGMASTERS"
LOG_DIR="${ROOT}/logs"
NUM_EPOCHS="${1:-8}"
ANALYSIS_EPOCH="${2:-0}"
SPLIT="${3:-validation}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash run_all_experiments.sh [num_epochs] [split]

Runs all 3 x 3 experiment pairs:
  models: distilbert, bert, roberta
  tasks: emotion, clinc_oos, ag_news

Arguments:
  num_epochs: number of training epochs for each run (default: 8)
  split: evaluation split for checkpoint evaluation (default: validation)

Example:
  bash run_all_experiments.sh 8 validation
EOF
  exit 0
fi

mkdir -p "${LOG_DIR}"

total_runs=9
current_run=0
SPLIT="${2:-validation}"

for model in distilbert bert roberta; do
  for task in emotion clinc_oos ag_news; do
    current_run=$((current_run + 1))
    log_file="${LOG_DIR}/${model}_${task}.log"
    echo "=================================================="
    echo "Run ${current_run}/${total_runs}"
    echo "Running model=${model} task=${task} epochs=${NUM_EPOCHS} split=${SPLIT}"
    echo "Log: ${log_file}"
    echo "=================================================="
    bash "${ROOT}/run_experiment.sh" "${model}" "${task}" "${NUM_EPOCHS}" "skip" "${SPLIT}" | tee "${log_file}"
    echo "Finished run ${current_run}/${total_runs}: model=${model} task=${task}"
  done
done

echo "All experiments completed."
