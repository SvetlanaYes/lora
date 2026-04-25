# Thesis Phase Plan

This file describes the practical workflow from the current full fine-tuning results to the LoRA comparison stage.

## Scope Decision

For now, RoBERTa is optional. The core thesis experiments should use:

- `distilbert`
- `bert`

with the three tasks:

- `emotion`
- `clinc_oos`
- `ag_news`

RoBERTa can be reintroduced after its unstable runs are cleaned up.

## Phase 1: Full Fine-Tuning Results

Goal:

- establish strong full fine-tuning baselines
- save per-epoch checkpoints
- select the best validation checkpoint

Collect the result table:

```bash
python3 scripts/collect_results.py --core-only
```

Output:

- `outputs/tables/full_finetuning_results.csv`

## Phase 2: Rank and Compressibility Analysis

Goal:

- compute rank metrics for `ΔW = W_finetuned - W_pretrained`
- compare layer types and tasks
- measure truncated-SVD reconstruction errors

Run analysis per completed run:

```bash
python3 scripts/analyze_updates.py --model-key distilbert --task-key emotion
```

Aggregate analysis:

```bash
python3 scripts/aggregate_analysis.py
```

Outputs:

- `artifacts/<model>_<task>/analysis/weight_update_ranks.csv`
- `artifacts/<model>_<task>/analysis/layer_type_summary.csv`
- `outputs/analysis/all_runs_layer_metrics.csv`
- `outputs/analysis/all_runs_group_summary.csv`

Main thesis question:

- Are update ranks uniform across layers and tasks?

Expected current signal:

- FFN layers tend to be less compressible than attention layers.
- CLINC OOS tends to produce higher-rank updates than Emotion.

## Phase 3: Fixed-Rank LoRA Baselines

Goal:

- establish standard LoRA baselines before proposing adaptive rank allocation

Default comparison:

- fixed `r = 8`
- fixed `r = 16`

Run all core fixed-rank LoRA baselines:

```bash
bash run_core_lora_baselines.sh 8
```

Check that a LoRA configuration builds correctly without training:

```bash
python3 scripts/train_lora.py --model-key distilbert --task-key emotion --mode fixed --rank 8 --dry-run
```

Output root:

- `artifacts_lora/`

Each LoRA run writes:

- `checkpoints/`
- `best_model/best_model.pth`
- `best_model/adapter/`
- `best_model/metrics_report.txt`
- `lora_training_summary.json`

## Phase 4: Adaptive LoRA

Goal:

- allocate different LoRA ranks to different target layers using full fine-tuning rank-analysis signals

Current adaptive rule:

- use `rank_at_90_ratio` from `weight_update_ranks.csv`
- map each target layer to ranks between `4` and `32`
- snap to common LoRA ranks

Example:

```bash
python3 scripts/train_lora.py \
  --model-key bert \
  --task-key clinc_oos \
  --mode adaptive \
  --rank 8 \
  --min-rank 4 \
  --max-rank 32 \
  --num-epochs 8
```

Dry-run an adaptive configuration before training:

```bash
python3 scripts/train_lora.py \
  --model-key bert \
  --task-key clinc_oos \
  --mode adaptive \
  --dry-run
```

The `--rank` value remains the fallback/default rank. The adaptive per-layer ranks come from the analysis CSV.

## Phase 5: Evaluation

Compare:

- full fine-tuning
- fixed LoRA `r = 8`
- fixed LoRA `r = 16`
- adaptive LoRA

Report:

- validation accuracy
- test accuracy
- trainable parameters
- trainable parameter percentage
- assigned rank pattern for adaptive LoRA

Collect LoRA result rows:

```bash
python3 scripts/collect_lora_results.py
```

Build one comparison table containing full fine-tuning and LoRA:

```bash
python3 scripts/build_method_comparison.py
```

Outputs:

- `outputs/tables/lora_results.csv`
- `outputs/tables/method_comparison.csv`

## Phase 6: Thesis Interpretation

The main interpretation should connect three pieces:

1. Full fine-tuning updates have nonuniform rank structure.
2. Fixed-rank LoRA gives every target layer the same capacity.
3. Adaptive LoRA tests whether rank-analysis signals can guide capacity allocation.

Careful claim:

- The rank analysis motivates adaptive LoRA.

Avoid overclaiming:

- Do not claim that LoRA must reconstruct the full fine-tuning update exactly.
