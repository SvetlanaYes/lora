# MASTERINGMASTERS

Research workspace for full fine-tuning, checkpoint evaluation, and rank-structure analysis of transformer updates.

## Layout

- `src/masteringmasters/`: reusable experiment code
- `scripts/`: thin CLI entrypoints
- `artifacts/`: run outputs, checkpoints, summaries
- `outputs/analysis/`: optional exported analysis files
- `run_experiment.sh`: sequential training -> evaluation -> rank-analysis pipeline
- `ANALYSIS.md`: explanation of the rank-analysis workflow and metrics
- `THESIS_PHASES.md`: phase-by-phase plan from full fine-tuning to LoRA evaluation

## Thesis Matrix

Models:

- `distilbert` -> `distilbert-base-uncased`
- `bert` -> `bert-base-uncased`
- `roberta` -> `FacebookAI/roberta-base`

Tasks:

- `emotion` -> `dair-ai/emotion`
- `clinc_oos` -> `clinc/clinc_oos` / `plus`
- `ag_news` -> `ag_news`

This gives a 3 x 3 experiment matrix:

- `distilbert x emotion`
- `distilbert x clinc_oos`
- `distilbert x ag_news`
- `bert x emotion`
- `bert x clinc_oos`
- `bert x ag_news`
- `roberta x emotion`
- `roberta x clinc_oos`
- `roberta x ag_news`

## Entry Points

Train:

```bash
python3 scripts/train.py --model-key distilbert --task-key emotion
```

Evaluate checkpoints:

```bash
python3 scripts/evaluate_checkpoints.py --model-key distilbert --task-key emotion --num-epochs 8 --split validation
```

Analyze weight updates:

```bash
python3 scripts/analyze_updates.py \
  --model-key distilbert \
  --task-key emotion
```

By default this analyzes:

- `artifacts/<model>_<task>/best_model/best_model.pth`

To analyze a specific epoch instead:

```bash
python3 scripts/analyze_updates.py \
  --model-key distilbert \
  --task-key emotion \
  --epoch 2
```

Run the full pipeline sequentially:

```bash
bash run_experiment.sh distilbert emotion 8 0 validation
```

Arguments:

- argument 1: model key
- argument 2: task key
- argument 3: number of epochs
- argument 4: checkpoint epoch used for rank analysis, or `0` for best validation checkpoint
- argument 5: evaluation split

Example matrix runs:

```bash
bash run_experiment.sh distilbert emotion 8 0 validation
bash run_experiment.sh bert clinc_oos 8 0 validation
bash run_experiment.sh roberta ag_news 8 0 validation
```

Run all 9 training jobs without SVD analysis:

```bash
bash run_all_experiments.sh 8 validation
```

Artifacts are stored under:

- `artifacts/<model>_<task>/checkpoints/`
- `artifacts/<model>_<task>/best_model/`
- `artifacts/<model>_<task>/training_summary.json`

Best-model outputs:

- `best_model/best_model.pth`: checkpoint with the highest validation accuracy
- `best_model/metrics_report.txt`: epoch-by-epoch train, validation, and test accuracies

If you later want SVD analysis, run it separately with:

```bash
python3 scripts/analyze_updates.py --model-key distilbert --task-key emotion
```

Aggregate analysis across multiple completed runs:

```bash
python3 scripts/aggregate_analysis.py
```

Analysis outputs:

- `artifacts/<model>_<task>/analysis/weight_update_ranks.csv`
- `artifacts/<model>_<task>/analysis/layer_type_summary.csv`
- `artifacts/<model>_<task>/analysis/analysis_summary.json`
- `outputs/analysis/all_runs_layer_metrics.csv`
- `outputs/analysis/all_runs_group_summary.csv`

For metric definitions and interpretation guidance, see [`ANALYSIS.md`](/Users/admin/Desktop/MASTERINGMASTERS/ANALYSIS.md).

Collect full fine-tuning result tables:

```bash
python3 scripts/collect_results.py --core-only
```

Train core fixed-rank LoRA baselines:

```bash
bash run_core_lora_baselines.sh 8
```

Train one adaptive LoRA run:

```bash
python3 scripts/train_lora.py --model-key bert --task-key clinc_oos --mode adaptive --num-epochs 8
```

Validate a LoRA configuration without training:

```bash
python3 scripts/train_lora.py --model-key distilbert --task-key emotion --mode fixed --rank 8 --dry-run
```

Collect LoRA and method-comparison result tables:

```bash
python3 scripts/collect_lora_results.py
python3 scripts/build_method_comparison.py
```

For the full phase plan, see [`THESIS_PHASES.md`](/Users/admin/Desktop/MASTERINGMASTERS/THESIS_PHASES.md).

Split behavior:

- if a dataset already has train/validation/test, those are used directly
- if validation is missing, it is created from train
- if test is missing, it is created from validation
- this gives every experiment explicit train, validation, and test splits

Official split status:

- `emotion`: official train/validation/test
- `clinc_oos`: official train/validation/test
- `ag_news`: official train/test, validation synthesized from train

## Why this refactor

The old code mixed configuration, training, evaluation, and analysis in single-purpose scripts. The new layout keeps the experiment logic reusable so new models, datasets, and analysis variants can be added without cloning entire files.
# lora
