# Analysis Workflow

This document explains how the rank-analysis stage works, what files it produces, and how the metrics should be interpreted in the thesis.

## Goal

The analysis stage studies the update matrix

- `ΔW = W_finetuned - W_pretrained`

for each weight matrix in the fine-tuned model. The purpose is not only to say whether updates are "low-rank", but to measure how compressible they are and whether that compressibility changes across:

- layers
- layer types
- tasks
- models

This is the empirical basis for later adaptive LoRA design.

## Run Analysis For One Experiment

Use the best validation checkpoint by default:

```bash
python3 scripts/analyze_updates.py --model-key distilbert --task-key emotion
```

Analyze a specific epoch checkpoint:

```bash
python3 scripts/analyze_updates.py --model-key distilbert --task-key emotion --epoch 2
```

## Aggregate Analysis Across Runs

After per-run analysis files exist, combine them into a single cross-run table:

```bash
python3 scripts/aggregate_analysis.py
```

## Output Files

For each run, analysis outputs are saved under:

- `artifacts/<model>_<task>/analysis/`

The main files are:

- `weight_update_ranks.csv`
- `layer_type_summary.csv`
- `analysis_summary.json`

Cross-run outputs are saved under:

- `outputs/analysis/`

The main files are:

- `all_runs_layer_metrics.csv`
- `all_runs_group_summary.csv`

## Per-Layer Metrics

The main per-layer CSV includes one row per update matrix.

Key columns:

- `layer_name`: exact parameter name from the model
- `layer_type`: coarse semantic grouping such as `attention`, `ffn`, `classifier`, `embeddings`, or `other`
- `layer_depth`: transformer block index when available
- `rows`, `cols`: matrix shape after reshaping to 2D
- `full_rank`: maximum possible matrix rank

Rank metrics:

- `effective_rank`: entropy-based rank; useful for describing how spread the singular values are
- `stable_rank`: Frobenius-norm-based rank surrogate; robust and standard in matrix analysis
- `participation_ratio`: energy-distribution rank surrogate; larger values indicate less concentrated singular spectra
- `rank_at_90`, `rank_at_95`, `rank_at_99`: smallest truncated rank preserving 90%, 95%, or 99% of squared singular-value energy

Normalized metrics:

- `effective_rank_ratio`
- `stable_rank_ratio`
- `participation_ratio_ratio`
- `rank_at_90_ratio`
- `rank_at_95_ratio`
- `rank_at_99_ratio`

These divide by `full_rank` so differently sized layers can be compared more fairly.

Magnitude metric:

- `frobenius_norm`: overall size of the update matrix

Spectral scale metric:

- `top_singular_value`: largest singular value of the update matrix

## Reconstruction-Error Metrics

The analysis also computes truncated-SVD approximation errors for fixed ranks:

- `relative_error_r4`
- `relative_error_r8`
- `relative_error_r16`
- `relative_error_r32`
- `relative_error_r64`

Interpretation:

- smaller reconstruction error means the update matrix can be represented well at that rank
- if error drops quickly at small ranks, the update is strongly compressible
- if error stays large until higher ranks, a uniform low-rank assumption may be too aggressive for that layer

These metrics are especially useful because they are easy to connect to LoRA rank budgets.

## Layer-Type Summary

The grouped summary CSV averages metrics by `layer_type`. This is helpful when writing results such as:

- attention layers are consistently more compressible than feed-forward layers
- classifier heads behave differently from backbone layers
- one task produces higher-rank updates in later layers than another

## Recommended Thesis Use

The strongest way to use these outputs is:

1. Compare per-layer rank profiles within one model-task run.
2. Compare the same layer type across tasks for one model.
3. Compare the same task across models.
4. Use reconstruction error to justify whether a fixed LoRA rank is adequate.

## What To Claim Carefully

You should avoid claiming that one metric alone defines the "true" rank of a fine-tuning update.

A stronger claim is:

- multiple standard rank surrogates consistently indicate that update compressibility varies by layer, task, and model

That is a much more defensible thesis statement than relying on a single handcrafted heuristic.
