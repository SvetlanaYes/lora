# Dataset Splits

This document records the dataset split policy used in the MASTERINGMASTERS experiments so the methodology can be described clearly in the thesis.

## Overview

The project currently uses three text classification tasks:

- `Emotion` (`dair-ai/emotion`)
- `CLINC OOS` (`clinc/clinc_oos`, configuration: `plus`)
- `AG News` (`ag_news`)

For every experiment, the training pipeline operates with three splits:

- `train`
- `validation`
- `test`

When a dataset already provides official `train`, `validation`, and `test` splits, those splits are used directly. When one split is missing, the code creates the missing split deterministically from the available labeled data using a fixed random seed.

## Split Policy

The split preparation logic is implemented in:

- [`data.py`](/Users/admin/Desktop/MASTERINGMASTERS/src/masteringmasters/data.py)

The current rules are:

1. If `train`, `validation`, and `test` already exist, use them unchanged.
2. If `train` and `validation` exist but `test` does not, split the validation set into new `validation` and `test` subsets.
3. If `train` and `test` exist but `validation` does not, split the training set into new `train` and `validation` subsets.
4. If only `train` exists, split it into `train`, `validation`, and `test`.

All synthesized splits are created with:

- random seed: `42`
- validation fraction: `0.1`
- test fraction: `0.1`

Important clarification:

- “Synthesized split” does not mean synthetic data generation.
- It means a reproducible random partition of existing labeled examples.

## Dataset-Specific Notes

### Emotion

Dataset:

- `dair-ai/emotion`

Task type:

- emotion classification

Number of labels:

- `6`

Split status:

- official `train`
- official `validation`
- official `test`

Methodological note:

- No split synthesis is required for this dataset.

### CLINC OOS

Dataset:

- `clinc/clinc_oos`
- configuration: `plus`

Task type:

- intent classification

Label field:

- `intent`

Number of labels:

- `151`

Split status:

- official `train`
- official `validation`
- official `test`

Methodological note:

- No split synthesis is required for this dataset.

### AG News

Dataset:

- `ag_news`

Task type:

- news topic classification

Number of labels:

- `4`

Split status:

- official `train`
- official `test`
- synthesized `validation`

Current handling:

- the validation split is created from the original training set using the deterministic split procedure described above

Methodological note:

- This should be stated explicitly in the thesis, since the validation split is not part of the original benchmark packaging used in the codebase

## Thesis Wording Suggestion

The following wording can be adapted for the methodology section:

> All experiments were conducted using train, validation, and test splits. For datasets with official train/validation/test partitions, the provided splits were used directly. For datasets lacking a validation split, a validation partition was created from the training data using a fixed random seed to ensure reproducibility.

If stricter benchmark comparability is required in the final thesis version, `AG News` is the main dataset that may need replacement with a benchmark providing official `train`, `validation`, and `test` splits.
