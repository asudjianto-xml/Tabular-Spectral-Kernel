# Reproduction protocol

The public package ports the paper's executed routines without changing their
spectral maps, metric parameterization, optimization objectives or ridge solves.
Paths and devices are configured by the CLI. Input caches and outputs are never
written into the installed package. A wheel includes recorded tensors and
prediction arrays so that source-checkout paths are unnecessary.

## Source map

| Paper component | Package source |
|---|---|
| Canonical metric, initial California search, tree baselines | `_paper/identifiable_spectral_california.py` |
| Single-bank checkpoint adapter | `_paper/single_bank.py` |
| Exact free-intercept ridge solver | `_paper/solvers.py` |
| California geometry refinement | `_paper/spectral_competitive_train.py` |
| California outer-pool geometry refit | `_paper/spectral_competitive_finalize.py` |
| Frozen-initial-geometry exact-head ablation | `_paper/spectral_exact_california.py` |
| Bike/Taiwan/Adult training, calibration and evaluation | `_paper/spectral_stage1_extended.py` |
| Data download and hashes | `datasets.py`, `taiwan_signature.json` |
| Replay, verification and CLI | `cli.py` |

All paths in the table are relative to `src/tabular_spectral_kernel/`.
The historical single-bank adapter retains a `members.0` state layout and an
empty fusion-coordinate tensor for checkpoint compatibility. It rejects multiple
banks. This is not a mixture-of-experts release.

## Recorded reference environment

The paper used Python 3.12, PyTorch 2.13.0 with CUDA 13.0, NumPy 1.26.4,
SciPy 1.17.1, pandas 2.3.3, scikit-learn 1.6.1, CatBoost 1.2.10 and LightGBM 4.7.0,
with four CPU threads and an NVIDIA GB10. `requirements-reference.txt` records the
library versions. The reference CUDA build is hardware-specific; do not install
it blindly on a different platform. The package declares broader compatible
version ranges for installation. `environment.json` records each new run.

Saved-prediction verification does not require a matching GPU. Prediction replay
uses tolerances for floating-point library differences. New optimization runs can
differ more, particularly when early stopping or validation selection changes.
A failed replay tolerance is reported instead of silently replacing the paper's
result. No exact cross-platform training equivalence is promised.

## Splits and leakage controls

- California: fixed three random outer splits; inner fitting/validation split;
  original inner-fitted scaling retained through the final outer-pool geometry
  refit. The refinement schedule and ridge choices use the recorded validation
  protocol. Historical holdouts were reused during development.
- Bike Sharing: chronological split by date; `casual`, `registered`, `instant`
  and `dteday` are excluded from predictors. The same test chronology is used
  across optimization seeds.
- Taiwan: stratified fitting/calibration/validation/test partitions. Original
  workbook amounts are used, not a pre-log-transformed third-party CSV.
- Adult: official test partition, with fitting/calibration/validation splits of
  official training rows. Missing categories are retained. `fnlwgt` is an input,
  not a sample weight.

The additional tasks do not refit on calibration or validation observations.
Both model families receive the same rows. Classification uses squared-error
kernel ridge fitting followed by a sigmoid learned on separate calibration rows;
CatBoost receives the same separate calibration treatment. Hyperparameters and
checkpoints are selected by validation AUC; ranking and probability metrics are
reported separately. Training budgets differ between model families.

## Data provenance

- [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), H. Fanaee-T, DOI 10.24432/C5W894
- [UCI default of credit card clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), I-Cheng Yeh, DOI 10.24432/C55S3H
- [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), Barry Becker and Ronny Kohavi, DOI 10.24432/C5XW20

Consult the source pages for dataset terms and attribution. Downloaded archives
remain in the user-selected cache. Archive members are read by name rather than
extracted wholesale. Taiwan arrays are explicitly converted to the paper's
float32/int64 storage representation before hash checking; model arithmetic is
float64.

## Interpreting the commands

`verify` recomputes metrics for 36 saved prediction sets covering the primary
results and California decoder ablations. It checks artifact hashes and disjoint
row indices; it does not claim to replay optimization.

`reproduce --mode replay` rebuilds exact heads from frozen primary spectral
geometries and replays the additional-task CatBoost models. It checks predictions
against the saved arrays. It can be run for a single dataset and seed. Dense
training Gram matrices and Cholesky factors are rebuilt rather than distributed.

`reproduce --mode train` repeats fitting and validation selection. California
starts from its published inner-trained initialization unless
`--initialization search` is specified. Its historical adaptive refinement search
is documented by the retained selection record, rather than presented as an
exhaustive reproducible grid. The single-bank refinement recipe is fixed.

The original baseline, frozen geometry, full refit and additional-task summaries
are retained under `recorded/`. The paper's tables can be recovered from their
`summary.json` files after `verify` checks the underlying predictions. File hashes
are in `recorded/manifest.json`. Later specialist and fusion experiments are not
included.
