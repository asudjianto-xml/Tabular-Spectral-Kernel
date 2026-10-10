# Context selection and spectral kernel fusion

These experiments extend the single-kernel implementation with historical kernel
libraries, query-conditioned mixture weights and regression on labeled context.
They include synthetic repayment examples, attention comparisons, learned window
selection, chronological Bike Sharing evaluation and the subsequent controls.
The repository distributes code and recorded numerical results, not manuscripts.

## Install and verify

Install the repository with `python -m pip install -e ".[dev]"`. The
`tsk-context` command is also included in the wheel and works outside the checkout.

```bash
tsk-context verify
tsk-context list
tsk-context export --output "runs/context workspace" --recorded
tsk-context run --workspace "runs/context workspace" --device cpu \
  bikeshare_context_fusion/smoke.py -- --output "smoke check"
```

`export` creates a writable workspace and refuses an existing destination.
Without `--recorded`, it exports source only. With `--recorded`, it also copies
the original checkpoints, predictions and metrics. Those artifacts allow
comparison and diagnostic scripts to run without repeating all training.
The bundle is about 150 MB uncompressed; wheel installation includes it.
The smoke test trains a small weighting gate through a fused ridge solve and
checks finite gradients, PSD and decreasing loss. It is not a benchmark run.

Script arguments after `--` belong to the selected script. Relative paths in
those arguments are relative to the exported workspace. CLI-level `--workspace`,
`--data-dir`, `--bike-csv` and `--output` are resolved relative to the caller.
The command runs in a subprocess, so scripts with similarly named local modules
do not contaminate each other's imports.

## Read the recorded comparisons

```bash
tsk-context run --workspace "runs/context workspace" --device cpu \
  --output "runs/context summary" review_experiments/summarize.py
```

This recomputes summary CSVs, paired temporal bootstrap intervals and plots from
saved predictions. It also generates local LaTeX tables in the workspace for
users who need them; these generated files are not distributed as manuscripts.
The numerical source records remain in their experiment directories.

The synthetic selection study finds that a learned spectral selector improves
on using the latest window, while a capacity-matched MLP and recent-context
Gaussian KRR supplied with unemployment history are competitive. The Bike
Sharing study finds a fixed-support gain from learned fusion, comparable error
from prediction averaging and no gain from the learned historical selectors.
Increasing the number of recent labels has a larger effect than learning
mixture weights at the smaller context size. See `review_experiments/summary/`
for the recorded values and uncertainty estimates.

## Generate synthetic repayment data and train the banks

A source-only workspace is sufficient for this command:

```bash
tsk-context export --output runs/new-synthetic
tsk-context run --workspace runs/new-synthetic --device cpu \
  synthetic_drift/run_credit_card.py -- --unemployment \
  --seeds 0 1 2 --steps 400 --meta-steps 600 \
  --output synthetic_drift/unemployment_results
```

The response is a payment-to-statement-balance fraction. Inputs are utilization,
credit score, cash-buffer months, prior repayment fraction and unemployment.
The simulator uses current unemployment and accumulated stress to create lagged
repayment responses. The lending relationships are hypothetical. They do not
model selective loan approvals, censored outcomes or delayed labels.

The older `synthetic_drift/run.py` example generates an IID feature stream with
changing response functions. `run_credit_card.py` without `--unemployment`
produces the earlier credit-card example. The other synthetic scripts retain
their original command-line arguments; append `-- --help` to inspect a runner.
Long training runs are separate from the installation smoke test.

## Reproduce context selection

Use an exported recorded workspace to supply the historical banks and preceding
selector calibrations. The selector scripts take one seed per invocation:

```bash
tsk-context run --workspace "runs/context workspace" --device cpu \
  synthetic_drift/run_spectral_selector.py -- --seed 0
```

This command deliberately refuses the already recorded
`synthetic_drift/spectral_selector_results/seed0` directory. Before retraining,
move that directory to a backup **inside your exported workspace**. Repeat for
seeds 1 and 2. `run_context_selector.py` and `run_selector_penalties.py` provide
the preceding descriptor and sparse-linear controls. Their saved outputs are
inputs to the spectral-selector comparison; retraining them changes that chain
of inputs and must be documented as a new run.

Training scripts with an explicit `--output` or `--out` accept a fresh destination
instead. Original recorded directories are included to reproduce the numerical
comparisons, not as overwrite targets. Comparison scripts usually refuse an
existing comparison directory as well.

## Bike Sharing

Raw data are downloaded separately from UCI and verified against the expected
SHA-256 digest. The installed downloader is shared with the single-kernel study.

```bash
tsk download --dataset bike --data-dir data
tsk-context run --workspace "runs/context workspace" --device cpu \
  --data-dir data bikeshare_context_fusion/run.py -- \
  --out new-bike-run
```

Use `-- --help` for the bank steps, gate epochs and seed options. `--device cuda`
requires a CUDA-capable PyTorch installation; `auto` selects CUDA when available.
Synthetic historical training remains on CPU as in the original implementation.
The Bike Sharing and review-control runners honor the selected device. All
checkpoint loads are mapped explicitly so saved GPU tensors can be read on CPU.

A user-supplied copy of `hour.csv` can be selected with `--bike-csv`. Direct
script execution also supports `TSK_DEVICE`, `TSK_DATA_DIR` and `TSK_BIKE_CSV`.
There are no dependencies on a particular user's home directory or private
research checkout. Missing data produce a download instruction; importing a
module never downloads data.

The three historical kernels cover January–June 2011. Gate training uses
July–September and calibration uses October–December. Evaluation rolls over
2012 using only labels from completed prior days, with contemporaneous weather
as an input. It is conditional demand prediction, not a weather-forecasting
experiment.

## Review controls

The following runs use recorded earlier checkpoints and predictions. A fresh
output is required:

```bash
tsk-context run --workspace "runs/context workspace" --device cpu \
  --data-dir data --output runs/new-bike-controls \
  review_experiments/bike_revision.py
tsk-context run --workspace "runs/context workspace" --device cpu \
  --output runs/new-synthetic-controls review_experiments/synthetic_controls.py
tsk-context run --workspace "runs/context workspace" --device cpu \
  --data-dir data --output runs/new-tree-control review_experiments/rolling_tree.py
```

The CLI output override applies only to the selected review script, not to its
imported dependencies. `ridge_limit.py` reads the recorded revised Bike results;
its override selects a directory containing that calibration. `mlp_exact_width.py`
reads the recorded synthetic-control search and writes its additional control.
To summarize retrained
results, place them at the documented experiment-relative locations in a new
workspace. Merely changing a training output path does not change downstream
inputs.

These controls were developed after inspecting the initial evaluations. Their
hyperparameters use the historical validation sets, but the comparisons are
not independent external replications.

## Source map

| Directory or module | Role |
|---|---|
| `synthetic_drift/run.py` | Shared synthetic generator, spectral model training and KRR helpers |
| `run_credit_card.py`, `unemployment_data.py` | Repayment cohorts and unemployment-dependent drift |
| `run_meta.py`, `run_retrieval.py` | Summary encoders and query-dependent retrieval |
| `run_transformer.py`, `psd_spectral_attention.py`, `simple_spectral_gate.py`, `mlp_ablation.py` | Attention, PSD encoders and linear-head comparisons |
| `run_optimized_fusion.py`, `run_oracle_references.py` | Context weight optimization and retrospective references |
| `run_context_selector.py`, `run_selector_penalties.py`, `run_spectral_selector.py` | Historical context scoring and selection |
| `compare_*.py`, `plot_unemployment.py` | Recorded comparison and diagnostic scripts |
| `bikeshare_context_fusion/` | Chronological real-data runner, report and causality tests |
| `review_experiments/` | Ridge, oracle, support-size, history-feature, MLP and tree controls |
| `context_runtime.py` | Data paths, device choice and provenance validation |

All active Python files from these three experiment directories are included.
Duplicated historical source snapshots, logs and rendered manuscript assets are
excluded. The shared `CanonicalSpectral` implementation is imported from the
installed package rather than loaded from an external directory.

## Integrity and tests

`release_manifest.json` records the hash of every distributed source or artifact.
For migrated files it also retains the pre-portability hash. Recorded JSON path
identifiers were converted to relative paths, and runnable sources were adapted
for portable imports and device selection. The release therefore does not claim
byte identity between ported source and the historical training source.
Comparison scripts validate declared transformations against this manifest;
unmodified checkpoint and prediction bytes remain unchanged. `tsk-context verify`
checks the manifest and recomputes hourly errors from saved predictions.

Run the experiment families in separate processes to keep their local `run`
modules separate:

```bash
python -m pytest -q "runs/context workspace/synthetic_drift"
TSK_DEVICE=cpu TSK_DATA_DIR="$PWD/data" python -m pytest -q \
  "runs/context workspace/bikeshare_context_fusion"
TSK_DEVICE=cpu python -m pytest -q "runs/context workspace/review_experiments"
```

The first and third commands need no downloaded dataset when exported with
`--recorded`. Tests cover causal contexts, label-leakage interventions, ridge
solves, kernel PSD, attention construction, selectors and oracle calculations.
Small numerical differences across devices and PyTorch versions are possible;
the manifest check and recorded-error replay are exact-artifact checks, not
claims that retraining is bitwise invariant across hardware.
