# Tabular Spectral Kernel

Code and recorded results for **Learning Spectral Geometry for Kernel Regression**,
by **Agus Sudjianto and Aijun Zhang**. The model learns Fourier frequencies, a
trace-normalized positive-definite metric and a Laplace bandwidth, then fits
kernel ridge regression. The paper studies one spectral bank per model.

## Install

Python 3.10 or newer is required. Python 3.12 is the reference environment.

```bash
git clone https://github.com/asudjianto-xml/Tabular-Spectral-Kernel.git
cd Tabular-Spectral-Kernel
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
For an NVIDIA GPU, install a compatible PyTorch build using the
[official installation selector](https://pytorch.org/get-started/locally/)
before installing this package. CPU execution is supported. CatBoost and
LightGBM baselines use CPU execution in the paper protocol.

## Check installation without downloading data

```bash
tsk smoke --device cpu --output "runs/install check"
tsk verify --output runs/recorded-audit
pytest -q
```

The smoke test trains a small synthetic mixed-input model, differentiates through
the matrix exponential and ridge solve, and predicts held-out rows. It is not a
paper benchmark. `verify` checks the bundled prediction arrays against all
reported per-run metrics and validates their hashes and split separation.
`verify` also exports means and sample standard deviations to `paper_results.csv`.
These commands work from any working directory after installation.

To regenerate the learned-geometry figure from the recorded diagnostics:

```bash
tsk assets --output runs/paper-assets
```

## Download the public datasets

```bash
tsk download --dataset all --data-dir data
```

The downloader fetches California Housing through scikit-learn and the original
UCI archives for Bike Sharing, Taiwan credit default and Adult. It verifies
Bike/Adult file hashes and Taiwan array hashes. Taiwan is read from the original
Excel workbook, with the identifier removed and the numerical representation
matched to the paper. Existing verified files are reused. Raw datasets are not
bundled in the repository. No account or local research-directory cache is needed.
Network access is needed only for installation and the first data download.

## Rebuild predictions from the recorded models

Start with the smaller Bike Sharing benchmark:

```bash
tsk reproduce --dataset bike --mode replay --seeds 0 --device auto \
  --data-dir data --output runs/bike-replay
```

Replay all reported primary spectral models and the additional-task CatBoost models:

```bash
tsk reproduce --dataset all --mode replay --device cuda \
  --data-dir data --output runs/paper-replay
```

Replay reconstructs preprocessing from the same training rows, loads the saved
geometry and rebuilds the **exact** ridge head. It compares predictions with the
recorded arrays and reports numerical differences. It does not retune models.
California replay reconstructs the primary full-refit spectral model; its
baseline and decoder-ablation prediction arrays are checked by `tsk verify`.
`--seeds 0`, `--seeds 1` or `--seeds 2` can replay one split. Use `--device cpu`
when a suitable GPU is unavailable; full-data CPU solves can be slow.

## Rerun training

```bash
tsk reproduce --dataset adult --mode train --device cuda \
  --data-dir data --output runs/adult-training
tsk reproduce --dataset california --mode train --device cuda \
  --data-dir data --output runs/california-training
```

Each training run uses all three paper seeds. Bike, Taiwan and Adult repeat the
preprocessing search, geometry training, ridge selection, CatBoost tuning and
separate calibration, then evaluate the test partitions after selections freeze.
California uses the published inner-training warm-start checkpoints, retrains the
selected refinement recipe, refits geometry on the outer training pool and
repeats the original CatBoost/LightGBM searches and decoder ablations.
To also regenerate California's initialization search:

```bash
tsk reproduce --dataset california --mode train --initialization search \
  --device cuda --data-dir data --output runs/california-from-scratch
```

California's historical refinement search was adaptive. The release reproduces
its selected single-bank recipe, not the abandoned search branches. An
initialization search on a different numerical stack can select a different
configuration. If its frequency count or preprocessing differs, training stops
with an explanation; use the default recorded initialization to run the paper
recipe. Exact historical replay uses the bundled checkpoints. Training
results need not be bitwise identical across PyTorch versions, CPU/GPU platforms
or library backends. Use a new output directory when changing configuration.
Existing selections can be resumed under the same configuration; incomplete
geometry-training stages restart from their initialization.

## Compute requirements

The CPU smoke test uses small matrices. The paper benchmarks do not: exact ridge
regression stores a dense n-by-n matrix and performs a Cholesky factorization.
For Adult, a single float64 training Gram matrix is approximately 5.4 GB; the
factor, temporary arrays, representation and allocator overhead require more.
Budget **at least 32 GB of available memory for full replay**, with additional
headroom for training and device-specific workspace. A low-memory GPU is not
sufficient merely because it supports CUDA. CPU mode uses system RAM instead.
The recorded experiments used an NVIDIA GB10 with a large unified-memory pool.
There is no out-of-core or hundred-million-row training claim in this release.

## Paper and recorded evidence

- [Paper PDF](paper/identifiable_spectral_regression.pdf)
- [LaTeX source](paper/identifiable_spectral_regression.tex)
- [Reproduction details and source map](REPRODUCIBILITY.md)
- [Recorded results](src/tabular_spectral_kernel/recorded/)
- [Release validation](validation.json)

The original single-model findings are dataset-dependent: the reported regression
comparisons favor the spectral model, Taiwan has higher ranking AUC with similar
probability quality, and Adult favors CatBoost. California holdouts were reused
and tuning budgets differ between model families. Mixtures of experts,
segmentation, replica fusion and later Adult frequency experiments are outside
this paper release.

## Citation

See [CITATION.cff](CITATION.cff). The manuscript is a research preprint; no journal
acceptance or DOI is claimed.

## Build the manuscript

With a LaTeX installation that provides `pdflatex` and `bibtex`:

```bash
cd paper
pdflatex identifiable_spectral_regression.tex
bibtex identifiable_spectral_regression
pdflatex identifiable_spectral_regression.tex
pdflatex identifiable_spectral_regression.tex
```
