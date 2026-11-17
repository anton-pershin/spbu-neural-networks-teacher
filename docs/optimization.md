# Optimization pipelines

## Purpose

M2 provides three Hydra-configured teaching pipelines. Each pipeline has a producer, numerical transformer filters, and a report builder. The builder returns a `ReportModel`; only the shared `ReportWriter` writes HTML and figures.

The executable scripts are thin orchestrators. They connect the filters, pass `result_dir` to the writer, and close all Matplotlib figures after output succeeds or fails.

## Run the pipelines

Set up the environment and select an output root as described in the repository README. Run these commands from a source checkout:

```bash
python -m spbunn.scripts.schedulers
python -m spbunn.scripts.loss_landscape
python -m spbunn.scripts.simple_loss_optimization
```

The corresponding production configurations are:

- `config/config_schedulers.yaml`;
- `config/config_loss_landscape.yaml`;
- `config/config_simple_loss_optimization.yaml`.

The two landscape-based configurations share `config/functions/optimization.yaml`. This group contains the active linear objective and the documented alternative objectives. Common M1 settings define `result_dir` and Hydra logging. Override values on the command line when needed:

```bash
python -m spbunn.scripts.loss_landscape \
  result_dir=/tmp/spbunn-example hydra.run.dir=/tmp/spbunn-example \
  grid_size=25 full_batch_steps=50
```

## Reports and figures

Each run writes one HTML report and one SVG or PNG artifact per figure directly into `result_dir`:

| Pipeline | HTML artifact | Figures |
|---|---|---|
| Schedulers | `schedulers.html` | Four scheduler curves |
| Loss landscape | `loss_landscape.html` | Five optimizer figures per function |
| Simple optimization | `simple_loss_optimization.html` | One mini-batch figure per function |

Figure stems retain the legacy names, such as `scheduler_linearlr`, `loss_landscape_func_linear_adam`, and `sgd_batch_16_lr_1e_2_linear`. The shared writer selects SVG at or below 102,400 serialized SVG bytes and PNG above that limit. It never retains both formats for one stem after a successful rerun. See [reporting.md](reporting.md) for the complete output contract.

The reports contain explanatory HTML and LaTeX. The embedded MathJax runtime renders mathematics without network access.

## Numerical scope

### Learning-rate schedules

The scheduler pipeline preserves the legacy 40-epoch linear, step, reduce-on-plateau, and cosine-annealing curves. These are synthetic teaching curves, not measurements from a training run. Their parameters and annotations are in `config_schedulers.yaml`.

### Loss landscape and full-batch optimization

The producer creates ordered, noiseless `float64` data and endpoint-inclusive parameter grids. The transformer computes an MSE matrix indexed as `(theta_0, theta_1)` and full-batch histories for SGD, momentum, Nesterov momentum, RMSprop, and Adam. Parameter histories include the initial point. Loss histories contain the value before each update. Figures for one function share axis limits so the optimizer paths can be compared.

### Simple mini-batch optimization

The production configuration performs exactly 25 SGD updates with batch size 16 and learning rate 0.01. This preserves the effective update count of the legacy loop without its off-by-one condition. A new NumPy generator with seed 42 permutes the training grid for each function. DataLoader shuffling remains disabled.

Training and landscape data remain separate. Training uses 1,024 points on `[-10, 10]` by default. The landscape uses 1,024 ordered points on the function's configured interval, which is `[6, 10]` for the active linear objective. The figure uses the correct `theta_0` and `theta_1` labels and includes both the grid and trajectory in its axis limits.

## Tests

Small test configurations reduce sample, grid, and update counts:

- `config/config_test_schedulers.yaml`;
- `config/config_test_loss_landscape.yaml`;
- `config/config_test_simple_loss_optimization.yaml`.

For example:

```bash
python -m spbunn.scripts.simple_loss_optimization \
  --config-name config_test_simple_loss_optimization \
  result_dir=/tmp/spbunn-test hydra.run.dir=/tmp/spbunn-test
```

The automated tests execute all three modules in subprocesses and verify report and figure placement. Unit tests cover numerical values, history alignment, deterministic data, optimizer settings, report models, and figure cleanup. The M1 writer tests retain exact SVG/PNG threshold coverage.

Run all local checks:

```bash
MPLBACKEND=Agg python -m pytest
./run_linters.sh
python -m build
```

Human review of all three rendered reports is still required for educational acceptance.
