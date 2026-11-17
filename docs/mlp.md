# MLP architecture and propagation pipelines

## Purpose

M3 provides four Hydra-configured teaching pipelines. Each has an explicit producer, numerical transformer, and report builder. The builder returns a `ReportModel`; only the shared `ReportWriter` writes HTML and figures. Numerical results do not depend on the run directory.

The executable scripts are thin orchestrators. They connect the filters, pass `result_dir` to the writer, and close all Matplotlib figures after output succeeds or fails.

## Run the pipelines

Set up the environment and select an output root as described in the repository README. Run these commands from a source checkout:

```bash
python -m spbunn.scripts.mlp_width_experiment
python -m spbunn.scripts.mlp_depth_experiment
python -m spbunn.scripts.hidden_norms_propagation
python -m spbunn.scripts.balduzzi_analysis
```

The production configurations are:

- `config/config_mlp_width_experiment.yaml`;
- `config/config_mlp_depth_experiment.yaml`;
- `config/config_hidden_norms.yaml`;
- `config/config_balduzzi_analysis.yaml`.

Common M1 settings define `result_dir` and Hydra logging. Override values on the command line when needed. Both the Hydra run directory and `result_dir` must be selected when an explicit output directory is required:

```bash
python -m spbunn.scripts.balduzzi_analysis \
  result_dir=/tmp/spbunn-balduzzi hydra.run.dir=/tmp/spbunn-balduzzi
```

## Reports and figures

Each run writes one HTML report and one SVG or PNG artifact per figure directly into `result_dir`.

| Pipeline | HTML artifact | Figures and legacy stems |
|---|---|---|
| MLP width | `mlp_width_experiment.html` | Eight figures: `mlp_conv_wrt_loss_<function>` and `mlp_conv_wrt_mae_<function>` |
| MLP depth | `mlp_depth_experiment.html` | Eight figures: `mlp_depth_conv_wrt_loss_<function>` and `mlp_depth_conv_wrt_mae_<function>` |
| Hidden propagation | `hidden_norms_propagation.html` | Six figures: initialization forward/backward norms, plain initialization sensitivity/Jacobian conditioning, and residual sensitivity/Jacobian conditioning |
| Balduzzi analysis | `balduzzi_analysis.html` | `balduzzi_analysis_plain_network` and `balduzzi_analysis_res_network` |

The function names are `cubic`, `gaussian`, `discontinuous`, and `periodic`. Hidden-propagation stems have the configured activation suffix, normally `relu`.

The shared writer selects SVG at or below 102,400 serialized SVG bytes and PNG above that limit. It never retains both formats for one stem after a successful rerun. The legacy hidden-propagation script wrote both formats; the migrated pipeline intentionally follows the shared contract instead. Hidden-propagation figures retain their 150 DPI setting for PNG output.

The reports contain explanatory HTML and LaTeX. The embedded MathJax runtime renders mathematics without network access.

## Numerical scope

### Width and depth experiments

Both pipelines generate 100 ordered, noiseless, float32 points on `[-2, 2]` by default. The four functions are the cubic `x^3-x`, a Gaussian, a cubic/positive-step discontinuity, and the sum `sin(2x)+cos(8x)+sin(16x)`.

The width pipeline uses one hidden ReLU layer with widths `[8, 16, 32, 64, 128, 256, 512, 1024]`. The prediction panel shows width 1024. The depth pipeline uses hidden width 8, plain ReLU networks with depths `[1, 2, 4, 8, 16, 32]`, and a prediction panel at depth 32.

The shared search preserves this order: function, architecture size, learning rate, step-size fraction, and seed. It tests ten legacy learning rates and step fractions `0.1`, `0.25`, and `0.5`. Each search run uses full-batch AdamW for 3,000 epochs and StepLR with gamma 0.1. The score is the mean pre-update MSE in the last 10 percent of epochs, averaged over seeds 42, 43, and 44. Non-finite tails are excluded from final selection as infinite scores.

Final training runs for 30,000 epochs for every seed. The run with the lowest post-training MSE is retained. Prediction curves contain pre-update outputs at epochs `(1, 10, 50, 100, 1000, 10000)`. The convergence panel uses raw MAE before epoch 1000 and an MAE EMA with beta 0.9 at epochs 1000 and later.

The legacy working-directory JSON learning-rate caches are not read or written. Search results now pass only through explicit in-memory pipes. Consequently, a production run repeats the complete search and is computationally expensive. Use the small test configurations for automated checks and quick examples.

### Hidden-norm and Jacobian propagation

The production pipeline uses input width 16, hidden width 256, batch size 256, and seeds `[0, 1, 2, 3, 4]`. Models are initialized in the default Torch dtype and then converted to float64, preserving legacy random-number consumption. Inputs and output adjoints use their legacy global and separate seeded generators.

The initialization panel compares standard deviations `1/sqrt(N)`, `sqrt(2/N)`, and `1` at depth 32. It reports mean forward pre-activation norms and backward input-adjoint norms with population-standard-deviation bands. The report explains the ReLU forward factor `N*sigma^2/2`, the Kaiming critical scale, and the zero-mean approximation limits.

Conditioning evaluates depths `[1, 2, 4, 8, 16, 32, 64]`. It compares three plain initialization scales and plain/residual variants with branch scales `1`, `1/sqrt(M)`, and `1/M`. For 16 inputs and 8 normalized perturbations per input, the finite-difference sensitivity uses perturbation norm `1e-3`. Exact input-Jacobian singular values are computed separately through autograd. The condition number is the mean of per-input singular-value ratios, followed by the mean across seeds.

The manual backward formulas retain the legacy ReLU activation mask. The production and validated configuration uses `relu`; the formulas are ReLU-specific.

### Balduzzi input gradients

The input grid contains 256 float32 points on `[-1, 1]`. For each point, autograd computes the derivative of that point's scalar network output with respect to its input by differentiating the sum of all outputs. The plain network is the existing `DeepReLUModel` with hidden width 200 and 24 layers.

The legacy residual-network import is absent from the repository and its available history. M3 therefore supplies a self-contained residual network with an input Linear layer, 23 square `Linear + ReLU` residual blocks, and an output Linear layer. Each square block computes `h + ReLU(W h + b)` with unit branch scale. All weights use the same Kaiming-uniform initialization as the plain network. The newly deterministic producer sets Torch seed 42 and constructs the plain network before the residual network. Axis limits remain automatic.

## Small test configurations

The reduced configurations are:

- `config/config_test_mlp_width_experiment.yaml`;
- `config/config_test_mlp_depth_experiment.yaml`;
- `config/config_test_hidden_norms.yaml`;
- `config/config_test_balduzzi_analysis.yaml`.

For example:

```bash
python -m spbunn.scripts.mlp_width_experiment \
  --config-name config_test_mlp_width_experiment \
  result_dir=/tmp/spbunn-test hydra.run.dir=/tmp/spbunn-test
```

The architecture test configurations retain all four functions but reduce data, sweep, search, and final-training counts. The propagation configuration reduces dimensions, depths, inputs, perturbations, and seeds while preserving both experiment paths and all six figures.

## Tests and review

The automated tests compose every production configuration and execute every pipeline with its test configuration in a subprocess. They verify artifact placement, figure references, SVG/PNG exclusivity, embedded MathJax, working-directory cleanliness, and figure cleanup. Numerical tests include manual training and propagation calculations, deterministic repetition, exact production grids, model shapes, Jacobian metrics, and independent autograd comparisons.

Run all local checks:

```bash
MPLBACKEND=Agg python -m pytest
./run_linters.sh
python -m build
```

Human review of all four rendered reports is required for educational acceptance. At least one report must be opened with network access disabled to verify LaTeX rendering.
