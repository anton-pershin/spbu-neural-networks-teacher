# Gradient flow in vanilla RNNs and LSTMs

## Purpose

The M7 gradient-flow pipeline trains matched vanilla-RNN and LSTM arms on the copy memory task. It measures per-position learning, full recurrent-state gradients, the LSTM direct cell-path contribution, and forget-gate activation before and after training.

The experiment distinguishes two statements:

1. Training can make the LSTM direct cell path much less lossy.
2. Better gradient flow does not guarantee task learning within a fixed budget.

The existing `bptt_instability.py` report remains separate. It audits untrained vanilla RNNs on the copy and adding tasks. This pipeline trains models and adds the LSTM comparison.

## Run the pipeline

From a source checkout, run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m spbunn.scripts.gradient_flow_audit
```

Select an explicit output directory with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m spbunn.scripts.gradient_flow_audit \
  result_dir=/tmp/spbunn-gradient-flow hydra.run.dir=/tmp/spbunn-gradient-flow
```

The production configuration is `config/config_gradient_flow_audit.yaml`. Runs use the CPU and need no external dataset or network access. Cases execute sequentially. One investigated gap-200 arm took about 18 minutes on one CPU thread; the complete four-gap grid has a substantial runtime.

A different thread configuration can change the post-plateau training trajectory. Use the one-thread command to reproduce the documented local observations. Automated tests do not require exact trained losses across CPUs.

## Copy task and matched arms

Each sequence contains:

1. Ten random ordinary symbols.
2. One delimiter.
3. A fixed blank gap.
4. Ten blank reproduction positions.

There are eight ordinary symbols. Cross-entropy applies only at reproduction. The optimal memoryless uniform predictor has loss `ln(8) = 2.079442`.

Separate models train at gaps 20, 50, 100, and 200. The three arms are:

- Default vanilla `tanh` RNN.
- Vanilla RNN with `torch.nn.init.orthogonal_` applied to the recurrent matrix without a scale factor.
- LSTM with ordinary non-forget initialization and both PyTorch forget-bias vectors set to zero.

Every arm uses 64 hidden units, fresh training batches of 128 sequences, Adam with learning rate 0.001, seed 42, and 5,000 updates. Training data, validation data, and audit data use separate fixed seed schedules. All arms at one gap receive identical data schedules.

Equal hidden size does not give equal parameter count. Each production RNN has 5,514 parameters; the LSTM has 20,106. The matched budget means equal data and optimizer updates, not equal parameters or runtime.

Packed `nn.RNN` and `nn.LSTM` forward passes make optimization efficient. Explicit recurrences use the same parameters for retained-state audits. Tests verify packed/explicit parity.

## Model equations

GetSomeMath defines the vanilla recurrent layer as:

```text
h_t = a(b_h + W_h h_{t-1} + W_x x_t)
```

The probe uses `a = tanh`. The LSTM forget gate and cell state are:

```text
f_t = sigmoid(b_f + W_xf x_t + W_hf h_{t-1})
c_t = f_t * c_{t-1} + i_t * c~_t
h_t = o_t * tanh(c_t)
```

Here `*` between vectors is the element-wise Hadamard product, written with `\odot` in the HTML report.

Orthogonal initialization is motivated by norm preservation in a linear map. It does not make the full nonlinear recurrent Jacobian orthogonal, and training does not preserve orthogonality.

## Measurement definitions

Let `t_*` be the last blank state before reproduction and `L^(b)` the loss for sequence `b`. The HTML report shows these definitions in LaTeX.

The **relative state-gradient norm** is:

```text
(1/B) sum_b ||grad(h[t_* - tau], L^(b))||_2 / ||grad(h[t_*], L^(b))||_2
```

The **relative cell-gradient norm** replaces `h` with `c`:

```text
(1/B) sum_b ||grad(c[t_* - tau], L^(b))||_2 / ||grad(c[t_*], L^(b))||_2
```

Each sequence is normalized before the batch average. Production audits use 128 fixed sequences. Autograd uses the model dtype; norms and ratios are calculated in float64 afterward.

The boundary reference avoids the direct readout-loss terms that occur at reproduction positions. The implementation retains every separate RNN hidden state or LSTM cell state before stacking. It performs one backward pass per snapshot. Retaining only a packed output tensor would not expose the internal recurrent adjoints.

The **direct cell-path contribution** uses the intervening forget gates element-wise:

```text
f[t+1] * ... * f[t_*] * grad(c[t_*], L)
```

It is normalized in the same way as the full cell gradient. The full gradient also includes paths through the hidden state and other gate equations. The two quantities are not identical in general. A cumulative implementation computes every direct-path ratio in linear time and is tested against a naive product calculation.

The **mean forget-gate activation** at time `t` is:

```text
f_bar[t] = (1 / (B N_h)) sum_b sum_j (f[t])_(b,j)
```

It is a scalar mean over batch and hidden components at one time step. The scalar product of these means is a reference. It can differ from the direct vector norm because it removes gate variation and alignment with the upstream adjoint.

The `0.5^tau` curve is also a reference. Zero forget bias does not force each gate to 0.5 because the forget weights remain active. Near-one gates and a delimiter dip are observations, not required outcomes.

## Production observations

### Gap 20

After 5,000 updates:

| Arm | Validation loss | Accuracy |
|---|---:|---:|
| Default RNN | 2.08032 | 13.1% |
| Orthogonal RNN | 2.08027 | 13.1% |
| Zero-bias LSTM | 1.65421 | 33.6% |

At lag 20, the full relative gradient was `4.635e-10` for the default RNN and `2.162e-12` for the orthogonal arm. The LSTM full-cell and direct-path ratios were 2.525 and 0.576. The corrected mean forget activation over the gap increased from 0.499 before training to 0.825 after training.

Longer runs show why this is not a universal ranking. At 10,000 updates, the default RNN, orthogonal RNN, and LSTM had validation losses 1.80859, 2.07961, and 0.90302. The default-RNN gradient curve changed after it started learning. A separate 20,000-update LSTM run reached loss 0.55364 and accuracy 77.1%; its mean gap activation was 0.908.

### Gap 200

After 5,000 updates, all arms remained near memoryless loss and had about 13.1% accuracy. Every per-position loss remained near `ln(8)`.

| Arm | Validation loss | Full gradient at lag 200 | Direct cell path |
|---|---:|---:|---:|
| Default RNN, float64 check | 2.08034 | `3.778e-85` | — |
| Orthogonal RNN, float64 check | 2.08027 | `6.264e-116` | — |
| Zero-bias LSTM | 2.08071 | `8.555e-14` | `8.563e-14` |
| Forget-bias-1 investigation control | 2.08070 | `4.067e-6` | `3.680e-6` |

For the zero-bias LSTM, training increased the direct-path ratio at lag 200 from `9.356e-54` to `8.563e-14`. The corrected mean forget activation over the gap increased from 0.499 to 0.731. The full and direct curves agreed closely at measured blank-gap lags 20 and 200, while the scalar product of mean gates was `6.094e-28`.

The bias-1 control had mean gap activation 0.902. It improved gradient flow but did not learn the task within the same budget. It is investigation evidence, not a fourth production training arm.

## Matched-weight float64 checks

Initial and trained longest-gap audits are repeated in float64 for every arm. The model is copied from its float32 state and converted, and the same generated float32 batch is cast to float64. Weights and data therefore match the primary run.

RNN gradients can underflow to exact zero in float32 while remaining finite and nonzero in float64. Result objects and tables preserve exact zero and non-finite outcomes. Only log-scaled figures use the disclosed display floor `1e-160`.

## Report artifacts

The run directory contains `gradient_flow_audit.html` and one SVG or PNG artifact for each stem:

| Stem | Content |
|---|---|
| `gradient_flow_training` | Validation history for every fixed gap and arm |
| `gradient_flow_per_position` | Trained loss at all ten reproduction positions |
| `gradient_flow_states` | Full RNN hidden-state and LSTM cell-state relative norms |
| `gradient_flow_cell_path` | Longest-gap full/direct/mean-gate/reference comparison before and after training |
| `gradient_flow_forget_gates` | Per-time mean forget activation at the delimiter and through each gap |

Figures have detailed HTML captions, no internal titles, large labels, and linewidth 2. Displayed long curves are pruned to at most 65 uniformly spaced points with preserved endpoints. Stored numerical arrays remain complete. The shared writer applies the normal 102,400-byte SVG/PNG threshold.

The approved visual investigation is retained separately at `.internal/investigations/gradient-flow-audit/index.html`. It is a standalone offline-capable HTML file with embedded figures, data, JavaScript, and MathJax. It contains the same three measurement definitions, three controlled animations, and the five confirmed decisions. It is evidence for the investigation, not an input to the executable pipeline.

## Reduced test configuration

`config/config_test_gradient_flow_audit.yaml` uses four sequences, four hidden units, two updates, gaps 2 and 4, and one validation batch. It writes `test_gradient_flow_audit.html` and all five figures. Run it with:

```bash
python -m spbunn.scripts.gradient_flow_audit \
  --config-name config_test_gradient_flow_audit \
  result_dir=/tmp/spbunn-test-gradient-flow \
  hydra.run.dir=/tmp/spbunn-test-gradient-flow
```

## Tests and review

Run all local checks:

```bash
MPLBACKEND=Agg python -m pytest
./run_linters.sh
python -m build
```

The tests cover packed/explicit parity, initialization and global-RNG restoration, deterministic matched training data, hand-calculated copy loss, per-position evaluation, independent automatic differentiation, naive direct-path references, corrected gate aggregation, zero/non-finite handling, initial scientific regressions, configuration validation, reduced pipeline execution, report content, figure pruning and scaling, standalone investigation content, and orchestrator cleanup.

A human reviewer must inspect the production report and approve its mathematical and educational correctness. At least one report must be opened with external network access blocked.
