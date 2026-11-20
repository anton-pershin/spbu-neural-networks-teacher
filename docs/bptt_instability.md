# BPTT instability in vanilla RNNs

## Purpose

M7 provides a two-part teaching audit of backpropagation through time (BPTT). The copy memory task demonstrates exponentially vanishing hidden-state adjoints under default and orthogonal initializations. The adding problem demonstrates exponentially growing adjoints under the documented Xavier `tanh` initialization while forward states remain bounded.

The pipeline does not train a model. It creates deterministic synthetic batches, performs one forward and backward pass per case, and writes an HTML report and two figures through the shared report writer. Concrete input/target/output examples and two inline HTML animations show every per-step untrained RNN prediction. It does not use Alice, n-gram perplexity, truncated BPTT, gradient clipping, or LSTMs.

## Run the pipeline

Set up the environment and optional output root as described in the README. From a source checkout, run:

```bash
python -m spbunn.scripts.bptt_instability
```

The production configuration is `config/config_bptt_instability.yaml`. Select an explicit output directory by overriding both Hydra paths:

```bash
python -m spbunn.scripts.bptt_instability \
  result_dir=/tmp/spbunn-bptt hydra.run.dir=/tmp/spbunn-bptt
```

The pipeline runs on CPU and needs no network resource or external dataset. Cases execute sequentially. Production uses 128 sequences, 64 hidden units, seeds 7, 42, and 123, and task lengths or gaps 20, 50, 100, and 200.

Exact float32 copy regression values are pinned to one PyTorch intra-op thread. A 200-step `tanh` recurrence can amplify small BLAS reduction-order differences from other thread counts. To reproduce the documented local reference values, run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m spbunn.scripts.bptt_instability
```

The scientific regression test applies `torch.set_num_threads(1)` only within a fixture and restores the previous setting afterward. Other tests and caller settings are not changed. The long adding recurrence can also amplify numerical differences between CPUs and BLAS implementations even at one thread. Its automated regression therefore checks finite forward/backward outcomes, bounded states, large parameter gradients, and ordered backward growth rather than exact pointwise loss or norm values. The documented adding values below are local one-thread reference observations, not cross-machine constants.

## RNN and adjoint definitions

For input `x_t`, the one-layer vanilla RNN uses:

```text
h_t = tanh(W_x x_t + W_h h_{t-1} + b).
```

For loss `L`, the hidden-state adjoint is `g_t = ∂L/∂h_t`. With `D_t = diag(1 - h_t^2)`, the local recurrent Jacobian is `J_t = D_t W_h`, so:

```text
g_{t-1} = J_t^T g_t = W_h^T D_t g_t.
```

Products of these Jacobians can shrink or amplify the adjoint exponentially.

The implementation uses an explicit `torch.nn.RNNCell` time loop and calls `retain_grad()` on every separate hidden state. This detail is necessary. Retaining the gradient of the single packed output tensor returned by `nn.RNN` records the direct decoder gradient at that tensor boundary, not each internal recurrent adjoint.

For sequence `i` and lag `tau`, the audit calculates:

```text
r_i(tau) = ||dL_i/dh_{T-1-tau}||_2 / ||dL_i/dh_{T-1}||_2.
```

It then averages `r_i(tau)` over valid sequences. Autograd uses the configured model dtype. Norms and ratios are converted to float64 afterward. Exact zeros and non-finite values remain unchanged in result objects and HTML tables. Only the log-scaled figures replace exact zeros with `1e-45`, and the report discloses this display floor.

## Copy memory task

There are eight ordinary symbols, a blank, and a delimiter. Each sequence contains:

1. Ten random ordinary symbols.
2. One delimiter.
3. A configured blank gap.
4. Ten blank reproduction positions.

The targets are the original ten symbols. Cross-entropy applies only at the reproduction positions. The optimal memoryless uniform predictor has loss `ln(8) = 2.079442`; it is a reference, not a fitted model.

Two initialization arms are audited:

- **Default:** the ordinary PyTorch `RNNCell` initialization.
- **Orthogonal:** `torch.nn.init.orthogonal_` applied to the recurrent matrix without a scale factor.

Orthogonal initialization is motivated by norm preservation in a linear map: `Q^T Q = I` and `||Qx||_2 = ||x||_2`. This avoids systematic initial shrinkage or amplification from the recurrent matrix itself when that matrix is applied many times. It does not make the complete recurrent Jacobian orthogonal. The `tanh` derivative factors still shrink or redirect the adjoint, so orthogonal initialization delays decay but cannot guarantee long-range gradient flow.

At gap 200 with seed 42, the production implementation gives:

| Initialization | g(20)/g(0) | g(50)/g(0) | g(100)/g(0) | g(200)/g(0) |
|---|---:|---:|---:|---:|
| Default float32 | 1.624e-3 | 1.309e-10 | 2.428e-22 | 0 |
| Orthogonal float32 | 0.473 | 1.140e-2 | 3.284e-5 | 7.551e-10 |

The default float64 check uses identical float32 weights and data, converted to float64. It gives `g(200)/g(0) = 9.550e-46`, confirming continued decay beyond the float32 underflow point. Orthogonal initialization delays decay but does not remove it.

## Adding problem

Each length-`T` input has two channels:

- A value channel sampled independently and uniformly from `[0, 1]`.
- A marker channel equal to one at two distinct random positions and zero elsewhere.

The target is the sum of the two marked values. The model produces one scalar at the final position, and mean squared error applies only there. Predicting the known target mean of one has MSE `1/6`; this is another reference rather than a trained model.

Input and recurrent weights use:

```python
nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain("tanh"))
```

The `tanh` gain is `5/3`. Recurrent biases are zero, and the linear readout keeps its ordinary initialization. There is no selected recurrent spectral radius or additional scale.

At length 200 with seed 42, the production run has finite loss 1.196123, maximum absolute hidden state 0.999930, and global parameter-gradient norm `1.281e7`. Its relative gradient norms are:

| Lag | 20 | 50 | 100 | 199 |
|---|---:|---:|---:|---:|
| g(lag)/g(0) | 3.783 | 39.02 | 1.455e3 | 1.950e7 |

Thus the forward state stays within the `tanh` bounds while the backward adjoint grows sharply. This observation is specific to the configured task and initialization. The report does not claim that every reasonable initialization explodes.

## Prediction examples and HTML animations

The report contains one small deterministic example for each task. Each example has an input-sequence row, a target-output row, and an actual untrained-RNN-output row. The copy example uses three symbols and a two-step blank gap. The adding example uses six value/marker pairs. These small examples are for inspection; the scientific audit grids retain their configured production sizes.

Each example also has an inline HTML animation. **Play**, **Pause**, **Previous**, and **Next** controls move through every time step. The table shows the current input, the RNN prediction at that step, and the target when the task defines loss at that position. Playing from the final frame restarts at the first frame; manual navigation pauses playback. The animations use inline JavaScript, text DOM updates, and no external script or animation artifact. Initial frames and static examples remain visible before playback starts.

Predictions can be poor because the models are untrained. The animations explain model behavior and are not evidence of task quality.

## Matched-weight float64 checks

Every longest case is repeated in float64. The model is first constructed in float32 and then converted, and task values are generated in float32 before conversion. Weights and data therefore match the primary float32 run exactly. These checks distinguish numerical range effects from the underlying recurrent trend; they do not replace the primary observations with a different random model or batch.

## Report artifacts

The run directory contains `bptt_instability.html` and one SVG or PNG artifact for each stem:

| Stem | Content |
|---|---|
| `bptt_copy_gradient_flow` | Long-gap adjoint curves and first-state ratios for both initialization arms and all seeds |
| `bptt_adding_gradient_flow` | Long-sequence adjoint curves and first-state ratios for all seeds |

The HTML contains the recurrence and adjoint equations, task and initialization definitions, concrete prediction examples, two controlled HTML animations, numerical diagnostics, matched-weight verification, and interpretation limits. Long-case figure curves are uniformly pruned to at most 32 points, including both endpoints; stored numerical results remain complete. Default and orthogonal copy curves share panels and use solid and dashed line styles. Log axes use at most 12 major ticks and no minor grid. SVG labels are serialized as SVG text through a figure-local plotting context that restores global settings. These choices keep both production figures below the shared 102,400-byte threshold, so the writer selects SVG without any shared-writer change. Figures use log-scaled vertical axes and local font settings: 14-point base text, 15-point labels and titles, and 13-point tick and legend text. MathJax is embedded and needs no external script or web font.

## Reduced test configuration

`config/config_test_bptt_instability.yaml` inherits the production configuration. It uses eight sequences, eight hidden units, two seeds, copy gaps 5 and 20, and adding lengths 10 and 20. It writes `test_bptt_instability.html` and both figures.

```bash
python -m spbunn.scripts.bptt_instability \
  --config-name config_test_bptt_instability \
  result_dir=/tmp/spbunn-test-bptt hydra.run.dir=/tmp/spbunn-test-bptt
```

Integer-valued settings must be finite and integral. Seeds and lengths must be sorted and unique. The primary dtype is float32, and the optional verification dtype is float64.

## Tests and review

Run all local checks:

```bash
MPLBACKEND=Agg python -m pytest
./run_linters.sh
python -m build
```

The recurrent tests cover exact task structure, deterministic matched-dtype batches, complete per-step prediction examples, initialization rules, global random-state restoration, hand-calculated and independent autograd adjoints, zero and non-finite diagnostics, production scientific regressions, configuration validation, static input/output/target content, all animation controls and inline playback data, report content and figure fonts, curve pruning and SVG text serialization, subprocess artifacts, and figure cleanup after writer success and failure.

A human reviewer must open the production report and approve its mathematical and educational content. Offline browser review of mathematical rendering is also required.
