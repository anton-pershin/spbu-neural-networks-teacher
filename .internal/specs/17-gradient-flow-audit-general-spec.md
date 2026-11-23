## Gradient flow audit

### 1. Executive summary

#### 1.1 Spec description

Implement one Hydra-configured pipeline that trains default-initialized vanilla RNN, orthogonally initialized vanilla RNN, and zero-forget-bias LSTM arms on separate fixed-gap copy memory tasks. The pipeline compares per-position learning and retained-state gradients before and after training, separates the full LSTM cell gradient from its direct cell-path contribution, and writes an educational HTML report through the shared report writer. It also preserves the approved visual investigation as a standalone HTML artifact.

The experiments, measurement corrections, and maintainer decisions are recorded in the [gradient flow audit investigation](../investigations/gradient-flow-audit.md) and [issue #17](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/17).

#### 1.2 Spec motivation

The course needs an executable demonstration of recurrent gradient decay and the LSTM additive cell-state mechanism. Matched training and audit batches make the comparison reproducible, while the direct-path and full-gradient distinction prevents an incorrect claim that all LSTM cell gradients are just a product of mean forget gates.

#### 1.3 Implementation repos

- `anton-pershin/spbu-neural-networks-teacher` — solution implementation and management repo.

### 2. Requirement analysis

#### 2.1 Functional requirements

**FR1. Deterministic fixed-gap copy tasks.** Reuse `generate_copy_batch` with eight ordinary symbols, a blank, a delimiter, ten symbols to memorize, and loss only at the ten reproduction positions. Train separate models for gaps 20, 50, 100, and 200 in production. Use explicit local data generators and fixed training, validation, and audit seed schedules.

**FR2. Three matched arms and equivalent recurrences.** Use one-layer PyTorch vanilla `tanh` RNNs with default or recurrent-only orthogonal initialization and an LSTM with both forget biases set to zero. Use 64 hidden units and the same linear-readout convention in every arm. Match training batches, optimizer updates, and Adam settings across arms; do not claim equal parameter counts. Train with packed `nn.RNN` or `nn.LSTM` forward passes. For audits, explicitly unroll the same parameters with equations matching GetSomeMath and expose each separate hidden or cell state. Test packed/explicit parity.

**FR3. Training and per-position evaluation.** Production uses 5,000 updates per arm and gap, batches of 128 fresh training sequences, Adam with learning rate 0.001, seed 42, and one documented CPU thread. Record training loss and validation metrics at update 1 and every 500 updates. Before and after training, evaluate 512 fixed validation sequences. Report overall cross-entropy, accuracy, and loss at each reproduction position against the closed-form memoryless loss `ln(8)`. Preserve the full history and negative results; no model-performance ordering is required.

**FR4. Full relative state- and cell-gradient norms.** Audit the same fixed batch of at least 128 production sequences before and after training. Use one backward pass per snapshot. Let `t_*` denote the last blank state before reproduction and let `L^(b)` denote the loss for sequence `b`. Define and show the relative state-gradient norm

```text
(1/B) * sum_b ||grad(h[t_* - tau], L^(b))||_2 / ||grad(h[t_*], L^(b))||_2
```

and the relative cell-gradient norm obtained by replacing `h` with `c`, using GetSomeMath's bold-vector and nabla notation in the reports. Retain separate states before stacking, normalize each sequence before averaging, calculate norms and ratios in float64, and preserve exact zero and non-finite outcomes in numerical results. Do not retain only a packed output tensor.

**FR5. Direct cell path and forget-gate diagnostics.** For the LSTM, also calculate the vector-valued direct cell-path contribution

```text
f[t+1] ... f[t_*] * grad(c[t_*], L)
```

at every earlier cell state, using element-wise products and per-sequence normalization identical to the full cell curve. Calculate this contribution from the retained forget gates and the same single backward pass; do not present it as equal to the full gradient in general. Show the direct-term and full-cell curves before and after training. Include the scalar product of mean gates and the `0.5^tau` curve only as references. Define and show mean forget-gate activation at each time step as the average over batch and hidden components. Do not require a delimiter dip, near-one gates, or agreement with `0.5^tau`.

**FR6. Matched-weight float64 verification.** Repeat the initial and trained longest-gap audit for every arm in float64. Construct or copy the float32 model before conversion and cast the same generated batch, so weights and data match. Use the checks to distinguish numerical underflow from the recurrent trend.

**FR7. Hydra pipeline and filter boundaries.** Expose `python -m spbunn.scripts.gradient_flow_audit` with production and reduced test configurations. Put all task, gap, batch, hidden-size, training, seed, validation, audit, verification, figure-limit, and artifact settings in configuration. A producer returns validated typed settings. A transformer trains and audits models without Hydra, Matplotlib, or artifact output. A dedicated builder creates the report model. The thin orchestrator only connects filters, invokes the shared writer, and closes figures on success and failure.

**FR8. Final educational report and figures.** Build `gradient_flow_audit.html` through the existing `ReportModel` and `ReportWriter`. Explain the copy task, GetSomeMath RNN/LSTM equations, the normalization point, all three measurement formulas, the direct-path distinction, initialization limits, and numerical fidelity. Include detailed-caption figures with stems `gradient_flow_training`, `gradient_flow_per_position`, `gradient_flow_states`, `gradient_flow_cell_path`, and `gradient_flow_forget_gates`. Use log scales where appropriate, prune displayed long curves without changing stored results, use no internal figure titles, and keep production figures below the shared SVG threshold where possible without changing writer behavior. State that gap-200 task learning did not occur within the investigated budget and that good gradient flow does not guarantee learning.

**FR9. Investigation artifact.** Preserve the approved visual investigation as the standalone, offline-capable `.internal/investigations/gradient-flow-audit/index.html`, including its animations, figures, data, mathematical expressions for all three measurements, and confirmation points. Embed its assets so the retained artifact has no external dependency. Link it from the investigation record. Keep it distinct from the generated final pipeline report.

**FR10. Documentation, checks, and scope.** Document execution, one-thread production settings, task and model definitions, formulas, configuration, artifacts, investigated observations, and tests in `docs/gradient_flow_audit.md` and the README. Extend the existing M7 design and roadmap entry and required terminology. Add focused tests and lint paths. Do not change the existing BPTT report's numerical behavior, add gate ablations, add another task or repository, or integrate GetSomeMath content.

#### 2.2 Non-functional requirements

**NFR1. Reproducibility and local state.** Use local `torch.Generator` instances for data and isolated random-number contexts for model construction. Identical settings must produce identical results on the documented runtime without changing the caller's global random state. Run on CPU. Document that a different thread count can change post-plateau training trajectories; portable tests must not require exact trained losses.

**NFR2. Numerical fidelity and resource safety.** Use float32 for primary training and audits, with float64 norms and matched-weight verification. Execute cases sequentially, retain live graphs only for the current forward/backward audit, clear parameter gradients between phases, and release models and batches before the next case. Keep only scalar histories and detached numerical audit arrays. Use small reduced configurations in CI.

**NFR3. Validation and maintainability.** Reject invalid, empty, duplicate, unsorted, non-integral, or inconsistent settings with clear errors. Keep model construction, training, evaluation, state audits, configuration, and report rendering in small single-purpose components. Reuse existing task and reporting components and avoid unrelated reformatting.

### 3. Acceptance criteria

**AC1. Model and training correctness (FR1–FR3, NFR1).** Tests verify exact copy structure, deterministic schedules, default/orthogonal/zero-forget-bias initialization, packed/explicit RNN and LSTM parity, parameter counts, changed parameters after a tiny training run, exact history lengths, per-position metrics, identical data schedules across arms, and global-RNG restoration. Tiny reference updates or independent calculations check the training and evaluation contract.

**AC2. Gradient and gate correctness (FR4–FR6, NFR2).** Independent `torch.autograd.grad` checks verify retained hidden and full cell adjoints, per-sequence normalization, curve orientation, and valid/zero/non-finite diagnostics. A naive direct-product reference verifies the cumulative direct cell-path calculation. Tests verify mean gates over batch and hidden components, scalar mean-gate products, the `0.5^tau` reference orientation, packed/explicit state parity, and matched-weight dtype conversion. Selected initial production cases check exponential decay and the direct-path reference without asserting trained performance thresholds.

**AC3. Configuration and numerical regressions (FR1–FR6, NFR1–NFR3).** Compose production and reduced configurations and verify case counts and settings. Reject invalid settings. Deterministic reduced experiments produce identical results and verification can be disabled. A scoped one-thread fixture covers exact initial scientific regressions and restores the previous setting. No test requires exact long-training losses across CPUs.

**AC4. Pipeline and final report outputs (FR7, FR8).** Run the reduced executable from a different working directory. Verify the HTML report, all five referenced figures, SVG/PNG exclusivity, artifact placement, embedded MathJax with bold-vector macro support, no external script, and figure cleanup after success and controlled writer failure. Builder tests verify required equations, explanations, all configured arms/gaps, before/after results, limitations, log scales, captions, point pruning, and large fonts. A production run writes all five figures as SVG below 102,400 bytes and records the investigated observations.

**AC5. Investigation artifact and project records (FR9, FR10).** The standalone investigation HTML is present, contains its three controlled animations and all three measurement formulas, embeds every asset, and passes DOM/offline-browser checks. The investigation and spec link each other and the artifact. Constitution, terminology, documentation, README, and focused static-check paths reflect the approved scope.

**AC6. Quality and scope (FR10, NFR3).** The complete test suite, focused formatting, import-order, lint, type checks, and package build pass. A diff review confirms no unrelated pipeline, task, repository, or GetSomeMath changes.

**AC7. Human educational review.** A human reviewer must open the final production report, inspect equations, figures, numerical diagnostics, animations in the retained investigation artifact, and interpretation, and explicitly approve their mathematical and educational correctness. At least one final report must be opened with external network access blocked. Automated checks alone do not establish acceptance.

### 4. High-level solution design

M7 adds the following constitution flow. Yellow components are new; the task generator, report models, and report writer are existing shared components.

**Gradient flow audit — `gradient_flow_audit.py` orchestrator**

```mermaid
flowchart LR
  O["Gradient flow orchestrator"] -. invokes .-> P["Fixed-gap settings producer"]
  P -- "Validated audit settings" --> T["Matched copy trainer and retained-state auditor"]
  T -- "Histories, losses, gradients, and gate diagnostics" --> B["Gradient flow report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

`spbunn.rnn.models` owns packed training models and their explicit retained-state recurrences. `spbunn.rnn.copy_training` owns deterministic optimization and validation. `spbunn.rnn.gradient_audit` owns full and direct-path measurements. `spbunn.rnn.gradient_flow_pipeline` validates settings and executes each gap and arm sequentially. `spbunn.reporting.gradient_flow` converts detached results into explanatory HTML and five figures. The existing synthetic-task generator and shared writer retain their current contracts.

The retained investigation artifact is project evidence, not an executable-pipeline input. It has no runtime coupling to the numerical or reporting components.
