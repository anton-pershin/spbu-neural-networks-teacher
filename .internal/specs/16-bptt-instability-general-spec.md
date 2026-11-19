## BPTT instability for vanilla RNNs

### 1. Executive summary

#### 1.1 Spec description

Implement one Hydra-configured pipeline that audits backpropagation through time (BPTT) in a one-layer vanilla `tanh` RNN. The copy memory task demonstrates exponential gradient decay under default and orthogonal initializations. The adding problem demonstrates exponential gradient growth under the documented Xavier `tanh` initialization. The pipeline writes an educational HTML report and log-scaled figures through the shared report writer.

The selected experiments and rejected alternatives are recorded in the [BPTT instability investigation](../investigations/bptt-instability.md).

#### 1.2 Spec motivation

GetSomeMath explains BPTT but has no executable demonstration of the two directions of recurrent gradient instability. Small synthetic tasks with exact random generators and retained hidden-state adjoints make these mechanisms measurable without selected corpora, model fitting, or mitigation methods.

#### 1.3 Implementation repos

- `anton-pershin/spbu-neural-networks-teacher` — solution implementation and management repo.

### 2. Requirement analysis

#### 2.1 Functional requirements

**FR1. Deterministic synthetic tasks.** Generate each batch with an explicit local random generator. For the copy memory task, use eight ordinary symbols, a blank, a delimiter, ten symbols to memorize, and loss only at the ten reproduction positions. For the adding problem, use two channels containing a uniform `[0, 1]` value and a marker at two distinct positions; the target is the sum of the marked values, and mean squared error applies only to the final scalar output. Production uses batches of 128 sequences.

**FR2. Vanilla RNN and exact adjoint audit.** Use a one-layer `torch.nn.RNNCell` with `tanh` and 64 hidden units in production. Unroll it explicitly, retain every separate hidden state, perform one backward pass, and calculate each sequence's state-gradient L2 norms. Normalize each sequence by its final-state norm and then average the ratios over the batch. Also retain loss, maximum absolute hidden state, global parameter-gradient norm, and finite/zero diagnostics. Do not infer internal adjoints from the single packed output of `nn.RNN`.

**FR3. Copy-task decay experiment.** Audit default PyTorch initialization and ordinary orthogonal recurrent initialization with seeds 7, 42, and 123 and blank gaps 20, 50, 100, and 200. Use no recurrent scale factor. Include long-gap curves and first-state gradient ratios for every configured case.

**FR4. Adding-task growth experiment.** Audit seeds 7, 42, and 123 and sequence lengths 20, 50, 100, and 200. Initialize input and recurrent weights with Xavier normal initialization using `nn.init.calculate_gain("tanh")`, use zero recurrent biases, and retain the ordinary linear readout initialization. Include long-sequence curves and first-state gradient ratios for every configured case.

**FR5. Matched-weight float64 checks.** Repeat the longest configured case for every initialization and seed in float64. Create the model in float32 and then convert it, and cast the same generated batch, so the primary and verification runs have identical initial weights and data. Use these checks to distinguish numerical underflow or overflow from the underlying gradient trend.

**FR6. Hydra pipeline and filter boundaries.** Expose `python -m spbunn.scripts.bptt_instability` with production and reduced test configurations composed from the common run-directory settings. Put task, initialization, seed, length, batch, hidden-size, dtype-verification, and artifact settings in configuration. A producer must return validated typed experiment input. A transformer must generate batches and calculate audit results without Hydra, Matplotlib, or artifact output. The orchestrator must only connect filters, invoke the shared writer, and close figures on success and failure.

**FR7. Educational report.** Build `bptt_instability.html` through the existing `ReportModel` and shared `ReportWriter`. Explain the RNN recurrence, the recurrent-Jacobian product, both tasks, initialization rules, per-sequence normalization, and why separate retained states are necessary. Show the measured long-case curves, length trends, finite/zero diagnostics, and matched-weight float64 results. State that the observations are specific to these tasks and initializations, not universal rankings. Do not claim that every reasonable initialization explodes or that orthogonal initialization removes decay. Include figures with stems `bptt_copy_gradient_flow` and `bptt_adding_gradient_flow`. Use at least 14-point base figure text, 15-point axis labels and titles, and 13-point tick and legend text through a local plotting context.

**FR8. Documentation and checks.** Document execution, task definitions, initialization, audit equations, production observations, configuration, artifacts, and tests in `docs/bptt_instability.md` and the README. Include new source and tests in focused formatting, lint, type, and test checks. Preserve all existing behavior and checks.

**FR9. Approved scope.** Add only the two-task vanilla-RNN audit and its supporting SDD records, configuration, documentation, and tests. Add the approved M7 recurrent-network design and roadmap entry and required terminology. Do not add Alice or n-gram evaluation, truncated BPTT, gradient clipping, LSTMs, model training, a generic pipeline framework, GetSomeMath changes, or another repository change.

#### 2.2 Non-functional requirements

**NFR1. Reproducibility and local state.** Use local `torch.Generator` instances for data and an isolated random-number context for model construction. Identical settings must produce identical batches, weights, and results without changing the caller's global random state. Run on CPU.

**NFR2. Numerical fidelity.** Calculate gradient norms and ratios in float64 after autograd, even for float32 models. Preserve exact zeros and non-finite outcomes in results. Do not replace them with an arbitrary plotting floor in numerical tables. A display floor is permitted only in log-scaled figures and must be identified in the report.

**NFR3. Validation and resource safety.** Reject empty or duplicate seeds and lengths, non-positive task sizes, invalid initialization names, unsupported dtypes, inconsistent verification settings, and task-length metadata that disagrees with the batch time dimension, with clear errors. Integer-valued settings must be finite and integral before conversion. Execute cases sequentially and release each computation graph before the next case. Derive report summary lags from the configured grids. Keep artifact names safe and close all figures after writer success or failure.

**NFR4. Separation and focused maintenance.** Producers validate and type settings. Transformers own task generation and numerical audits. Builders only create escaped presentation and figures. Use small dataclasses and functions, and do not reformat unrelated modules.

### 3. Acceptance criteria

**AC1. Task and initialization correctness (FR1–FR4, NFR1).** Tests must verify exact copy sequence structure, reproduction targets, two distinct adding markers, marked-value sums, deterministic batches, all initialization rules, and restoration of global random state. Tests must cover invalid task settings.

**AC2. Gradient-audit correctness (FR2, NFR2).** Hand-calculated or independent automatic-differentiation checks must verify retained-state adjoints, per-sequence normalization, curve orientation, loss, maximum hidden state, and parameter-gradient norms. Tests must cover exact zero and non-finite diagnostics without plotting-floor contamination.

**AC3. Production scientific regression (FR3–FR5).** Selected production checks must reproduce the investigated copy decay and adding growth at seed 42, including bounded adding hidden states and finite losses. Pin exact float32 regression cases to one PyTorch intra-op thread and restore the previous setting afterward so BLAS reduction-order differences do not make the suite machine-dependent. Float64 checks must use identical weights and data and must retain finite nonzero copy gradients beyond selected float32 underflow points. These checks detect unintended numerical changes; they are not model-performance thresholds.

**AC4. Pipeline and report outputs (FR6, FR7, NFR3, NFR4).** Compose both configurations and run the reduced executable pipeline in a subprocess from a different working directory. Verify the configured HTML, two referenced figures, SVG/PNG exclusivity, artifact placement, embedded MathJax, no external script, and figure cleanup after success and controlled writer failure. Builder tests must verify equations, task and initialization explanations, numerical tables with configuration-derived lag columns and no out-of-grid NaN placeholders, scope limitations, log scales, display-floor disclosure, and figure font sizes. Audit tests must reject task-length metadata that disagrees with the batch time dimension.

**AC5. Quality and approved scope (FR8, FR9).** The complete test suite, focused static checks, and package build must pass. A diff review must confirm the approved M7 and terminology additions and no Alice, TBPTT, clipping, LSTM, training, unrelated pipeline, or shared-writer changes.

**AC6. Human educational review.** A human reviewer must open the production report, inspect equations, figures, numerical diagnostics, and interpretation, and explicitly approve its mathematical and educational correctness. At least one report must be opened with network access disabled. Automated checks alone do not establish acceptance.

### 4. High-level solution design

M7 adds the following constitution flow. Yellow components are new; the report writer and report models are existing shared components.

**BPTT instability — `bptt_instability.py` orchestrator**

```mermaid
flowchart LR
  O["BPTT instability orchestrator"] -. invokes .-> P["Synthetic task and settings producer"]
  P -- "Validated audit settings" --> T["Recurrent gradient auditor"]
  T -- "Gradient curves and numerical diagnostics" --> B["BPTT report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

`spbunn.rnn` owns deterministic task batches, vanilla-RNN construction, the explicit `RNNCell` recurrence, and typed gradient-audit results. `spbunn.rnn.pipeline` validates Hydra settings and executes the configured cases sequentially. It creates the primary float32 result and a matched-weight float64 result for each longest case.

`spbunn.reporting.rnn` converts results into explanatory HTML, tables, and two log-scaled Matplotlib figures. `spbunn.scripts.bptt_instability` contains only Hydra entry-point orchestration and figure cleanup. Production and reduced test configurations share the same data flow; the test configuration reduces batch size, hidden size, lengths, and seeds.
