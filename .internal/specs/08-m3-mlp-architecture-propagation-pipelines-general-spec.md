## M3 MLP architecture and propagation pipelines

### 1. Executive summary

#### 1.1 Spec description

Implement roadmap item M3 for the MLP width, MLP depth, hidden-norms propagation, and Balduzzi analysis experiments. Migrate each legacy script to a thin Hydra orchestrator with explicit producer, numerical transformer, and report builder filters. Each run must create a complete HTML report with LaTeX support and its figures through the shared report writer in the configured run directory.

#### 1.2 Spec motivation

M3 is the next roadmap step after the merged M1 foundation and M2 optimization pipelines. The migration preserves the architecture and propagation teaching experiments while separating numerical work, presentation, configuration, and artifact output. This separation makes the experiments testable and keeps report persistence and figure-format selection in the shared foundation.

#### 1.3 Implementation repos

- `anton-pershin/spbu-neural-networks-teacher` — solution implementation and management repo.

### 2. Requirement analysis

#### 2.1 Functional requirements

**FR1. Hydra pipeline entry points.** Expose `spbunn/scripts/mlp_width_experiment.py`, `spbunn/scripts/mlp_depth_experiment.py`, `spbunn/scripts/hidden_norms_propagation.py`, and `spbunn/scripts/balduzzi_analysis.py` as executable Hydra pipelines. Compose the common M1 run-directory and logging settings. Put pipeline parameters, random seeds, grids, training settings, tolerances, artifact names, and the run directory in configuration. Supply a small test configuration for each pipeline.

**FR2. MLP width pipeline.** Produce the legacy function-regression data and width-sweep settings. Preserve the legacy hyperparameter search, shallow ReLU model construction, widths, seeds, training procedure, convergence metrics, and predictions. Expose the numerical work as producer and transformer filters independent of Matplotlib, Hydra, and file output. Preserve the legacy prediction and convergence figure content and figure-name stems.

**FR3. MLP depth pipeline.** Produce the legacy function-regression data and depth-sweep settings. Preserve the legacy hyperparameter search, deep plain ReLU model construction, depths, seeds, training procedure, convergence metrics, and predictions. Expose the numerical work as producer and transformer filters independent of Matplotlib, Hydra, and file output. Preserve the legacy prediction and convergence figure content and figure-name stems.

**FR4. Hidden-norms propagation pipeline.** Produce random MLPs, inputs, output gradients, and perturbations for the configured initialization, depth, and residual variants. Measure forward hidden norms, backward gradient norms, finite-difference empirical sensitivity, and exact input-Jacobian singular-value diagnostics. Preserve legacy random-seed use, grids, initialization scales, residual scaling, finite-difference step, numerical tolerances, result ordering, and figure-name stems. Keep all numerical filters independent of Matplotlib, Hydra, and file output.

**FR5. Balduzzi analysis pipeline.** Construct the existing plain deep ReLU network and a self-contained residual deep ReLU network whose square hidden blocks compute `h + ReLU(W h + b)` with branch scale 1. This residual definition replaces the legacy import of a class that is absent from the repository and its available history. Construct the legacy input grid and compute gradients of summed network outputs with respect to each input. Use a new explicit production seed of 42 because the non-runnable legacy script has no fixed seed. Preserve legacy network dimensions, input grid, curve ordering, labels, automatic axis limits, and figure-name stems. Keep the network producer and gradient evaluator independent of Matplotlib, Hydra, and file output.

**FR6. Pipeline report builders.** Give each pipeline a report builder that converts in-memory results into the existing report model. Include accurate explanatory HTML and LaTeX for the actual teaching experiment and all legacy figures needed to interpret its results. Send every report model to the existing shared report writer. Do not duplicate HTML persistence, MathJax setup, figure-format selection, or stale-format cleanup.

**FR7. Documentation and developer checks.** Document the four execution commands, their production and test configurations, outputs, figure names, and numerical scope. Include the new numerical modules, report builders, orchestrators, and tests in the focused formatting, lint, type, and test checks. Keep the M1 report-writer threshold tests and the M2 pipeline tests unchanged and passing.

**FR8. Milestone boundary.** Migrate only the four M3 scripts. Do not migrate `simple_convolution.py` or other M4 behavior, add a validation subproject, change the constitution spec, or alter the shared report-writer output contract. No cross-repository change is required.

#### 2.2 Non-functional requirements

**NFR1. Filter isolation.** Producers only create experiment input and model instances. Transformers only compute numerical results. Report builders own presentation and explanatory content. Orchestrators only resolve configuration, connect explicit arguments and return values, invoke the shared writer, and release figures. Filters must not communicate through global state or the run directory.

**NFR2. Deterministic numerical behavior.** Preserve legacy data types, endpoint-inclusive grids, seeded model and data generation, training and evaluation order, and finite-difference and Jacobian calculations. Seed every relevant random generator explicitly and avoid changing random-number consumption outside the configured experiment. Use numerical tolerances in tests where Torch and linear-algebra implementations can affect rounding. Do not introduce a model-performance threshold.

**NFR3. Input and resource safety.** Validate empty sweeps, invalid sample, grid, depth, width, seed, and training counts, and invalid finite-difference steps with clear errors. Keep artifact names unique and safe. Avoid retaining computation graphs after results are collected. Close all report figures after writing and when report writing raises an exception.

**NFR4. Reproducible and isolated checks.** Use temporary run directories, small numerical cases, a headless Matplotlib backend, and no network access. Test configurations must reduce training and sweep sizes without changing the tested data flow. End-to-end tests must not leave figures in the process working directory or depend on timestamped output discovery.

**NFR5. Focused maintenance.** Use small typed data objects and functions instead of a generic pipeline inheritance framework. Reuse the existing MLP, activation, hook, reporting, and plotting utilities where they match the legacy behavior. Share only genuinely common numerical or presentation logic and do not reformat unrelated legacy modules.

### 3. Acceptance criteria

**AC1. Configuration and execution (FR1, FR6, NFR4).** Automated tests must compose the production and test Hydra configurations for all four pipelines. Each executable script must run successfully with its test configuration and an explicit temporary run directory. The configured HTML artifact and every referenced figure must exist in that directory, with exactly one SVG or PNG artifact per figure stem. No legacy figure may appear in the process working directory.

**AC2. Width and depth behavior (FR2, FR3, NFR2).** Numerical tests must use small deterministic cases to verify data generation, model shapes, width and depth sweep ordering, seeded repetition, hyperparameter selection, training-history alignment, predictions, and convergence metrics. Tests must verify that production configurations retain the legacy widths, depths, seeds, grids, learning-rate candidates, and training counts. Builder tests must verify ordered prediction and convergence figures, labels, explanatory LaTeX, and stable unique legacy-compatible stems.

**AC3. Hidden-norms propagation behavior (FR4, NFR2, NFR3).** Numerical tests must verify the configured initialization variants and depths, hidden-norm and gradient-norm alignment across seeds, finite-difference sensitivity against a hand-calculated or independently calculated small case, and exact input-Jacobian singular values for plain and residual networks. Tests must preserve production grids, seeds, initialization scales, residual scaling, finite-difference step, and tolerances. Builder tests must verify complete ordered diagnostics figures, labels, explanatory LaTeX, and stable unique stems.

**AC4. Balduzzi analysis behavior (FR5, NFR2).** Numerical tests must verify plain and residual network construction, the input grid, and input gradients against automatic differentiation or a hand-calculated small case. Tests must verify the new explicit production seed 42, legacy network shapes, unit residual branch scale, and grid. Builder tests must verify both ordered curves, labels, automatic axis limits, explanatory LaTeX, and stable unique stems.

**AC5. Shared reporting contract (FR6, NFR1, NFR3).** Builder and pipeline tests must verify report titles, explanatory HTML with LaTeX, figure counts, safe unique artifact names, and figure references. The generated HTML must contain the M1 embedded MathJax runtime without an external script. Existing M1 tests must continue to verify SVG output at or below 102,400 bytes, PNG-only output above that size, and stale-format removal. Tests must also verify figure cleanup after success and writer failure.

**AC6. Quality and scope (FR7, FR8, NFR1–NFR5).** The complete test suite, focused static checks, and package build must pass. A diff review must confirm that only M3 pipeline behavior and supporting documentation changed, that numerical filters perform no file output, that M4 remains untouched, and that the constitution and shared report-writer contract are unchanged.

**AC7. Human educational review.** A human reviewer must open all four generated reports, inspect their explanations, equations, figures, and output files for educational correctness, and explicitly approve each pipeline. At least one report must be opened with network access disabled to confirm that LaTeX renders. Automated checks alone do not establish acceptance.

### 4. High-level solution design

M3 implements the four architecture and propagation flows from the constitution. Yellow components are created or migrated in this spec. The report writer and report models are existing M1 components.

**Hidden norms propagation — `hidden_norms_propagation.py` orchestrator**

```mermaid
flowchart LR
  O["Hidden norms propagation orchestrator"] -. invokes .-> P["Random MLP input producer"]
  P -- "Inputs, output gradients, and experiment settings" --> T["Propagation diagnostics evaluator"]
  T -- "Norm and conditioning results" --> B["Propagation diagnostics report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

**Balduzzi analysis — `balduzzi_analysis.py` orchestrator**

```mermaid
flowchart LR
  O["Balduzzi analysis orchestrator"] -. invokes .-> P["Network and input grid producer"]
  P -- "Networks and input grid" --> T["Input gradient evaluator"]
  T -- "Plain and residual input-gradient curves" --> B["Balduzzi analysis report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

**MLP width experiment — `mlp_width_experiment.py` orchestrator**

```mermaid
flowchart LR
  O["MLP width orchestrator"] -. invokes .-> P["Function regression data producer"]
  P -- "Training data and width sweep settings" --> T["Width sweep trainer"]
  T -- "Predictions and convergence metrics" --> B["MLP width report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

**MLP depth experiment — `mlp_depth_experiment.py` orchestrator**

```mermaid
flowchart LR
  O["MLP depth orchestrator"] -. invokes .-> P["Function regression data producer"]
  P -- "Training data and depth sweep settings" --> T["Depth sweep trainer"]
  T -- "Predictions and convergence metrics" --> B["MLP depth report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

At component level, `spbunn.mlp.architecture` contains the width/depth input and result types, data producers, model construction, hyperparameter search, and training evaluators. `spbunn.mlp.propagation` contains the propagation and Balduzzi input and result types, network producers, norm and sensitivity evaluators, Jacobian diagnostics, and input-gradient evaluator. These modules reuse the existing MLP utilities and do not import the report writer or save artifacts.

`spbunn.reporting.mlp` contains the four pipeline-specific report builders, axes-level plotting helpers, and explanatory HTML. Builders create Matplotlib figures and figure models but do not serialize them. The four scripts contain only Hydra entry points and orchestration. They pass `cfg.result_dir` explicitly to `ReportWriter` and close every figure in a `finally` block.

Production configurations retain all legacy numerical settings and figure-name stems. Dedicated small test configurations reduce data, grid, sweep, and training counts without changing the tested data flow. Each report uses a configured descriptive HTML artifact name and legacy-compatible figure stems. The legacy width/depth working-directory JSON caches are removed; search results pass only through explicit in-memory pipes, so a full production run repeats its configured search. The non-runnable Balduzzi residual import is replaced by the explicit unit-scale residual definition in FR5, with seed 42 recorded in configuration and documentation. No new terminology or constitution-spec change is necessary.
