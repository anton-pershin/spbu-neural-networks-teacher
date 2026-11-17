## M2 Optimization pipelines

### 1. Executive summary

#### 1.1 Spec description

Implement roadmap item M2 for the learning-rate schedulers, loss-landscape comparison, and simple mini-batch optimization experiments. Migrate each legacy script to a thin Hydra orchestrator with an explicit producer, numerical transformer, and report builder. Each run must create a complete HTML report with LaTeX support and its figures through the shared report writer in the configured run directory.

#### 1.2 Spec motivation

M2 is the next roadmap step after the merged M1 foundation. The migration preserves the teaching experiments while separating numerical work, presentation, configuration, and artifact output. This separation makes the experiments testable and prevents pipeline-specific report-writing rules.

#### 1.3 Implementation repos

- `anton-pershin/spbu-neural-networks-teacher` — solution implementation and management repo.

### 2. Requirement analysis

#### 2.1 Functional requirements

**FR1. Hydra pipeline entry points.** Expose `spbunn/scripts/schedulers.py`, `spbunn/scripts/loss_landscape.py`, and `spbunn/scripts/simple_loss_optimization.py` as executable Hydra pipelines. Compose the common M1 run-directory and logging settings. Put pipeline parameters, seeds, grid sizes, artifact names, and the run directory in configuration. Supply a small test configuration for each pipeline.

**FR2. Learning-rate scheduler pipeline.** Produce the configured epoch grid and scheduler settings. Evaluate the existing linear, step, reduce-on-plateau, and cosine-annealing teaching curves without changing their numerical values or transition points. Preserve their configured annotations, axis labels, and stable figure-name stems. Make the schedule evaluator independent of Matplotlib and Hydra.

**FR3. Loss-landscape pipeline.** Produce noiseless `float64` regression data, parameter grids, and optimizer settings. Compute the MSE landscape with matrix index order `(theta_0, theta_1)`. Compute full-batch trajectories and pre-update loss histories for SGD, SGD with momentum, SGD with Nesterov momentum, RMSprop, and Adam. Preserve the configured initial point, learning rates, momentum, Adam betas, optimizer order, contour levels, and comparison axis limits.

**FR4. Simple loss-optimization pipeline.** Produce a seed-42 permutation of the training grid and a separate ordered landscape data set by default. Preserve the legacy distinction between training inputs on `[-10, 10]` and landscape inputs on the function's configured interval. Run mini-batch SGD and compute its loss landscape. Preserve the effective 25 updates of the legacy production configuration, but express the update count directly and stop after exactly that count. Record the initial parameter point, each post-update point, and each pre-update loss. Repair the legacy missing-landscape-limits failure and label the parameter axes as `theta_0` and `theta_1`.

**FR5. Pipeline report builders.** Give each pipeline a report builder that converts in-memory results into the existing report model. Include accurate explanatory HTML and LaTeX for the actual teaching algorithms. Build one figure per scheduler, one figure per optimizer and function in the loss-landscape report, and one figure per function in the simple-optimization report. Send every report model to the existing shared report writer. Do not duplicate HTML persistence, MathJax setup, figure-format selection, or stale-format cleanup.

**FR6. Documentation and developer checks.** Document the three execution commands, their configurations, outputs, and numerical scope. Include the new numerical modules, report builders, orchestrators, and tests in the focused formatting, lint, type, and test checks. Keep the M1 report-writer threshold tests unchanged and passing.

**FR7. Milestone boundary.** Migrate only the three M2 scripts. Do not migrate M3 or M4 scripts, add a validation subproject, change the constitution spec, or alter the shared report-writer output contract. No cross-repository change is required.

#### 2.2 Non-functional requirements

**NFR1. Filter isolation.** Producers only create experiment input. Transformers only compute numerical results. Report builders own presentation and explanatory content. Orchestrators only resolve configuration, connect explicit arguments and return values, invoke the shared writer, and release figures. Filters must not communicate through global state or the run directory.

**NFR2. Deterministic numerical behavior.** Preserve `float64` objective calculations, endpoint-inclusive grids, deterministic ordered data, and seeded mini-batch ordering. Use numerical tolerances in tests where Torch optimizer versions can affect rounding. Do not introduce a model-performance threshold.

**NFR3. Input and resource safety.** Validate unsupported schedule or optimizer names, empty grids, invalid sample counts, and invalid update counts with clear errors. Keep artifact names unique and safe. Close all report figures after writing and when report writing raises an exception.

**NFR4. Reproducible and isolated checks.** Use temporary run directories, small numerical cases, a headless Matplotlib backend, and no network access. End-to-end tests must not leave figures in the process working directory or depend on timestamped output discovery.

**NFR5. Focused maintenance.** Use small typed data objects and functions instead of a generic pipeline inheritance framework. Share landscape calculation and axes-level plotting between the two optimization pipelines. Do not reformat unrelated legacy modules.

### 3. Acceptance criteria

**AC1. Configuration and execution (FR1, FR5, NFR4).** Automated tests must compose the production and test Hydra configurations for all three pipelines. Each executable script must run successfully with its test configuration and an explicit temporary run directory. The configured HTML artifact and every referenced figure must exist in that directory, with exactly one SVG or PNG artifact per figure stem. No legacy figure may appear in the process working directory.

**AC2. Scheduler behavior (FR2, NFR2).** Numerical tests must check all four complete 40-point curves or their defining segments, including the linear plateau, step transitions, reduce-on-plateau transitions, and four repeated cosine quarters. Builder tests must check four ordered figures, annotations, axis labels, unique legacy-compatible stems, and explanatory LaTeX content.

**AC3. Landscape and full-batch optimization (FR3, NFR2).** A small hand-calculated case must verify MSE values and matrix orientation. Tests must verify the five optimizer settings and order, deterministic full-batch input, the initial trajectory point, `n_steps + 1` parameter points, `n_steps` pre-update losses, and a known first SGD update. Builder tests must verify one figure per optimizer, shared landscape limits, logarithmic loss axes, and stable unique stems.

**AC4. Mini-batch optimization (FR4, NFR2).** Tests must verify seeded permutation order, separate training and landscape intervals, exact configured update counts, parameter-history and loss-history alignment, and production configuration equivalence to 25 updates. A small end-to-end run must verify that the repaired landscape rendering completes and uses `theta_0` and `theta_1` labels.

**AC5. Shared reporting contract (FR5, NFR1, NFR3).** Builder and pipeline tests must verify report titles, explanatory HTML with LaTeX, figure counts, safe unique artifact names, and figure references. The generated HTML must contain the M1 embedded MathJax runtime without an external script. Existing M1 tests must continue to verify SVG output at or below 102,400 bytes, PNG-only output above that size, and stale-format removal. Tests must also verify figure cleanup after success and writer failure.

**AC6. Quality and scope (FR6, FR7, NFR1–NFR5).** The complete test suite, focused static checks, and package build must pass. A diff review must confirm that only M2 pipeline behavior and supporting documentation changed, that numerical filters perform no file output, and that the shared report-writer contract is unchanged.

**AC7. Human educational review.** A human reviewer must open all three generated reports, inspect their explanations, equations, figures, and output files for educational correctness, and explicitly approve each pipeline. At least one report must be opened with network access disabled to confirm that LaTeX renders. Automated checks alone do not establish acceptance.

### 4. High-level solution design

M2 implements the three optimization flows from the constitution. Yellow components are created or migrated in this spec. The report writer and report models are existing M1 components.

**Learning-rate schedulers — `schedulers.py` orchestrator**

```mermaid
flowchart LR
  O["Scheduler orchestrator"] -. invokes .-> P["Epoch grid producer"]
  P -- "Epoch grid and scheduler settings" --> T["Learning-rate schedule evaluator"]
  T -- "Learning-rate curves" --> B["Scheduler report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

**Loss landscape — `loss_landscape.py` orchestrator**

```mermaid
flowchart LR
  O["Loss landscape orchestrator"] -. invokes .-> P["Function data and parameter grid producer"]
  P -- "Regression data, parameter grids, and optimizer settings" --> T["Landscape and trajectory evaluator"]
  T -- "Loss landscapes, trajectories, and loss curves" --> B["Loss landscape report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

**Simple loss optimization — `simple_loss_optimization.py` orchestrator**

```mermaid
flowchart LR
  O["Simple loss optimization orchestrator"] -. invokes .-> P["Function data and parameter grid producer"]
  P -- "Training data, ordered landscape data, and parameter grid" --> T["Mini-batch optimization evaluator"]
  T -- "Loss landscape, parameter history, and loss history" --> B["Simple optimization report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

At component level, `spbunn.optimization.schedulers` contains scheduler input/result types, the producer, and pure schedule evaluators. `spbunn.optimization.landscape` contains shared regression/grid inputs, landscape calculation, full-batch optimization, and optimizer result types. `spbunn.optimization.mini_batch` contains simple-optimization input creation and the exact-step mini-batch evaluator. These modules do not import the report writer or save artifacts.

`spbunn.optimization.plotting` contains axes-level landscape, trajectory, and loss-curve helpers shared by the report builders. `spbunn.reporting.optimization` contains the three pipeline-specific report builders and their explanatory HTML. Builders create Matplotlib figures and figure models but do not serialize them. The three scripts contain only Hydra entry points and orchestration. They pass `cfg.result_dir` explicitly to `ReportWriter` and close every figure in a `finally` block.

Production configurations retain the active linear-function experiment and its commented alternatives. A shared Hydra function group prevents configuration drift between the two landscape-based pipelines. Dedicated small test configurations reduce sample, grid, and step counts without changing the tested data flow. Each report uses a configured descriptive HTML artifact name and legacy-compatible figure stems.
