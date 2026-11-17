## M4 Convolution pipeline

### 1. Executive summary

#### 1.1 Spec description

Implement roadmap item M4 for the simple convolution teaching experiment. Migrate the legacy `simple_convolution.py` script to a thin Hydra orchestrator with an explicit image and kernel producer, numerical convolution evaluator, and convolution report builder. Each run must create a complete HTML report with LaTeX support and three figures through the shared report writer in the configured run directory.

#### 1.2 Spec motivation

M4 is the remaining roadmap step after the merged M1 foundation. The migration preserves the horizontal, vertical, and diagonal convolution examples while separating input loading, numerical work, presentation, configuration, and artifact output. This separation makes the experiment testable and removes legacy working-directory input and output assumptions.

#### 1.3 Implementation repos

- `anton-pershin/spbu-neural-networks-teacher` — solution implementation and management repo.

### 2. Requirement analysis

#### 2.1 Functional requirements

**FR1. Hydra pipeline entry point.** Expose `spbunn/scripts/simple_convolution.py` as an executable Hydra pipeline. Compose the common M1 run-directory and logging settings. Put the RGB and grayscale input paths, kernel size, convolution repetition count, report metadata, and run directory in configuration. Supply a small test configuration. Resolve relative input paths from the original invocation directory rather than a Hydra-changed working directory.

**FR2. Image and kernel producer.** Load the configured RGB image without an alpha channel and the configured grayscale image with one channel. Add a batch dimension and convert both images to `float64`. Preserve the legacy RGB normalization calculation before convolution; leave the grayscale image unnormalized. Construct centered horizontal and vertical kernels and an identity diagonal kernel with the configured odd size. Replicate each RGB kernel into a three-output-channel, one-input-channel-per-group filter for grouped convolution, and construct the corresponding one-channel diagonal filter.

**FR3. Convolution evaluator.** Apply the configured number of `torch.nn.functional.conv2d` operations with stride 1, `padding="same"`, no bias, and groups 3 for RGB and 1 for grayscale. Preserve the legacy normalization after every RGB convolution and its absence from the diagonal experiment. Return the original and convolved images with their experiment settings through explicit in-memory results. Perform no rendering or file output.

**FR4. Convolution report builder.** Convert the results into the existing report model with accurate explanatory HTML and LaTeX for same-padding grouped convolution, the three kernels, repetition, and normalization. Build two 2-by-3 RGB channel figures for the horizontal and vertical experiments and one 2-by-1 grayscale figure for the diagonal experiment. Each figure must show the original image above the convolved image, use the legacy reverse-grayscale presentation, show the configured repetition count, and use the stable stems `stripes_convoluted_h`, `stripes_convoluted_v`, and `waves_convoluted`.

**FR5. Shared report output.** Send the report model to the existing shared report writer with `cfg.result_dir`. Write no figure directly from the orchestrator, producer, evaluator, or builder. Preserve the M1 HTML, embedded MathJax, SVG/PNG threshold, and stale-format cleanup behavior. Close all report figures after writing and when writing raises an exception.

**FR6. Documentation and developer checks.** Document the execution command, required image-path overrides, production and test configurations, outputs, figure names, and numerical behavior. Include the new numerical module, report builder, orchestrator, and tests in the focused formatting, lint, type, and test checks. Update the roadmap status of the merged M1–M3 milestones and of M4 when its implementation is complete.

**FR7. Milestone boundary.** Migrate only the M4 convolution script and its supporting components. Do not change other pipelines, add a validation subproject, or alter the shared report-writer output contract. No cross-repository change is required. Remove the legacy dead code, unused imports, hard-coded script input paths, and direct file-based `savefig` calls.

#### 2.2 Non-functional requirements

**NFR1. Filter isolation.** The producer only loads experiment input and constructs kernels. The evaluator only computes numerical results. The report builder owns presentation and explanatory content. The orchestrator only resolves configuration, connects explicit arguments and return values, invokes the shared writer, and releases figures. Filters must not communicate through global state or the run directory.

**NFR2. Deterministic numerical behavior.** Preserve legacy `float64` calculations, image tensor layout, normalization reduction dimensions, kernel values, convolution order, and result ordering. Use exact comparisons for kernels and simple hand-calculated convolutions, and numerical tolerances where Torch operations can affect rounding. Do not introduce a model-performance threshold.

**NFR3. Input and resource safety.** Validate positive odd kernel sizes, positive repetition counts, non-empty configured paths, expected image dimensions and channel counts, and compatible filters with clear errors. Keep artifact names unique and safe. Avoid mutation of producer-owned input tensors by the evaluator. Close all report figures after success and writer failure.

**NFR4. Reproducible and isolated checks.** Use temporary input images and run directories, small numerical cases, a headless Matplotlib backend, and no network access. End-to-end tests must not leave figures in the process working directory or depend on timestamped output discovery. Test configuration composition must not require input files to exist until the producer runs.

**NFR5. Focused maintenance.** Use small typed data objects and functions instead of a generic pipeline inheritance framework. Reuse the M1 report models and writer without changing their interface. Do not reformat unrelated legacy modules.

### 3. Acceptance criteria

**AC1. Configuration and execution (FR1, FR5, NFR4).** Automated tests must compose the production and test Hydra configurations. The executable script must run successfully with its test configuration, temporary RGB and grayscale input images, and an explicit temporary run directory. The configured HTML artifact and all three referenced figures must exist in that directory, with exactly one SVG or PNG artifact per figure stem. No legacy figure may appear in the process working directory.

**AC2. Producer behavior (FR2, NFR2, NFR3).** Tests must use small generated images to verify alpha removal, grayscale loading, batch and channel dimensions, `float64` conversion, legacy RGB normalization, and the absence of grayscale normalization. Tests must verify the exact horizontal, vertical, and diagonal kernel values and filter shapes for the configured size. Production configuration tests must verify size 7, 20 repetitions, and both configurable input paths.

**AC3. Numerical behavior (FR3, NFR2, NFR3).** A small hand-calculated case must verify same-padding convolution values, grouping, repetition count, and normalization behavior. Tests must verify result order and original-image preservation and must reject invalid kernel sizes, repetition counts, image shapes, and incompatible filters.

**AC4. Report content (FR4, FR5).** Builder tests must verify the report title and artifact name, explanatory HTML with LaTeX, three ordered figures with stable unique stems, 2-by-3 and 2-by-1 layouts, original and convolved panels, reverse-grayscale channel images, and titles containing the configured repetition count. Pipeline tests must verify figure references and the M1 embedded MathJax runtime without an external script.

**AC5. Shared output and resource contract (FR5, NFR1, NFR3).** Existing M1 tests must continue to verify SVG output at or below 102,400 bytes, PNG-only output above that size, and stale-format removal. M4 tests must verify that all output passes through the shared writer and that figures close after success and writer failure.

**AC6. Quality and scope (FR6, FR7, NFR1–NFR5).** The complete test suite, focused static checks, and package build must pass. Documentation and roadmap statuses must reflect the implementation. A diff review must confirm that only M4 behavior and supporting documentation changed, that numerical filters perform no file output, and that the shared report-writer contract is unchanged.

**AC7. Human educational review.** A human reviewer must open the generated report, inspect its explanations, equations, three figures and their original/convolved panels, and output files for educational correctness, and explicitly approve the pipeline. The report must be opened with network access disabled to confirm that LaTeX renders. Automated checks alone do not establish acceptance.

### 4. High-level solution design

M4 implements the simple convolution flow from the constitution. Yellow components are created or migrated in this spec. The report writer and report models are existing M1 components.

**Simple convolution — `simple_convolution.py` orchestrator**

```mermaid
flowchart LR
  O["Simple convolution orchestrator"] -. invokes .-> P["Image and kernel producer"]
  P -- "Images and convolution kernels" --> T["Convolution evaluator"]
  T -- "Original and convolved images" --> B["Convolution report builder"]
  B -- "Report model" --> C["Shared report writer<br/>EXISTING"]
  classDef current fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
  class O,P,T,B current;
```

At component level, `spbunn.convolution` contains typed experiment and result objects, the configured image and kernel producer, and the pure repeated-convolution evaluator. It does not import Matplotlib or the report writer and does not save artifacts. Image paths are configuration values resolved by the producer against Hydra's original invocation directory.

`spbunn.reporting.convolution` contains the convolution report builder, channel-level plotting helpers, and explanatory HTML. The builder creates Matplotlib figures and figure models but does not serialize them. `spbunn.scripts.simple_convolution` contains only the Hydra entry point and orchestration. It passes `cfg.result_dir` explicitly to `ReportWriter` and closes every figure in a `finally` block.

The production configuration retains the legacy kernel size, repetition count, and figure-name stems. Its configurable image-path defaults retain the legacy file names, while command-line overrides support absolute paths and paths relative to the original invocation directory; no image assets are added to the repository. The test configuration reduces the repetition count, and tests supply generated image paths explicitly. No new terminology or constitution design change is necessary; only roadmap statuses and M4 usage documentation change.
