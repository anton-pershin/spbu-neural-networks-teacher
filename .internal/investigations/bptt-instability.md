## BPTT instability

**Status**: active

### Objective

Provide an executable teaching example that demonstrates exponentially vanishing and exploding BPTT gradients in vanilla RNNs without a selected unstable weight scale. Link the approved discussion in [issue #16](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/16) and the implementation contract in [the BPTT instability general spec](../specs/16-bptt-instability-general-spec.md).

### Current constraints

- Follow the constitution's Hydra pipeline, explicit-pipe, HTML-report, figure, and shared-report-writer requirements.
- Use simple synthetic tasks, ordinary documented initializations, fixed seeds, and at least 128 sequences in the production audit.
- Measure separate hidden-state adjoints with one backward pass and use matched-weight float64 checks for numerical range effects.
- Keep the approved increment limited to the copy memory task and adding problem. Do not add Alice evaluation, TBPTT, clipping, LSTMs, or model training.
- Keep GetSomeMath integration outside this repository change.

### Current state

The repository has no recurrent pipeline. M1 supplies the shared report model and writer. Earlier pipelines provide thin Hydra orchestrators, typed producers and transformers, report builders, and end-to-end test patterns. Issue #17 independently proposes an LSTM comparison on the copy task; the approved #16 increment contains only vanilla-RNN instability.

### Contexts

| ID | Solution implementation | Validation setup | Runtime and environment | Baseline |
|----|-------------------------|------------------|-------------------------|----------|
| C1 | Temporary Alice word-level RNN probe | N-gram sentence split; full and accumulated TBPTT; controlled recurrent spectral radii | Python 3.12.13; PyTorch 2.15.0+cpu; one CPU thread | No RNN example |
| C2 | Temporary copy-task RNNCell probe | Eight symbols, ten copied symbols, gaps 20–200; default and orthogonal initialization; batch 128 | Same runtime; explicit retained states | C1 |
| C3 | Temporary adding-task RNNCell probe | Two marked values, lengths 20–200; several activation and initialization choices; batch 128 | Same runtime; explicit retained states | C2 |
| C4 | Repository implementation on this branch | Spec-defined unit, scientific regression, report, subprocess, static, and package checks | Project Python 3.12 virtual environment; CPU | C3 |

### Current understanding

#### Supported by evidence

- A selected recurrent spectral radius of 2.0 produces large Alice gradients, but this is not convincing evidence of instability under ordinary initialization. The maintainer rejected this design.
- On the copy task, default PyTorch initialization produces exponential decay. Orthogonal initialization delays decay but does not remove it. Three seeds and float64 checks support this result.
- Retaining the single packed output tensor of `nn.RNN` records only its direct decoder gradient. An explicit `RNNCell` loop with separate retained states is necessary for the intended adjoint audit.
- A 20-step TBPTT window can delete the complete copy-signal gradient. Clipping cannot restore deleted graph edges. These mechanisms are outside the final approved scope.
- The adding problem with activation-aware Xavier `tanh` initialization produces backward growth while forward states remain bounded by one and losses remain finite. Ten tested seeds showed growth at length 200.
- ReLU with He initialization and Xavier `tanh` with gain 1 did not reliably explode. A linear RNN also exploded in the forward state, so these candidates are less suitable.

#### Tentative or unresolved

- Human educational review of the rendered production report remains necessary.
- Browser review with network access disabled remains necessary for the complete report, although the shared MathJax runtime is already tested.
- The observations do not establish that every reasonable initialization has the same failure direction. The report must retain this limitation.

### Experiments and decisions

| ID | Idea or question | Context | Result and evidence links | Decision and rationale |
|----|------------------|---------|---------------------------|------------------------|
| E1 | Can an Alice RNN with a controlled recurrent scale demonstrate explosion? | C1 | Radius 2.0 gave full-BPTT global gradient 1,411 on a 180-event sentence; truncation reduced it. [First investigation](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/16#issuecomment-7002390301) | Reject after maintainer review. The selected scale does not demonstrate ordinary-initialization instability. |
| E2 | Does equal-update TBPTT isolate truncation on Alice? | C1 | Accumulating chunk gradients gave every arm 1,052 updates per epoch. Clipping slowed early learning in the probe and did not improve validation perplexity. Same issue evidence | Do not claim a universal performance ordering. Remove TBPTT and clipping when the maintainer narrows scope. |
| E3 | Does the copy task demonstrate vanishing under ordinary initialization? | C2 | At gap 200, default-arm ratios at lag 200 were below `1e-38` in float64; orthogonal-arm float32 ratios were `7.86e-10` to `3.56e-7`. [Copy-task investigation](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/16#issuecomment-7007383458) | Use the copy task for exponential decay with default and orthogonal initialization. |
| E4 | Why did the first copy audit show only ten nonzero output states? | C2 | The packed `nn.RNN` output retained only direct decoder gradients. Separate `RNNCell` states exposed the recurrent adjoints and the expected curves. Same issue evidence | Use an explicit time loop and retain each hidden state. Add a regression test for this measurement contract. |
| E5 | Can training at gap 20 distinguish full BPTT from TBPTT? | C2 | Full default, full orthogonal, and TBPTT arms remained near the memoryless baseline after 3,000 Adam updates. Same issue evidence | Do not base the specification on an unobserved training success. The final increment audits gradients without fitting. |
| E6 | Can ReLU with He initialization or Xavier `tanh` with gain 1 demonstrate explosion? | C3 | Neither choice reliably exploded on the copy or adding task. [Adding-task investigation](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/16#issuecomment-7009680568) | Reject these candidates; retain the negative results and avoid a universal initialization claim. |
| E7 | Does a linear RNN give a clean explosion example? | C3 | Gradient growth occurred with forward-state growth, including very large losses and hidden values. Same issue evidence | Reject because forward and backward instability are mixed. |
| E8 | Does the adding problem isolate backward explosion under a documented initialization? | C3 | With Xavier `tanh` gain `5/3`, seed-42 first-state ratios were 11.85, 162.77, `6.07e3`, and `1.02e7` at lengths 20, 50, 100, and 200. Ten seeds gave ratios `2.76e2` to `6.10e7` at length 200; hidden states stayed bounded and losses finite. Same issue evidence | Select the adding problem with activation-aware Xavier `tanh` initialization and matched-weight float64 checks. |
| E9 | Does the repository implementation preserve the approved numerical behavior? | C4 | The implementation uses deterministic local generators, isolated model seeds, explicit retained `RNNCell` states, and matched-weight dtype checks. Independent autograd tests agree with the public audit. The production seed-42 adding case has finite loss 1.196123, bounded hidden states, parameter-gradient norm `1.281e7`, and first-state ratio `1.950e7`. The default copy float64 check retains nonzero gradients after float32 underflow. The complete run passed 275 tests, all static checks for 74 source files, and the package build. The production pipeline generated `bptt_instability.html` and two PNG figures. [Numerical tests](../../tests/test_rnn_audit.py), [scientific and pipeline tests](../../tests/test_rnn_pipeline.py), [report tests](../../tests/test_rnn_reporting.py), and [usage documentation](../../docs/bptt_instability.md) define the repeatable evidence. | The implementation satisfies the automated spec checks. Keep the investigation active for human educational review; do not treat gradient ratios as model-performance thresholds. |
| E10 | Is the exact scientific regression stable under a different BLAS thread count? | C4 | Independent review found that eight-thread BLAS changed the adding loss by about `4e-4` relative, exceeding the exact regression tolerance. A scoped fixture now sets one PyTorch intra-op thread and restores the previous value. The same review also identified hard-coded report lags and unchecked task-length metadata. The revision derives all summary lag columns from configuration and validates metadata against batch time dimensions. The full suite passed 276 tests with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`; static checks and the package build pass. [Review regression test](../../tests/test_rnn_pipeline.py), [audit shape checks](../../tests/test_rnn_audit.py), and [environment documentation](../../docs/bptt_instability.md) contain the evidence. | Resolve the review no-pass without widening tolerances. Keep exact regressions single-threaded and make the two maintenance fixes in the same revision. |
| E11 | Can the report expose concrete predictions and remain compact enough for SVG? | C4 | The maintainer requested input/output/target examples, orthogonal-initialization motivation, smaller SVG figures, and controlled HTML prediction animations. The revision adds deterministic complete per-step examples for both tasks, two inline animations with Play/Pause/Previous/Next controls, and an explanation of norm preservation and `tanh` derivative limits. Display curves use at most 32 points with preserved endpoints, combined copy panels, simplified major log ticks, and figure-local SVG text serialization; numerical arrays remain complete. Production SVG sizes are 66,899 bytes for copy and 47,144 bytes for adding, below the shared threshold. A DOM harness verified all controls, timer advancement, boundaries, and restart for both animations. The complete run passed 282 tests with eight threads, static checks for 77 files, and the package build. [Prediction examples](../../spbunn/rnn/examples.py), [animation presentation](../../spbunn/reporting/rnn_animation.py), and [report tests](../../tests/test_rnn_reporting.py) contain the evidence. | Satisfy all four educational-review requests without changing the numerical experiment or shared writer. Keep animations explicitly untrained and leave final browser approval to the maintainer. |
| E12 | Are exact adding values portable across CPUs with one BLAS thread? | C4 | GitHub CI on the animation revision produced finite adding loss 1.208894 instead of the local one-thread value 1.196123, even though the eight-thread local suite passed. This confirms cross-CPU numerical amplification beyond thread-count effects. The scientific regression now retains exact copy checks but asserts only finite outcomes, bounded states, large parameter gradients, and ordered backward growth for the chaotic length-200 adding case. Documentation identifies the displayed adding values as local reference observations. The revised local suite passed 282 tests with eight threads. [Scientific regression](../../tests/test_rnn_pipeline.py) and [environment documentation](../../docs/bptt_instability.md) contain the evidence. | Remove invalid cross-machine pointwise assertions without weakening the scientific growth check. Keep the single-thread fixture for exact copy regressions. |
| E13 | Do the final figures remain compact after the requested readability changes? | C4 | The maintainer requested larger fonts, no in-figure titles, detailed captions, linewidth 2, and larger markers. The revision uses 18-point base text, 20-point axis labels, 16-point ticks and legends, linewidth 2, and markersize 10 on length-trend curves. All axes titles are empty. Detailed task-specific captions explain both panels, initialization line styles, lag direction, log scales, pruning, and the zero floor. `FigureModel` now has an optional escaped caption rendered below the image; caption-less output and SVG/PNG selection are unchanged. Production SVG sizes are 65,431 bytes for copy and 45,733 bytes for adding. The full suite passed 283 tests with eight threads, static checks for 77 files, and the package build. [Figure presentation](../../spbunn/reporting/rnn.py), [shared caption tests](../../tests/test_reporting.py), and [BPTT report tests](../../tests/test_rnn_reporting.py) contain the evidence. | Satisfy the readability review with real HTML captions and a minimal backward-compatible shared reporting extension. |

### Next candidates

- [x] Complete the spec-defined automated checks and inspect the production report.
- [ ] Obtain human educational approval of the rendered report, including offline mathematical rendering.
- [x] Coordinate issue #17 through the [gradient flow audit investigation](gradient-flow-audit.md) without expanding this increment.
- [ ] Integrate the approved report into GetSomeMath separately if the maintainer wants it.

### Linked specs

- [16 — BPTT instability general spec](../specs/16-bptt-instability-general-spec.md): defines and bounds this implementation increment.
