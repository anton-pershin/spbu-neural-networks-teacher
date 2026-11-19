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

### Next candidates

- [x] Complete the spec-defined automated checks and inspect the production report.
- [ ] Obtain human educational approval of the rendered report, including offline mathematical rendering.
- [ ] Coordinate any future #17 LSTM audit through shared recurrent utilities without expanding this increment.
- [ ] Integrate the approved report into GetSomeMath separately if the maintainer wants it.

### Linked specs

- [16 — BPTT instability general spec](../specs/16-bptt-instability-general-spec.md): defines and bounds this implementation increment.
