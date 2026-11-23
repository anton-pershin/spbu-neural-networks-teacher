## Gradient flow audit

**Status**: active

### Objective

Provide an executable copy-memory experiment that separates vanilla-RNN gradient decay from the LSTM direct cell-state mechanism and measures what training changes. The approved scope and maintainer decisions are in [issue #17](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/17) and the [gradient flow audit general spec](../specs/17-gradient-flow-audit-general-spec.md).

### Current constraints

- Follow the constitution's Hydra pipeline, explicit-pipe, HTML-report, figure, and shared-report-writer requirements.
- Reuse the merged BPTT copy task and retain the existing BPTT report behavior.
- Use separate fixed-gap runs, matched data and update budgets, zero forget bias for the main LSTM arm, and at least 128 audit sequences.
- Normalize at the last blank state before reproduction. Keep full cell gradients, vector-valued direct cell-path contributions, and scalar mean-gate products distinct.
- Treat `0.5^tau`, near-one forget gates, and a delimiter dip as references or observations, not mandatory outcomes.
- Report negative long-gap learning outcomes and avoid portable performance thresholds.
- Preserve the approved visual investigation, including all three measurement formulas and controlled animations, as a standalone artifact.

### Current state

PR #18 supplies deterministic copy batches, vanilla-RNN construction, explicit retained states, and matched-weight float64 checks. Issue #16 intentionally excluded training and LSTMs. The new pipeline needs trained-model APIs, an explicit LSTM gate recurrence, optimization and validation, a boundary-relative audit, and a dedicated report.

### Contexts

| ID | Solution implementation | Validation setup | Runtime and environment | Baseline |
|----|-------------------------|------------------|-------------------------|----------|
| C1 | Temporary RNN/LSTM probes | Copy task; gaps 20 and 200; batch 128; hidden size 64; fixed validation and audit batches | Python 3.12.13; PyTorch 2.15.0; CPU; one thread unless noted | Merged BPTT audit |
| C2 | Approved visual investigation | Same measured data; controlled HTML animations; standalone offline artifact | Chromium and jsdom checks | C1 |
| C3 | Repository implementation | Spec-defined unit, reduced-pipeline, report, static, and package checks | Project Python 3.12 environment; CPU | C2 |

### Current understanding

#### Supported by evidence

- An explicit recurrence must retain every separate state. A packed output alone does not expose internal recurrent adjoints.
- The full LSTM cell adjoint contains direct cell-state and other recurrent/gate paths. The direct contribution is a vector-valued forget-gate product applied to the upstream cell adjoint.
- A product of scalar mean gates can differ greatly from the direct-path vector norm because it removes gate variation and alignment with the upstream adjoint.
- With zero forget bias, the scalar mean-gate product is close to `0.5^tau`; neither measured vector norm is generally required to follow it.
- At gap 20 after 5,000 matched updates, validation loss was 1.65421 for the zero-bias LSTM, 2.08032 for the default RNN, and 2.08027 for the orthogonal arm. At 10,000 updates, the values were 0.90302, 1.80859, and 2.07961. The default-RNN gradient curve changed in the longer run.
- At gap 200 after 5,000 updates, all arms remained near memoryless loss. Matched-weight float64 ratios at lag 200 were `3.778e-85` for the default RNN and `6.264e-116` for the orthogonal arm. The zero-bias LSTM full/direct ratios were `8.555e-14` and `8.563e-14`; the bias-1 control gave `4.067e-6` and `3.680e-6`.
- Training increased the zero-bias LSTM direct-path ratio at lag 200 from `9.356e-54` to `8.563e-14`. This improved gradient flow without producing copy-task learning within the budget.
- The corrected mean forget activation at gap 200 was 0.499 before training, 0.731 after zero-bias training, and 0.902 in the bias-1 control. Near-one means and a delimiter dip are not guaranteed.
- Thread configuration can change the post-plateau training trajectory. Exact trained losses are not portable regression targets.

#### Tentative or unresolved

- Gaps 50 and 100 use the same approved fixed-gap configuration but were not separately investigated before specification.
- The production implementation must demonstrate that its packed/explicit recurrences preserve the probe's numerical behavior.
- Human educational and offline-browser review of the final production report remains necessary.

### Experiments and decisions

| ID | Idea or question | Context | Result and evidence links | Decision and rationale |
|----|------------------|---------|---------------------------|------------------------|
| E1 | What does the merged BPTT increment supply? | C1 | PR #18 supplies copy batches, vanilla-RNN construction, retained-state audits, and float64 checks but no training or LSTM. [Issue investigation](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/17#issuecomment-7039821869) | Reuse shared task/numerical behavior in a separate pipeline; leave the BPTT report unchanged. |
| E2 | Is the full cell adjoint just a product of forget gates? | C1 | Full autograd also includes hidden-state and gate paths. The direct cell-path term can be calculated separately from one backward pass. | Plot full and direct curves with the same boundary normalization; do not call them identical. |
| E3 | Does zero forget bias imply a measured `0.5^tau` norm? | C1 | Short-lag direct values were close, but long-lag vector norms diverged. The scalar mean-gate product followed the reference. | Keep `0.5^tau` only as a mean-gate reference. Measure gates and report observed behavior. |
| E4 | Can the three arms learn gap 20 under a matched budget? | C1 | The zero-bias LSTM learned after a plateau; both RNN arms remained near baseline at update 5,000. The default RNN started learning later. [Training evidence](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/17#issuecomment-7039821869) | Use 5,000 updates for the approved snapshot and show training history without a universal ranking claim. |
| E5 | Does the LSTM learn gap 200 with the same budget? | C1 | No arm learned the task. The LSTM nevertheless retained much more gradient, and its full curve matched the vector-valued direct term at measured blank-gap lags. [Gap-200 evidence](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/17#issuecomment-7039821869) | Keep the negative result. Do not claim that gradient flow guarantees learning. Continue training investigation only if long-gap task success is required. |
| E6 | Did the first gate aggregation have the intended axes? | C1 | Visual preparation exposed a time/hidden-axis error. Recalculation averages over batch and hidden components per time step. Full/direct curves and loss results are unchanged. | Use corrected gate summaries and record the correction explicitly. [Visual-investigation note](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/17#issuecomment-7090002730) |
| E7 | Can the investigation be explained visually with course notation? | C2 | The hosted page used GetSomeMath notation, three controlled animations, five measured figures, and five confirmation points. Fifty DOM checks passed; Chromium rendered all mathematics without external requests. The maintainer accepted all five points and requested explicit measurement formulas plus retention of the artifact. [Maintainer approval](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/17#issuecomment-7090454123) | Preserve the standalone page in [the investigation artifact](gradient-flow-audit/index.html), add the three formulas to both reports, and implement the approved spec. |

### Next candidates

- [ ] Implement and check the approved pipeline and production report.
- [ ] Obtain human educational approval of the final report with external network access blocked.
- [ ] Continue long-gap training only if the maintainer requires LSTM task success at gap 200.
- [ ] Integrate approved reports into GetSomeMath separately if requested.

### Linked specs

- [17 — Gradient flow audit general spec](../specs/17-gradient-flow-audit-general-spec.md): defines this implementation increment.
- [16 — BPTT instability general spec](../specs/16-bptt-instability-general-spec.md): supplies the existing recurrent-task and numerical baseline.
