## N-gram pipeline

**Status**: active

### Objective

Provide an executable teaching example that shows the value and limitations of probabilistic n-gram language models for the GetSomeMath `rnn/intro_to_sequence_modelling` material. Link the approved requirements in [issue #11](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/11) and the implementation contract in [the n-gram general spec](../specs/11-ngram-language-model-general-spec.md).

### Current constraints

- Follow the constitution's Hydra pipeline, explicit-pipe, HTML-report, figure, and shared-report-writer requirements.
- Run without network access and without a new runtime NLP dependency.
- Use the official NLTK Gutenberg corpus archive as the acquisition source, but vendor and hash-check the Alice resource for offline execution.
- Show held-out performance, sparsity, normalized top-k conditional probabilities, and all configured seeded samples without performance thresholds or selective sample filtering.
- Keep GetSomeMath integration outside this repository change.

### Current state

M1 supplies the shared report model and writer. M2 and M3 provide thin Hydra orchestrators, typed producers and transformers, report builders, and end-to-end test patterns. No n-gram pipeline existed before this investigation. The approved candidate is a word-level unigram/bigram/trigram comparison on NLTK's `carroll-alice.txt`, with validation-selected add-alpha smoothing.

### Contexts

| ID | Solution implementation | Validation setup | Runtime and environment | Baseline |
|----|-------------------------|------------------|-------------------------|----------|
| C1 | In-memory investigation prototype using GITenberg Alice or Peter Rabbit | 80/20 sentence split; full vocabulary; alpha 0.1; orders 1, 2, 3 | Python 3.12 standard library; no repository implementation | No n-gram example |
| C2 | In-memory investigation prototype using GITenberg Alice | 70/15/15 split; minimum token count 2; validation-selected alpha grid | Python 3.12 standard library; seed 42 | C1 |
| C3 | In-memory investigation prototype using pinned official NLTK `carroll-alice.txt` | Same model and settings as C2 | Python 3.12 standard library; NLTK data commit `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a` | C2 |
| C4 | Repository implementation on this branch | Spec-defined unit, report, subprocess, static, and package checks | Project Python 3.12 virtual environment; offline packaged corpus | C3 |

### Current understanding

#### Supported by evidence

- A bigram reduces test perplexity relative to the unigram on the selected fixed split.
- The trigram has lower training perplexity but higher test perplexity and much greater sparsity. This is evidence about this example, not a universal ordering of n-gram models.
- NLTK distributes `gutenberg/carroll-alice.txt` in an official archive whose metadata identifies the corpus as public domain. Pinning the archive commit and SHA-256 gives a durable acquisition record.
- Add-alpha smoothing produces a normalized distribution for seen and unseen contexts. Top-k rows alone do not sum to one, so the report must include the remaining probability mass.
- Seeded samples can contain local patterns but are often incoherent or truncated. Showing poor samples is necessary for an honest teaching example.
- Existing scheduler and optimization pipelines provide the required architecture without a generic framework.

#### Tentative or unresolved

- Human educational review of the rendered production report remains necessary.
- Browser review with network access disabled remains necessary for the complete report, although the shared MathJax runtime is already tested.

### Experiments and decisions

| ID | Idea or question | Context | Result and evidence links | Decision and rationale |
|----|------------------|---------|---------------------------|------------------------|
| E1 | Which built-in story gives a stable teaching example? | C1 | Alice had 301 test sentences and 5.5% unseen unigram events; Peter Rabbit had 12 test sentences and 30.8%. [Issue investigation](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/11#issuecomment-6954696312) | Select Alice; Peter Rabbit's test set and unknown-word rate are less suitable. |
| E2 | Can one example show both local-context value and higher-order sparsity? | C2 | Test perplexities were 195.37, 128.61, and 375.51 for orders 1, 2, and 3. Trigram training perplexity was 14.94 and 68.10% of test events were unseen. Same issue evidence | Use a 70/15/15 split, minimum training count 2, and validation-selected add-alpha smoothing. |
| E3 | Can the corpus source be more persistent, as the maintainer requested? | C3 | Official NLTK archive commit `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`; archive SHA-256 `2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611`; Alice SHA-256 `5ce9042c0d8536b072b61d4c7c4e6d70c72d9dbb5f3d816261326d9280749266`. [NLTK source update](https://github.com/anton-pershin/spbu-neural-networks-teacher/issues/11#issuecomment-6967392658) | Vendor the unchanged NLTK file with metadata and hashes. Do not install NLTK or download at runtime. |
| E4 | Does the NLTK source preserve the teaching result? | C3 | Test perplexities were 201.83, 125.21, and 348.83. Trigram training perplexity was 14.78; 66.62% of test events and 27.52% of test contexts were unseen. Same NLTK source update | Use these NLTK-based observations as the production example evidence, not as acceptance thresholds. |
| E5 | How should probabilistic modelling be visible beyond perplexity? | C3 | Calculated normalized top-five tables for the unigram, bigram after `the`, and trigram after `of the`; generated all samples for seeds 7, 42, and 123. Same NLTK source update | Show counts, smoothed probabilities, remaining mass, every raw sample, its seed, and its stop reason. Generation uses the full distribution. |
| E6 | Which existing architecture fits the pipeline? | Repository inspection | `schedulers.py` and the optimization/reporting modules provide thin orchestration, typed filters, separate builders, and shared writing. | Use small typed objects and functions under `spbunn.ngram`; do not copy the legacy convolution script or add a framework. |
| E7 | Does the repository implementation preserve the approved design and numerical evidence? | C4 | The final local run passed 155 tests, including 43 n-gram tests; formatting, import ordering, pylint, and mypy passed for 46 source files. Wheel and source-distribution inspection found the corpus and all provenance resources; an isolated wheel import also loaded and tokenized the packaged corpus offline. The production run generated `ngram.html` and four SVG figures with the investigated metrics and samples. [Numerical tests](../../tests/test_ngram_model.py), [pipeline tests](../../tests/test_ngram_pipelines.py), [report tests](../../tests/test_ngram_reporting.py), and [usage documentation](../../docs/ngram.md) define the repeatable evidence. | The implementation satisfies the automated spec checks. Keep the investigation active for human educational review; do not treat the observed metrics as performance acceptance thresholds. |
| E8 | Did independent implementation review find remaining gaps? | C4 | Review found silent truncation of fractional numeric settings and missing tiny-corpus coverage for OOV encoding and unseen-context fractions. The revision added finite-integral configuration validation, six fractional-setting rejection cases, and three direct tiny-corpus tests. The revised run passed 164 tests, including 52 n-gram tests, all static checks, and the package build. [Configuration validation](../../spbunn/ngram/pipeline.py) and the linked test modules contain the evidence. | Resolve all review findings before review readiness. Production numerical results are unchanged; the revision affects validation errors and acceptance coverage only. |
| E9 | Which additional trigram context and explanations improve human educational review? | C4 | The maintainer requested an `<unk>` explanation, a rarer trigram example, concrete generation contexts, and larger figure text. In the unchanged production model, context counts were 81 for `of the`, 40 for `mock turtle`, 14 for `the caterpillar`, and zero for `the hare`. The `the caterpillar` distribution has five `<eos>` events, two each for `and` and `took`, and one each for `angrily` and `called` in its top five. [Pipeline regression tests](../../tests/test_ngram_pipelines.py) and [report tests](../../tests/test_ngram_reporting.py) record the selected behavior. The complete merged run passed 231 tests, all static checks for 63 source files, and the package build. | Add `the caterpillar` because it is seen but sparse and gives a more informative example than an unseen uniform context. Explain special tokens with real-word context examples and enlarge figure typography. Do not change the fitted models, generated samples, or metric definitions. |

| E10 | What initial contexts make the generation table instructive? | C4 | The maintainer accepted the other report revisions but rejected `<bos>` table cells for orders 2 and 3. The actual contexts `alice` and `alice was` have 261 and 13 training events respectively. Explicit-context sampling gives deterministic continuations; for example, bigram seed 7 starts from `alice` and produces `could chin presently being ordered grin and once`. The focused revision passed 56 n-gram tests, including exact continuation, context-shifting, invalid-context, configuration, and HTML checks. The complete merged run passed 235 tests, all static checks for 63 source files, and the package build. | Keep only the unigram context empty. Generate bigrams from `alice` and trigrams from `alice was`; store and display the actual initial context and label the output as a continuation. Do not relabel `<bos>` cells without changing generation or select samples for readability. |

### Next candidates

- [x] Complete the spec-defined automated checks and inspect the production report.
- [ ] Obtain human educational approval of the rendered report, including offline mathematical rendering.
- [ ] Integrate the approved report into GetSomeMath separately if the maintainer wants it.

### Linked specs

- [11 — N-gram language-model pipeline general spec](../specs/11-ngram-language-model-general-spec.md): defines and bounds this implementation increment.
