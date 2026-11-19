# N-gram language-model pipeline

## Purpose

M6 provides an executable introduction to probabilistic language modelling with word-level unigram, bigram, and trigram models. The pipeline trains exact count models, selects add-alpha smoothing on validation data, evaluates held-out perplexity and sparsity, displays conditional next-token distributions, and generates seeded text.

The numerical filters have no file-output or plotting responsibility. `spbunn.reporting.ngram` builds the report, and the shared `ReportWriter` persists HTML and figures.

## Run the pipeline

Set up the environment and optional output root as described in the README. From a source checkout, run:

```bash
python -m spbunn.scripts.ngram
```

The production configuration is `config/config_ngram.yaml`. Select an explicit output directory by overriding both Hydra paths:

```bash
python -m spbunn.scripts.ngram \
  result_dir=/tmp/spbunn-ngram hydra.run.dir=/tmp/spbunn-ngram
```

No NLTK installation, `nltk.download()` call, network connection, or user `nltk_data` directory is needed. The pipeline reads and hash-checks package data.

## Corpus source and integrity

The unchanged resource `spbunn/data/carroll-alice.txt` contains *Alice's Adventures in Wonderland* by Lewis Carroll. Its acquisition source is the official NLTK Gutenberg archive at commit:

```text
550b6625bcef1f2abff2ff770a5a0d272c9c6b2a
```

The pinned archive is:

```text
https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/corpora/gutenberg.zip
```

Integrity values are:

| Resource | Bytes | SHA-256 |
|---|---:|---|
| NLTK `gutenberg.zip` | 4,251,829 | `2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611` |
| `gutenberg/carroll-alice.txt` | 144,395 | `5ce9042c0d8536b072b61d4c7c4e6d70c72d9dbb5f3d816261326d9280749266` |

The retrieval date is 2026-11-18. The original NLTK package metadata declares the corpus public domain. `spbunn/data/PROVENANCE.md`, `nltk-gutenberg-README.txt`, and `nltk-gutenberg.xml` preserve the source details and notices. A production test checks the exact Alice hash.

## Preprocessing and data splits

The producer removes the one-line bracketed NLTK title, lowercases the story, and retains words with internal apostrophes or hyphens. `.`, `!`, and `?` terminate sentences. The abbreviations `Mr`, `Mrs`, `Ms`, `Dr`, and `St` do not terminate a sentence. Sentences shorter than three words are discarded.

Sentence indices are shuffled with seed 42 and split at rounded cumulative boundaries:

| Split | Fraction | Sentences | Words before `<eos>` |
|---|---:|---:|---:|
| Training | 0.70 | 1,052 | 18,185 |
| Validation | 0.15 | 226 | 4,263 |
| Test | 0.15 | 225 | 3,852 |

There are 1,503 retained sentences. The vocabulary uses training counts only. Words with count below two become `<unk>`. The `<unk>` token means "unknown word": it groups many different rare or previously unseen original words into one vocabulary item, so the model cannot recover which original word it represents. Validation and test words absent from the training vocabulary also become `<unk>`. It can receive probability mass and be sampled, but it is not an English word. `<bos>` supplies left context but cannot be predicted. `<eos>` is appended to each encoded sentence and is included in perplexity events. The sorted prediction vocabulary contains 1,163 tokens, including `<unk>` and `<eos>` and excluding `<bos>`.

## Count models and evaluation

For order `n`, the model uses `n - 1` context tokens. For context `c`, candidate token `w`, and positive smoothing value alpha, the complete conditional distribution is:

```text
P(w | c) = (count(c, w) + alpha) / (count(c) + alpha * |V|).
```

An unseen context gives probability `1 / |V|` to every prediction token. The alpha grid is:

```text
0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0
```

Each order independently selects the candidate with minimum validation perplexity. An exact tie selects the earlier configured value. Test data does not participate in selection. Perplexity is `exp(-mean(natural_log_probability))` over all next-token events, including `<eos>`.

The pipeline also measures the fraction of evaluation events with an unseen exact n-gram and the fraction with an unseen context. The latter is zero for unigrams.

## Top-k probability displays

The report shows counts and selected-alpha probabilities for five next tokens in each of these distributions:

1. Unigram with empty context.
2. Bigram after `the`.
3. Trigram after `of the`.
4. Trigram after the rarer context `the caterpillar`.

Rows are sorted by decreasing probability and then token text. Each table includes `all other tokens` with the remaining probability mass. The top rows alone do not sum to one. Top-k is a display only; evaluation and generation use the full vocabulary.

For example, the bigram after `the` assigns approximately 9.864% to `<unk>`, 3.842% to `queen`, 3.755% to `king`, 3.406% to `gryphon`, and 3.319% to `mock`. The remaining tokens share approximately 75.814%.

The additional trigram example has 14 training events for `the caterpillar`, compared with 81 for `of the`. Its top rows are `<eos>` with count 5 and probability 19.547%, `and` and `took` with count 2 and probability 7.842% each, and `angrily` and `called` with count 1 and probability 3.941% each. The remaining mass is 56.886%. This makes the effect of a small context count visible without using an unseen context whose distribution would be uniform.

## Seeded generation

A generation context is the previous word or words used to predict the next word. This report uses actual input words for the bigram and trigram samples:

| Order | Initial context |
|---:|---|
| 1 | Empty; the unigram does not use context |
| 2 | `alice` |
| 3 | `alice was` |

Both contexts occur in training: `alice` has 261 bigram events and `alice was` has 13 trigram events. The initial words are inputs, not generated output, so the table shows them separately from the generated continuation. `<bos>` remains a boundary marker for first-word training and evaluation events, but it is not used as the bigram or trigram prompt in this table.

For the bigram sample with seed 7, the first prediction uses `alice` and produces `could`; the next uses `could` and produces `chin`; then the context `chin` produces `presently`. The report derives this walkthrough from the stored sample and shows its actual initial context.

Production generation uses seeds 7, 42, and 123 for every order. At each step, the model samples from the full smoothed distribution, appends the token, shifts the context, and repeats. It stops at `<eos>` or after 30 generated continuation tokens. The report retains empty, incoherent, and length-limited continuations without wording corrections or readability filtering. Every row identifies its order, initial context, seed, continuation, and stop reason.

Starting from `alice`, the bigram continuation with seed 7 is:

```text
could chin presently being ordered grin and once
```

The samples demonstrate local word patterns and their limits; they are not presented as coherent writing.

## Production observations

The deterministic production example gives:

| Order | Observed n-gram types | Observed contexts | Alpha | Training PP | Validation PP | Test PP | Unseen test events | Unseen test contexts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,163 | 1 | 0.001 | 238.83 | 189.18 | 201.83 | 0.00% | 0.00% |
| 2 | 9,315 | 1,163 | 0.03 | 40.90 | 123.20 | 125.21 | 28.53% | 0.00% |
| 3 | 15,336 | 8,905 | 0.01 | 14.78 | 349.15 | 348.83 | 66.62% | 27.52% |

The bigram has the lowest test perplexity on this fixed split. The trigram has much lower training perplexity but higher test perplexity and more unseen events. These observations show local-context value, sparsity, and overfitting. They are specific to this corpus, preprocessing, split, and smoothing grid. They are not universal model rankings or performance thresholds.

## Report artifacts

The run directory contains `ngram.html` and one SVG or PNG artifact for each stem:

| Stem | Content |
|---|---|
| `ngram_validation_perplexity` | Validation perplexity across the alpha grid |
| `ngram_train_test_perplexity` | Training and test perplexity by order |
| `ngram_test_sparsity` | Unseen test n-gram events and contexts |
| `ngram_top_k_probabilities` | Top-k rows and remaining mass for all four contexts |

The figures use 14-point base text, 15-point axis labels and titles, and 13-point tick labels and legends. These settings are applied in a local Matplotlib context, which restores global plotting defaults after report construction.

The shared writer selects SVG at or below 102,400 serialized SVG bytes and PNG otherwise. The HTML includes the equation, provenance, preprocessing rules, corpus statistics, result tables, all top-k tables, all samples, and interpretation. It embeds MathJax and needs no external script, CDN, or web font. Dynamic text and special tokens are HTML-escaped.

## Small test configuration

`config/config_test_ngram.yaml` inherits the production configuration and uses the same real corpus, models, probability queries, initial generation contexts, and four figures. It reduces generation to seeds 7 and 42 and a 12-token continuation limit, and writes `test_ngram.html`. Integer-valued settings must be finite and integral; fractional values are rejected rather than truncated.

```bash
python -m spbunn.scripts.ngram \
  --config-name config_test_ngram \
  result_dir=/tmp/spbunn-test-ngram hydra.run.dir=/tmp/spbunn-test-ngram
```

## Tests and review

Run all local checks:

```bash
MPLBACKEND=Agg python -m pytest
./run_linters.sh
python -m build
```

The n-gram tests cover exact resource integrity, tokenization, deterministic disjoint splits, training-only vocabulary, hand-calculated counts and probabilities, normalization for seen and unseen contexts, perplexity, sparsity, validation-only alpha selection, top-k ordering and remaining mass, deterministic sampling and stop reasons, production regressions, HTML escaping, special-token and generation-context explanations, enlarged figure fonts, end-to-end artifacts, and figure cleanup after success and writer failure.

Package inspection must confirm that the Alice corpus and provenance resources are included. A human reviewer must open the production report and approve its mathematical and educational content. Offline browser review of mathematical rendering is also required.
