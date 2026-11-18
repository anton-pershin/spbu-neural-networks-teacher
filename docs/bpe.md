# BPE pipeline

## Purpose

The M5 pipeline explains byte-pair encoding (BPE) in two stages:

1. A four-word corpus exposes every pair count, merge, new token, and corpus change.
2. A larger book corpus shows how learned tokens reduce training and held-out text length.

Both stages use the same project-owned, character-initialized implementation. There is no tokenizer-library dependency, random state, model training, or runtime download.

## Run the pipeline

Set up the environment and output root as described in the repository README. From a source checkout, run:

```bash
python -m spbunn.scripts.bpe
```

The production configuration is `config/config_bpe.yaml`. The run writes these artifacts directly into `result_dir`:

| Artifact | Contents |
|---|---|
| `bpe.html` | Two-part teaching report with embedded MathJax |
| `bpe_toy_dynamics.svg` or `.png` | Toy vocabulary growth and weighted corpus-token count |
| `bpe_large_corpus_compression.svg` or `.png` | Training and held-out tokens remaining at each vocabulary checkpoint |

The shared report writer selects SVG at or below 102,400 serialized SVG bytes and PNG above that limit.

For a faster run, use the small test configuration:

```bash
python -m spbunn.scripts.bpe \
  --config-name config_test_bpe \
  result_dir=/tmp/spbunn-bpe-test hydra.run.dir=/tmp/spbunn-bpe-test
```

This configuration performs four toy merges and 20 book merges while preserving the complete data flow.

## Algorithm

Each word starts as its Unicode characters followed by the end-of-word marker `</w>`. For each training step, the pipeline:

1. Counts every adjacent token pair, weighted by each word's frequency.
2. Selects a pair with maximum frequency.
3. Breaks equal-count ties by lexicographic order of the first token and then the second token.
4. Replaces all non-overlapping occurrences from left to right.
5. Adds the concatenated token to the vocabulary if it is new and records the ordered merge rule.

Training stops after the configured number of merges or when no adjacent pair remains. Tokenization starts again from characters and applies the learned rules in rank order. A checkpoint is therefore a prefix of the single learned merge sequence, not a separately trained model.

A sample word containing a character absent from the initial vocabulary causes a clear error. This increment does not introduce an unknown-token fallback or UTF-8 byte-level BPE.

## Toy example

The production toy corpus is:

```text
low     × 5
lower   × 2
newest  × 6
widest  × 3
```

Its initial vocabulary has 11 tokens, and its weighted corpus has 95 character-or-boundary tokens. The first three merges are:

```text
(e, s)       -> es       frequency 9
(es, t)      -> est      frequency 9
(est, </w>)  -> est</w>  frequency 9
```

After 12 merges, the vocabulary has 23 tokens and the weighted corpus has 22 tokens. The report shows the complete corpus after each step. It also tokenizes known words and new words. In particular, `lowest`, which is absent from training, becomes:

```text
low | est</w>
```

## Larger corpus

`data/alice_in_wonderland.txt` is the unmodified `carroll-alice.txt` file from the NLTK Project Gutenberg selections archive. The work is Lewis Carroll's public-domain *Alice's Adventures in Wonderland*. `data/README.md` records the download source and SHA-256, and `data/gutenberg-README.txt` retains the collection's original notices.

The loader supports both chapter-heading layouts in the file. It normalizes curly apostrophes to straight apostrophes, lowercases the text, and extracts ASCII-letter words with optional internal apostrophes. Punctuation is excluded. Chapters 1–10 provide weighted training word frequencies; chapters 11–12 remain held out.

The production data has:

- 22,577 training words and 2,389 distinct training words;
- 4,031 held-out words;
- 28 initial tokens, including the end-of-word marker.

### Production checkpoints

The default configuration learns 1,000 merges and evaluates prefixes of 0, 100, 250, 500, and 1,000 merges. The current deterministic results are:

| Merges | Vocabulary | Training tokens | Held-out tokens | Held-out tokens / word |
|---:|---:|---:|---:|---:|
| 0 | 28 | 113,985 | 20,598 | 5.110 |
| 100 | 128 | 57,293 | 10,341 | 2.565 |
| 250 | 278 | 44,402 | 8,182 | 2.030 |
| 500 | 528 | 36,199 | 6,705 | 1.663 |
| 1,000 | 1,028 | 29,903 | 5,766 | 1.430 |

The final held-out token count is 72.0% below the character baseline. The figure shows training and held-out token counts as percentages of their own baselines, so the two curves are directly comparable.

The report also shows:

- the held-out sentence `the knave of hearts he stole those tarts and took them quite away` at every checkpoint;
- frequent learned tokens and their occurrences in the final training tokenization;
- held-out words absent from training and their final tokenizations.

Fewer tokens do not by themselves prove better language-model quality. The experiment demonstrates token construction and sequence-length reduction, not downstream model performance.

## Configuration overrides

Corpus paths, merge counts, checkpoints, sample words, and report names are in Hydra configuration. For example, to run 500 book merges with matching checkpoints:

```bash
python -m spbunn.scripts.bpe \
  large.n_merges=500 'large.checkpoints=[0,100,250,500]' \
  result_dir=/tmp/spbunn-bpe hydra.run.dir=/tmp/spbunn-bpe
```

Checkpoints must be unique, strictly increasing, non-negative, and no greater than the actual merge count. The configured sentence words must occur in the held-out chapters.

## Components and tests

- `spbunn/sequence/bpe.py`: pure training, tokenization, checkpoint metrics, frequent-token ranking, and unseen-word selection.
- `spbunn/sequence/pipeline.py`: Hydra input validation, local corpus loading, and experiment filters.
- `spbunn/reporting/bpe.py`: static token-block HTML, explanatory content, tables, and two figures.
- `spbunn/scripts/bpe.py`: thin orchestration, shared-writer invocation, and figure cleanup.

Tests include hand-calculated merge traces, overlapping pairs, tie-breaking, early termination, unknown characters, checkpoint equivalence to separately trained prefixes, both book-heading layouts, exact held-out metrics, HTML escaping, figure data, subprocess output, the full 1,000-merge production experiment, and figure cleanup after writer failure.

Run all local checks:

```bash
MPLBACKEND=Agg python -m pytest
./run_linters.sh
python -m build
```

Human review of the rendered two-part report is still required for educational acceptance.
