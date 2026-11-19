# SpbU Neural Networks Teacher

Executable teaching materials for the Neural Networks course at SpbU. The repository uses Hydra for configuration and shared report components for HTML, SVG, and PNG outputs.

> **Milestone status:** M1 provides the project and reporting foundation. M2 provides the schedulers, loss-landscape, and simple loss-optimization pipelines. M3 provides the MLP width, MLP depth, hidden-propagation, and Balduzzi HTML-reporting pipelines. M4 provides the simple convolution HTML-reporting pipeline. M5 provides the two-part BPE pipeline. M6 provides the offline n-gram language-model pipeline.

## Repository structure

```text
config/                    Hydra pipeline and common configuration
  hydra/                   Hydra run-directory and logging overrides
  user_settings/           Environment-specific paths
data/                      Bundled public-domain BPE corpus and source notices
docs/                      Component and usage documentation
spbunn/                    Python package
  data/                    Hash-checked offline corpus and provenance
  ngram/                   N-gram corpus, count models, evaluation, and sampling
  optimization/            Numerical filters and shared plotting helpers
  mlp/                     MLP models, architecture experiments, and propagation filters
  convolution.py           Image producer and repeated-convolution evaluator
  reporting/               Report models, builders, and shared report writer
  scripts/                 Executable experiment scripts
  sequence/                BPE training, tokenization, and corpus evaluation
  utils/                   Shared project utilities
tests/                     Automated tests
pyproject.toml             Package and static-analysis configuration
requirements.txt           Runtime dependencies
requirements_dev.txt       Development dependencies
run_linters.sh             Focused formatting, lint, and type checks
```

## Environment setup

Use Python 3.12 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements_dev.txt
```

Set `SPBUNN_HYDRA_PATH` to choose the parent directory for Hydra outputs:

```bash
export SPBUNN_HYDRA_PATH=/path/to/spbunn-outputs
```

The default is `/tmp/spbunn_hydra`. Common settings in `config/user_settings/user_settings.yaml` place each run in a date-and-time subdirectory. The same directory is exposed as `result_dir` for report outputs.

## Hydra configuration

A pipeline configuration can select the common groups and expose the run directory as follows:

```yaml
defaults:
  - _self_
  - user_settings: user_settings
  - hydra: base

result_dir: ${user_settings.result_dir}
hydra_root: ${user_settings.hydra_root}
hydra_dir: ${user_settings.hydra_dir}
```

Hydra entry points can use `spbunn.utils.common.get_config_path()` as their absolute configuration path. `config/config_test.yaml` is a smoke-test configuration for the common groups; it is not a course pipeline.

## Optimization pipelines

Run the three M2 pipelines from a source checkout:

```bash
python -m spbunn.scripts.schedulers
python -m spbunn.scripts.loss_landscape
python -m spbunn.scripts.simple_loss_optimization
```

Each command writes an HTML report and its figures into `result_dir`. The default HTML names are `schedulers.html`, `loss_landscape.html`, and `simple_loss_optimization.html`. Small `config_test_*.yaml` configurations support fast automated runs.

See [docs/optimization.md](docs/optimization.md) for configuration overrides, numerical behavior, figure names, and test execution.

## MLP architecture and propagation pipelines

Run the four M3 pipelines from a source checkout:

```bash
python -m spbunn.scripts.mlp_width_experiment
python -m spbunn.scripts.mlp_depth_experiment
python -m spbunn.scripts.hidden_norms_propagation
python -m spbunn.scripts.balduzzi_analysis
```

Each command writes an HTML report and its figures into `result_dir`. The default HTML names are `mlp_width_experiment.html`, `mlp_depth_experiment.html`, `hidden_norms_propagation.html`, and `balduzzi_analysis.html`. Small `config_test_*.yaml` configurations support fast automated runs. The full width and depth production configurations repeat their expensive hyperparameter searches in memory; they do not use the legacy working-directory JSON caches.

See [docs/mlp.md](docs/mlp.md) for configuration overrides, numerical behavior, figure names, the reconstructed Balduzzi residual network, and test execution.

## Convolution pipeline

Run the M4 pipeline from the source-checkout root:

```bash
python -m spbunn.scripts.simple_convolution
```

This uses the repository's default `climate_change.png` for the horizontal and vertical kernels and `waves.png` for the diagonal kernel. Both paths remain configurable:

```bash
python -m spbunn.scripts.simple_convolution \
  images.rgb=/path/to/rgb.png \
  images.grayscale=/path/to/grayscale.png
```

Relative override paths are resolved from the original invocation directory. The command writes `simple_convolution.html` and the horizontal, vertical, and diagonal convolution figures into `result_dir`.

See [docs/convolution.md](docs/convolution.md) for input and output overrides, numerical behavior, figure names, the small test configuration, and test execution.

## BPE pipeline

Run the M5 byte-pair encoding demonstration from a source checkout:

```bash
python -m spbunn.scripts.bpe
```

The default run writes `bpe.html`, `bpe_toy_dynamics.svg` or `.png`, and `bpe_large_corpus_compression.svg` or `.png` into `result_dir`. The first report section exposes every merge in a four-word corpus. The second trains on chapters 1–10 of a bundled public-domain copy of *Alice's Adventures in Wonderland* and evaluates 1,000 merges on held-out chapters 11–12. It shows token-count reduction, sentence tokenization at selected checkpoints, frequent learned tokens, and words absent from training.

The corpus and its source notices are in `data/`. Runs need no network access or tokenizer library. See [docs/bpe.md](docs/bpe.md) for the algorithm, preprocessing, production results, configuration overrides, and tests.

## N-gram language-model pipeline

Run the M6 pipeline from a source checkout:

```bash
python -m spbunn.scripts.ngram
```

The pipeline uses a hash-checked Alice corpus vendored from a pinned official NLTK Gutenberg archive. It needs no NLTK installation or network access. The default `ngram.html` report explains unigram, bigram, and trigram models with conditional-probability tables, perplexity and sparsity figures, and all seeded generation samples. It explains `<unk>`, starts bigram samples from `alice` and trigram samples from `alice was`, includes a sparse trigram probability display, and uses enlarged figure text. Four figures are written beside the report through the shared writer.

Use `config/config_test_ngram.yaml` for a reduced-generation test run. See [docs/ngram.md](docs/ngram.md) for source hashes, preprocessing, model formulas, configuration, exact example results, artifact names, and tests.

## Reporting

A report builder creates a `ReportModel`. The shared `ReportWriter` then creates the run directory, writes the figures, adds their final file names to the HTML, and writes the HTML report.

```python
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.reporting import FigureModel, ReportModel, ReportWriter


def write_example(cfg: DictConfig) -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    report = ReportModel(
        title="Linear relation",
        html_content=r"<p>The relation \(y = x\) is linear.</p>",
        figures=(FigureModel(figure=figure, artifact_name="linear-relation"),),
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        plt.close(figure)
```

The default HTML artifact name is `report.html`. Figure artifact names are stems without a directory or extension. The writer selects `.svg` when the serialized SVG size is at most 102,400 bytes; otherwise, it selects `.png`. It never keeps both formats for the same stem after a rerun.

MathJax supplies LaTeX rendering without network access. Each HTML report embeds the pinned MathJax 3.2.2 full SVG runtime, including its glyph paths. It needs no external MathJax script, CDN, or web font. The embedded runtime adds approximately 2.3 MB to each HTML report. Source and license details are in `spbunn/reporting/static/README.md`.

See [docs/reporting.md](docs/reporting.md) for the complete interface and artifact rules.

## Tests and developer checks

```bash
python -m pytest
./run_linters.sh
python -m build
```

The checks cover the M1 foundation, M2 optimization pipelines, M3 architecture and propagation pipelines, the M4 convolution pipeline, the M5 BPE pipeline, the M6 n-gram pipeline, and tests. They intentionally do not reformat or lint unrelated legacy numerical modules. The tests include exact checks below, at, and above the 102,400-byte SVG threshold, embedded MathJax support, numerical behavior, deterministic BPE traces, held-out tokenization, corpus integrity, probability distributions, deterministic generation, and end-to-end pipeline output.

The GitHub Actions **CI/CD** workflow runs these checks on pushes to `main` and on pull requests. It builds the source and wheel distributions and uploads them as the `python-distributions` workflow artifact.
