# SpbU Neural Networks Teacher

Executable teaching materials for the Neural Networks course at SpbU. The repository uses Hydra for configuration and shared report components for HTML, SVG, and PNG outputs.

> **Milestone status:** M1 provides the project and reporting foundation. The existing experiment scripts are legacy implementations. Their migration to complete HTML-reporting pipelines is planned for M2–M4.

## Repository structure

```text
config/                    Hydra pipeline and common configuration
  hydra/                   Hydra run-directory and logging overrides
  user_settings/           Environment-specific paths
docs/                      Component and usage documentation
spbunn/                    Python package
  reporting/               Shared report models and report writer
  scripts/                 Executable experiment scripts
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

MathJax supplies LaTeX rendering. The generated HTML loads MathJax from its CDN, so a browser needs network access to render the notation.

See [docs/reporting.md](docs/reporting.md) for the complete interface and artifact rules.

## Tests and developer checks

```bash
python -m pytest
./run_linters.sh
python -m build
```

The checks cover the M1 foundation and tests. They intentionally do not reformat or lint unrelated legacy numerical modules. The tests include exact checks below, at, and above the 102,400-byte SVG threshold.
