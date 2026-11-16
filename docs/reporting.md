# Reporting foundation

## Purpose

The reporting package separates pipeline-specific content from artifact persistence. Producers and transformers compute experiment results. A report builder converts those results into a report model. `ReportWriter` is the single component that writes the HTML report and its figures.

M1 does not connect the legacy experiment scripts to this interface. That work belongs to M2–M4.

## Models

Import the public interface from `spbunn.reporting`:

- `FigureModel(figure, artifact_name)` contains a Matplotlib `Figure` and a file-name stem.
- `ReportModel(html_content, figures=(), title=..., artifact_name=...)` contains an HTML body fragment, a tuple of figure models, a document title, and the HTML file name.
- `ReportWriter(run_directory)` receives the output directory explicitly, normally from `cfg.result_dir`.

The dataclasses are frozen. The caller still owns each mutable Matplotlib figure and must close it when appropriate.

`html_content` is trusted HTML, not escaped text. Use HTML elements and LaTeX notation in this field. `title` is escaped by the writer. Figure elements are appended after the supplied body content in model order.

## Output contract

`ReportWriter.write(report)` performs these operations:

1. Validate all artifact names and reject duplicate figure stems before creating the run directory.
2. Create the run directory and missing parents.
3. Serialize each figure to an in-memory SVG representation.
4. Write the SVG bytes when their length is at most `SVG_SIZE_THRESHOLD_BYTES` (102,400 bytes).
5. Write only a PNG instead when the SVG is larger than the threshold.
6. Remove a stale artifact with the alternative extension for the same figure stem.
7. Add an image element with the selected file name to the HTML body.
8. Write the HTML as UTF-8 and return its `Path`.

Given figure stem `loss-landscape`, the result is either `loss-landscape.svg` or `loss-landscape.png`, never both after a successful rerun. The default HTML result is `report.html`; set `ReportModel.artifact_name` to use another file name.

All artifact names must be non-empty file names without path components. For example, `loss-landscape` is valid, while `figures/loss-landscape` and `../loss-landscape` are invalid. This rule keeps every artifact directly in the selected run directory.

## HTML and LaTeX

The writer creates a complete HTML document with a UTF-8 charset, responsive image styling, the supplied body, and figure elements. It configures MathJax for inline notation delimited by `\(...\)` or `$...$`. Standard MathJax display notation is also available.

The HTML references the MathJax 3 CDN at `cdn.jsdelivr.net`. No network access is needed to generate or test the file, but network access is needed for MathJax rendering in a browser.

## Example

```python
from pathlib import Path

import matplotlib.pyplot as plt

from spbunn.reporting import FigureModel, ReportModel, ReportWriter

run_directory = Path("/tmp/spbunn_hydra/example")
figure, axis = plt.subplots()
axis.plot([0, 1, 2], [0, 1, 4])
axis.set_xlabel("$x$")
axis.set_ylabel("$y$")

report = ReportModel(
    title="Quadratic example",
    html_content=(
        r"<h1>Quadratic example</h1>"
        r"<p>The figure shows \(y = x^2\) for three points.</p>"
    ),
    figures=(FigureModel(figure=figure, artifact_name="quadratic"),),
    artifact_name="index.html",
)

try:
    html_path = ReportWriter(run_directory).write(report)
finally:
    plt.close(figure)

print(html_path)
```

This example creates `index.html` and normally `quadratic.svg` in `/tmp/spbunn_hydra/example`. A sufficiently complex figure can produce `quadratic.png` instead.

## Tests

`tests/test_reporting.py` uses temporary run directories. Controlled figure doubles provide exact SVG sizes of 102,399, 102,400, and 102,401 bytes. A real Matplotlib figure checks integration with HTML generation. Other tests cover multiple figures, stale-format removal, and invalid or duplicate artifact names.

Run all checks from the repository root:

```bash
python -m pytest
./run_linters.sh
```
