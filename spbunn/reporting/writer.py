from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from spbunn.reporting.models import FigureModel, ReportModel

SVG_SIZE_THRESHOLD_BYTES = 102_400


class ReportWriter:
    """Write a report model's HTML and figures into one run directory."""

    def __init__(self, run_directory: str | Path) -> None:
        self._run_directory = Path(run_directory)

    def write(self, report: ReportModel) -> Path:
        """Write all report artifacts and return the HTML artifact path."""
        self._validate_report(report)
        self._run_directory.mkdir(parents=True, exist_ok=True)

        figure_markup = "\n".join(
            self._write_figure_markup(figure) for figure in report.figures
        )
        report_path = self._run_directory / report.artifact_name
        report_path.write_text(
            self._render_html(report, figure_markup), encoding="utf-8"
        )
        return report_path

    def _write_figure_markup(self, model: FigureModel) -> str:
        figure_path = self._write_figure(model)
        source = escape(quote(figure_path.name), quote=True)
        alternative_text = escape(model.artifact_name, quote=True)
        return (
            "<figure>\n"
            f'  <img src="{source}" alt="{alternative_text}">\n'
            "</figure>"
        )

    def _write_figure(self, model: FigureModel) -> Path:
        svg_buffer = BytesIO()
        model.figure.savefig(svg_buffer, format="svg")
        svg_data = svg_buffer.getvalue()

        svg_path = self._run_directory / f"{model.artifact_name}.svg"
        png_path = self._run_directory / f"{model.artifact_name}.png"
        if len(svg_data) <= SVG_SIZE_THRESHOLD_BYTES:
            png_path.unlink(missing_ok=True)
            svg_path.write_bytes(svg_data)
            return svg_path

        svg_path.unlink(missing_ok=True)
        model.figure.savefig(png_path, format="png")
        return png_path

    def _validate_report(self, report: ReportModel) -> None:
        self._validate_artifact_name(report.artifact_name)
        figure_names = set()
        for figure in report.figures:
            self._validate_artifact_name(figure.artifact_name)
            if figure.artifact_name in figure_names:
                raise ValueError(
                    f"Duplicate figure artifact name: {figure.artifact_name}"
                )
            figure_names.add(figure.artifact_name)

    @staticmethod
    def _validate_artifact_name(name: str) -> None:
        if (
            not name
            or name in {".", ".."}
            or Path(name).name != name
            or "/" in name
            or "\\" in name
        ):
            raise ValueError(
                f"Invalid artifact name {name!r}; use a non-empty file name "
                "without path components"
            )

    @staticmethod
    def _render_html(report: ReportModel, figure_markup: str) -> str:
        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '  <meta charset="utf-8">',
                (
                    '  <meta name="viewport" content="width=device-width, '
                    'initial-scale=1">'
                ),
                f"  <title>{escape(report.title)}</title>",
                "  <style>",
                "    img { max-width: 100%; height: auto; }",
                "    figure { margin: 1rem 0; }",
                "  </style>",
                "  <script>",
                "    window.MathJax = {",
                '      tex: {inlineMath: [["\\\\(", "\\\\)"], ["$", "$"]]}',
                "    };",
                "  </script>",
                (
                    "  <script defer "
                    'src="https://cdn.jsdelivr.net/npm/mathjax@3/'
                    'es5/tex-mml-chtml.js"></script>'
                ),
                "</head>",
                "<body>",
                "  <main>",
                report.html_content,
                figure_markup,
                "  </main>",
                "</body>",
                "</html>",
                "",
            ]
        )
