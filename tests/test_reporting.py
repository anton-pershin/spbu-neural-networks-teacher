import re
from importlib.resources import files
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from matplotlib.figure import Figure

from spbunn.reporting import FigureModel, ReportModel, ReportWriter


class ControlledFigure:
    """Write deterministic image bytes for exact threshold tests."""

    def __init__(self, svg_size: int) -> None:
        self._svg_size = svg_size

    def savefig(self, target: object, *args: object, **kwargs: object) -> None:
        del args
        image_format = kwargs["format"]
        if image_format == "svg":
            data = b"<svg>" + b"x" * (self._svg_size - 11) + b"</svg>"
        elif image_format == "png":
            data = b"\x89PNG\r\n\x1a\ncontrolled"
        else:
            raise AssertionError(f"Unexpected image format: {image_format}")

        if isinstance(target, (str, Path)):
            Path(target).write_bytes(data)
        else:
            cast(BytesIO, target).write(data)


def controlled_figure(svg_size: int) -> Figure:
    return cast(Figure, ControlledFigure(svg_size))


def test_write_creates_html_with_latex_support_and_figure(tmp_path: Path) -> None:
    run_directory = tmp_path / "nested" / "run"
    figure = Figure()
    axis = figure.add_subplot()
    axis.plot([0, 1], [0, 1])
    report = ReportModel(
        title="Example <report>",
        html_content=r"<p>Linear relation \(y = x\).</p>",
        figures=(FigureModel(figure=figure, artifact_name="line"),),
        artifact_name="index.html",
    )

    report_path = ReportWriter(run_directory).write(report)

    assert report_path == run_directory / "index.html"
    html = report_path.read_text(encoding="utf-8")
    assert r"<p>Linear relation \(y = x\).</p>" in html
    assert "<title>Example &lt;report&gt;</title>" in html
    assert "window.MathJax" in html
    mathjax_runtime = (
        files("spbunn.reporting.static")
        .joinpath("tex-svg-full.js")
        .read_text(encoding="utf-8")
    )
    assert html.count(mathjax_runtime) == 1
    assert 'id="mathjax-runtime"' in html
    assert "options: {enableMenu: false}" in html
    assert re.search(r"<script\b[^>]*\bsrc=", html) is None
    assert "cdn.jsdelivr.net/npm/mathjax" not in html
    assert '<img src="line.svg"' in html
    assert (run_directory / "line.svg").is_file()
    assert not (run_directory / "line.png").exists()
    assert set(run_directory.iterdir()) == {report_path, run_directory / "line.svg"}


@pytest.mark.parametrize("svg_size", [102_399, 102_400])
def test_write_keeps_svg_at_or_below_threshold(tmp_path: Path, svg_size: int) -> None:
    report = ReportModel(
        html_content="<p>Boundary test</p>",
        figures=(
            FigureModel(figure=controlled_figure(svg_size), artifact_name="boundary"),
        ),
    )

    ReportWriter(tmp_path).write(report)

    svg_path = tmp_path / "boundary.svg"
    assert svg_path.stat().st_size == svg_size
    assert not (tmp_path / "boundary.png").exists()


def test_write_uses_png_above_threshold(tmp_path: Path) -> None:
    report = ReportModel(
        html_content="<p>Fallback test</p>",
        figures=(
            FigureModel(figure=controlled_figure(102_401), artifact_name="large"),
        ),
    )

    ReportWriter(tmp_path).write(report)

    assert (tmp_path / "large.png").read_bytes().startswith(b"\x89PNG")
    assert not (tmp_path / "large.svg").exists()
    assert '<img src="large.png"' in (tmp_path / "report.html").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("svg_size", "stale_name", "expected_name"),
    [
        (100, "figure.png", "figure.svg"),
        (102_401, "figure.svg", "figure.png"),
    ],
)
def test_write_removes_stale_alternative_format(
    tmp_path: Path, svg_size: int, stale_name: str, expected_name: str
) -> None:
    stale_path = tmp_path / stale_name
    stale_path.write_bytes(b"stale")
    report = ReportModel(
        html_content="",
        figures=(
            FigureModel(figure=controlled_figure(svg_size), artifact_name="figure"),
        ),
    )

    ReportWriter(tmp_path).write(report)

    assert not stale_path.exists()
    assert (tmp_path / expected_name).is_file()


def test_write_places_multiple_figures_in_run_directory(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    report = ReportModel(
        html_content="<p>Two figures</p>",
        figures=(
            FigureModel(figure=controlled_figure(100), artifact_name="small"),
            FigureModel(figure=controlled_figure(102_401), artifact_name="large"),
        ),
    )

    ReportWriter(run_directory).write(report)

    assert {path.name for path in run_directory.iterdir()} == {
        "report.html",
        "small.svg",
        "large.png",
    }


@pytest.mark.parametrize(
    "artifact_name", ["", ".", "..", "../figure", "nested/figure", "nested\\figure"]
)
def test_write_rejects_unsafe_figure_artifact_names(
    tmp_path: Path, artifact_name: str
) -> None:
    report = ReportModel(
        html_content="",
        figures=(
            FigureModel(figure=controlled_figure(100), artifact_name=artifact_name),
        ),
    )

    with pytest.raises(ValueError, match="artifact name"):
        ReportWriter(tmp_path / "run").write(report)

    assert not (tmp_path / "run").exists()


def test_write_rejects_path_in_report_artifact_name(tmp_path: Path) -> None:
    report = ReportModel(html_content="", artifact_name="nested/report.html")

    with pytest.raises(ValueError, match="artifact name"):
        ReportWriter(tmp_path / "run").write(report)

    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("report_name", "svg_size"),
    [("figure.svg", 100), ("figure.png", 102_401)],
)
def test_write_rejects_report_and_figure_artifact_collision(
    tmp_path: Path, report_name: str, svg_size: int
) -> None:
    report = ReportModel(
        html_content="",
        figures=(
            FigureModel(figure=controlled_figure(svg_size), artifact_name="figure"),
        ),
        artifact_name=report_name,
    )

    with pytest.raises(ValueError, match="collides"):
        ReportWriter(tmp_path / "run").write(report)

    assert not (tmp_path / "run").exists()


def test_write_rejects_duplicate_figure_artifact_names(tmp_path: Path) -> None:
    report = ReportModel(
        html_content="",
        figures=(
            FigureModel(figure=controlled_figure(100), artifact_name="same"),
            FigureModel(figure=controlled_figure(100), artifact_name="same"),
        ),
    )

    with pytest.raises(ValueError, match="Duplicate figure artifact name"):
        ReportWriter(tmp_path / "run").write(report)

    assert not (tmp_path / "run").exists()
