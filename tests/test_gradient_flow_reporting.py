"""Report content, figures, and retained investigation-artifact checks."""

from collections.abc import Sized
from dataclasses import replace
from io import BytesIO
from typing import cast

import matplotlib.pyplot as plt

from spbunn.reporting.gradient_flow import (
    DISPLAY_ZERO_FLOOR,
    FIGURE_STEMS,
    build_gradient_flow_report,
)
from spbunn.utils.common import get_project_path
from tests.rnn_helpers import reduced_gradient_flow_result


def test_report_contains_equations_all_cases_and_limitations() -> None:
    result = reduced_gradient_flow_result()

    report = build_gradient_flow_report(
        result,
        title="Test gradient flow",
        artifact_name="test_gradient_flow.html",
    )

    assert report.artifact_name == "test_gradient_flow.html"
    assert tuple(figure.artifact_name for figure in report.figures) == FIGURE_STEMS
    html = report.html_content
    for content in (
        "relative state-gradient norm",
        "relative cell-gradient norm",
        "mean forget-gate activation",
        "t_*",
        "L^{(b)}",
        "overline{f}_t",
        "bm{h}",
        "bm{c}",
        "bm{f}",
        "Direct cell-path contribution",
        "Default RNN",
        "Orthogonal RNN",
        "LSTM",
        "gap 200",
        "Better gradient flow does not guarantee task learning",
        ".internal/investigations/gradient-flow-audit/index.html",
    ):
        assert content.lower() in html.lower(), content
    assert html.count("<tr><td>2</td>") == 6
    assert html.count("<tr><td>4</td>") == 6
    for figure in report.figures:
        assert figure.caption
        for axis in figure.figure.axes:
            assert axis.get_title() == ""
    plt.close("all")


def test_figure_scales_pruning_fonts_and_local_svg_text() -> None:
    original = reduced_gradient_flow_result()
    settings = replace(original.settings, figure_max_points=5)
    result = replace(original, settings=settings)
    previous_font_size = plt.rcParams["font.size"]
    previous_svg_fonttype = plt.rcParams["svg.fonttype"]

    report = build_gradient_flow_report(result, title="Test", artifact_name="test.html")

    assert plt.rcParams["font.size"] == previous_font_size
    for figure_index, figure in enumerate(report.figures):
        for axis in figure.figure.axes:
            assert axis.get_xlabel()
            for line in axis.lines:
                assert line.get_linewidth() >= 2
                if figure_index == 1 and line.get_linestyle() != "--":
                    assert _point_count(line) == 10
                    assert line.get_markersize() >= 7
                elif figure_index != 1:
                    assert _point_count(line) <= 5
    for axis in report.figures[2].figure.axes:
        assert axis.get_yscale() == "log"
        assert axis.get_ylim()[0] == DISPLAY_ZERO_FLOOR
    for axis in report.figures[3].figure.axes:
        assert axis.get_yscale() == "log"
    buffer = BytesIO()
    report.figures[2].figure.savefig(buffer, format="svg")
    assert b"<text" in buffer.getvalue()
    assert plt.rcParams["svg.fonttype"] == previous_svg_fonttype
    plt.close("all")


def test_investigation_artifact_is_standalone_and_complete() -> None:
    artifact = (
        get_project_path()
        / ".internal"
        / "investigations"
        / "gradient-flow-audit"
        / "index.html"
    )

    html = artifact.read_text(encoding="utf-8")

    assert artifact.stat().st_size > 2_000_000
    assert 'id="mathjax-runtime"' in html
    assert 'macros: {bm: ["\\\\boldsymbol{#1}", 1]}' in html
    assert "<script src=" not in html
    assert html.count('src="data:image/svg+xml;base64,') == 5
    assert html.count('class="confirm-card"') == 5
    for animation in ("copy", "highway", "journey"):
        for control in ("play", "pause", "prev", "next"):
            assert f'id="{animation}-{control}"' in html
    for expression in (
        "Relative state-gradient norm",
        "Relative cell-gradient norm",
        "Mean forget-gate activation",
        "nabla_{\\bm{h}",
        "nabla_{\\bm{c}",
        "overline{f}_t",
    ):
        assert expression in html
    assert '<section id="confirm">' in html
    assert html.rfind('<section id="confirm">') > html.rfind('<section id="proposal">')


def _point_count(line) -> int:
    return len(cast(Sized, line.get_xdata()))
