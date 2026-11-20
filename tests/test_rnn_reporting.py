"""Presentation and educational-content checks for the BPTT report."""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

from spbunn.reporting.rnn import BPTT_FIGURE_STEMS, build_bptt_report
from tests.rnn_helpers import reduced_bptt_result


def test_report_contains_both_experiments_verification_and_scope_limits() -> None:
    result = reduced_bptt_result()

    report = build_bptt_report(result, "BPTT test title", "bptt.html")
    try:
        assert report.title == "BPTT test title"
        assert report.artifact_name == "bptt.html"
        assert (
            tuple(figure.artifact_name for figure in report.figures)
            == BPTT_FIGURE_STEMS
        )
        assert [len(figure.figure.axes) for figure in report.figures] == [2, 2]
        copy_figure, adding_figure = report.figures
        assert "Figure 1 — Copy memory task" in copy_figure.caption
        assert "solid lines" in copy_figure.caption
        assert "dashed lines" in copy_figure.caption
        assert "Right: first-input-state" in copy_figure.caption
        assert "Figure 2 — Adding problem" in adding_figure.caption
        assert "backward amplification" in adding_figure.caption
        assert "Xavier-tanh initialization" in adding_figure.caption
        html = report.html_content
        assert "BPTT instability in vanilla RNNs" in html
        assert r"h_t=\tanh(W_xx_t+W_hh_{t-1}+b)" in html
        assert r"g_{t-1}=J_t^Tg_t=W_h^TD_tg_t" in html
        assert "Copy memory task: gradient decay" in html
        assert "Adding problem: gradient growth" in html
        assert "torch.nn.RNNCell" in html
        assert "single packed output tensor" in html
        assert "Default" in html and "Orthogonal" in html
        assert "norm preservation" in html
        assert "Q<sup>T</sup>Q = I" in html
        assert "does not make the full recurrent Jacobian orthogonal" in html
        assert 'nn.init.calculate_gain("tanh") = 5/3' in html
        assert "Matched-weight float64 verification" in html
        assert "Exact zeros" in html
        assert "1e-45" in html
        assert "forward finite=" in html
        assert "backward finite=" in html
        assert "valid=" in html
        assert "<th>g(5)/g(0)</th>" in html
        assert "<th>g(19)/g(0)</th>" in html
        assert "<th>g(200)/g(0)</th>" not in html
        assert "<td>NaN</td>" not in html
        assert "not universal rankings" in html
        assert "does not claim that every reasonable initialization" in html
        assert "truncated BPTT, gradient clipping, LSTMs, or model" in html
        assert "The models are not trained" in html
        for verification in result.verification_results:
            assert f"<td>{verification.seed}</td>" in html
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


def test_report_contains_prediction_examples_and_animation_controls() -> None:
    report = build_bptt_report(
        reduced_bptt_result(), "Animation inspection", "animations.html"
    )
    try:
        html = report.html_content
        assert "Copy task: input, target, and RNN output example" in html
        assert "Adding task: input, target, and RNN output example" in html
        assert "Input sequence" in html
        assert "Target output" in html
        assert "Untrained RNN output" in html
        for task in ("copy", "adding"):
            for control in ("play", "pause", "prev", "next"):
                assert f'id="{task}-{control}"' in html
            assert f'id="{task}-prediction"' in html
        assert html.count(">Play</button>") == 2
        assert html.count(">Pause</button>") == 2
        assert html.count(">Previous</button>") == 2
        assert html.count(">Next</button>") == 2
        assert "setInterval" in html and "clearInterval" in html
        assert "addEventListener" in html and "textContent" in html
        assert "\\u003cblank\\u003e" in html
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


def test_report_figures_use_log_scales_and_readable_local_fonts() -> None:
    original_font_size = plt.rcParams["font.size"]
    report = build_bptt_report(
        reduced_bptt_result(), "Figure inspection", "figures.html"
    )
    try:
        assert plt.rcParams["font.size"] == original_font_size
        for report_figure in report.figures:
            for axis in report_figure.figure.axes:
                assert axis.get_yscale() == "log"
                assert axis.get_title() == ""
                assert float(axis.xaxis.label.get_fontsize()) >= 20
                assert float(axis.yaxis.label.get_fontsize()) >= 20
                assert float(axis.get_xticklabels()[0].get_fontsize()) >= 16
                assert float(axis.get_yticklabels()[0].get_fontsize()) >= 16
                assert axis.get_legend() is not None
                for line in axis.lines:
                    assert line.get_linewidth() == 2
        copy_figure, adding_figure = report.figures
        assert all(len(axis.lines) == 4 for axis in copy_figure.figure.axes)
        assert all(len(axis.lines) == 2 for axis in adding_figure.figure.axes)
        for axis in (copy_figure.figure.axes[0],):
            for line in axis.lines:
                x_data = np.asarray(line.get_xdata())
                assert len(x_data) <= 24
                assert x_data[0] == 0
                assert x_data[-1] == 40
        for line in adding_figure.figure.axes[0].lines:
            x_data = np.asarray(line.get_xdata())
            assert len(x_data) <= 24
        for figure_model in (copy_figure, adding_figure):
            for line in figure_model.figure.axes[1].lines:
                assert line.get_markersize() >= 10
        original_svg_font_type = plt.rcParams["svg.fonttype"]
        svg_buffer = BytesIO()
        copy_figure.figure.savefig(svg_buffer, format="svg")
        assert b"<text" in svg_buffer.getvalue()
        assert plt.rcParams["svg.fonttype"] == original_svg_font_type
    finally:
        for figure in report.figures:
            plt.close(figure.figure)
