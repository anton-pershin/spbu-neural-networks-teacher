"""Presentation and educational-content checks for the BPTT report."""

import matplotlib.pyplot as plt

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
        assert [len(figure.figure.axes) for figure in report.figures] == [4, 2]
        html = report.html_content
        assert "BPTT instability in vanilla RNNs" in html
        assert r"h_t=\tanh(W_xx_t+W_hh_{t-1}+b)" in html
        assert r"g_{t-1}=J_t^Tg_t=W_h^TD_tg_t" in html
        assert "Copy memory task: gradient decay" in html
        assert "Adding problem: gradient growth" in html
        assert "torch.nn.RNNCell" in html
        assert "single packed output tensor" in html
        assert "Default" in html and "Orthogonal" in html
        assert 'nn.init.calculate_gain("tanh") = 5/3' in html
        assert "Matched-weight float64 verification" in html
        assert "Exact zeros" in html
        assert "1e-45" in html
        assert "forward finite=" in html
        assert "backward finite=" in html
        assert "valid=" in html
        assert "not universal rankings" in html
        assert "does not claim that every reasonable initialization" in html
        assert "truncated BPTT, gradient clipping, LSTMs, or model" in html
        assert "The models are not trained" in html
        for verification in result.verification_results:
            assert f"<td>{verification.seed}</td>" in html
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
                assert float(axis.xaxis.label.get_fontsize()) >= 15
                assert float(axis.yaxis.label.get_fontsize()) >= 15
                assert float(axis.title.get_fontsize()) >= 15
                assert float(axis.get_xticklabels()[0].get_fontsize()) >= 13
                assert float(axis.get_yticklabels()[0].get_fontsize()) >= 13
                assert axis.get_legend() is not None
        copy_figure, adding_figure = report.figures
        assert all(len(axis.lines) == 2 for axis in copy_figure.figure.axes)
        assert all(len(axis.lines) == 2 for axis in adding_figure.figure.axes)
        assert "longest gap 20" in copy_figure.figure.axes[0].get_title()
        assert "longest length 20" in adding_figure.figure.axes[0].get_title()
    finally:
        for figure in report.figures:
            plt.close(figure.figure)
