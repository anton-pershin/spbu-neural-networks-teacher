from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spbunn.reporting.bpe import build_bpe_report
from spbunn.reporting.models import ReportModel
from spbunn.sequence.bpe import training_token_counts, training_vocabulary_sizes
from spbunn.sequence.pipeline import (
    LargeExperimentResult,
    ToyExperimentResult,
    produce_large_input,
    produce_toy_input,
    run_large_experiment,
    run_toy_experiment,
)
from tests.sequence_helpers import compose_sequence_config


@pytest.fixture(scope="module", name="experiment_results")
def _experiment_results() -> tuple[ToyExperimentResult, LargeExperimentResult]:
    cfg = compose_sequence_config("config_test_bpe")
    return (
        run_toy_experiment(produce_toy_input(cfg)),
        run_large_experiment(produce_large_input(cfg)),
    )


def close_report(report: ReportModel) -> None:
    for figure in report.figures:
        plt.close(figure.figure)


def test_report_has_complete_toy_and_large_teaching_content(
    experiment_results: tuple[ToyExperimentResult, LargeExperimentResult],
) -> None:
    toy, large = experiment_results
    report = build_bpe_report(
        toy, large, title="BPE <demonstration>", artifact_name="test_bpe.html"
    )

    try:
        html = report.html_content
        assert report.artifact_name == "test_bpe.html"
        assert len(report.figures) == 2
        assert [figure.artifact_name for figure in report.figures] == [
            "bpe_toy_dynamics",
            "bpe_large_corpus_compression",
        ]
        assert "BPE &lt;demonstration&gt;" in html
        assert "BPE <demonstration>" not in html
        assert "1. How BPE works" in html
        assert "2. What BPE achieves on a larger corpus" in html
        assert r"c(u,v)" in html
        assert (
            "Fewer tokens do not by themselves prove better language-model quality"
            in html
        )
        assert "&lt;/w&gt;" in html
        assert "</w>" not in html
        assert html.count('class="merge-row"') == len(toy.training.steps)
        for step in toy.training.steps:
            assert escape(step.token) in html
            assert f' data-merge="{step.rank}"' in html
        for sample in toy.samples:
            assert f'data-toy-sample="{escape(sample.word, quote=True)}"' in html
        for checkpoint in large.evaluation.checkpoints:
            assert f'data-checkpoint="{checkpoint.merge_count}"' in html
            assert f'data-sentence-checkpoint="{checkpoint.merge_count}"' in html
        for token_item in large.evaluation.frequent_tokens:
            assert escape(token_item.token) in html
        for word_item in large.evaluation.unseen_words:
            assert f'data-unseen-word="{escape(word_item.word, quote=True)}"' in html
        assert "Chapters 1–10" in html
        assert "Chapters 11–12" in html
        assert "public-domain" in html
    finally:
        close_report(report)


def test_report_figures_contain_exact_metric_series(
    experiment_results: tuple[ToyExperimentResult, LargeExperimentResult],
) -> None:
    toy, large = experiment_results
    report = build_bpe_report(toy, large, title="BPE", artifact_name="bpe.html")

    try:
        toy_figure, large_figure = report.figures
        toy_steps = np.arange(len(toy.training.steps) + 1)
        np.testing.assert_array_equal(
            toy_figure.figure.axes[0].lines[0].get_xdata(), toy_steps
        )
        np.testing.assert_array_equal(
            toy_figure.figure.axes[0].lines[0].get_ydata(),
            training_vocabulary_sizes(toy.training),
        )
        np.testing.assert_array_equal(
            toy_figure.figure.axes[1].lines[0].get_ydata(),
            training_token_counts(toy.training),
        )
        checkpoints = large.evaluation.checkpoints
        baseline_training = checkpoints[0].training_token_count
        baseline_heldout = checkpoints[0].heldout_token_count
        np.testing.assert_array_equal(
            large_figure.figure.axes[0].lines[0].get_xdata(),
            [item.vocabulary_size for item in checkpoints],
        )
        training_percent = np.asarray(
            large_figure.figure.axes[0].lines[0].get_ydata(), dtype=np.float64
        )
        heldout_percent = np.asarray(
            large_figure.figure.axes[0].lines[1].get_ydata(), dtype=np.float64
        )
        expected_training = np.asarray(
            [
                100 * item.training_token_count / baseline_training
                for item in checkpoints
            ],
            dtype=np.float64,
        )
        expected_heldout = np.asarray(
            [100 * item.heldout_token_count / baseline_heldout for item in checkpoints],
            dtype=np.float64,
        )
        np.testing.assert_allclose(training_percent, expected_training)
        np.testing.assert_allclose(heldout_percent, expected_heldout)
        assert training_percent[0] == 100
        assert heldout_percent[0] == 100
        assert heldout_percent[-1] < 100
    finally:
        close_report(report)
