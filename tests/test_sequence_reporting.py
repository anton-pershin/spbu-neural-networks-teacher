import re
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spbunn.reporting.bpe import build_bpe_report
from spbunn.reporting.models import ReportModel
from spbunn.sequence.bpe import (
    TokenizedWord,
    training_token_counts,
    training_vocabulary_sizes,
)
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


def expected_token_html(token: str, new_token: str | None = None) -> str:
    if token == new_token:
        css_class = "new-token"
    elif token == "</w>" or len(token) == 1:
        css_class = "base-token"
    else:
        css_class = "learned-token"
    return f'<span class="token {css_class}">{escape(token)}</span>'


def expected_tokens_html(tokens: tuple[str, ...], new_token: str | None = None) -> str:
    content = "".join(expected_token_html(token, new_token) for token in tokens)
    return f'<span class="token-stream">{content}</span>'


def extract_row(html: str, attribute: str, value: object) -> str:
    match = re.search(
        rf'<tr[^>]*{re.escape(attribute)}="{re.escape(str(value))}"[^>]*>'
        rf"(.*?)</tr>",
        html,
        flags=re.DOTALL,
    )
    assert match is not None, f"missing row {attribute}={value}"
    return match.group(1)


def table_cells(row: str) -> list[str]:
    return re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL)


def expected_corpus_row_html(item: TokenizedWord, new_token: str | None = None) -> str:
    return (
        '<div class="corpus-row">'
        f'<span class="word-label">{escape(item.word)} × {item.frequency}</span>'
        f"{expected_tokens_html(item.tokens, new_token)}</div>"
    )


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
        initial_cells = table_cells(extract_row(html, "data-merge", 0))
        for item in toy.training.initial_corpus:
            assert expected_corpus_row_html(item) in initial_cells[-1]
        for step in toy.training.steps:
            cells = table_cells(extract_row(html, "data-merge", step.rank))
            assert len(cells) == 4
            assert expected_tokens_html(step.pair) in cells[1]
            assert f"frequency {step.frequency}" in cells[1]
            assert expected_token_html(step.token, step.token) in cells[2]
            assert step.corpus is not None
            for item in step.corpus:
                assert expected_corpus_row_html(item, step.token) in cells[-1]
        for initial_sample, final_sample in zip(toy.initial_samples, toy.samples):
            row = extract_row(html, "data-toy-sample", final_sample.word)
            assert expected_tokens_html(initial_sample.tokens) in row
            assert expected_tokens_html(final_sample.tokens) in row
        assert "has not yet reached" in html
        baseline = large.evaluation.checkpoints[0]
        for checkpoint in large.evaluation.checkpoints:
            row = extract_row(html, "data-checkpoint", checkpoint.merge_count)
            reduction = 100 * (
                1 - checkpoint.heldout_token_count / baseline.heldout_token_count
            )
            expected_values = (
                str(checkpoint.merge_count),
                f"{checkpoint.vocabulary_size:,}",
                f"{checkpoint.training_token_count:,}",
                f"{checkpoint.heldout_token_count:,}",
                f"{checkpoint.heldout_tokens_per_word:.3f}",
                f"{reduction:.1f}%",
            )
            for value in expected_values:
                assert f'<td class="metric">{value}</td>' in row
            sentence_match = re.search(
                '<details class="sentence-checkpoint" '
                f'data-sentence-checkpoint="{checkpoint.merge_count}">'
                r"(.*?)</details>",
                html,
                flags=re.DOTALL,
            )
            assert sentence_match is not None
            for sample in checkpoint.sentence:
                assert (
                    f'<span class="word-label">{escape(sample.word)}</span>'
                    f"{expected_tokens_html(sample.tokens)}" in sentence_match.group(1)
                )
        for token_item in large.evaluation.frequent_tokens:
            assert (
                f"<tr><td>{expected_token_html(token_item.token)}</td>"
                f'<td class="metric">{token_item.frequency:,}</td>' in html
            )
        for word_item in large.evaluation.unseen_words:
            row = extract_row(html, "data-unseen-word", word_item.word)
            cells = table_cells(row)
            assert cells[1] == str(word_item.frequency)
            assert expected_tokens_html(word_item.tokens) in cells[2]
        assert "Chapters 1–10" in html
        assert "Chapters 11–12" in html
        assert "public-domain" in html
    finally:
        close_report(report)


def test_production_toy_trace_has_expected_lowest_note_and_custom_names(
    experiment_results: tuple[ToyExperimentResult, LargeExperimentResult],
) -> None:
    _, large = experiment_results
    cfg = compose_sequence_config("config_test_bpe", ["toy.n_merges=12"])
    toy = run_toy_experiment(produce_toy_input(cfg))
    lowest = next(item for item in toy.samples if item.word == "lowest")
    assert lowest.tokens == ("low", "est</w>")
    report = build_bpe_report(
        toy,
        large,
        title="BPE",
        artifact_name="bpe.html",
        toy_figure_artifact_name="custom_toy",
        large_figure_artifact_name="custom_large",
    )

    try:
        assert "It reuses the learned <code>low</code>" in report.html_content
        assert "has not yet reached" not in report.html_content
        assert [figure.artifact_name for figure in report.figures] == [
            "custom_toy",
            "custom_large",
        ]
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
