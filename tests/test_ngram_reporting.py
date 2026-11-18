"""Presentation, educational-content, and escaping checks for n-gram reports."""

from dataclasses import replace
from html import escape

import matplotlib.pyplot as plt

from spbunn.ngram.model import GeneratedSample, TopKItem
from spbunn.reporting.ngram import NGRAM_FIGURE_STEMS, build_ngram_report
from tests.ngram_helpers import production_ngram_result


def test_report_contains_all_educational_results_and_four_figures() -> None:
    result = production_ngram_result()

    report = build_ngram_report(result, "N-gram test title", "ngram.html")
    try:
        assert report.title == "N-gram test title"
        assert report.artifact_name == "ngram.html"
        assert (
            tuple(figure.artifact_name for figure in report.figures)
            == NGRAM_FIGURE_STEMS
        )
        assert [len(figure.figure.axes) for figure in report.figures] == [1, 1, 1, 3]
        html = report.html_content
        assert "N-gram language models" in html
        assert r"P_\alpha(w\mid c)" in html
        assert result.source_sha256 in html
        assert "Top-k conditional probability distributions" in html
        assert "Seeded text generation" in html
        assert "This observation is specific" in html
        assert "not a" in html and "universal ranking" in html
        assert "<unk>" not in html
        assert "&lt;unk&gt;" in html
        for distribution in result.probability_distributions:
            assert f"Order {distribution.order}" in html
            for item in distribution.items:
                assert f"<code>{escape(item.token)}</code>" in html
        for sample in result.generated_samples:
            if sample.tokens:
                assert escape(" ".join(sample.tokens)) in html
        assert "(empty sequence)" in html
        assert "maximum token limit" in html
        assert "said alice because four sense repeated turning call it" in html
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


def test_report_escapes_source_tokens_and_generated_text() -> None:
    result = production_ngram_result()
    malicious_sample = GeneratedSample(
        order=1,
        seed=999,
        tokens=("<img src=x onerror=alert(1)>",),
        stop_reason="eos",
    )
    malicious_distribution = replace(
        result.probability_distributions[0],
        items=(TopKItem(token="<script>alert(2)</script>", count=0, probability=0.4),),
        remaining_probability=0.6,
    )
    malicious_result = replace(
        result,
        source_title="<script>alert(1)</script>",
        generated_samples=(malicious_sample,),
        probability_distributions=(malicious_distribution,),
    )

    report = build_ngram_report(malicious_result, "Escaped", "escaped.html")
    try:
        html = report.html_content
        assert "<script>alert(1)</script>" not in html
        assert "<script>alert(2)</script>" not in html
        assert "<img src=x onerror=alert(1)>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "&lt;script&gt;alert(2)&lt;/script&gt;" in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


def test_report_figure_titles_and_axes_describe_probability_and_sparsity() -> None:
    report = build_ngram_report(
        production_ngram_result(), "Figure inspection", "figures.html"
    )
    try:
        validation, perplexity, sparsity, top_k = report.figures
        assert validation.figure.axes[0].get_ylabel() == "Validation perplexity"
        assert perplexity.figure.axes[0].get_yscale() == "log"
        assert sparsity.figure.axes[0].get_ylabel() == "Test events (%)"
        assert sparsity.figure.axes[0].get_ylim() == (0.0, 100.0)
        assert "context: the" in top_k.figure.axes[1].get_title()
        assert "context: of the" in top_k.figure.axes[2].get_title()
        assert top_k.figure.axes[0].get_xlabel() == "Conditional probability"
    finally:
        for figure in report.figures:
            plt.close(figure.figure)
