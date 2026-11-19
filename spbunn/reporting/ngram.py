"""HTML and figures for the n-gram language-model teaching report."""

from html import escape

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from spbunn.ngram.model import GeneratedSample, TopKDistribution
from spbunn.ngram.pipeline import NGramExperimentResult, NGramOrderResult
from spbunn.reporting.models import FigureModel, ReportModel

NGRAM_FIGURE_STEMS = (
    "ngram_validation_perplexity",
    "ngram_train_test_perplexity",
    "ngram_test_sparsity",
    "ngram_top_k_probabilities",
)
NLTK_DATA_COMMIT = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"


def build_ngram_report(
    result: NGramExperimentResult, title: str, artifact_name: str
) -> ReportModel:
    """Build the complete educational report without writing artifacts."""
    sections = [
        "<h1>N-gram language models</h1>",
        (
            r"<p>An order-\(n\) language model predicts each token from the "
            r"previous \(n-1\) tokens. It is trained by counting events in "
            "text. The unigram has an empty context; the bigram uses one "
            "token; the trigram uses two.</p>"
        ),
        (
            r"<p>For context \(c\), candidate token \(w\), and vocabulary "
            r"size \(|V|\), add-alpha smoothing gives</p>"
            r"\[P_\alpha(w\mid c)=\frac{\operatorname{count}(c,w)+\alpha}"
            r"{\operatorname{count}(c)+\alpha|V|}.\]"
            r"<p>The numerator and denominator include every token in the "
            r"prediction vocabulary. An unseen context therefore gives a "
            r"uniform distribution. Smoothing is selected independently for "
            r"each order by validation perplexity.</p>"
        ),
        _corpus_section(result),
        _metrics_section(result.order_results),
        _probability_section(result.probability_distributions),
        _generation_section(result),
        _interpretation_section(result),
    ]
    with plt.rc_context(
        {
            "font.size": 14.0,
            "axes.titlesize": 15.0,
            "axes.labelsize": 15.0,
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
            "legend.fontsize": 13.0,
        }
    ):
        figures = (
            _validation_figure(result.order_results),
            _perplexity_figure(result.order_results),
            _sparsity_figure(result.order_results),
            _top_k_figure(result.probability_distributions),
        )
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=figures,
        artifact_name=artifact_name,
    )


def _corpus_section(result: NGramExperimentResult) -> str:
    corpus = result.corpus
    settings = result.settings
    rows = (
        (
            "Training",
            corpus.training_sentence_count,
            corpus.training_word_count,
        ),
        (
            "Validation",
            corpus.validation_sentence_count,
            corpus.validation_word_count,
        ),
        ("Test", corpus.test_sentence_count, corpus.test_word_count),
    )
    table_rows = "".join(
        f"<tr><td>{name}</td><td>{sentences}</td><td>{words}</td></tr>"
        for name, sentences, words in rows
    )
    return (
        "<section><h2>Offline corpus and preprocessing</h2>"
        f"<p>The source is <em>{escape(result.source_title)}</em>, vendored "
        f"as <code>{escape(result.source_resource)}</code> from the official "
        f"NLTK Gutenberg distribution at commit <code>{NLTK_DATA_COMMIT}</code>. "
        "The pipeline reads no network resource and requires no NLTK "
        "installation. The packaged file SHA-256 is "
        f"<code>{escape(result.source_sha256)}</code>.</p>"
        f"<p>Sentence shuffling uses seed {settings.split_seed}. Sentences "
        "are split by index, not by individual words. The prediction "
        f"vocabulary has {corpus.vocabulary_size} tokens.</p>"
        "<p><code>&lt;unk&gt;</code> means <strong>unknown word</strong>. "
        "It groups many different original words into one vocabulary item. "
        f"Training words that occur fewer than {settings.minimum_token_count} "
        "times and validation or test words absent from the training "
        "vocabulary are replaced by <code>&lt;unk&gt;</code>. For example, "
        "if <code>abide</code> and <code>absence</code> are both below this "
        "limit, the model sees the same <code>&lt;unk&gt;</code> token for "
        "both and cannot recover which original word appeared. "
        "<code>&lt;unk&gt;</code> can receive probability mass and be "
        "sampled, but it is not an English word.</p>"
        "<p><code>&lt;bos&gt;</code> means beginning of sentence and supplies "
        "temporary left context. <code>&lt;eos&gt;</code> means end of "
        "sentence and is an evaluation and generation stop event. Word counts "
        "below exclude <code>&lt;eos&gt;</code>.</p>"
        "<table><thead><tr><th>Split</th><th>Sentences</th><th>Words</th></tr>"
        f"</thead><tbody>{table_rows}</tbody></table>"
        f"<p>The retained corpus has {corpus.sentence_count} sentences. "
        "Punctuation <code>. ! ?</code> terminates sentences; common titles "
        "such as <code>Mr</code> and <code>Dr</code> are protected. Sentences "
        "with fewer than three words are discarded.</p></section>"
    )


def _metrics_section(order_results: tuple[NGramOrderResult, ...]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{result.order}</td>"
        f"<td>{result.observed_ngram_type_count}</td>"
        f"<td>{result.observed_context_count}</td>"
        f"<td>{result.selected_alpha:g}</td>"
        f"<td>{result.training_metrics.perplexity:.2f}</td>"
        f"<td>{result.validation_metrics.perplexity:.2f}</td>"
        f"<td>{result.test_metrics.perplexity:.2f}</td>"
        f"<td>{100 * result.test_metrics.unseen_ngram_fraction:.2f}%</td>"
        f"<td>{100 * result.test_metrics.unseen_context_fraction:.2f}%</td>"
        "</tr>"
        for result in order_results
    )
    return (
        "<section><h2>Training, validation, and test results</h2>"
        r"<p>Perplexity is \(\exp(-\bar{\log P})\) over next-token events, "
        r"including <code>&lt;eos&gt;</code>. Lower perplexity on the same "
        r"held-out events means that the model assigned greater probability "
        r"to those events. The alpha grid is selected only on validation "
        r"data; test data does not choose a model.</p>"
        "<table><thead><tr><th>Order</th><th>N-gram types</th><th>Contexts</th>"
        "<th>Alpha</th><th>Training PP</th><th>Validation PP</th><th>Test PP</th>"
        "<th>Unseen test events</th><th>Unseen test contexts</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p>The first figure shows every validation candidate. The next two "
        "figures compare training/test perplexity and exact zero-count "
        "fractions on the test split.</p></section>"
    )


def _probability_section(distributions: tuple[TopKDistribution, ...]) -> str:
    sections = ["<section><h2>Top-k conditional probability distributions</h2>"]
    for distribution in distributions:
        context = _html_context(distribution.context)
        rows = "".join(
            "<tr>"
            f"<td><code>{escape(item.token)}</code></td>"
            f"<td>{item.count}</td><td>{100 * item.probability:.3f}%</td>"
            "</tr>"
            for item in distribution.items
        )
        rows += (
            "<tr><td>all other tokens</td><td>—</td>"
            f"<td>{100 * distribution.remaining_probability:.3f}%</td></tr>"
        )
        sections.append(
            f"<h3>Order {distribution.order}, {context}</h3>"
            f"<p>The training context count is {distribution.context_count}; "
            f"alpha is {distribution.alpha:g}. The table contains the "
            f"{len(distribution.items)} largest rows and the exact remaining "
            "mass before rounding.</p>"
            "<table><thead><tr><th>Next token</th><th>Count</th>"
            f"<th>Smoothed probability</th></tr></thead><tbody>{rows}</tbody>"
            "</table>"
        )
    sections.append(
        "<p>Top-k is only a display. Evaluation and generation use the full "
        "vocabulary. The horizontal-bar figure includes the remaining mass "
        "so that each distribution sums to one.</p></section>"
    )
    return "\n".join(sections)


def _generation_section(result: NGramExperimentResult) -> str:
    rows = "".join(_sample_row(sample) for sample in result.generated_samples)
    return (
        "<section><h2>Seeded text generation</h2>"
        "<p>A context is the previous word or words used to predict the next "
        "word. For example, after the words <code>alice was</code>, a trigram "
        "uses exactly those two words as its context and assigns a probability "
        "to every possible next word.</p>"
        "<p>The unigram has no context, so its initial context is empty. "
        "The bigram and trigram rows instead start from actual words. These "
        "initial words are inputs, so the table shows them separately from "
        "the generated continuation. The <code>&lt;bos&gt;</code> marker is "
        "still used for first-word training and evaluation events, but it is "
        "not the initial context for these samples.</p>"
        f"{_generation_walkthrough(result.generated_samples)}"
        "<p>For every row below, the model samples one token from the full "
        "smoothed distribution, appends it, shifts the context, and repeats. "
        "It stops at <code>&lt;eos&gt;</code> or the configured limit of "
        f"{result.settings.maximum_generated_tokens} tokens. Output is not "
        "corrected or filtered for readability.</p>"
        "<table><thead><tr><th>Order</th><th>Initial context</th><th>Seed</th>"
        "<th>Generated continuation</th><th>Stop reason</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "</section>"
    )


def _generation_walkthrough(samples: tuple[GeneratedSample, ...]) -> str:
    sample = next(
        (item for item in samples if item.order == 2 and len(item.tokens) >= 3),
        None,
    )
    if sample is None:
        return ""
    initial = escape(" ".join(sample.initial_context))
    first, second, third = (escape(token) for token in sample.tokens[:3])
    return (
        f"<p>For the bigram sample with seed {sample.seed}, the first "
        f"prediction uses the real-word initial context <code>{initial}</code> "
        f"and produces <code>{first}</code>. The next prediction uses the "
        f"real-word context <code>{first}</code> and produces "
        f"<code>{second}</code>; then the context <code>{second}</code> "
        f"produces <code>{third}</code>.</p>"
    )


def _initial_context(context: tuple[str, ...]) -> str:
    if not context:
        return "empty (no previous token)"
    return f"<code>{escape(' '.join(context))}</code>"


def _sample_row(sample: GeneratedSample) -> str:
    if sample.tokens:
        text = escape(" ".join(sample.tokens))
    else:
        text = "(empty sequence)"
    stop_reason = (
        "<code>&lt;eos&gt;</code>"
        if sample.stop_reason == "eos"
        else "maximum token limit"
    )
    return (
        f"<tr><td>{sample.order}</td>"
        f"<td>{_initial_context(sample.initial_context)}</td>"
        f"<td>{sample.seed}</td><td><code>{text}</code></td>"
        f"<td>{stop_reason}</td></tr>"
    )


def _interpretation_section(result: NGramExperimentResult) -> str:
    best = min(result.order_results, key=lambda item: item.test_metrics.perplexity)
    unigram = next((item for item in result.order_results if item.order == 1), None)
    comparison = (
        f"Order {best.order} has the lowest test perplexity in this run "
        f"({best.test_metrics.perplexity:.2f})."
    )
    if unigram is not None and best.order != 1:
        comparison += f" The unigram value is {unigram.test_metrics.perplexity:.2f}."
    return (
        "<section><h2>What the example shows</h2><ul>"
        "<li><strong>Simple and inspectable:</strong> counts and conditional "
        "probabilities can be calculated directly from a corpus.</li>"
        "<li><strong>Useful local context:</strong> conditioning changes the "
        "next-token distribution and can improve held-out probability.</li>"
        "<li><strong>Sparsity:</strong> many valid higher-order combinations "
        "do not occur in training. Smoothing prevents zero probabilities but "
        "does not recover missing evidence.</li>"
        "<li><strong>Overfitting and growth:</strong> higher orders can fit "
        "training events closely while generalizing worse and storing many "
        "more observed n-gram types.</li>"
        "<li><strong>Limited meaning:</strong> no order in this experiment "
        "uses context beyond its fixed window. Local word patterns do not "
        "guarantee a globally coherent sentence.</li>"
        "</ul>"
        f"<p>{comparison} This observation is specific to the selected "
        "corpus, preprocessing, split, and smoothing grid. It is not a "
        "universal ranking of n-gram orders or a model-performance target. "
        "Poor, empty, and length-limited samples are retained because they "
        "show the limitations of the model.</p></section>"
    )


def _html_context(context: tuple[str, ...]) -> str:
    if not context:
        return "empty unigram context"
    return f"after <code>{escape(' '.join(context))}</code>"


def _validation_figure(
    order_results: tuple[NGramOrderResult, ...],
) -> FigureModel:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for result in order_results:
        axis.semilogx(
            [point.alpha for point in result.validation_curve],
            [point.perplexity for point in result.validation_curve],
            "o-",
            label=f"n = {result.order}",
        )
    axis.set_xlabel(r"Smoothing $\alpha$")
    axis.set_ylabel("Validation perplexity")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=NGRAM_FIGURE_STEMS[0])


def _perplexity_figure(
    order_results: tuple[NGramOrderResult, ...],
) -> FigureModel:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    positions = np.arange(len(order_results))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [result.training_metrics.perplexity for result in order_results],
        width,
        label="Training",
    )
    axis.bar(
        positions + width / 2,
        [result.test_metrics.perplexity for result in order_results],
        width,
        label="Test",
    )
    _set_order_axis(axis, positions, order_results)
    axis.set_ylabel("Perplexity (log scale)")
    axis.set_yscale("log")
    axis.legend()
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=NGRAM_FIGURE_STEMS[1])


def _sparsity_figure(
    order_results: tuple[NGramOrderResult, ...],
) -> FigureModel:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    positions = np.arange(len(order_results))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [100 * result.test_metrics.unseen_ngram_fraction for result in order_results],
        width,
        label="Unseen n-gram events",
    )
    axis.bar(
        positions + width / 2,
        [100 * result.test_metrics.unseen_context_fraction for result in order_results],
        width,
        label="Unseen contexts",
    )
    _set_order_axis(axis, positions, order_results)
    axis.set_ylabel("Test events (%)")
    axis.set_ylim(0, 100)
    axis.legend()
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=NGRAM_FIGURE_STEMS[2])


def _set_order_axis(
    axis: Axes, positions: np.ndarray, order_results: tuple[NGramOrderResult, ...]
) -> None:
    axis.set_xticks(positions, [f"n = {result.order}" for result in order_results])
    axis.grid(True, axis="y", alpha=0.3)


def _top_k_figure(
    distributions: tuple[TopKDistribution, ...],
) -> FigureModel:
    figure, axes = plt.subplots(
        len(distributions),
        1,
        figsize=(10, 3.8 * len(distributions)),
        squeeze=False,
    )
    for axis, distribution in zip(axes.flat, distributions, strict=True):
        labels = [item.token for item in distribution.items] + ["all other tokens"]
        probabilities = [item.probability for item in distribution.items] + [
            distribution.remaining_probability
        ]
        positions = np.arange(len(labels))
        colors = ["C0"] * len(distribution.items) + ["lightgrey"]
        axis.barh(positions, probabilities, color=colors)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xlabel("Conditional probability")
        context = (
            "empty" if not distribution.context else " ".join(distribution.context)
        )
        axis.set_title(
            f"n = {distribution.order}; context: {context}; "
            f"alpha = {distribution.alpha:g}"
        )
        axis.grid(True, axis="x", alpha=0.3)
        for position, probability in zip(positions, probabilities, strict=True):
            axis.text(
                probability + 0.01,
                position,
                f"{100 * probability:.2f}%",
                va="center",
            )
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=NGRAM_FIGURE_STEMS[3])
