from html import escape
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from spbunn.reporting.models import FigureModel, ReportModel
from spbunn.sequence.bpe import (
    END_OF_WORD,
    BPETraining,
    SampleTokenization,
    TokenizedWord,
    training_token_counts,
    training_vocabulary_sizes,
)
from spbunn.sequence.pipeline import (
    LargeExperimentResult,
    ToyExperimentResult,
)

REPORT_STYLE = """
<style>
  .token-stream { display: inline-flex; flex-wrap: wrap; gap: 0.2rem; }
  .token { display: inline-block; padding: 0.12rem 0.35rem; border-radius: 0.25rem;
    font-family: monospace; white-space: nowrap; }
  .base-token { background: #eceff1; color: #263238; }
  .learned-token { background: #dbeafe; color: #1e3a8a; }
  .new-token { background: #ffedd5; color: #9a3412; font-weight: bold; }
  .word-label { display: inline-block; min-width: 7rem; font-weight: 600; }
  .corpus-row, .sentence-word { margin: 0.3rem 0; }
  .sentence-word { display: block; }
  table { border-collapse: collapse; margin: 0.8rem 0; }
  th, td { border: 1px solid #cbd5e1; padding: 0.4rem 0.55rem;
    text-align: left; vertical-align: top; }
  th { background: #f1f5f9; }
  .metric { text-align: right; font-variant-numeric: tabular-nums; }
  details { margin: 0.65rem 0; }
  summary { cursor: pointer; font-weight: 600; }
  .note { border-left: 0.25rem solid #60a5fa; padding-left: 0.8rem; }
</style>
"""


def _token_markup(token: str, new_token: str | None = None) -> str:
    if token == new_token:
        css_class = "new-token"
    elif token == END_OF_WORD or len(token) == 1:
        css_class = "base-token"
    else:
        css_class = "learned-token"
    return f'<span class="token {css_class}">{escape(token)}</span>'


def _tokens_markup(tokens: Iterable[str], new_token: str | None = None) -> str:
    content = "".join(_token_markup(token, new_token) for token in tokens)
    return f'<span class="token-stream">{content}</span>'


def _corpus_markup(
    corpus: Iterable[TokenizedWord], new_token: str | None = None
) -> str:
    return "".join(
        '<div class="corpus-row">'
        f'<span class="word-label">{escape(item.word)} × {item.frequency}</span>'
        f"{_tokens_markup(item.tokens, new_token)}</div>"
        for item in corpus
    )


def _toy_trace(training: BPETraining) -> str:
    rows = [
        '<tr class="initial-row" data-merge="0">'
        '<td class="metric">0</td><td>Initial characters</td><td>—</td>'
        f"<td>{_corpus_markup(training.initial_corpus)}</td></tr>"
    ]
    for step in training.steps:
        if step.corpus is None:
            raise ValueError("the toy report requires a complete merge trace")
        pair = _tokens_markup(step.pair)
        rows.append(
            f'<tr class="merge-row" data-merge="{step.rank}">'
            f'<td class="metric">{step.rank}</td>'
            f"<td>{pair}<br>frequency {step.frequency}</td>"
            f"<td>{_token_markup(step.token, step.token)}</td>"
            f"<td>{_corpus_markup(step.corpus, step.token)}</td></tr>"
        )
    return (
        '<table class="trace-table"><thead><tr><th>Step</th>'
        "<th>Selected pair</th><th>New token</th>"
        "<th>Complete corpus after the merge</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _sample_table(
    initial_samples: tuple[SampleTokenization, ...],
    final_samples: tuple[SampleTokenization, ...],
) -> str:
    rows = []
    for initial, final in zip(initial_samples, final_samples):
        if initial.word != final.word:
            raise ValueError("initial and final sample words must align")
        word = escape(final.word, quote=True)
        rows.append(
            f'<tr data-toy-sample="{word}"><td>{escape(final.word)}</td>'
            f"<td>{_tokens_markup(initial.tokens)}</td>"
            f"<td>{_tokens_markup(final.tokens)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Word</th><th>Character baseline</th>"
        f"<th>Learned tokenization</th></tr></thead><tbody>{''.join(rows)}"
        "</tbody></table>"
    )


def _toy_figure(training: BPETraining, artifact_name: str) -> FigureModel:
    steps = np.arange(len(training.steps) + 1)
    figure, (vocabulary_axis, token_axis) = plt.subplots(1, 2, figsize=(9, 3.6))
    vocabulary_axis.plot(
        steps, training_vocabulary_sizes(training), "o-", color="tab:blue"
    )
    vocabulary_axis.set_title("Vocabulary grows")
    vocabulary_axis.set_ylabel("Vocabulary size")
    token_axis.plot(steps, training_token_counts(training), "o-", color="tab:orange")
    token_axis.set_title("Corpus becomes shorter")
    token_axis.set_ylabel("Weighted corpus tokens")
    for axis in (vocabulary_axis, token_axis):
        axis.set_xlabel("Merge step")
        axis.set_xticks(steps)
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=artifact_name)


def _chapter_label(chapters: tuple[int, ...]) -> str:
    if len(chapters) == 1:
        return f"Chapter {chapters[0]}"
    return f"Chapters {chapters[0]}–{chapters[-1]}"


def _checkpoint_table(result: LargeExperimentResult) -> str:
    checkpoints = result.evaluation.checkpoints
    baseline = checkpoints[0].heldout_token_count
    rows = []
    for item in checkpoints:
        reduction = 100 * (1 - item.heldout_token_count / baseline)
        rows.append(
            f'<tr data-checkpoint="{item.merge_count}">'
            f'<td class="metric">{item.merge_count}</td>'
            f'<td class="metric">{item.vocabulary_size:,}</td>'
            f'<td class="metric">{item.training_token_count:,}</td>'
            f'<td class="metric">{item.heldout_token_count:,}</td>'
            f'<td class="metric">{item.heldout_tokens_per_word:.3f}</td>'
            f'<td class="metric">{reduction:.1f}%</td></tr>'
        )
    return (
        "<table><thead><tr><th>Merges</th><th>Vocabulary</th>"
        "<th>Training tokens</th><th>Held-out tokens</th>"
        "<th>Tokens / word</th><th>Held-out reduction</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _large_figure(result: LargeExperimentResult, artifact_name: str) -> FigureModel:
    checkpoints = result.evaluation.checkpoints
    vocabulary_sizes = [item.vocabulary_size for item in checkpoints]
    baseline_training = checkpoints[0].training_token_count
    baseline_heldout = checkpoints[0].heldout_token_count
    training_percent = [
        100 * item.training_token_count / baseline_training for item in checkpoints
    ]
    heldout_percent = [
        100 * item.heldout_token_count / baseline_heldout for item in checkpoints
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        vocabulary_sizes,
        training_percent,
        "o-",
        label=_chapter_label(result.training_chapters) + " (training)",
    )
    axis.plot(
        vocabulary_sizes,
        heldout_percent,
        "o-",
        label=_chapter_label(result.heldout_chapters) + " (held out)",
    )
    for item, percent in zip(checkpoints, heldout_percent):
        axis.annotate(
            f"{item.merge_count} merges",
            (item.vocabulary_size, percent),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("Vocabulary size (initial tokens + learned tokens)")
    axis.set_ylabel("Tokens remaining (% of character baseline)")
    axis.set_title("Larger vocabularies shorten training and unseen text")
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=artifact_name)


def _sentence_checkpoints(result: LargeExperimentResult) -> str:
    sections = []
    for checkpoint in result.evaluation.checkpoints:
        token_count = sum(len(item.tokens) for item in checkpoint.sentence)
        words = "".join(
            '<span class="sentence-word">'
            f'<span class="word-label">{escape(item.word)}</span>'
            f"{_tokens_markup(item.tokens)}</span>"
            for item in checkpoint.sentence
        )
        sections.append(
            '<details class="sentence-checkpoint" '
            f'data-sentence-checkpoint="{checkpoint.merge_count}">'
            f"<summary>{checkpoint.merge_count} merges · "
            f"{token_count} tokens</summary>{words}</details>"
        )
    return "".join(sections)


def _frequent_tokens_table(result: LargeExperimentResult) -> str:
    rows = []
    for item in result.evaluation.frequent_tokens:
        visible_length = len(item.token.replace(END_OF_WORD, ""))
        position = "word-final" if item.token.endswith(END_OF_WORD) else "word-internal"
        rows.append(
            f"<tr><td>{_token_markup(item.token)}</td>"
            f'<td class="metric">{item.frequency:,}</td>'
            f'<td class="metric">{visible_length}</td><td>{position}</td></tr>'
        )
    return (
        "<table><thead><tr><th>Learned token</th>"
        "<th>Occurrences in final training tokenization</th>"
        "<th>Characters</th><th>Form</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _unseen_words_table(result: LargeExperimentResult) -> str:
    rows = []
    for item in result.evaluation.unseen_words:
        word = escape(item.word, quote=True)
        rows.append(
            f'<tr data-unseen-word="{word}"><td>{escape(item.word)}</td>'
            f'<td class="metric">{item.frequency}</td>'
            f"<td>{_tokens_markup(item.tokens)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Held-out word absent from training</th>"
        "<th>Held-out occurrences</th><th>Final tokenization</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _lowest_note(toy: ToyExperimentResult) -> str:
    lowest = next((item for item in toy.samples if item.word == "lowest"), None)
    if lowest is None:
        return ""
    expected_tokens = ("low", f"est{END_OF_WORD}")
    if lowest.tokens == expected_tokens:
        return (
            "<p><code>lowest</code> is absent from training. It reuses "
            "the learned <code>low</code> and word-final "
            "<code>est</code> tokens.</p>"
        )
    return (
        "<p>At the configured merge count, <code>lowest</code> tokenizes as "
        f"{_tokens_markup(lowest.tokens)}. It has not yet reached the "
        "production 12-merge form <code>low</code> plus word-final "
        "<code>est</code>.</p>"
    )


def build_bpe_report(
    toy: ToyExperimentResult,
    large: LargeExperimentResult,
    title: str,
    artifact_name: str,
    toy_figure_artifact_name: str = "bpe_toy_dynamics",
    large_figure_artifact_name: str = "bpe_large_corpus_compression",
) -> ReportModel:
    """Build the complete two-part BPE teaching report."""
    if not large.evaluation.checkpoints:
        raise ValueError("the large report requires at least one checkpoint")
    initial = large.evaluation.checkpoints[0]
    final = large.evaluation.checkpoints[-1]
    heldout_reduction = 100 * (
        1 - final.heldout_token_count / initial.heldout_token_count
    )
    training_word_count = sum(item.frequency for item in large.training.initial_corpus)
    training_label = _chapter_label(large.training_chapters)
    heldout_label = _chapter_label(large.heldout_chapters)
    sentence_text = " ".join(item.word for item in initial.sentence)
    marker = escape(END_OF_WORD)
    html_content = "\n".join(
        [
            REPORT_STYLE,
            f"<h1>{escape(title)}</h1>",
            (
                "<p>This report uses character-initialized byte-pair encoding "
                f"(BPE). Every word starts as characters plus <code>{marker}</code>, "
                "an end-of-word marker. BPE is not UTF-8 byte-level encoding "
                "in this demonstration.</p>"
            ),
            (
                r"<p>For adjacent tokens \(u\) and \(v\), the weighted pair "
                r"count is \(c(u,v)=\sum_w f_w n_w(u,v)\), where \(f_w\) is "
                r"the word frequency and \(n_w(u,v)\) counts adjacent "
                r"occurrences. BPE selects a most frequent pair, resolves ties "
                r"by lexicographic token order, and replaces non-overlapping "
                r"occurrences from left to right.</p>"
            ),
            '<section id="toy-example">',
            "<h2>1. How BPE works</h2>",
            "<h3>Initial corpus and vocabulary</h3>",
            _corpus_markup(toy.training.initial_corpus),
            (
                f"<p>Initial vocabulary ({len(toy.training.initial_vocabulary)} "
                f"tokens): {_tokens_markup(toy.training.initial_vocabulary)}</p>"
            ),
            "<h3>Every merge and corpus change</h3>",
            (
                "<p>Gray blocks are initial characters or the word boundary. "
                "Blue blocks are earlier learned tokens. Orange blocks show "
                "the token learned in the current row.</p>"
            ),
            _toy_trace(toy.training),
            "<h3>Applying the learned rules</h3>",
            _sample_table(toy.initial_samples, toy.samples),
            _lowest_note(toy),
            "</section>",
            '<section id="large-example">',
            "<h2>2. What BPE achieves on a larger corpus</h2>",
            (
                "<p>The second experiment uses the bundled public-domain text "
                "of Lewis Carroll's <em>Alice's Adventures in Wonderland</em>. "
                "Curly apostrophes are normalized, text is lowercased, and "
                "ASCII words with internal apostrophes are retained. "
                "Punctuation is excluded.</p>"
            ),
            "<table><tbody>",
            (
                f"<tr><th>Training text</th><td>{training_label}</td></tr>"
                f"<tr><th>Held-out text</th><td>{heldout_label}</td></tr>"
            ),
            (
                f"<tr><th>Training words</th><td>{training_word_count:,} "
                f"({len(large.training.initial_corpus):,} distinct)</td></tr>"
                "<tr><th>Held-out words</th>"
                f"<td>{initial.heldout_word_count:,}</td></tr>"
                "<tr><th>Learned merges</th>"
                f"<td>{len(large.training.steps):,}</td></tr>"
            ),
            "</tbody></table>",
            "<h3>Token-count reduction</h3>",
            _checkpoint_table(large),
            (
                f"<p>At the final checkpoint, held-out text needs "
                f"<strong>{final.heldout_token_count:,} tokens</strong>, down from "
                f"{initial.heldout_token_count:,}: a <strong>{heldout_reduction:.1f}% "
                "reduction</strong>. Each checkpoint applies a prefix of the "
                "same learned merge sequence; no model is retrained.</p>"
            ),
            "<h3>One held-out sentence at every checkpoint</h3>",
            f"<p><code>{escape(sentence_text)}</code></p>",
            _sentence_checkpoints(large),
            "<h3>Frequent learned tokens</h3>",
            (
                "<p>These frequencies count tokens that remain in the final "
                "training tokenization. They are not the original pair counts.</p>"
            ),
            _frequent_tokens_table(large),
            "<h3>Words not seen during training</h3>",
            _unseen_words_table(large),
            (
                '<p class="note"><strong>Limitation:</strong> Fewer tokens do '
                "not by themselves prove better language-model quality. BPE "
                "uses pair frequencies, not word meanings. This demonstration "
                "also rejects characters absent from the initial vocabulary "
                "instead of introducing an unknown-token fallback.</p>"
            ),
            "</section>",
            "<section><h2>Figures</h2><p>The first figure summarizes the "
            "complete toy trace. The second compares training and held-out "
            "token counts at the saved large-corpus checkpoints.</p></section>",
        ]
    )
    return ReportModel(
        title=title,
        html_content=html_content,
        figures=(
            _toy_figure(toy.training, toy_figure_artifact_name),
            _large_figure(large, large_figure_artifact_name),
        ),
        artifact_name=artifact_name,
    )
