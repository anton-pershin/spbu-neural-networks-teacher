"""Count-based n-gram models and numerical operations."""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log
from random import Random
from typing import Literal

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNKNOWN_TOKEN = "<unk>"

Token = str
Sentence = tuple[Token, ...]
Context = tuple[Token, ...]
NGram = tuple[Token, ...]
StopReason = Literal["eos", "length_limit"]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Probability and sparsity metrics for one encoded evaluation split."""

    perplexity: float
    unseen_ngram_fraction: float
    unseen_context_fraction: float
    event_count: int


@dataclass(frozen=True)
class ValidationPoint:
    """One smoothing candidate and its validation perplexity."""

    alpha: float
    perplexity: float


@dataclass(frozen=True)
class TopKItem:
    """One next-token count and smoothed conditional probability."""

    token: Token
    count: int
    probability: float


@dataclass(frozen=True)
class TopKDistribution:
    """The largest probability rows and the remaining probability mass."""

    order: int
    context: Context
    context_count: int
    alpha: float
    items: tuple[TopKItem, ...]
    remaining_probability: float


@dataclass(frozen=True)
class GeneratedSample:
    """One unedited seeded sample and the reason generation stopped."""

    order: int
    seed: int
    tokens: tuple[Token, ...]
    stop_reason: StopReason


@dataclass(frozen=True)
class NGramModel:
    """Exact n-gram and context counts with add-alpha probabilities."""

    order: int
    vocabulary: tuple[Token, ...]
    ngram_counts: dict[NGram, int]
    context_counts: dict[Context, int]

    def __post_init__(self) -> None:
        if self.order <= 0:
            raise ValueError("The n-gram order must be positive")
        if not self.vocabulary:
            raise ValueError("The prediction vocabulary must not be empty")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("The prediction vocabulary must contain unique tokens")
        if tuple(sorted(self.vocabulary)) != self.vocabulary:
            raise ValueError("The prediction vocabulary must be sorted")
        if BOS_TOKEN in self.vocabulary:
            raise ValueError("The beginning-of-sentence token cannot be predicted")
        if any(len(ngram) != self.order for ngram in self.ngram_counts):
            raise ValueError("Every n-gram count key must match the model order")
        if any(len(context) != self.order - 1 for context in self.context_counts):
            raise ValueError("Every context count key must match the model order")

    def count(self, context: Context, token: Token) -> int:
        """Return one exact next-token count."""
        self._validate_context(context)
        return self.ngram_counts.get(context + (token,), 0)

    def context_count(self, context: Context) -> int:
        """Return the number of training events for a context."""
        self._validate_context(context)
        return self.context_counts.get(context, 0)

    def probability(self, context: Context, token: Token, alpha: float) -> float:
        """Return add-alpha probability for one token in the full vocabulary."""
        _validate_alpha(alpha)
        self._validate_context(context)
        if token not in self.vocabulary:
            raise ValueError(f"Token is outside the prediction vocabulary: {token!r}")
        denominator = self.context_count(context) + alpha * len(self.vocabulary)
        return (self.count(context, token) + alpha) / denominator

    def distribution(
        self, context: Context, alpha: float
    ) -> tuple[tuple[Token, int, float], ...]:
        """Return token, count, and probability rows in vocabulary order."""
        self._validate_context(context)
        _validate_alpha(alpha)
        return tuple(
            (token, self.count(context, token), self.probability(context, token, alpha))
            for token in self.vocabulary
        )

    def _validate_context(self, context: Context) -> None:
        if len(context) != self.order - 1:
            raise ValueError(
                f"Order {self.order} requires a context of length {self.order - 1}"
            )


def encode_sentences(
    sentences: Iterable[Sentence], known_tokens: frozenset[Token]
) -> tuple[Sentence, ...]:
    """Map out-of-vocabulary words and append the end-of-sentence event."""
    encoded = tuple(
        tuple(token if token in known_tokens else UNKNOWN_TOKEN for token in sentence)
        + (EOS_TOKEN,)
        for sentence in sentences
    )
    if not encoded:
        raise ValueError("At least one sentence is required")
    return encoded


def fit_ngram_model(
    encoded_sentences: Iterable[Sentence], order: int, vocabulary: Sequence[Token]
) -> NGramModel:
    """Fit exact n-gram and context counts from encoded training sentences."""
    if order <= 0:
        raise ValueError("The n-gram order must be positive")
    sentences = tuple(encoded_sentences)
    if not sentences:
        raise ValueError("Training requires at least one encoded sentence")
    vocabulary_tuple = tuple(vocabulary)
    observed_tokens = {token for sentence in sentences for token in sentence}
    missing_tokens = observed_tokens - set(vocabulary_tuple)
    if missing_tokens:
        raise ValueError(
            "Encoded training tokens are absent from the vocabulary: "
            f"{sorted(missing_tokens)}"
        )
    ngram_counts: Counter[NGram] = Counter()
    context_counts: Counter[Context] = Counter()
    for sentence in sentences:
        padded = (BOS_TOKEN,) * (order - 1) + sentence
        for index in range(order - 1, len(padded)):
            context = padded[index - order + 1 : index]
            target = padded[index]
            ngram_counts[context + (target,)] += 1
            context_counts[context] += 1
    return NGramModel(
        order=order,
        vocabulary=vocabulary_tuple,
        ngram_counts=dict(ngram_counts),
        context_counts=dict(context_counts),
    )


def evaluate_ngram_model(
    model: NGramModel, encoded_sentences: Iterable[Sentence], alpha: float
) -> EvaluationMetrics:
    """Calculate perplexity and exact zero-count fractions for one split."""
    _validate_alpha(alpha)
    sentences = tuple(encoded_sentences)
    if not sentences:
        raise ValueError("Evaluation requires at least one encoded sentence")
    total_log_probability = 0.0
    event_count = 0
    unseen_ngram_count = 0
    unseen_context_count = 0
    for sentence in sentences:
        padded = (BOS_TOKEN,) * (model.order - 1) + sentence
        for index in range(model.order - 1, len(padded)):
            context = padded[index - model.order + 1 : index]
            target = padded[index]
            total_log_probability += log(model.probability(context, target, alpha))
            event_count += 1
            unseen_ngram_count += model.count(context, target) == 0
            unseen_context_count += (
                model.order > 1 and model.context_count(context) == 0
            )
    if event_count == 0:
        raise ValueError("Evaluation requires at least one target event")
    return EvaluationMetrics(
        perplexity=exp(-total_log_probability / event_count),
        unseen_ngram_fraction=unseen_ngram_count / event_count,
        unseen_context_fraction=unseen_context_count / event_count,
        event_count=event_count,
    )


def calculate_top_k_distribution(
    model: NGramModel, context: Context, alpha: float, k: int
) -> TopKDistribution:
    """Calculate normalized top-k rows and all remaining probability mass."""
    if k <= 0:
        raise ValueError("The top-k value must be positive")
    rows = model.distribution(context, alpha)
    ordered_rows = sorted(rows, key=lambda row: (-row[2], row[0]))
    selected = ordered_rows[: min(k, len(ordered_rows))]
    items = tuple(
        TopKItem(token=token, count=count, probability=probability)
        for token, count, probability in selected
    )
    remaining = (
        0.0
        if len(selected) == len(rows)
        else 1.0 - sum(item.probability for item in items)
    )
    return TopKDistribution(
        order=model.order,
        context=context,
        context_count=model.context_count(context),
        alpha=alpha,
        items=items,
        remaining_probability=remaining,
    )


def generate_sample(
    model: NGramModel, alpha: float, seed: int, max_tokens: int
) -> GeneratedSample:
    """Sample from the full smoothed distribution until EOS or a length limit."""
    _validate_alpha(alpha)
    if seed < 0:
        raise ValueError("The generation seed must be non-negative")
    if max_tokens <= 0:
        raise ValueError("The generation token limit must be positive")
    random_generator = Random(seed)
    context = (BOS_TOKEN,) * (model.order - 1)
    generated: list[Token] = []
    stop_reason: StopReason = "length_limit"
    for _ in range(max_tokens):
        rows = model.distribution(context, alpha)
        token = random_generator.choices(
            [row[0] for row in rows],
            weights=[row[2] for row in rows],
            k=1,
        )[0]
        if token == EOS_TOKEN:
            stop_reason = "eos"
            break
        generated.append(token)
        context = (context + (token,))[-(model.order - 1) :] if model.order > 1 else ()
    return GeneratedSample(
        order=model.order,
        seed=seed,
        tokens=tuple(generated),
        stop_reason=stop_reason,
    )


def _validate_alpha(alpha: float) -> None:
    if not isfinite(alpha) or alpha <= 0:
        raise ValueError("The smoothing alpha must be finite and positive")
