"""Hydra-facing corpus production and the in-memory n-gram experiment."""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from math import isclose, isfinite
from pathlib import PurePosixPath
from random import Random
from typing import Any

from omegaconf import DictConfig

from spbunn.ngram.model import (
    BOS_TOKEN,
    EOS_TOKEN,
    UNKNOWN_TOKEN,
    Context,
    EvaluationMetrics,
    GeneratedSample,
    NGramModel,
    Sentence,
    TopKDistribution,
    ValidationPoint,
    calculate_top_k_distribution,
    encode_sentences,
    evaluate_ngram_model,
    fit_ngram_model,
    generate_sample,
)

ALICE_HEADER = "[Alice's Adventures in Wonderland by Lewis Carroll 1865]"
ABBREVIATION_PATTERN = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St)\.", flags=re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z]+(?:['-][a-z]+)*|[.!?]")


@dataclass(frozen=True)
class ProbabilityQuery:
    """One model order and context selected for a probability display."""

    order: int
    context: Context


@dataclass(frozen=True)
class NGramSettings:
    """Validated numerical settings independent of Hydra."""

    split_seed: int
    training_fraction: float
    validation_fraction: float
    minimum_token_count: int
    orders: tuple[int, ...]
    alphas: tuple[float, ...]
    probability_queries: tuple[ProbabilityQuery, ...]
    top_k: int
    generation_seeds: tuple[int, ...]
    maximum_generated_tokens: int

    @property
    def test_fraction(self) -> float:
        """Return the remaining test fraction."""
        return 1.0 - self.training_fraction - self.validation_fraction


@dataclass(frozen=True)
class NGramExperimentInput:
    """Raw disjoint corpus splits and all numerical experiment settings."""

    source_title: str
    source_resource: str
    source_sha256: str
    training_sentences: tuple[Sentence, ...]
    validation_sentences: tuple[Sentence, ...]
    test_sentences: tuple[Sentence, ...]
    known_tokens: frozenset[str]
    settings: NGramSettings

    @property
    def prediction_vocabulary(self) -> tuple[str, ...]:
        """Return the sorted prediction vocabulary, excluding BOS."""
        return tuple(sorted(self.known_tokens | {UNKNOWN_TOKEN, EOS_TOKEN}))

    @property
    def sentence_count(self) -> int:
        """Return the retained sentence count before splitting."""
        return (
            len(self.training_sentences)
            + len(self.validation_sentences)
            + len(self.test_sentences)
        )


@dataclass(frozen=True)
class CorpusStatistics:
    """Sentence, word, and vocabulary sizes displayed in the report."""

    sentence_count: int
    training_sentence_count: int
    validation_sentence_count: int
    test_sentence_count: int
    training_word_count: int
    validation_word_count: int
    test_word_count: int
    vocabulary_size: int


@dataclass(frozen=True)
class NGramOrderResult:
    """Training, selection, and evaluation results for one model order."""

    order: int
    observed_ngram_type_count: int
    observed_context_count: int
    validation_curve: tuple[ValidationPoint, ...]
    selected_alpha: float
    training_metrics: EvaluationMetrics
    validation_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics


@dataclass(frozen=True)
class NGramExperimentResult:
    """Complete in-memory input to the n-gram report builder."""

    source_title: str
    source_resource: str
    source_sha256: str
    corpus: CorpusStatistics
    settings: NGramSettings
    order_results: tuple[NGramOrderResult, ...]
    probability_distributions: tuple[TopKDistribution, ...]
    generated_samples: tuple[GeneratedSample, ...]


def load_packaged_corpus(resource: str, expected_sha256: str) -> str:
    """Read and hash-check one corpus resource without network access."""
    if not resource or PurePosixPath(resource).name != resource:
        raise ValueError("The corpus resource must be a file name without a path")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("The corpus SHA-256 must contain 64 lowercase hex digits")
    corpus_bytes = files("spbunn.data").joinpath(resource).read_bytes()
    actual_sha256 = sha256(corpus_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Corpus hash mismatch for {resource!r}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return corpus_bytes.decode("utf-8")


def tokenize_alice_corpus(text: str) -> tuple[Sentence, ...]:
    """Remove the NLTK title and tokenize the story into short sentences."""
    try:
        header, body = text.split("\n", 1)
    except ValueError as exc:
        raise ValueError("The NLTK Alice corpus must contain a title and body") from exc
    if header != ALICE_HEADER:
        raise ValueError(f"Unexpected NLTK Alice title header: {header!r}")
    body = ABBREVIATION_PATTERN.sub(lambda match: match.group(0)[:-1], body.lstrip())
    raw_tokens = TOKEN_PATTERN.findall(body.lower())
    sentences: list[Sentence] = []
    sentence: list[str] = []
    for token in raw_tokens:
        if token in ".!?":
            if len(sentence) >= 3:
                sentences.append(tuple(sentence))
            sentence = []
        else:
            sentence.append(token)
    if len(sentence) >= 3:
        sentences.append(tuple(sentence))
    if not sentences:
        raise ValueError("The corpus produced no sentences of at least three words")
    return tuple(sentences)


def split_sentences(
    sentences: Sequence[Sentence],
    seed: int,
    training_fraction: float,
    validation_fraction: float,
) -> tuple[tuple[Sentence, ...], tuple[Sentence, ...], tuple[Sentence, ...]]:
    """Shuffle sentence indices and split at rounded cumulative boundaries."""
    if seed < 0:
        raise ValueError("The split seed must be non-negative")
    _validate_split_fractions(training_fraction, validation_fraction)
    indices = list(range(len(sentences)))
    Random(seed).shuffle(indices)
    training_end = round(len(sentences) * training_fraction)
    validation_end = round(len(sentences) * (training_fraction + validation_fraction))
    splits = (
        tuple(sentences[index] for index in indices[:training_end]),
        tuple(sentences[index] for index in indices[training_end:validation_end]),
        tuple(sentences[index] for index in indices[validation_end:]),
    )
    if any(not split for split in splits):
        raise ValueError("Training, validation, and test splits must not be empty")
    return splits


def build_training_vocabulary(
    training_sentences: Sequence[Sentence], minimum_token_count: int
) -> frozenset[str]:
    """Build the non-special vocabulary from training token counts only."""
    if minimum_token_count <= 0:
        raise ValueError("The minimum token count must be positive")
    if not training_sentences:
        raise ValueError("Vocabulary construction requires training sentences")
    counts = Counter(token for sentence in training_sentences for token in sentence)
    vocabulary = frozenset(
        token for token, count in counts.items() if count >= minimum_token_count
    )
    if not vocabulary:
        raise ValueError("The minimum token count produced an empty vocabulary")
    reserved = vocabulary & {BOS_TOKEN, EOS_TOKEN, UNKNOWN_TOKEN}
    if reserved:
        raise ValueError(
            f"Corpus tokens collide with special tokens: {sorted(reserved)}"
        )
    return vocabulary


def produce_ngram_input(cfg: DictConfig) -> NGramExperimentInput:
    """Resolve configuration and produce a typed, offline experiment input."""
    settings = _parse_settings(cfg)
    source_title = str(cfg.corpus.title)
    source_resource = str(cfg.corpus.resource)
    source_sha256 = str(cfg.corpus.sha256)
    if not source_title:
        raise ValueError("The corpus title must not be empty")
    text = load_packaged_corpus(source_resource, source_sha256)
    sentences = tokenize_alice_corpus(text)
    training, validation, test = split_sentences(
        sentences,
        seed=settings.split_seed,
        training_fraction=settings.training_fraction,
        validation_fraction=settings.validation_fraction,
    )
    known_tokens = build_training_vocabulary(training, settings.minimum_token_count)
    return NGramExperimentInput(
        source_title=source_title,
        source_resource=source_resource,
        source_sha256=source_sha256,
        training_sentences=training,
        validation_sentences=validation,
        test_sentences=test,
        known_tokens=known_tokens,
        settings=settings,
    )


def evaluate_ngram_experiment(
    experiment: NGramExperimentInput,
) -> NGramExperimentResult:
    """Train, select smoothing, evaluate, inspect, and sample each order."""
    settings = experiment.settings
    vocabulary = experiment.prediction_vocabulary
    encoded_training = encode_sentences(
        experiment.training_sentences, experiment.known_tokens
    )
    encoded_validation = encode_sentences(
        experiment.validation_sentences, experiment.known_tokens
    )
    encoded_test = encode_sentences(experiment.test_sentences, experiment.known_tokens)
    models: dict[int, NGramModel] = {}
    selected_alphas: dict[int, float] = {}
    order_results: list[NGramOrderResult] = []
    for order in settings.orders:
        model = fit_ngram_model(encoded_training, order, vocabulary)
        models[order] = model
        validation_by_alpha = {
            alpha: evaluate_ngram_model(model, encoded_validation, alpha)
            for alpha in settings.alphas
        }
        curve = tuple(
            ValidationPoint(
                alpha=alpha, perplexity=validation_by_alpha[alpha].perplexity
            )
            for alpha in settings.alphas
        )
        selected_index = 0
        for index in range(1, len(curve)):
            if curve[index].perplexity < curve[selected_index].perplexity:
                selected_index = index
        selected_alpha = settings.alphas[selected_index]
        selected_alphas[order] = selected_alpha
        order_results.append(
            NGramOrderResult(
                order=order,
                observed_ngram_type_count=len(model.ngram_counts),
                observed_context_count=len(model.context_counts),
                validation_curve=curve,
                selected_alpha=selected_alpha,
                training_metrics=evaluate_ngram_model(
                    model, encoded_training, selected_alpha
                ),
                validation_metrics=validation_by_alpha[selected_alpha],
                test_metrics=evaluate_ngram_model(model, encoded_test, selected_alpha),
            )
        )
    distributions = tuple(
        calculate_top_k_distribution(
            models[query.order],
            query.context,
            selected_alphas[query.order],
            settings.top_k,
        )
        for query in settings.probability_queries
    )
    samples = tuple(
        generate_sample(
            models[order],
            selected_alphas[order],
            seed,
            settings.maximum_generated_tokens,
        )
        for order in settings.orders
        for seed in settings.generation_seeds
    )
    corpus = CorpusStatistics(
        sentence_count=experiment.sentence_count,
        training_sentence_count=len(experiment.training_sentences),
        validation_sentence_count=len(experiment.validation_sentences),
        test_sentence_count=len(experiment.test_sentences),
        training_word_count=sum(map(len, experiment.training_sentences)),
        validation_word_count=sum(map(len, experiment.validation_sentences)),
        test_word_count=sum(map(len, experiment.test_sentences)),
        vocabulary_size=len(vocabulary),
    )
    return NGramExperimentResult(
        source_title=experiment.source_title,
        source_resource=experiment.source_resource,
        source_sha256=experiment.source_sha256,
        corpus=corpus,
        settings=settings,
        order_results=tuple(order_results),
        probability_distributions=distributions,
        generated_samples=samples,
    )


def _parse_settings(cfg: DictConfig) -> NGramSettings:
    orders = _parse_orders(cfg)
    alphas = _parse_alphas(cfg)
    queries = _parse_probability_queries(cfg, orders)
    split_seed = _finite_integer(cfg.split.seed, "The split seed")
    training_fraction = float(cfg.split.training_fraction)
    validation_fraction = float(cfg.split.validation_fraction)
    if split_seed < 0:
        raise ValueError("The split seed must be non-negative")
    _validate_split_fractions(training_fraction, validation_fraction)
    minimum_token_count = _finite_integer(
        cfg.minimum_token_count, "The minimum token count"
    )
    if minimum_token_count <= 0:
        raise ValueError("The minimum token count must be positive")
    top_k = _finite_integer(cfg.top_k, "The top-k value")
    if top_k <= 0:
        raise ValueError("The top-k value must be positive")
    generation_seeds = tuple(
        _finite_integer(seed, "The generation seed") for seed in cfg.generation.seeds
    )
    if not generation_seeds or any(seed < 0 for seed in generation_seeds):
        raise ValueError("Generation seeds must be non-empty and non-negative")
    maximum_generated_tokens = _finite_integer(
        cfg.generation.max_tokens, "The generation token limit"
    )
    if maximum_generated_tokens <= 0:
        raise ValueError("The generation token limit must be positive")
    return NGramSettings(
        split_seed=split_seed,
        training_fraction=training_fraction,
        validation_fraction=validation_fraction,
        minimum_token_count=minimum_token_count,
        orders=orders,
        alphas=alphas,
        probability_queries=queries,
        top_k=top_k,
        generation_seeds=generation_seeds,
        maximum_generated_tokens=maximum_generated_tokens,
    )


def _parse_orders(cfg: DictConfig) -> tuple[int, ...]:
    orders = tuple(_finite_integer(order, "The n-gram order") for order in cfg.orders)
    if not orders or any(order <= 0 for order in orders):
        raise ValueError("The n-gram orders must be non-empty and positive")
    if len(set(orders)) != len(orders):
        raise ValueError("The n-gram orders must be unique")
    return orders


def _parse_alphas(cfg: DictConfig) -> tuple[float, ...]:
    alphas = tuple(float(alpha) for alpha in cfg.smoothing_alphas)
    if not alphas or any(not isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("The smoothing grid must contain finite positive values")
    if len(set(alphas)) != len(alphas):
        raise ValueError("The smoothing grid must contain unique values")
    return alphas


def _parse_probability_queries(
    cfg: DictConfig, orders: tuple[int, ...]
) -> tuple[ProbabilityQuery, ...]:
    queries = tuple(
        ProbabilityQuery(
            order=_finite_integer(query.order, "The probability query order"),
            context=tuple(str(token) for token in query.context),
        )
        for query in cfg.probability_queries
    )
    if not queries:
        raise ValueError("At least one probability query is required")
    for query in queries:
        if query.order not in orders:
            raise ValueError(f"Probability query uses unconfigured order {query.order}")
        if len(query.context) != query.order - 1:
            raise ValueError(
                f"Order {query.order} probability query requires "
                f"{query.order - 1} context tokens"
            )
    return queries


def _finite_integer(value: Any, name: str) -> int:
    """Convert an integer-valued Hydra scalar without silent truncation."""
    error = ValueError(f"{name} must be a finite integer")
    if isinstance(value, bool):
        raise error
    if isinstance(value, int):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise error from exc
    if not isfinite(numeric) or not numeric.is_integer():
        raise error
    return int(numeric)


def _validate_split_fractions(
    training_fraction: float, validation_fraction: float
) -> None:
    fractions = (
        training_fraction,
        validation_fraction,
        1.0 - training_fraction - validation_fraction,
    )
    if any(not isfinite(fraction) or not 0 < fraction < 1 for fraction in fractions):
        raise ValueError("Each split fraction must be finite and between zero and one")
    if not isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The split fractions must sum to one")
