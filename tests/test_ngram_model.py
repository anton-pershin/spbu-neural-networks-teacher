"""Independent numerical checks for count-based n-gram models."""

from dataclasses import replace
from math import exp, log

import pytest

from spbunn.ngram.model import (
    BOS_TOKEN,
    EOS_TOKEN,
    EvaluationMetrics,
    NGramModel,
    calculate_top_k_distribution,
    encode_sentences,
    evaluate_ngram_model,
    fit_ngram_model,
    generate_sample,
)
from spbunn.ngram.pipeline import evaluate_ngram_experiment
from tests.ngram_helpers import tiny_experiment


def test_encoding_maps_oov_words_and_appends_eos() -> None:
    experiment = tiny_experiment()

    encoded = encode_sentences((("the", "new", "cat"),), experiment.known_tokens)

    assert encoded == (("the", "<unk>", "cat", EOS_TOKEN),)


def test_fit_bigram_counts_and_add_alpha_probabilities() -> None:
    experiment = tiny_experiment()
    training = encode_sentences(experiment.training_sentences, experiment.known_tokens)
    model = fit_ngram_model(training, 2, experiment.prediction_vocabulary)

    assert model.context_count((BOS_TOKEN,)) == 3
    assert model.count((BOS_TOKEN,), "the") == 3
    assert model.context_count(("the",)) == 3
    assert model.count(("the",), "cat") == 2
    assert model.count(("the",), "dog") == 1
    assert model.count(("cat",), EOS_TOKEN) == 1
    assert model.count(("sat",), EOS_TOKEN) == 2

    denominator = 3 + 0.1 * 6
    assert model.probability(("the",), "cat", 0.1) == pytest.approx(2.1 / denominator)
    assert model.probability(("the",), "dog", 0.1) == pytest.approx(1.1 / denominator)
    assert model.probability(("the",), "<unk>", 0.1) == pytest.approx(0.1 / denominator)
    assert sum(
        probability for _, _, probability in model.distribution(("the",), 0.1)
    ) == pytest.approx(1.0)


def test_trigram_padding_and_unseen_context_are_exact() -> None:
    experiment = tiny_experiment()
    training = encode_sentences(experiment.training_sentences, experiment.known_tokens)
    model = fit_ngram_model(training, 3, experiment.prediction_vocabulary)

    assert model.count((BOS_TOKEN, BOS_TOKEN), "the") == 3
    assert model.count((BOS_TOKEN, "the"), "cat") == 2
    assert model.count((BOS_TOKEN, "the"), "dog") == 1
    assert model.count(("the", "cat"), "sat") == 1
    assert model.count(("the", "cat"), EOS_TOKEN) == 1

    unseen_context = ("of", "the")
    assert model.context_count(unseen_context) == 0
    assert model.distribution(unseen_context, 0.1) == tuple(
        (token, 0, pytest.approx(1 / 6)) for token in experiment.prediction_vocabulary
    )


def test_perplexity_and_sparsity_match_independent_calculation() -> None:
    experiment = tiny_experiment()
    training = encode_sentences(experiment.training_sentences, experiment.known_tokens)
    test = encode_sentences(experiment.test_sentences, experiment.known_tokens)
    model = fit_ngram_model(training, 2, experiment.prediction_vocabulary)

    metrics = evaluate_ngram_model(model, test, 0.1)
    probabilities = (
        model.probability((BOS_TOKEN,), "the", 0.1),
        model.probability(("the",), "dog", 0.1),
        model.probability(("dog",), EOS_TOKEN, 0.1),
    )
    expected = exp(-sum(log(probability) for probability in probabilities) / 3)

    assert probabilities[-1] == pytest.approx(0.1 / 1.6)
    assert metrics.perplexity == pytest.approx(expected)
    assert metrics.event_count == 3
    assert metrics.unseen_ngram_fraction == pytest.approx(1 / 3)
    assert metrics.unseen_context_fraction == 0.0


def test_unigram_has_zero_unseen_context_fraction_with_oov_events() -> None:
    experiment = tiny_experiment()
    training = encode_sentences(experiment.training_sentences, experiment.known_tokens)
    evaluation = encode_sentences((("new",),), experiment.known_tokens)
    model = fit_ngram_model(training, 1, experiment.prediction_vocabulary)

    metrics = evaluate_ngram_model(model, evaluation, 0.1)

    assert metrics.event_count == 2
    assert metrics.unseen_ngram_fraction == pytest.approx(0.5)
    assert metrics.unseen_context_fraction == 0.0


def test_trigram_unseen_context_fraction_matches_tiny_corpus() -> None:
    experiment = tiny_experiment()
    training = encode_sentences(experiment.training_sentences, experiment.known_tokens)
    evaluation = encode_sentences((("cat", "dog"),), experiment.known_tokens)
    model = fit_ngram_model(training, 3, experiment.prediction_vocabulary)

    metrics = evaluate_ngram_model(model, evaluation, 0.1)

    assert metrics.event_count == 3
    assert metrics.unseen_ngram_fraction == pytest.approx(1.0)
    assert metrics.unseen_context_fraction == pytest.approx(2 / 3)


def test_top_k_probability_order_ties_and_remaining_mass() -> None:
    experiment = tiny_experiment()
    training = encode_sentences(experiment.training_sentences, experiment.known_tokens)
    model = fit_ngram_model(training, 2, experiment.prediction_vocabulary)

    distribution = calculate_top_k_distribution(model, ("the",), 0.1, 3)

    assert [item.token for item in distribution.items] == ["cat", "dog", EOS_TOKEN]
    assert [item.count for item in distribution.items] == [2, 1, 0]
    assert distribution.items[0].probability == pytest.approx(2.1 / 3.6)
    assert distribution.items[1].probability == pytest.approx(1.1 / 3.6)
    assert distribution.items[2].probability == pytest.approx(0.1 / 3.6)
    assert distribution.remaining_probability == pytest.approx(0.3 / 3.6)
    assert sum(
        item.probability for item in distribution.items
    ) + distribution.remaining_probability == pytest.approx(1.0)


def test_top_k_uses_token_text_to_break_equal_probability_ties() -> None:
    experiment = tiny_experiment()
    training = encode_sentences(experiment.training_sentences, experiment.known_tokens)
    model = fit_ngram_model(training, 3, experiment.prediction_vocabulary)

    distribution = calculate_top_k_distribution(model, ("of", "the"), 0.1, 3)

    assert [item.token for item in distribution.items] == [EOS_TOKEN, "<unk>", "cat"]
    assert all(item.probability == pytest.approx(1 / 6) for item in distribution.items)
    assert distribution.remaining_probability == pytest.approx(0.5)


def test_alpha_selection_uses_validation_not_test_data() -> None:
    experiment = tiny_experiment()
    changed_test = replace(
        experiment,
        test_sentences=(("not", "in", "vocabulary"),),
    )

    first = evaluate_ngram_experiment(experiment)
    second = evaluate_ngram_experiment(changed_test)

    assert [item.selected_alpha for item in first.order_results] == [
        item.selected_alpha for item in second.order_results
    ]
    assert [item.validation_curve for item in first.order_results] == [
        item.validation_curve for item in second.order_results
    ]
    assert [item.test_metrics for item in first.order_results] != [
        item.test_metrics for item in second.order_results
    ]


def test_alpha_tie_selects_earlier_configured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def tied_metrics(
        model: NGramModel, sentences: object, alpha: float
    ) -> EvaluationMetrics:
        del model, sentences, alpha
        return EvaluationMetrics(
            perplexity=10.0,
            unseen_ngram_fraction=0.0,
            unseen_context_fraction=0.0,
            event_count=1,
        )

    monkeypatch.setattr("spbunn.ngram.pipeline.evaluate_ngram_model", tied_metrics)

    result = evaluate_ngram_experiment(tiny_experiment())

    assert [item.selected_alpha for item in result.order_results] == [0.1, 0.1, 0.1]
    assert all(
        [point.perplexity for point in item.validation_curve] == [10.0, 10.0]
        for item in result.order_results
    )


def test_experiment_generation_and_probability_results_are_deterministic() -> None:
    first = evaluate_ngram_experiment(tiny_experiment())
    second = evaluate_ngram_experiment(tiny_experiment())

    assert first.generated_samples == second.generated_samples
    assert first.probability_distributions == second.probability_distributions
    assert len(first.generated_samples) == 3
    assert [sample.initial_context for sample in first.generated_samples] == [
        (),
        ("the",),
        ("the", "cat"),
    ]


def test_generation_can_stop_immediately_at_eos() -> None:
    model = NGramModel(
        order=1,
        vocabulary=(EOS_TOKEN,),
        ngram_counts={(EOS_TOKEN,): 1},
        context_counts={(): 1},
    )

    sample = generate_sample(model, alpha=0.1, seed=0, max_tokens=5, initial_context=())

    assert not sample.tokens
    assert sample.stop_reason == "eos"


def test_generation_enforces_length_limit() -> None:
    model = NGramModel(
        order=1,
        vocabulary=("a",),
        ngram_counts={("a",): 1},
        context_counts={(): 1},
    )

    sample = generate_sample(model, alpha=0.1, seed=0, max_tokens=4, initial_context=())

    assert sample.tokens == ("a", "a", "a", "a")
    assert sample.stop_reason == "length_limit"


def test_bigram_generation_shifts_real_initial_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = NGramModel(
        order=2,
        vocabulary=(EOS_TOKEN, "a", "alice"),
        ngram_counts={
            ("alice", "a"): 1,
            ("a", "a"): 1,
            ("a", EOS_TOKEN): 1,
        },
        context_counts={("alice",): 1, ("a",): 2},
    )
    calls: list[tuple[tuple[str, ...], tuple[float, ...]]] = []

    class FakeRandom:
        def __init__(self, seed: int) -> None:
            assert seed == 9
            self.outputs = iter(("a", EOS_TOKEN))

        def choices(
            self, population: list[str], weights: list[float], k: int
        ) -> list[str]:
            assert k == 1
            calls.append((tuple(population), tuple(weights)))
            return [next(self.outputs)]

    monkeypatch.setattr("spbunn.ngram.model.Random", FakeRandom)
    sample = generate_sample(
        model, alpha=1.0, seed=9, max_tokens=4, initial_context=("alice",)
    )

    assert sample.initial_context == ("alice",)
    assert sample.tokens == ("a",)
    assert sample.stop_reason == "eos"
    assert calls[0][1] == (
        pytest.approx(1 / 4),
        pytest.approx(1 / 2),
        pytest.approx(1 / 4),
    )
    assert calls[1][1] == (
        pytest.approx(2 / 5),
        pytest.approx(2 / 5),
        pytest.approx(1 / 5),
    )


def test_generation_rejects_invalid_initial_context() -> None:
    experiment = tiny_experiment()
    training = encode_sentences(experiment.training_sentences, experiment.known_tokens)
    model = fit_ngram_model(training, 2, experiment.prediction_vocabulary)

    with pytest.raises(ValueError, match="requires a context of length 1"):
        generate_sample(model, alpha=0.1, seed=0, max_tokens=1, initial_context=())
    with pytest.raises(ValueError, match="prediction vocabulary"):
        generate_sample(
            model, alpha=0.1, seed=0, max_tokens=1, initial_context=("missing",)
        )
    with pytest.raises(ValueError, match="must not contain the end-of-sentence"):
        generate_sample(
            model, alpha=0.1, seed=0, max_tokens=1, initial_context=(EOS_TOKEN,)
        )


def test_model_rejects_invalid_context_and_token() -> None:
    experiment = tiny_experiment()
    training = encode_sentences(experiment.training_sentences, experiment.known_tokens)
    model = fit_ngram_model(training, 2, experiment.prediction_vocabulary)

    with pytest.raises(ValueError, match="requires a context"):
        model.probability((), "cat", 0.1)
    with pytest.raises(ValueError, match="outside the prediction vocabulary"):
        model.probability(("the",), "missing", 0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        model.probability(("the",), "cat", 0.0)
