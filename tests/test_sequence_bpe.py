from collections import Counter

import pytest

from spbunn.sequence.bpe import (
    END_OF_WORD,
    BPETraining,
    WordFrequency,
    evaluate_bpe,
    tokenize_samples,
    tokenize_word,
    train_bpe,
    training_token_counts,
    training_vocabulary_sizes,
    weighted_token_count,
)

TOY_CORPUS = (
    WordFrequency("low", 5),
    WordFrequency("lower", 2),
    WordFrequency("newest", 6),
    WordFrequency("widest", 3),
)


def train_toy(n_merges: int = 12) -> BPETraining:
    return train_bpe(TOY_CORPUS, n_merges, keep_trace=True)


def test_toy_training_matches_hand_calculated_trace() -> None:
    result = train_toy()

    assert len(result.initial_vocabulary) == 11
    assert result.initial_vocabulary == (
        "d",
        "e",
        "i",
        "l",
        "n",
        "o",
        "r",
        "s",
        "t",
        "w",
        END_OF_WORD,
    )
    assert weighted_token_count(result.initial_corpus) == 95
    assert len(result.steps) == 12
    assert [(step.pair, step.token, step.frequency) for step in result.steps[:3]] == [
        (("e", "s"), "es", 9),
        (("es", "t"), "est", 9),
        (("est", END_OF_WORD), f"est{END_OF_WORD}", 9),
    ]
    assert len(result.final_vocabulary) == 23
    assert result.final_vocabulary[:11] == result.initial_vocabulary
    assert result.final_vocabulary[11:] == tuple(step.token for step in result.steps)
    assert weighted_token_count(result.final_corpus) == 22
    assert training_vocabulary_sizes(result)[0] == 11
    assert training_vocabulary_sizes(result)[-1] == 23
    assert training_token_counts(result)[0] == 95
    assert training_token_counts(result)[-1] == 22
    assert all(step.corpus is not None for step in result.steps)


def test_overlapping_pair_is_replaced_from_left_to_right() -> None:
    result = train_bpe((WordFrequency("aaa", 1),), 1, keep_trace=True)

    assert result.steps[0].pair == ("a", "a")
    assert result.steps[0].frequency == 2
    assert result.final_corpus[0].tokens == ("aa", "a", END_OF_WORD)
    assert weighted_token_count(result.initial_corpus) == 4
    assert weighted_token_count(result.final_corpus) == 3


def test_equal_pair_counts_use_lexicographic_tie_breaking() -> None:
    result = train_bpe(
        (WordFrequency("cd", 1), WordFrequency("ab", 1)),
        1,
    )

    assert result.steps[0].pair == ("a", "b")


def test_tie_breaking_uses_second_token_when_first_tokens_match() -> None:
    result = train_bpe(
        (WordFrequency("ac", 1), WordFrequency("ab", 1)),
        1,
    )

    assert result.steps[0].pair == ("a", "b")


def test_training_stops_when_no_adjacent_pair_remains() -> None:
    result = train_bpe((WordFrequency("a", 1),), 5)

    assert len(result.steps) == 1
    assert result.steps[0].pair == ("a", END_OF_WORD)
    assert result.final_corpus[0].tokens == (f"a{END_OF_WORD}",)


def test_zero_merges_preserves_initial_corpus() -> None:
    result = train_toy(0)

    assert not result.steps
    assert result.final_corpus == result.initial_corpus
    assert result.final_vocabulary == result.initial_vocabulary


def test_unseen_word_reuses_toy_merge_rules() -> None:
    result = train_toy()

    assert tokenize_word("lowest", result.initial_vocabulary, result.steps) == (
        "low",
        f"est{END_OF_WORD}",
    )


def test_tokenization_preserves_each_word_and_word_boundary() -> None:
    result = train_toy()
    words = ("low", "lower", "newest", "lowest", "widest", "wide")

    samples = tokenize_samples(words, result)

    assert tuple(sample.word for sample in samples) == words
    for sample in samples:
        assert sample.tokens[-1].endswith(END_OF_WORD)
        assert "".join(sample.tokens).replace(END_OF_WORD, "") == sample.word


def test_unknown_sample_character_has_clear_error() -> None:
    result = train_toy()

    with pytest.raises(ValueError, match="unknown initial tokens.*z"):
        tokenize_samples(("lowz",), result)


@pytest.mark.parametrize(
    ("corpus", "n_merges", "message"),
    [
        ((), 1, "corpus must not be empty"),
        ((WordFrequency("low", 1),), -1, "merge count must be non-negative"),
        ((WordFrequency("", 1),), 1, "word must not be empty"),
        ((WordFrequency("two words", 1),), 1, "word must not contain whitespace"),
        ((WordFrequency("low", 0),), 1, "frequency must be positive"),
        (
            (WordFrequency("low", 1), WordFrequency("low", 2)),
            1,
            "duplicate word",
        ),
    ],
)
def test_invalid_training_input_has_clear_error(
    corpus: tuple[WordFrequency, ...], n_merges: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        train_bpe(corpus, n_merges)


def test_checkpoint_metrics_match_independent_prefix_models() -> None:
    result = train_toy()
    heldout_words = ("lowest", "lowest", "wide")
    sentence_words = ("lowest", "wide")
    checkpoints = (0, 4, 8, 12)

    evaluation = evaluate_bpe(
        result,
        heldout_words=heldout_words,
        checkpoints=checkpoints,
        sentence_words=sentence_words,
    )

    assert tuple(item.merge_count for item in evaluation.checkpoints) == checkpoints
    for checkpoint in evaluation.checkpoints:
        prefix_model = train_toy(checkpoint.merge_count)
        expected_counts = [
            len(
                tokenize_word(word, prefix_model.initial_vocabulary, prefix_model.steps)
            )
            for word in heldout_words
        ]
        assert checkpoint.heldout_token_count == sum(expected_counts)
        assert checkpoint.training_token_count == weighted_token_count(
            prefix_model.final_corpus
        )
        assert checkpoint.vocabulary_size == len(prefix_model.final_vocabulary)
        assert checkpoint.heldout_tokens_per_word == pytest.approx(
            sum(expected_counts) / len(heldout_words)
        )
        assert checkpoint.sentence == tokenize_samples(sentence_words, prefix_model)


def test_checkpoint_baseline_has_hand_calculated_counts() -> None:
    evaluation = evaluate_bpe(
        train_toy(),
        heldout_words=("lowest", "lowest", "wide"),
        checkpoints=(0, 12),
        sentence_words=("lowest",),
    )

    baseline, final = evaluation.checkpoints
    assert baseline.heldout_word_count == 3
    assert baseline.heldout_token_count == 19
    assert baseline.heldout_tokens_per_word == pytest.approx(19 / 3)
    assert final.heldout_token_count == 9
    assert final.heldout_tokens_per_word == 3


def test_frequent_tokens_and_unseen_examples_come_from_results() -> None:
    result = train_toy()
    heldout_words = ("lowest", "lowest", "wide", "low")

    evaluation = evaluate_bpe(
        result,
        heldout_words=heldout_words,
        checkpoints=(0,),
        sentence_words=("lowest",),
        frequent_token_limit=4,
        unseen_word_limit=2,
    )

    assert evaluation.frequent_tokens[0].token == f"newest{END_OF_WORD}"
    assert evaluation.frequent_tokens[0].frequency == 6
    assert [item.word for item in evaluation.unseen_words] == ["lowest", "wide"]
    assert [item.frequency for item in evaluation.unseen_words] == [2, 1]
    assert evaluation.unseen_words[0].tokens == (
        "low",
        f"est{END_OF_WORD}",
    )
    assert set(item.word for item in evaluation.unseen_words).isdisjoint(
        item.word for item in TOY_CORPUS
    )


@pytest.mark.parametrize(
    "checkpoints",
    [(1, 0), (0, 0), (-1,), (13,), (1, 2)],
)
def test_invalid_checkpoints_have_clear_error(checkpoints: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="checkpoints"):
        evaluate_bpe(
            train_toy(),
            heldout_words=("lowest",),
            checkpoints=checkpoints,
            sentence_words=("lowest",),
        )


def test_training_and_evaluation_are_deterministic() -> None:
    first = train_toy()
    second = train_toy()

    assert first == second
    assert evaluate_bpe(
        first,
        heldout_words=("lowest", "wide"),
        checkpoints=(0, 12),
        sentence_words=("lowest",),
    ) == evaluate_bpe(
        second,
        heldout_words=("lowest", "wide"),
        checkpoints=(0, 12),
        sentence_words=("lowest",),
    )


def test_unseen_example_order_uses_frequency_then_word() -> None:
    result = train_toy()
    frequencies = Counter(("wide", "wide", "lowest", "lowest", "low"))

    assert frequencies["wide"] == frequencies["lowest"]
    evaluation = evaluate_bpe(
        result,
        heldout_words=tuple(frequencies.elements()),
        checkpoints=(0,),
        sentence_words=("lowest",),
        unseen_word_limit=2,
    )

    assert [item.word for item in evaluation.unseen_words] == ["lowest", "wide"]
