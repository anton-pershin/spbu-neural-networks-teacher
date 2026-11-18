from pathlib import Path

import pytest

from spbunn.sequence.pipeline import (
    extract_words,
    load_book_corpus,
    produce_large_input,
    produce_toy_input,
    run_large_experiment,
    run_toy_experiment,
    split_chapters,
)
from spbunn.utils.common import get_project_path
from tests.sequence_helpers import compose_sequence_config

BOOK_PATH = get_project_path() / "data" / "alice_in_wonderland.txt"


def test_word_extraction_normalizes_case_and_apostrophes() -> None:
    assert extract_words("Alice’s DON'T rabbit-hole _I_") == (
        "alice's",
        "don't",
        "rabbit",
        "hole",
        "i",
    )


def test_chapter_parser_accepts_both_book_heading_layouts() -> None:
    chapters = split_chapters(BOOK_PATH.read_text(encoding="utf-8"))

    assert tuple(chapters) == tuple(range(1, 13))
    assert "Down the Rabbit-Hole" not in chapters[1]
    assert "Alice was beginning" in chapters[1]
    assert "Who Stole the Tarts" not in chapters[11]
    assert "The King and Queen of Hearts" in chapters[11]
    assert "Alice's Evidence" not in chapters[12]
    assert "pack of cards" in chapters[12]


def test_missing_chapter_has_clear_error() -> None:
    with pytest.raises(ValueError, match="missing requested chapters"):
        split_chapters("CHAPTER I. Only one\n\nSome text.")


def test_book_loader_uses_training_and_heldout_chapter_ranges() -> None:
    book = load_book_corpus(BOOK_PATH)

    assert book.training_chapters == tuple(range(1, 11))
    assert book.heldout_chapters == (11, 12)
    assert sum(item.frequency for item in book.corpus) > 20_000
    assert len(book.heldout_words) > 2_000
    assert "knave" in book.heldout_words
    assert "lowest" not in {item.word for item in book.corpus}
    assert [item.word for item in book.corpus] == sorted(
        item.word for item in book.corpus
    )
    assert all(item.frequency > 0 for item in book.corpus)


def test_production_configuration_composes_and_loads_inputs() -> None:
    cfg = compose_sequence_config("config_bpe")

    toy = produce_toy_input(cfg)
    large = produce_large_input(cfg)

    assert cfg.result_dir == cfg.hydra_dir
    assert cfg.report.artifact_name == "bpe.html"
    assert cfg.report.toy_figure_artifact_name == "bpe_toy_dynamics"
    assert cfg.report.large_figure_artifact_name == "bpe_large_corpus_compression"
    assert toy.n_merges == 12
    assert tuple(item.word for item in toy.corpus) == (
        "low",
        "lower",
        "newest",
        "widest",
    )
    assert toy.sample_words == (
        "low",
        "lower",
        "newest",
        "lowest",
        "widest",
        "wide",
    )
    assert large.n_merges == 1_000
    assert large.checkpoints == (0, 100, 250, 500, 1_000)
    assert large.sentence_words == (
        "the",
        "knave",
        "of",
        "hearts",
        "he",
        "stole",
        "those",
        "tarts",
        "and",
        "took",
        "them",
        "quite",
        "away",
    )
    assert large.training_chapters == tuple(range(1, 11))
    assert large.heldout_chapters == (11, 12)


def test_sentence_must_come_from_heldout_text() -> None:
    cfg = compose_sequence_config(
        "config_test_bpe", ["large.sentence_words.0=notintheheldouttext"]
    )

    with pytest.raises(ValueError, match="sentence words must occur in held-out"):
        produce_large_input(cfg)


def test_nonzero_baseline_override_is_rejected_through_experiment() -> None:
    cfg = compose_sequence_config("config_test_bpe", ["large.checkpoints=[10,20]"])
    experiment = produce_large_input(cfg)

    with pytest.raises(ValueError, match="zero-merge baseline"):
        run_large_experiment(experiment)


def test_small_real_corpus_experiments_use_same_data_flow() -> None:
    cfg = compose_sequence_config("config_test_bpe")

    toy = run_toy_experiment(produce_toy_input(cfg))
    large = run_large_experiment(produce_large_input(cfg))

    assert len(toy.training.steps) == 4
    assert len(toy.samples) == 6
    assert len(large.training.steps) == 20
    assert large.training.steps[-1].corpus is None
    assert tuple(
        checkpoint.merge_count for checkpoint in large.evaluation.checkpoints
    ) == (0, 10, 20)
    assert (
        large.evaluation.checkpoints[-1].heldout_token_count
        < large.evaluation.checkpoints[0].heldout_token_count
    )
    assert large.evaluation.frequent_tokens
    assert large.evaluation.unseen_words


def test_missing_corpus_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        load_book_corpus(tmp_path / "missing.txt")


def test_production_book_run_reaches_merges_and_compresses_heldout() -> None:
    cfg = compose_sequence_config("config_bpe")

    result = run_large_experiment(produce_large_input(cfg))

    assert len(result.training.steps) == 1_000
    checkpoints = result.evaluation.checkpoints
    assert tuple(item.merge_count for item in checkpoints) == (
        0,
        100,
        250,
        500,
        1_000,
    )
    assert [item.vocabulary_size for item in checkpoints] == [
        28,
        128,
        278,
        528,
        1_028,
    ]
    assert [item.training_token_count for item in checkpoints] == [
        113_985,
        57_293,
        44_402,
        36_199,
        29_903,
    ]
    assert [item.heldout_token_count for item in checkpoints] == [
        20_581,
        10_330,
        8_171,
        6_696,
        5_761,
    ]
    assert checkpoints[-1].heldout_token_count < checkpoints[0].heldout_token_count
    assert checkpoints[-1].training_token_count < checkpoints[0].training_token_count
    assert checkpoints[-1].sentence
    assert result.evaluation.frequent_tokens
    assert result.evaluation.unseen_words
    assert all(
        example.word not in {item.word for item in result.training.initial_corpus}
        for example in result.evaluation.unseen_words
    )
