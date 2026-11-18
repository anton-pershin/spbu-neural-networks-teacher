import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from spbunn.sequence.bpe import (
    BPEEvaluation,
    BPETraining,
    SampleTokenization,
    WordFrequency,
    evaluate_bpe,
    tokenize_samples,
    train_bpe,
)
from spbunn.utils.common import get_project_path

BOOK_CHAPTERS = tuple(range(1, 13))
TRAINING_CHAPTERS = tuple(range(1, 11))
HELDOUT_CHAPTERS = (11, 12)
ROMAN_NUMERALS = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}
CHAPTER_PATTERN = re.compile(
    r"(?m)^[ \t]*CHAPTER[ \t]+([IVXLCDM]+)(?:\.[^\n]*)?[ \t]*$"
)
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class BookCorpus:
    corpus: tuple[WordFrequency, ...]
    heldout_words: tuple[str, ...]
    training_chapters: tuple[int, ...]
    heldout_chapters: tuple[int, ...]


@dataclass(frozen=True)
class ToyExperimentInput:
    corpus: tuple[WordFrequency, ...]
    n_merges: int
    sample_words: tuple[str, ...]


@dataclass(frozen=True)
class ToyExperimentResult:
    training: BPETraining
    initial_samples: tuple[SampleTokenization, ...]
    samples: tuple[SampleTokenization, ...]


@dataclass(frozen=True)
class LargeExperimentInput:
    corpus: tuple[WordFrequency, ...]
    heldout_words: tuple[str, ...]
    n_merges: int
    checkpoints: tuple[int, ...]
    sentence_words: tuple[str, ...]
    frequent_token_limit: int
    unseen_word_limit: int
    training_chapters: tuple[int, ...]
    heldout_chapters: tuple[int, ...]


@dataclass(frozen=True)
class LargeExperimentResult:
    training: BPETraining
    evaluation: BPEEvaluation
    training_chapters: tuple[int, ...]
    heldout_chapters: tuple[int, ...]


def extract_words(text: str) -> tuple[str, ...]:
    normalized = text.replace("’", "'").lower()
    return tuple(WORD_PATTERN.findall(normalized))


def split_chapters(
    text: str, required_chapters: Sequence[int] = BOOK_CHAPTERS
) -> dict[int, str]:
    matches = list(CHAPTER_PATTERN.finditer(text))
    chapters = {}
    for index, match in enumerate(matches):
        chapter_number = ROMAN_NUMERALS[match.group(1)]
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chapters[chapter_number] = text[match.end() : end]
    missing = set(required_chapters).difference(chapters)
    if missing:
        raise ValueError(
            f"missing requested chapters: {', '.join(map(str, sorted(missing)))}"
        )
    return {number: chapters[number] for number in required_chapters}


def load_book_corpus(
    path: str | Path,
    training_chapters: Sequence[int] = TRAINING_CHAPTERS,
    heldout_chapters: Sequence[int] = HELDOUT_CHAPTERS,
) -> BookCorpus:
    """Load the local book and construct weighted training and held-out words."""
    text = Path(path).read_text(encoding="utf-8")
    requested_chapters = tuple(training_chapters) + tuple(heldout_chapters)
    chapters = split_chapters(text, requested_chapters)
    training_words = tuple(
        word
        for chapter in training_chapters
        for word in extract_words(chapters[chapter])
    )
    heldout_words = tuple(
        word
        for chapter in heldout_chapters
        for word in extract_words(chapters[chapter])
    )
    frequencies = Counter(training_words)
    corpus = tuple(
        WordFrequency(word=word, frequency=frequencies[word])
        for word in sorted(frequencies)
    )
    return BookCorpus(
        corpus=corpus,
        heldout_words=heldout_words,
        training_chapters=tuple(training_chapters),
        heldout_chapters=tuple(heldout_chapters),
    )


def produce_toy_input(cfg: DictConfig) -> ToyExperimentInput:
    return ToyExperimentInput(
        corpus=tuple(
            WordFrequency(word=str(item.word), frequency=int(item.frequency))
            for item in cfg.toy.corpus
        ),
        n_merges=int(cfg.toy.n_merges),
        sample_words=tuple(str(word) for word in cfg.toy.sample_words),
    )


def produce_large_input(cfg: DictConfig) -> LargeExperimentInput:
    corpus_path = Path(str(cfg.large.corpus_path))
    if not corpus_path.is_absolute():
        corpus_path = get_project_path() / corpus_path
    book = load_book_corpus(corpus_path)
    sentence_words = tuple(str(word) for word in cfg.large.sentence_words)
    missing_sentence_words = set(sentence_words).difference(book.heldout_words)
    if missing_sentence_words:
        raise ValueError(
            "sentence words must occur in held-out chapters: "
            + ", ".join(sorted(missing_sentence_words))
        )
    return LargeExperimentInput(
        corpus=book.corpus,
        heldout_words=book.heldout_words,
        n_merges=int(cfg.large.n_merges),
        checkpoints=tuple(int(item) for item in cfg.large.checkpoints),
        sentence_words=sentence_words,
        frequent_token_limit=int(cfg.large.frequent_token_limit),
        unseen_word_limit=int(cfg.large.unseen_word_limit),
        training_chapters=book.training_chapters,
        heldout_chapters=book.heldout_chapters,
    )


def run_toy_experiment(experiment: ToyExperimentInput) -> ToyExperimentResult:
    training = train_bpe(experiment.corpus, experiment.n_merges, keep_trace=True)
    return ToyExperimentResult(
        training=training,
        initial_samples=tokenize_samples(
            experiment.sample_words, training, merge_count=0
        ),
        samples=tokenize_samples(experiment.sample_words, training),
    )


def run_large_experiment(
    experiment: LargeExperimentInput,
) -> LargeExperimentResult:
    training = train_bpe(experiment.corpus, experiment.n_merges, keep_trace=False)
    evaluation = evaluate_bpe(
        training,
        heldout_words=experiment.heldout_words,
        checkpoints=experiment.checkpoints,
        sentence_words=experiment.sentence_words,
        frequent_token_limit=experiment.frequent_token_limit,
        unseen_word_limit=experiment.unseen_word_limit,
    )
    return LargeExperimentResult(
        training=training,
        evaluation=evaluation,
        training_chapters=experiment.training_chapters,
        heldout_chapters=experiment.heldout_chapters,
    )
