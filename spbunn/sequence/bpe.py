from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

END_OF_WORD = "</w>"
TokenPair = tuple[str, str]


@dataclass(frozen=True)
class WordFrequency:
    word: str
    frequency: int


@dataclass(frozen=True)
class TokenizedWord:
    word: str
    frequency: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class MergeStep:
    rank: int
    pair: TokenPair
    token: str
    frequency: int
    vocabulary_size: int
    corpus_token_count: int
    corpus: tuple[TokenizedWord, ...] | None = None


@dataclass(frozen=True)
class BPETraining:
    initial_vocabulary: tuple[str, ...]
    final_vocabulary: tuple[str, ...]
    initial_corpus: tuple[TokenizedWord, ...]
    final_corpus: tuple[TokenizedWord, ...]
    steps: tuple[MergeStep, ...]


@dataclass(frozen=True)
class SampleTokenization:
    word: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class TokenFrequency:
    token: str
    frequency: int


@dataclass(frozen=True)
class UnseenWordExample:
    word: str
    frequency: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointEvaluation:
    merge_count: int
    vocabulary_size: int
    training_token_count: int
    heldout_token_count: int
    heldout_word_count: int
    heldout_tokens_per_word: float
    sentence: tuple[SampleTokenization, ...]


@dataclass(frozen=True)
class BPEEvaluation:
    checkpoints: tuple[CheckpointEvaluation, ...]
    frequent_tokens: tuple[TokenFrequency, ...]
    unseen_words: tuple[UnseenWordExample, ...]


def _validate_word(word: str) -> None:
    if not word:
        raise ValueError("word must not be empty")
    if any(character.isspace() for character in word):
        raise ValueError(f"word must not contain whitespace: {word!r}")


def _initial_corpus(corpus: Sequence[WordFrequency]) -> tuple[TokenizedWord, ...]:
    if not corpus:
        raise ValueError("training corpus must not be empty")
    seen_words = set()
    tokenized = []
    for item in corpus:
        _validate_word(item.word)
        if item.word in seen_words:
            raise ValueError(f"duplicate word in training corpus: {item.word}")
        if item.frequency <= 0:
            raise ValueError(f"frequency must be positive for word {item.word!r}")
        seen_words.add(item.word)
        tokenized.append(
            TokenizedWord(
                word=item.word,
                frequency=item.frequency,
                tokens=tuple(item.word) + (END_OF_WORD,),
            )
        )
    return tuple(tokenized)


def _vocabulary_key(token: str) -> tuple[bool, str]:
    return token == END_OF_WORD, token


def weighted_token_count(corpus: Sequence[TokenizedWord]) -> int:
    return sum(len(item.tokens) * item.frequency for item in corpus)


def _pair_frequencies(corpus: Sequence[TokenizedWord]) -> Counter[TokenPair]:
    counts: Counter[TokenPair] = Counter()
    for item in corpus:
        for left, right in zip(item.tokens, item.tokens[1:]):
            counts[(left, right)] += item.frequency
    return counts


def _replace_pair(
    tokens: tuple[str, ...], pair: TokenPair, merged_token: str
) -> tuple[str, ...]:
    replaced = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            replaced.append(merged_token)
            index += 2
        else:
            replaced.append(tokens[index])
            index += 1
    return tuple(replaced)


def train_bpe(
    corpus: Sequence[WordFrequency], n_merges: int, keep_trace: bool = False
) -> BPETraining:
    """Learn deterministic character-initialized BPE merge rules."""
    if n_merges < 0:
        raise ValueError("merge count must be non-negative")
    initial = _initial_corpus(corpus)
    initial_tokens = {token for item in initial for token in item.tokens}
    vocabulary = tuple(sorted(initial_tokens, key=_vocabulary_key))
    known_vocabulary = set(vocabulary)
    current = initial
    steps = []

    for rank in range(1, n_merges + 1):
        pair_counts = _pair_frequencies(current)
        if not pair_counts:
            break
        max_frequency = max(pair_counts.values())
        pair = min(
            candidate
            for candidate, frequency in pair_counts.items()
            if frequency == max_frequency
        )
        merged_token = "".join(pair)
        current = tuple(
            TokenizedWord(
                word=item.word,
                frequency=item.frequency,
                tokens=_replace_pair(item.tokens, pair, merged_token),
            )
            for item in current
        )
        if merged_token not in known_vocabulary:
            vocabulary += (merged_token,)
            known_vocabulary.add(merged_token)
        steps.append(
            MergeStep(
                rank=rank,
                pair=pair,
                token=merged_token,
                frequency=pair_counts[pair],
                vocabulary_size=len(vocabulary),
                corpus_token_count=weighted_token_count(current),
                corpus=current if keep_trace else None,
            )
        )

    return BPETraining(
        initial_vocabulary=tuple(sorted(initial_tokens, key=_vocabulary_key)),
        final_vocabulary=vocabulary,
        initial_corpus=initial,
        final_corpus=current,
        steps=tuple(steps),
    )


def tokenize_word(
    word: str,
    initial_vocabulary: Sequence[str],
    merge_steps: Sequence[MergeStep],
) -> tuple[str, ...]:
    """Apply a prefix of learned merge rules to one word."""
    _validate_word(word)
    tokens = tuple(word) + (END_OF_WORD,)
    unknown = sorted(set(tokens).difference(initial_vocabulary))
    if unknown:
        raise ValueError(
            f"word {word!r} has unknown initial tokens: {', '.join(unknown)}"
        )
    for step in merge_steps:
        tokens = _replace_pair(tokens, step.pair, step.token)
    return tokens


def tokenize_samples(
    words: Sequence[str], training: BPETraining, merge_count: int | None = None
) -> tuple[SampleTokenization, ...]:
    actual_merge_count = len(training.steps)
    if merge_count is None:
        merge_count = actual_merge_count
    if not 0 <= merge_count <= actual_merge_count:
        raise ValueError("merge count must identify an available merge prefix")
    merge_steps = training.steps[:merge_count]
    return tuple(
        SampleTokenization(
            word=word,
            tokens=tokenize_word(word, training.initial_vocabulary, merge_steps),
        )
        for word in words
    )


def training_vocabulary_sizes(training: BPETraining) -> tuple[int, ...]:
    return (len(training.initial_vocabulary),) + tuple(
        step.vocabulary_size for step in training.steps
    )


def training_token_counts(training: BPETraining) -> tuple[int, ...]:
    return (weighted_token_count(training.initial_corpus),) + tuple(
        step.corpus_token_count for step in training.steps
    )


def _learned_token_frequencies(
    training: BPETraining, limit: int
) -> tuple[TokenFrequency, ...]:
    counts: Counter[str] = Counter()
    for item in training.final_corpus:
        for token in item.tokens:
            counts[token] += item.frequency
    learned_tokens = {step.token for step in training.steps}
    ordered = sorted(
        (
            TokenFrequency(token=token, frequency=counts[token])
            for token in learned_tokens
            if counts[token] > 0
        ),
        key=lambda item: (-item.frequency, item.token),
    )
    return tuple(ordered[:limit])


def _unseen_word_examples(
    training: BPETraining, heldout_words: Sequence[str], limit: int
) -> tuple[UnseenWordExample, ...]:
    frequencies = Counter(heldout_words)
    training_words = {item.word for item in training.initial_corpus}
    candidates = sorted(
        (
            (word, frequency)
            for word, frequency in frequencies.items()
            if word not in training_words
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(
        UnseenWordExample(
            word=word,
            frequency=frequency,
            tokens=tokenize_word(word, training.initial_vocabulary, training.steps),
        )
        for word, frequency in candidates[:limit]
    )


def _validate_checkpoints(checkpoints: Sequence[int], actual_merge_count: int) -> None:
    if not checkpoints:
        raise ValueError("checkpoints must not be empty")
    if any(
        checkpoint < 0 or checkpoint > actual_merge_count for checkpoint in checkpoints
    ) or any(later <= earlier for earlier, later in zip(checkpoints, checkpoints[1:])):
        raise ValueError(
            "checkpoints must be unique, increasing, non-negative, and no "
            "greater than the actual merge count"
        )


def evaluate_bpe(
    training: BPETraining,
    heldout_words: Sequence[str],
    checkpoints: Sequence[int],
    sentence_words: Sequence[str],
    frequent_token_limit: int = 20,
    unseen_word_limit: int = 6,
) -> BPEEvaluation:
    """Evaluate saved merge prefixes on held-out words and one sentence."""
    if not heldout_words:
        raise ValueError("held-out words must not be empty")
    if frequent_token_limit < 0 or unseen_word_limit < 0:
        raise ValueError("result limits must be non-negative")
    _validate_checkpoints(checkpoints, len(training.steps))
    vocabulary_sizes = training_vocabulary_sizes(training)
    token_counts = training_token_counts(training)
    evaluations = []
    for merge_count in checkpoints:
        tokenized_heldout = tokenize_samples(heldout_words, training, merge_count)
        heldout_token_count = sum(len(sample.tokens) for sample in tokenized_heldout)
        evaluations.append(
            CheckpointEvaluation(
                merge_count=merge_count,
                vocabulary_size=vocabulary_sizes[merge_count],
                training_token_count=token_counts[merge_count],
                heldout_token_count=heldout_token_count,
                heldout_word_count=len(heldout_words),
                heldout_tokens_per_word=(heldout_token_count / len(heldout_words)),
                sentence=tokenize_samples(sentence_words, training, merge_count),
            )
        )
    return BPEEvaluation(
        checkpoints=tuple(evaluations),
        frequent_tokens=_learned_token_frequencies(training, frequent_token_limit),
        unseen_words=_unseen_word_examples(training, heldout_words, unseen_word_limit),
    )
