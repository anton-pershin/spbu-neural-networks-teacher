"""Shared fixtures and configuration helpers for n-gram tests."""

from collections.abc import Sequence
from functools import lru_cache

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from spbunn.ngram.pipeline import (
    GenerationContext,
    NGramExperimentInput,
    NGramExperimentResult,
    NGramSettings,
    ProbabilityQuery,
    build_training_vocabulary,
    evaluate_ngram_experiment,
    produce_ngram_input,
)
from spbunn.utils.common import get_config_path


def compose_ngram_config(
    config_name: str = "config_test_ngram", overrides: Sequence[str] = ()
) -> DictConfig:
    """Compose one n-gram configuration from the real config directory."""
    with initialize_config_dir(
        config_dir=str(get_config_path()),
        job_name="ngram_test",
        version_base="1.3",
    ):
        return compose(config_name=config_name, overrides=list(overrides))


@lru_cache(maxsize=1)
def production_ngram_result() -> NGramExperimentResult:
    """Evaluate the real example once for independent regression checks."""
    cfg = compose_ngram_config("config_ngram")
    return evaluate_ngram_experiment(produce_ngram_input(cfg))


def tiny_experiment() -> NGramExperimentInput:
    """Return a small model input with hand-checkable counts."""
    training = (
        ("the", "cat", "sat"),
        ("the", "dog", "sat"),
        ("the", "cat"),
    )
    settings = NGramSettings(
        split_seed=0,
        training_fraction=0.6,
        validation_fraction=0.2,
        minimum_token_count=1,
        orders=(1, 2, 3),
        alphas=(0.1, 1.0),
        probability_queries=(
            ProbabilityQuery(order=1, context=()),
            ProbabilityQuery(order=2, context=("the",)),
            ProbabilityQuery(order=3, context=("of", "the")),
        ),
        top_k=3,
        generation_seeds=(0,),
        generation_contexts=(
            GenerationContext(order=1, context=()),
            GenerationContext(order=2, context=("the",)),
            GenerationContext(order=3, context=("the", "cat")),
        ),
        maximum_generated_tokens=4,
    )
    return NGramExperimentInput(
        source_title="Tiny test corpus",
        source_resource="in-memory",
        source_sha256="0" * 64,
        training_sentences=training,
        validation_sentences=(("the", "cat", "sat"),),
        test_sentences=(("the", "dog"),),
        known_tokens=build_training_vocabulary(training, 1),
        settings=settings,
    )
