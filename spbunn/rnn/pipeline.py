"""Hydra-facing settings and the in-memory BPTT instability experiment."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import torch
from omegaconf import DictConfig

from spbunn.rnn.audit import (
    DtypeName,
    GradientAudit,
    Initialization,
    TaskKind,
    audit_gradients,
)
from spbunn.rnn.examples import (
    PredictionExamples,
    PredictionExampleSettings,
    make_adding_prediction_example,
    make_copy_prediction_example,
)
from spbunn.rnn.tasks import generate_adding_batch, generate_copy_batch

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class CopyAuditSettings:
    """Copy-memory task and initialization grid."""

    symbol_count: int
    memorize_length: int
    gaps: tuple[int, ...]
    initializations: tuple[Initialization, ...]
    seeds: tuple[int, ...]
    data_seed: int


@dataclass(frozen=True)
class AddingAuditSettings:
    """Adding-problem task and initialization grid."""

    lengths: tuple[int, ...]
    initialization: Initialization
    seeds: tuple[int, ...]
    data_seed: int


@dataclass(frozen=True)
class BPTTSettings:
    """Validated numerical settings independent of Hydra."""

    batch_size: int
    hidden_size: int
    figure_max_points: int
    primary_dtype: DtypeName
    verification_dtype: DtypeName | None
    copy: CopyAuditSettings
    adding: AddingAuditSettings
    examples: PredictionExampleSettings


@dataclass(frozen=True)
class BPTTExperimentInput:
    """Typed producer output for the recurrent gradient audit."""

    settings: BPTTSettings


@dataclass(frozen=True)
class BPTTExperimentResult:
    """Primary and matched-weight verification audit results."""

    settings: BPTTSettings
    primary_results: tuple[GradientAudit, ...]
    verification_results: tuple[GradientAudit, ...]
    prediction_examples: PredictionExamples


def produce_bptt_input(cfg: DictConfig) -> BPTTExperimentInput:
    """Resolve and validate all numerical audit settings."""
    batch_size = _positive_config_integer(cfg.batch_size, "The batch size")
    hidden_size = _positive_config_integer(cfg.hidden_size, "The hidden size")
    figure_max_points = _positive_config_integer(
        cfg.report.figure_max_points, "The figure point limit"
    )
    if figure_max_points < 2:
        raise ValueError("The figure point limit must be at least two")
    primary_dtype = _parse_dtype(cfg.primary_dtype, "The primary dtype")
    if primary_dtype != "float32":
        raise ValueError("The primary audit dtype must be float32")
    if not isinstance(cfg.verification.enabled, bool):
        raise ValueError("The verification.enabled setting must be a boolean")
    verification_enabled = cfg.verification.enabled
    verification_dtype = (
        _parse_dtype(cfg.verification.dtype, "The verification dtype")
        if verification_enabled
        else None
    )
    if verification_dtype not in {None, "float64"}:
        raise ValueError("The verification audit dtype must be float64")

    copy = CopyAuditSettings(
        symbol_count=_positive_config_integer(
            cfg.copy_task.symbol_count, "The copy symbol count"
        ),
        memorize_length=_positive_config_integer(
            cfg.copy_task.memorize_length, "The memorized-symbol count"
        ),
        gaps=_integer_sequence(cfg.copy_task.gaps, "The copy gaps", positive=True),
        initializations=_copy_initializations(cfg.copy_task.initializations),
        seeds=_integer_sequence(cfg.copy_task.seeds, "The copy seeds", positive=False),
        data_seed=_non_negative_config_integer(
            cfg.copy_task.data_seed, "The copy data seed"
        ),
    )
    if copy.symbol_count < 2:
        raise ValueError("The copy symbol count must be at least two")

    adding_initialization = str(cfg.adding_task.initialization)
    if adding_initialization != "xavier_tanh":
        raise ValueError("The adding task requires xavier_tanh initialization")
    adding = AddingAuditSettings(
        lengths=_integer_sequence(
            cfg.adding_task.lengths, "The adding lengths", positive=True
        ),
        initialization=cast(Initialization, adding_initialization),
        seeds=_integer_sequence(
            cfg.adding_task.seeds, "The adding seeds", positive=False
        ),
        data_seed=_non_negative_config_integer(
            cfg.adding_task.data_seed, "The adding data seed"
        ),
    )
    if any(length < 2 for length in adding.lengths):
        raise ValueError("Every adding-sequence length must be at least two")

    settings = BPTTSettings(
        batch_size=batch_size,
        hidden_size=hidden_size,
        figure_max_points=figure_max_points,
        primary_dtype=primary_dtype,
        verification_dtype=verification_dtype,
        copy=copy,
        adding=adding,
        examples=_example_settings(cfg),
    )
    return BPTTExperimentInput(settings=settings)


def evaluate_bptt_experiment(
    experiment: BPTTExperimentInput,
) -> BPTTExperimentResult:
    """Execute every configured case sequentially without model fitting."""
    settings = experiment.settings
    primary_results = []
    verification_results = []
    longest_copy_gap = max(settings.copy.gaps)
    for initialization in settings.copy.initializations:
        for seed in settings.copy.seeds:
            for gap in settings.copy.gaps:
                primary_results.append(
                    _audit_case(
                        task="copy",
                        initialization=initialization,
                        seed=seed,
                        task_length=gap,
                        dtype_name=settings.primary_dtype,
                        settings=settings,
                    )
                )
                if settings.verification_dtype is not None and gap == longest_copy_gap:
                    verification_results.append(
                        _audit_case(
                            task="copy",
                            initialization=initialization,
                            seed=seed,
                            task_length=gap,
                            dtype_name=settings.verification_dtype,
                            settings=settings,
                        )
                    )

    longest_adding_length = max(settings.adding.lengths)
    for seed in settings.adding.seeds:
        for length in settings.adding.lengths:
            primary_results.append(
                _audit_case(
                    task="adding",
                    initialization=settings.adding.initialization,
                    seed=seed,
                    task_length=length,
                    dtype_name=settings.primary_dtype,
                    settings=settings,
                )
            )
            if (
                settings.verification_dtype is not None
                and length == longest_adding_length
            ):
                verification_results.append(
                    _audit_case(
                        task="adding",
                        initialization=settings.adding.initialization,
                        seed=seed,
                        task_length=length,
                        dtype_name=settings.verification_dtype,
                        settings=settings,
                    )
                )

    return BPTTExperimentResult(
        settings=settings,
        primary_results=tuple(primary_results),
        verification_results=tuple(verification_results),
        prediction_examples=PredictionExamples(
            copy=make_copy_prediction_example(
                symbol_count=settings.copy.symbol_count,
                memorize_length=settings.examples.copy_memorize_length,
                gap=settings.examples.copy_gap,
                hidden_size=settings.hidden_size,
                model_seed=settings.examples.model_seed,
                data_seed=settings.examples.data_seed,
            ),
            adding=make_adding_prediction_example(
                length=settings.examples.adding_length,
                hidden_size=settings.hidden_size,
                model_seed=settings.examples.model_seed,
                data_seed=settings.examples.data_seed,
            ),
        ),
    )


def _audit_case(
    *,
    task: TaskKind,
    initialization: Initialization,
    seed: int,
    task_length: int,
    dtype_name: DtypeName,
    settings: BPTTSettings,
) -> GradientAudit:
    dtype = _DTYPES[dtype_name]
    if task == "copy":
        batch = generate_copy_batch(
            symbol_count=settings.copy.symbol_count,
            memorize_length=settings.copy.memorize_length,
            gap=task_length,
            batch_size=settings.batch_size,
            seed=settings.copy.data_seed + task_length,
            dtype=dtype,
        )
    else:
        batch = generate_adding_batch(
            length=task_length,
            batch_size=settings.batch_size,
            seed=settings.adding.data_seed + task_length,
            dtype=dtype,
        )
    return audit_gradients(
        task=task,
        batch=batch,
        initialization=initialization,
        seed=seed,
        task_length=task_length,
        hidden_size=settings.hidden_size,
        dtype=dtype,
    )


def _example_settings(cfg: DictConfig) -> PredictionExampleSettings:
    settings = PredictionExampleSettings(
        copy_memorize_length=_positive_config_integer(
            cfg.examples.copy_memorize_length, "The example copy length"
        ),
        copy_gap=_non_negative_config_integer(
            cfg.examples.copy_gap, "The example copy gap"
        ),
        adding_length=_positive_config_integer(
            cfg.examples.adding_length, "The example adding length"
        ),
        model_seed=_non_negative_config_integer(
            cfg.examples.model_seed, "The example model seed"
        ),
        data_seed=_non_negative_config_integer(
            cfg.examples.data_seed, "The example data seed"
        ),
        animation_interval_ms=_positive_config_integer(
            cfg.report.animation_interval_ms, "The animation interval"
        ),
    )
    if settings.adding_length < 2:
        raise ValueError("The example adding length must be at least two")
    return settings


def _copy_initializations(values: Any) -> tuple[Initialization, ...]:
    initializations = tuple(str(value) for value in values)
    if not initializations:
        raise ValueError("The copy initialization sequence must not be empty")
    if len(set(initializations)) != len(initializations):
        raise ValueError("The copy initializations must be unique")
    if any(value not in {"default", "orthogonal"} for value in initializations):
        raise ValueError("Copy initializations must be default or orthogonal")
    return cast(tuple[Initialization, ...], initializations)


def _integer_sequence(values: Any, name: str, *, positive: bool) -> tuple[int, ...]:
    result = tuple(_finite_integer(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    if any(value < 0 for value in result) or (
        positive and any(value <= 0 for value in result)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must contain {qualifier} integers")
    if tuple(sorted(result)) != result:
        raise ValueError(f"{name} must be sorted in increasing order")
    return result


def _positive_config_integer(value: Any, name: str) -> int:
    result = _finite_integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_config_integer(value: Any, name: str) -> int:
    result = _finite_integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_integer(value: Any, name: str) -> int:
    error = ValueError(f"{name} must contain only finite integers")
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


def _parse_dtype(value: Any, name: str) -> DtypeName:
    dtype = str(value)
    if dtype not in _DTYPES:
        raise ValueError(f"{name} must be float32 or float64")
    return cast(DtypeName, dtype)
