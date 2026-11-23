"""Hydra-facing settings and the fixed-gap gradient-flow experiment."""

import copy
import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from omegaconf import DictConfig

from spbunn.rnn.copy_training import (
    CopyEvaluationSettings,
    CopyTaskSettings,
    CopyTrainingSettings,
    TrainingPoint,
    ValidationMetrics,
    evaluate_copy_model,
    make_copy_batch,
    train_copy_model,
)
from spbunn.rnn.gradient_audit import (
    LSTMGradientAudit,
    RNNGradientAudit,
    audit_lstm_gradients,
    audit_rnn_gradients,
)
from spbunn.rnn.models import (
    CopyLSTMModel,
    CopyRNNModel,
    ModelInitialization,
    ModelKind,
    build_copy_model,
    parameter_count,
)
from spbunn.rnn.tasks import TaskBatch

AnyGradientAudit = RNNGradientAudit | LSTMGradientAudit
ARMS = (
    ("rnn", "default"),
    ("rnn", "orthogonal"),
    ("lstm", "default"),
)


@dataclass(frozen=True)
class GradientFlowSettings:
    """Validated numerical settings independent of Hydra."""

    batch_size: int
    audit_batch_size: int
    hidden_size: int
    model_seed: int
    audit_data_seed: int
    figure_max_points: int
    symbol_count: int
    memorize_length: int
    gaps: tuple[int, ...]
    training: CopyTrainingSettings
    evaluation: CopyEvaluationSettings
    verification_dtype: str | None


@dataclass(frozen=True)
class GradientFlowExperimentInput:
    """Typed producer output."""

    settings: GradientFlowSettings


@dataclass(frozen=True)
class GradientFlowCaseResult:
    """Training and audit evidence for one arm at one fixed gap."""

    gap: int
    kind: ModelKind
    initialization: ModelInitialization
    parameter_count: int
    initial_validation: ValidationMetrics
    trained_validation: ValidationMetrics
    history: tuple[TrainingPoint, ...]
    initial_audit: AnyGradientAudit
    trained_audit: AnyGradientAudit
    initial_verification: AnyGradientAudit | None
    trained_verification: AnyGradientAudit | None


@dataclass(frozen=True)
class GradientFlowExperimentResult:
    """Complete detached numerical result."""

    settings: GradientFlowSettings
    cases: tuple[GradientFlowCaseResult, ...]


def produce_gradient_flow_input(cfg: DictConfig) -> GradientFlowExperimentInput:
    """Resolve and validate every numerical experiment setting."""
    updates = _positive_config_integer(cfg.training_updates, "The training updates")
    history_interval = _positive_config_integer(
        cfg.history_interval, "The history interval"
    )
    if history_interval > updates:
        raise ValueError("The history interval must not exceed the training updates")
    gaps = _integer_sequence(cfg.copy_task.gaps, "The copy gaps", positive=True)
    symbol_count = _positive_config_integer(
        cfg.copy_task.symbol_count, "The copy symbol count"
    )
    if symbol_count < 2:
        raise ValueError("The copy symbol count must be at least two")
    memorize_length = _positive_config_integer(
        cfg.copy_task.memorize_length, "The memorized-symbol count"
    )
    verification_enabled = cfg.verification.enabled
    if not isinstance(verification_enabled, bool):
        raise ValueError("The verification.enabled setting must be a boolean")
    verification_dtype = (
        _verification_dtype(cfg.verification.dtype) if verification_enabled else None
    )
    training = CopyTrainingSettings(
        updates=updates,
        learning_rate=_positive_config_float(cfg.learning_rate, "The learning rate"),
        batch_size=_positive_config_integer(cfg.batch_size, "The training batch size"),
        data_seed=_non_negative_config_integer(
            cfg.training_data_seed, "The training data seed"
        ),
        history_interval=history_interval,
    )
    evaluation = CopyEvaluationSettings(
        batch_size=_positive_config_integer(
            cfg.validation_batch_size, "The validation batch size"
        ),
        batches=_positive_config_integer(
            cfg.validation_batches, "The validation batch count"
        ),
        data_seed=_non_negative_config_integer(
            cfg.validation_data_seed, "The validation data seed"
        ),
    )
    settings = GradientFlowSettings(
        batch_size=training.batch_size,
        audit_batch_size=_positive_config_integer(
            cfg.audit_batch_size, "The audit batch size"
        ),
        hidden_size=_positive_config_integer(cfg.hidden_size, "The hidden size"),
        model_seed=_non_negative_config_integer(cfg.model_seed, "The model seed"),
        audit_data_seed=_non_negative_config_integer(
            cfg.audit_data_seed, "The audit data seed"
        ),
        figure_max_points=_positive_config_integer(
            cfg.report.figure_max_points, "The figure point limit"
        ),
        symbol_count=symbol_count,
        memorize_length=memorize_length,
        gaps=gaps,
        training=training,
        evaluation=evaluation,
        verification_dtype=verification_dtype,
    )
    if settings.figure_max_points < 2:
        raise ValueError("The figure point limit must be at least two")
    return GradientFlowExperimentInput(settings=settings)


def evaluate_gradient_flow_experiment(
    experiment: GradientFlowExperimentInput,
) -> GradientFlowExperimentResult:
    """Train and audit each fixed-gap arm sequentially."""
    settings = experiment.settings
    cases = []
    longest_gap = max(settings.gaps)
    for gap in settings.gaps:
        task = CopyTaskSettings(
            symbol_count=settings.symbol_count,
            memorize_length=settings.memorize_length,
            gap=gap,
        )
        audit_batch = make_copy_batch(
            task,
            batch_size=settings.audit_batch_size,
            seed=settings.audit_data_seed,
            dtype=torch.float32,
        )
        for kind_value, initialization_value in ARMS:
            kind = cast(ModelKind, kind_value)
            initialization = cast(ModelInitialization, initialization_value)
            model = build_copy_model(
                kind=kind,
                initialization=initialization,
                seed=settings.model_seed,
                hidden_size=settings.hidden_size,
                input_size=settings.symbol_count + 2,
            )
            initial_validation = evaluate_copy_model(model, task, settings.evaluation)
            initial_audit = _audit_model(model, audit_batch, task)
            initial_verification = (
                _matched_float64_audit(model, audit_batch, task)
                if settings.verification_dtype is not None and gap == longest_gap
                else None
            )
            history = train_copy_model(
                model, task, settings.training, settings.evaluation
            )
            trained_validation = evaluate_copy_model(model, task, settings.evaluation)
            trained_audit = _audit_model(model, audit_batch, task)
            trained_verification = (
                _matched_float64_audit(model, audit_batch, task)
                if settings.verification_dtype is not None and gap == longest_gap
                else None
            )
            cases.append(
                GradientFlowCaseResult(
                    gap=gap,
                    kind=kind,
                    initialization=initialization,
                    parameter_count=parameter_count(model),
                    initial_validation=initial_validation,
                    trained_validation=trained_validation,
                    history=history,
                    initial_audit=initial_audit,
                    trained_audit=trained_audit,
                    initial_verification=initial_verification,
                    trained_verification=trained_verification,
                )
            )
    return GradientFlowExperimentResult(settings=settings, cases=tuple(cases))


def _audit_model(
    model: CopyRNNModel | CopyLSTMModel,
    batch: TaskBatch,
    task: CopyTaskSettings,
) -> AnyGradientAudit:
    if isinstance(model, CopyRNNModel):
        return audit_rnn_gradients(model, batch, task)
    return audit_lstm_gradients(model, batch, task)


def _matched_float64_audit(
    model: CopyRNNModel | CopyLSTMModel,
    batch: TaskBatch,
    task: CopyTaskSettings,
) -> AnyGradientAudit:
    converted = copy.deepcopy(model).to(dtype=torch.float64)
    converted_batch = TaskBatch(
        inputs=batch.inputs.to(torch.float64), targets=batch.targets
    )
    return _audit_model(converted, converted_batch, task)


def _verification_dtype(value: Any) -> str:
    dtype = str(value)
    if dtype != "float64":
        raise ValueError("The verification audit dtype must be float64")
    return dtype


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
    return tuple(result)


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


def _positive_config_float(value: Any, name: str) -> float:
    error = ValueError(f"{name} must be a finite positive number")
    if isinstance(value, bool):
        raise error
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise error from exc
    if not math.isfinite(result) or result <= 0:
        raise error
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
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise error
    return int(numeric)
