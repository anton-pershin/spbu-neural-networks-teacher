"""Deterministic copy-task optimization and per-position evaluation."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from spbunn.rnn.models import CopyLSTMModel, CopyRNNModel
from spbunn.rnn.tasks import TaskBatch, generate_copy_batch

CopyModel = CopyRNNModel | CopyLSTMModel


@dataclass(frozen=True)
class CopyTaskSettings:
    """One fixed copy-memory task geometry."""

    symbol_count: int
    memorize_length: int
    gap: int


@dataclass(frozen=True)
class CopyTrainingSettings:
    """Optimization and training-data settings shared by all arms."""

    updates: int
    learning_rate: float
    batch_size: int
    data_seed: int
    history_interval: int


@dataclass(frozen=True)
class CopyEvaluationSettings:
    """Fixed validation-data settings."""

    batch_size: int
    batches: int
    data_seed: int


@dataclass(frozen=True)
class ValidationMetrics:
    """Overall and per-position copy-task validation metrics."""

    loss: float
    accuracy: float
    per_position_loss: tuple[float, ...]


@dataclass(frozen=True)
class TrainingPoint:
    """One recorded optimization and validation point."""

    step: int
    training_loss: float
    validation: ValidationMetrics


def make_copy_batch(
    task: CopyTaskSettings, *, batch_size: int, seed: int, dtype: torch.dtype
) -> TaskBatch:
    """Generate one deterministic batch for a fixed-gap task."""
    return generate_copy_batch(
        symbol_count=task.symbol_count,
        memorize_length=task.memorize_length,
        gap=task.gap,
        batch_size=batch_size,
        seed=seed,
        dtype=dtype,
    )


def copy_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy over only the final reproduction positions."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("Copy logits and targets have inconsistent dimensions")
    target_length = targets.shape[1]
    if target_length <= 0 or logits.shape[1] < target_length:
        raise ValueError("The copy target length is incompatible with the logits")
    reproduction = logits[:, -target_length:]
    if reproduction.shape[:2] != targets.shape:
        raise ValueError("Copy logit and target batch dimensions do not match")
    return F.cross_entropy(
        reproduction.reshape(-1, reproduction.shape[-1]), targets.reshape(-1)
    )


@torch.no_grad()
def evaluate_copy_model(
    model: CopyModel,
    task: CopyTaskSettings,
    settings: CopyEvaluationSettings,
) -> ValidationMetrics:
    """Evaluate overall loss, accuracy, and every reproduction position."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    per_position = torch.zeros(task.memorize_length, dtype=torch.float64)
    correct = 0
    target_count = 0
    for batch_index in range(settings.batches):
        batch = make_copy_batch(
            task,
            batch_size=settings.batch_size,
            seed=settings.data_seed + batch_index,
            dtype=torch.float32,
        )
        logits = model(batch.inputs)
        total_loss += float(copy_cross_entropy(logits, batch.targets))
        for position in range(task.memorize_length):
            position_loss = F.cross_entropy(
                logits[:, -task.memorize_length + position], batch.targets[:, position]
            )
            per_position[position] += float(position_loss)
        predictions = logits[:, -task.memorize_length :].argmax(dim=-1)
        correct += int((predictions == batch.targets).sum())
        target_count += batch.targets.numel()
    if was_training:
        model.train()
    batches = float(settings.batches)
    return ValidationMetrics(
        loss=total_loss / batches,
        accuracy=correct / target_count,
        per_position_loss=tuple(float(value / batches) for value in per_position),
    )


def train_copy_model(
    model: CopyModel,
    task: CopyTaskSettings,
    training: CopyTrainingSettings,
    evaluation: CopyEvaluationSettings,
) -> tuple[TrainingPoint, ...]:
    """Train one model and record matched optimization/validation snapshots."""
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    history = []
    model.train()
    for step in range(1, training.updates + 1):
        batch = make_copy_batch(
            task,
            batch_size=training.batch_size,
            seed=training.data_seed + step,
            dtype=torch.float32,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = copy_cross_entropy(model(batch.inputs), batch.targets)
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError(f"Non-finite copy training loss at update {step}")
        loss.backward()
        optimizer.step()
        if (
            step == 1
            or step % training.history_interval == 0
            or step == training.updates
        ):
            history.append(
                TrainingPoint(
                    step=step,
                    training_loss=float(loss.detach()),
                    validation=evaluate_copy_model(model, task, evaluation),
                )
            )
    return tuple(history)
