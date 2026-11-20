"""Deterministic, typed RNN prediction examples for HTML presentation."""

from dataclasses import dataclass

import torch

from spbunn.rnn.audit import build_recurrent_model
from spbunn.rnn.tasks import generate_adding_batch, generate_copy_batch


@dataclass(frozen=True)
class PredictionExampleSettings:
    """Sizes, seeds, and display timing for both prediction examples."""

    copy_memorize_length: int
    copy_gap: int
    adding_length: int
    model_seed: int
    data_seed: int
    animation_interval_ms: int


@dataclass(frozen=True)
class CopyPredictionExample:
    """One complete copy input and every untrained RNN class prediction."""

    inputs: tuple[int, ...]
    predictions: tuple[int, ...]
    targets: tuple[int, ...]
    symbol_count: int
    gap: int
    model_seed: int


@dataclass(frozen=True)
class AddingPredictionExample:
    """One complete adding input and every untrained scalar prediction."""

    values: tuple[float, ...]
    markers: tuple[int, ...]
    predictions: tuple[float, ...]
    target: float
    model_seed: int


@dataclass(frozen=True)
class PredictionExamples:
    """The copy and adding examples shown by the report."""

    copy: CopyPredictionExample
    adding: AddingPredictionExample


def make_copy_prediction_example(
    *,
    symbol_count: int,
    memorize_length: int,
    gap: int,
    hidden_size: int,
    model_seed: int,
    data_seed: int,
) -> CopyPredictionExample:
    """Calculate all class predictions for one deterministic copy sequence."""
    batch = generate_copy_batch(
        symbol_count=symbol_count,
        memorize_length=memorize_length,
        gap=gap,
        batch_size=1,
        seed=data_seed,
        dtype=torch.float32,
    )
    model = build_recurrent_model(
        input_size=symbol_count + 2,
        output_size=symbol_count + 2,
        hidden_size=hidden_size,
        initialization="default",
        seed=model_seed,
        dtype=torch.float32,
    )
    with torch.no_grad():
        _, logits = model(batch.inputs)
        predictions = logits.argmax(dim=-1)[0].tolist()
    return CopyPredictionExample(
        inputs=tuple(int(value) for value in batch.inputs.argmax(dim=-1)[0].tolist()),
        predictions=tuple(int(value) for value in predictions),
        targets=tuple(int(value) for value in batch.targets[0].tolist()),
        symbol_count=symbol_count,
        gap=gap,
        model_seed=model_seed,
    )


def make_adding_prediction_example(
    *,
    length: int,
    hidden_size: int,
    model_seed: int,
    data_seed: int,
) -> AddingPredictionExample:
    """Calculate all scalar predictions for one deterministic adding sequence."""
    batch = generate_adding_batch(
        length=length,
        batch_size=1,
        seed=data_seed,
        dtype=torch.float32,
    )
    model = build_recurrent_model(
        input_size=2,
        output_size=1,
        hidden_size=hidden_size,
        initialization="xavier_tanh",
        seed=model_seed,
        dtype=torch.float32,
    )
    with torch.no_grad():
        _, logits = model(batch.inputs)
        predictions = logits[0, :, 0].tolist()
    return AddingPredictionExample(
        values=tuple(float(value) for value in batch.inputs[0, :, 0].tolist()),
        markers=tuple(int(value) for value in batch.inputs[0, :, 1].tolist()),
        predictions=tuple(float(value) for value in predictions),
        target=float(batch.targets[0]),
        model_seed=model_seed,
    )
