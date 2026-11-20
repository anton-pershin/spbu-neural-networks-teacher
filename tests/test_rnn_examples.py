"""Deterministic prediction examples used by the report and animations."""

import math

import torch

from spbunn.rnn.examples import (
    make_adding_prediction_example,
    make_copy_prediction_example,
)


def test_copy_prediction_example_has_complete_input_output_and_target() -> None:
    example = make_copy_prediction_example(
        symbol_count=8,
        memorize_length=3,
        gap=2,
        hidden_size=8,
        model_seed=42,
        data_seed=2026,
    )

    assert len(example.inputs) == 9
    assert len(example.predictions) == 9
    assert len(example.targets) == 3
    assert example.inputs[:3] == example.targets
    assert example.inputs[3] == 9
    assert example.inputs[4:6] == (8, 8)
    assert example.inputs[6:] == (8, 8, 8)
    assert all(0 <= prediction < 10 for prediction in example.predictions)
    assert example.model_seed == 42


def test_adding_prediction_example_has_complete_input_output_and_target() -> None:
    example = make_adding_prediction_example(
        length=6, hidden_size=8, model_seed=42, data_seed=2026
    )

    assert len(example.values) == 6
    assert len(example.markers) == 6
    assert len(example.predictions) == 6
    assert sum(example.markers) == 2
    expected = sum(
        value
        for value, marker in zip(example.values, example.markers, strict=True)
        if marker
    )
    assert example.target == expected
    assert all(math.isfinite(prediction) for prediction in example.predictions)
    assert example.model_seed == 42


def test_prediction_examples_are_deterministic_and_preserve_global_rng() -> None:
    torch.manual_seed(991)
    state = torch.random.get_rng_state().clone()

    first_copy = make_copy_prediction_example(
        symbol_count=8,
        memorize_length=3,
        gap=2,
        hidden_size=8,
        model_seed=42,
        data_seed=2026,
    )
    second_copy = make_copy_prediction_example(
        symbol_count=8,
        memorize_length=3,
        gap=2,
        hidden_size=8,
        model_seed=42,
        data_seed=2026,
    )
    first_adding = make_adding_prediction_example(
        length=6, hidden_size=8, model_seed=42, data_seed=2026
    )
    second_adding = make_adding_prediction_example(
        length=6, hidden_size=8, model_seed=42, data_seed=2026
    )

    assert first_copy == second_copy
    assert first_adding == second_adding
    assert torch.equal(torch.random.get_rng_state(), state)
