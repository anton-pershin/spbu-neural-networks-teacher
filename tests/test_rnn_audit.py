"""Initialization and independent retained-adjoint audit checks."""

import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from spbunn.rnn.audit import (
    _state_gradient_norms,
    audit_gradients,
    build_recurrent_model,
)
from spbunn.rnn.tasks import generate_adding_batch, generate_copy_batch


def test_orthogonal_xavier_and_matched_dtype_initialization() -> None:
    orthogonal = build_recurrent_model(
        input_size=2,
        output_size=1,
        hidden_size=16,
        initialization="orthogonal",
        seed=42,
        dtype=torch.float32,
    )
    xavier32 = build_recurrent_model(
        input_size=2,
        output_size=1,
        hidden_size=16,
        initialization="xavier_tanh",
        seed=42,
        dtype=torch.float32,
    )
    xavier64 = build_recurrent_model(
        input_size=2,
        output_size=1,
        hidden_size=16,
        initialization="xavier_tanh",
        seed=42,
        dtype=torch.float64,
    )

    weights = orthogonal.cell.weight_hh
    identity = weights.T @ weights
    assert torch.allclose(identity, torch.eye(16), atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(xavier32.cell.bias_ih) == 0
    assert torch.count_nonzero(xavier32.cell.bias_hh) == 0
    assert xavier64.cell.weight_ih.dtype == torch.float64
    assert torch.equal(xavier64.cell.weight_ih, xavier32.cell.weight_ih.double())
    assert torch.equal(xavier64.cell.weight_hh, xavier32.cell.weight_hh.double())
    assert torch.equal(xavier64.readout.weight, xavier32.readout.weight.double())


def test_model_construction_restores_global_random_state() -> None:
    torch.manual_seed(991)
    expected_state = torch.random.get_rng_state().clone()

    build_recurrent_model(
        input_size=2,
        output_size=1,
        hidden_size=8,
        initialization="xavier_tanh",
        seed=42,
        dtype=torch.float32,
    )

    assert torch.equal(torch.random.get_rng_state(), expected_state)


def test_explicit_states_expose_hand_calculated_recurrent_adjoint() -> None:
    model = build_recurrent_model(
        input_size=2,
        output_size=1,
        hidden_size=1,
        initialization="default",
        seed=0,
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.cell.weight_ih.zero_()
        model.cell.weight_hh.fill_(0.2)
        model.cell.bias_ih.fill_(0.1)
        model.cell.bias_hh.zero_()
        model.readout.weight.fill_(1.0)
        model.readout.bias.zero_()
    inputs = torch.zeros(1, 2, 2, dtype=torch.float64)

    states, logits = model(inputs)
    loss = logits[0, -1, 0]
    loss.backward()

    first_value = math.tanh(0.1)
    final_value = math.tanh(0.1 + 0.2 * first_value)
    assert states[0].item() == pytest.approx(first_value)
    assert states[1].item() == pytest.approx(final_value)
    assert states[1].grad is not None
    assert states[0].grad is not None
    assert states[1].grad.item() == pytest.approx(1.0)
    assert states[0].grad.item() == pytest.approx(0.2 * (1 - final_value**2))


def test_public_audit_matches_independent_autograd_adjoints() -> None:
    batch = generate_adding_batch(length=3, batch_size=2, seed=8, dtype=torch.float64)
    result = audit_gradients(
        task="adding",
        batch=batch,
        initialization="default",
        seed=7,
        task_length=3,
        hidden_size=4,
        dtype=torch.float64,
    )
    model = build_recurrent_model(
        input_size=2,
        output_size=1,
        hidden_size=4,
        initialization="default",
        seed=7,
        dtype=torch.float64,
    )
    states, logits = model(batch.inputs)
    loss = nn.functional.mse_loss(logits[:, -1, 0], batch.targets)
    independent = torch.autograd.grad(
        loss, tuple(states) + tuple(model.parameters()), allow_unused=False
    )
    state_gradients = independent[: len(states)]
    parameter_gradients = independent[len(states) :]
    norms = torch.stack([gradient.norm(dim=-1) for gradient in state_gradients], dim=1)
    expected_ratios = (norms / norms[:, -1:]).mean(dim=0)
    expected_parameter_norm = math.sqrt(
        sum(float(gradient.pow(2).sum()) for gradient in parameter_gradients)
    )

    assert result.relative_state_gradients == pytest.approx(
        tuple(float(value) for value in expected_ratios)
    )
    assert result.relative_state_gradients[-1] == pytest.approx(1.0)
    assert result.loss == pytest.approx(float(loss.detach()))
    assert result.maximum_hidden_state == pytest.approx(
        max(float(state.detach().abs().max()) for state in states)
    )
    assert result.parameter_gradient_norm == pytest.approx(expected_parameter_norm)
    assert result.valid_sequence_count == 2
    assert result.zero_state_gradient_count == 0
    assert result.nonfinite_state_gradient_count == 0
    assert result.forward_finite and result.backward_finite


def test_zero_final_adjoint_is_preserved_and_marked_invalid() -> None:
    batch = generate_adding_batch(length=2, batch_size=2, seed=8, dtype=torch.float32)
    batch.targets.fill_(1.0)
    model = build_recurrent_model(
        input_size=2,
        output_size=1,
        hidden_size=2,
        initialization="default",
        seed=7,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    with patch("spbunn.rnn.audit.build_recurrent_model", return_value=model):
        result = audit_gradients(
            task="adding",
            batch=batch,
            initialization="default",
            seed=7,
            task_length=2,
            hidden_size=2,
            dtype=torch.float32,
        )

    assert result.valid_sequence_count == 0
    assert result.zero_state_gradient_count == 4
    assert all(math.isnan(value) for value in result.relative_state_gradients)
    assert result.nonfinite_state_gradient_count == 0
    assert result.forward_finite and result.backward_finite


def test_nonfinite_forward_and_backward_values_are_retained() -> None:
    batch = generate_adding_batch(length=2, batch_size=2, seed=8, dtype=torch.float32)
    model = build_recurrent_model(
        input_size=2,
        output_size=1,
        hidden_size=2,
        initialization="default",
        seed=7,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.readout.weight.fill_(float("inf"))

    with patch("spbunn.rnn.audit.build_recurrent_model", return_value=model):
        result = audit_gradients(
            task="adding",
            batch=batch,
            initialization="default",
            seed=7,
            task_length=2,
            hidden_size=2,
            dtype=torch.float32,
        )

    assert not result.forward_finite
    assert not result.backward_finite
    assert result.nonfinite_state_gradient_count > 0
    assert not math.isfinite(result.parameter_gradient_norm)


def test_missing_retained_gradient_is_an_explicit_error() -> None:
    state = torch.ones(1, 2, requires_grad=True)
    with pytest.raises(RuntimeError, match="Missing retained"):
        _state_gradient_norms((state,))


def test_task_length_must_match_batch_time_dimension() -> None:
    adding_batch = generate_adding_batch(
        length=2, batch_size=2, seed=8, dtype=torch.float32
    )
    with pytest.raises(ValueError, match="inconsistent with the batch time"):
        audit_gradients(
            task="adding",
            batch=adding_batch,
            initialization="default",
            seed=7,
            task_length=3,
            hidden_size=2,
            dtype=torch.float32,
        )

    copy_batch = generate_copy_batch(
        symbol_count=8,
        memorize_length=3,
        gap=5,
        batch_size=2,
        seed=8,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="inconsistent with the batch time"):
        audit_gradients(
            task="copy",
            batch=copy_batch,
            initialization="default",
            seed=7,
            task_length=6,
            hidden_size=2,
            dtype=torch.float32,
        )


def test_invalid_model_and_audit_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported recurrent initialization"):
        build_recurrent_model(
            input_size=2,
            output_size=1,
            hidden_size=8,
            initialization="selected",  # type: ignore[arg-type]
            seed=0,
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="model seed"):
        build_recurrent_model(
            input_size=2,
            output_size=1,
            hidden_size=8,
            initialization="default",
            seed=-1,
            dtype=torch.float32,
        )
    batch = generate_adding_batch(length=2, batch_size=2, seed=8, dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype must match"):
        audit_gradients(
            task="adding",
            batch=batch,
            initialization="default",
            seed=7,
            task_length=2,
            hidden_size=2,
            dtype=torch.float64,
        )
