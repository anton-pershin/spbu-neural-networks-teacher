"""Independent full-gradient, direct-path, and gate-audit checks."""

from collections.abc import Iterator

import pytest
import torch

from spbunn.rnn.copy_training import (
    CopyTaskSettings,
    copy_cross_entropy,
    make_copy_batch,
)
from spbunn.rnn.gradient_audit import audit_lstm_gradients, audit_rnn_gradients
from spbunn.rnn.models import build_copy_model


@pytest.fixture
def single_thread_torch() -> Iterator[None]:
    """Use one intra-op thread for exact initial-model scientific checks."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_rnn_audit_matches_independent_autograd() -> None:
    task = CopyTaskSettings(symbol_count=2, memorize_length=2, gap=2)
    batch = make_copy_batch(task, batch_size=3, seed=77, dtype=torch.float32)
    model = build_copy_model(
        kind="rnn", initialization="default", seed=42, hidden_size=3, input_size=4
    )
    reference_model = build_copy_model(
        kind="rnn", initialization="default", seed=42, hidden_size=3, input_size=4
    )
    unroll = reference_model.unroll(batch.inputs, retain_states=False)
    loss = copy_cross_entropy(unroll.logits, batch.targets)
    gradients = torch.autograd.grad(loss, unroll.states)

    result = audit_rnn_gradients(model, batch, task)

    norms = torch.stack(
        [gradient.detach().to(torch.float64).norm(dim=-1) for gradient in gradients],
        dim=1,
    )
    reference = task.memorize_length + task.gap
    expected = (norms[:, : reference + 1] / norms[:, reference : reference + 1]).mean(
        dim=0
    )
    torch.testing.assert_close(
        torch.tensor(result.relative_state_gradients, dtype=torch.float64),
        expected.flip(0).to(torch.float64),
    )
    assert result.loss == pytest.approx(float(loss.detach()))
    assert result.valid_sequence_count == 3
    assert result.relative_state_gradients[0] == pytest.approx(1.0)


def test_lstm_full_direct_and_gate_audits_match_independent_references() -> None:
    task = CopyTaskSettings(symbol_count=2, memorize_length=2, gap=2)
    batch = make_copy_batch(task, batch_size=3, seed=77, dtype=torch.float32)
    model = build_copy_model(
        kind="lstm", initialization="default", seed=42, hidden_size=3, input_size=4
    )
    reference_model = build_copy_model(
        kind="lstm", initialization="default", seed=42, hidden_size=3, input_size=4
    )
    unroll = reference_model.unroll(batch.inputs, retain_states=False)
    loss = copy_cross_entropy(unroll.logits, batch.targets)
    gradients = torch.autograd.grad(loss, unroll.cell_states)

    result = audit_lstm_gradients(model, batch, task)

    reference = task.memorize_length + task.gap
    norms = torch.stack(
        [gradient.detach().to(torch.float64).norm(dim=-1) for gradient in gradients],
        dim=1,
    )
    expected_full = (
        norms[:, : reference + 1] / norms[:, reference : reference + 1]
    ).mean(dim=0)
    torch.testing.assert_close(
        torch.tensor(result.relative_cell_gradients, dtype=torch.float64),
        expected_full.flip(0).to(torch.float64),
    )
    anchor = gradients[reference].to(torch.float64)
    direct_time = []
    scalar_time = []
    for time_index in range(reference + 1):
        product = torch.ones_like(anchor)
        scalar = torch.ones((), dtype=torch.float64)
        for step in range(time_index + 1, reference + 1):
            product = product * unroll.forget_gates[step].detach().to(torch.float64)
            scalar = (
                scalar * unroll.forget_gates[step].detach().to(torch.float64).mean()
            )
        direct_time.append(
            ((anchor * product).norm(dim=-1) / anchor.norm(dim=-1)).mean()
        )
        scalar_time.append(scalar)
    torch.testing.assert_close(
        torch.tensor(result.direct_cell_path_gradients, dtype=torch.float64),
        torch.stack(direct_time).flip(0),
    )
    torch.testing.assert_close(
        torch.tensor(result.mean_gate_products, dtype=torch.float64),
        torch.stack(scalar_time).flip(0),
    )
    expected_mean = torch.stack(unroll.forget_gates).detach().mean(dim=(1, 2))
    torch.testing.assert_close(
        torch.tensor(result.mean_forget_gates, dtype=torch.float64),
        expected_mean.to(torch.float64),
    )
    assert len(result.mean_forget_gates) == batch.inputs.shape[1]
    assert result.direct_cell_path_gradients[0] == pytest.approx(1.0)
    assert result.mean_gate_products[0] == pytest.approx(1.0)


def test_zero_reference_norm_is_preserved_without_a_plotting_floor() -> None:
    task = CopyTaskSettings(symbol_count=2, memorize_length=2, gap=1)
    batch = make_copy_batch(task, batch_size=2, seed=7, dtype=torch.float32)
    model = build_copy_model(
        kind="rnn", initialization="default", seed=42, hidden_size=3, input_size=4
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    result = audit_rnn_gradients(model, batch, task)

    assert result.valid_sequence_count == 0
    assert all(
        torch.isnan(torch.tensor(value)) for value in result.relative_state_gradients
    )
    assert result.zero_state_gradient_count == 2 * batch.inputs.shape[1]
    assert result.nonfinite_state_gradient_count == 0


def test_nonfinite_state_gradients_remain_visible() -> None:
    task = CopyTaskSettings(symbol_count=2, memorize_length=2, gap=1)
    batch = make_copy_batch(task, batch_size=2, seed=7, dtype=torch.float32)
    model = build_copy_model(
        kind="lstm", initialization="default", seed=42, hidden_size=3, input_size=4
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))

    result = audit_lstm_gradients(model, batch, task)

    assert result.valid_sequence_count == 0
    assert result.nonfinite_cell_gradient_count > 0
    assert all(
        torch.isnan(torch.tensor(value)) for value in result.relative_cell_gradients
    )


@pytest.mark.parametrize("kind", ["rnn", "lstm"])
def test_audit_rejects_inconsistent_gap_and_dtype(kind: str) -> None:
    task = CopyTaskSettings(symbol_count=2, memorize_length=2, gap=2)
    batch = make_copy_batch(task, batch_size=2, seed=7, dtype=torch.float32)
    mismatched = make_copy_batch(task, batch_size=2, seed=7, dtype=torch.float64)
    if kind == "rnn":
        rnn_model = build_copy_model(
            kind="rnn", initialization="default", seed=42, hidden_size=3, input_size=4
        )
        with pytest.raises(ValueError, match="gap does not match"):
            audit_rnn_gradients(rnn_model, batch, CopyTaskSettings(2, 2, 3))
        with pytest.raises(ValueError, match="dtype must match"):
            audit_rnn_gradients(rnn_model, mismatched, task)
    else:
        lstm_model = build_copy_model(
            kind="lstm", initialization="default", seed=42, hidden_size=3, input_size=4
        )
        with pytest.raises(ValueError, match="gap does not match"):
            audit_lstm_gradients(lstm_model, batch, CopyTaskSettings(2, 2, 3))
        with pytest.raises(ValueError, match="dtype must match"):
            audit_lstm_gradients(lstm_model, mismatched, task)


@pytest.mark.usefixtures("single_thread_torch")
def test_initial_production_scientific_regressions() -> None:
    task = CopyTaskSettings(symbol_count=8, memorize_length=10, gap=20)
    batch = make_copy_batch(task, batch_size=128, seed=800000, dtype=torch.float32)
    default = build_copy_model(
        kind="rnn", initialization="default", seed=42, hidden_size=64, input_size=10
    )
    orthogonal = build_copy_model(
        kind="rnn",
        initialization="orthogonal",
        seed=42,
        hidden_size=64,
        input_size=10,
    )
    lstm = build_copy_model(
        kind="lstm", initialization="default", seed=42, hidden_size=64, input_size=10
    )

    default_result = audit_rnn_gradients(default, batch, task)
    orthogonal_result = audit_rnn_gradients(orthogonal, batch, task)
    lstm_result = audit_lstm_gradients(lstm, batch, task)

    assert default_result.relative_state_gradients[20] == pytest.approx(
        9.58477756e-6, rel=1e-5
    )
    assert orthogonal_result.relative_state_gradients[20] == pytest.approx(
        0.101942082, rel=1e-5
    )
    assert lstm_result.relative_cell_gradients[20] == pytest.approx(
        1.11725485e-4, rel=1e-5
    )
    assert lstm_result.direct_cell_path_gradients[20] == pytest.approx(
        1.65867857e-6, rel=1e-5
    )
    assert lstm_result.mean_gate_products[20] == pytest.approx(8.98702044e-7, rel=1e-5)
    mean_gap = torch.tensor(lstm_result.mean_forget_gates[11:31]).mean()
    assert float(mean_gap) == pytest.approx(0.49851799, rel=1e-6)


@pytest.mark.usefixtures("single_thread_torch")
def test_initial_long_lag_scalar_reference_and_vector_norms() -> None:
    task = CopyTaskSettings(symbol_count=8, memorize_length=10, gap=200)
    batch = make_copy_batch(task, batch_size=128, seed=800000, dtype=torch.float32)
    model = build_copy_model(
        kind="lstm", initialization="default", seed=42, hidden_size=64, input_size=10
    )

    result = audit_lstm_gradients(model, batch, task)

    assert result.relative_cell_gradients[200] == pytest.approx(
        1.27228274e-35, rel=1e-5
    )
    assert result.direct_cell_path_gradients[200] == pytest.approx(
        9.3560724e-54, rel=1e-5
    )
    assert result.mean_gate_products[200] == pytest.approx(3.58894962e-61, rel=1e-5)
    assert result.mean_gate_products[200] == pytest.approx(0.5**200, rel=0.43)
