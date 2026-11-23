"""Packed/explicit recurrence and initialization tests for gradient flow."""

import pytest
import torch

from spbunn.rnn.copy_training import CopyTaskSettings, make_copy_batch
from spbunn.rnn.models import (
    CopyLSTMModel,
    CopyRNNModel,
    build_copy_model,
    parameter_count,
)


def test_rnn_packed_and_explicit_recurrences_match() -> None:
    task = CopyTaskSettings(symbol_count=8, memorize_length=3, gap=2)
    batch = make_copy_batch(task, batch_size=3, seed=7, dtype=torch.float32)
    model = build_copy_model(
        kind="rnn",
        initialization="default",
        seed=42,
        hidden_size=5,
        input_size=10,
    )

    packed = model(batch.inputs)
    unroll = model.unroll(batch.inputs, retain_states=False)

    torch.testing.assert_close(unroll.logits, packed, rtol=1e-5, atol=1e-6)
    assert len(unroll.states) == batch.inputs.shape[1]


def test_lstm_packed_and_explicit_recurrences_match() -> None:
    task = CopyTaskSettings(symbol_count=8, memorize_length=3, gap=2)
    batch = make_copy_batch(task, batch_size=3, seed=7, dtype=torch.float32)
    model = build_copy_model(
        kind="lstm",
        initialization="default",
        seed=42,
        hidden_size=5,
        input_size=10,
    )

    packed = model(batch.inputs)
    unroll = model.unroll(batch.inputs, retain_states=False)

    torch.testing.assert_close(unroll.logits, packed, rtol=1e-5, atol=1e-6)
    assert len(unroll.cell_states) == batch.inputs.shape[1]
    assert len(unroll.forget_gates) == batch.inputs.shape[1]
    for gate in unroll.forget_gates:
        assert bool(((gate >= 0) & (gate <= 1)).all())
    hidden_size = model.recurrent.hidden_size
    assert torch.count_nonzero(model.bias_ih[hidden_size : 2 * hidden_size]) == 0
    assert torch.count_nonzero(model.bias_hh[hidden_size : 2 * hidden_size]) == 0


def test_orthogonal_initialization_and_global_rng_restoration() -> None:
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()

    model = build_copy_model(
        kind="rnn",
        initialization="orthogonal",
        seed=42,
        hidden_size=8,
        input_size=10,
    )

    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    weights = model.weight_hh
    torch.testing.assert_close(weights @ weights.T, torch.eye(8), rtol=1e-5, atol=1e-6)


def test_lstm_rejects_orthogonal_nonforget_initialization() -> None:
    with pytest.raises(ValueError, match="LSTM arm requires default"):
        build_copy_model(
            kind="lstm",
            initialization="orthogonal",
            seed=42,
            hidden_size=4,
            input_size=10,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "gru"}, "Unsupported copy-model kind"),
        ({"initialization": "xavier"}, "Unsupported copy-model initialization"),
        ({"hidden_size": 0}, "hidden size"),
        ({"input_size": -1}, "input size"),
        ({"seed": -1}, "model seed"),
        ({"dtype": torch.float16}, "dtype"),
    ],
)
def test_invalid_model_settings_are_rejected(overrides: dict, message: str) -> None:
    settings = {
        "kind": "rnn",
        "initialization": "default",
        "seed": 42,
        "hidden_size": 4,
        "input_size": 10,
    }
    settings.update(overrides)
    with pytest.raises(ValueError, match=message):
        build_copy_model(**settings)  # type: ignore[call-overload]


def test_unroll_rejects_invalid_input_shapes() -> None:
    model = build_copy_model(
        kind="rnn",
        initialization="default",
        seed=42,
        hidden_size=4,
        input_size=10,
    )
    with pytest.raises(ValueError, match="batch, time, and feature"):
        model.unroll(torch.zeros(2, 10))
    with pytest.raises(ValueError, match="feature dimension"):
        model.unroll(torch.zeros(2, 3, 9))


def test_production_parameter_counts() -> None:
    rnn = build_copy_model(
        kind="rnn",
        initialization="default",
        seed=42,
        hidden_size=64,
        input_size=10,
    )
    lstm = build_copy_model(
        kind="lstm",
        initialization="default",
        seed=42,
        hidden_size=64,
        input_size=10,
    )
    assert isinstance(rnn, CopyRNNModel)
    assert isinstance(lstm, CopyLSTMModel)
    assert parameter_count(rnn) == 5514
    assert parameter_count(lstm) == 20106
