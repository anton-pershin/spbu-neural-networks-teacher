"""Copy-loss, deterministic training, and validation tests."""

import math

import pytest
import torch
from torch.nn import functional as F

from spbunn.rnn.copy_training import (
    CopyEvaluationSettings,
    CopyTaskSettings,
    CopyTrainingSettings,
    copy_cross_entropy,
    evaluate_copy_model,
    make_copy_batch,
    train_copy_model,
)
from spbunn.rnn.models import build_copy_model
from spbunn.rnn.tasks import copy_memoryless_loss


def test_copy_loss_uses_only_reproduction_positions() -> None:
    logits = torch.tensor([[[100.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]])
    targets = torch.tensor([[0, 1]])
    expected = (
        F.cross_entropy(logits[:, 1], targets[:, 0])
        + F.cross_entropy(logits[:, 2], targets[:, 1])
    ) / 2

    result = copy_cross_entropy(logits, targets)

    torch.testing.assert_close(result, expected)
    changed = logits.clone()
    changed[:, 0] = -1000
    torch.testing.assert_close(copy_cross_entropy(changed, targets), result)


@pytest.mark.parametrize(
    ("logits_shape", "targets_shape"),
    [((2, 3), (2, 2)), ((1, 3, 3), (1,)), ((1, 1, 3), (1, 2)), ((2, 3, 3), (1, 2))],
)
def test_copy_loss_rejects_invalid_shapes(
    logits_shape: tuple[int, ...], targets_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        copy_cross_entropy(torch.zeros(logits_shape), torch.zeros(targets_shape).long())


def test_evaluation_matches_independent_position_loop_and_restores_mode() -> None:
    task = CopyTaskSettings(symbol_count=8, memorize_length=3, gap=2)
    settings = CopyEvaluationSettings(batch_size=4, batches=2, data_seed=900)
    model = build_copy_model(
        kind="lstm",
        initialization="default",
        seed=42,
        hidden_size=5,
        input_size=10,
    )
    model.train()

    result = evaluate_copy_model(model, task, settings)

    assert model.training
    losses = []
    correct = 0
    count = 0
    with torch.no_grad():
        for batch_index in range(settings.batches):
            batch = make_copy_batch(
                task, batch_size=4, seed=900 + batch_index, dtype=torch.float32
            )
            logits = model(batch.inputs)[:, -3:]
            losses.append(
                [
                    float(
                        F.cross_entropy(logits[:, position], batch.targets[:, position])
                    )
                    for position in range(3)
                ]
            )
            correct += int((logits.argmax(dim=-1) == batch.targets).sum())
            count += batch.targets.numel()
    expected_positions = torch.tensor(losses).mean(dim=0)
    torch.testing.assert_close(
        torch.tensor(result.per_position_loss), expected_positions, rtol=1e-6, atol=1e-7
    )
    assert result.loss == pytest.approx(float(torch.tensor(losses).mean()), rel=1e-6)
    assert result.accuracy == pytest.approx(correct / count)
    assert copy_memoryless_loss(8) == math.log(8)
    model.eval()
    evaluate_copy_model(model, task, settings)
    assert not model.training


def test_tiny_training_is_deterministic_and_preserves_global_rng() -> None:
    task = CopyTaskSettings(symbol_count=8, memorize_length=2, gap=1)
    training = CopyTrainingSettings(
        updates=2,
        learning_rate=0.001,
        batch_size=4,
        data_seed=1000,
        history_interval=1,
    )
    evaluation = CopyEvaluationSettings(batch_size=4, batches=1, data_seed=2000)
    first = build_copy_model(
        kind="rnn", initialization="default", seed=42, hidden_size=4, input_size=10
    )
    second = build_copy_model(
        kind="rnn", initialization="default", seed=42, hidden_size=4, input_size=10
    )
    initial_parameters = [
        parameter.detach().clone() for parameter in first.parameters()
    ]
    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state()

    first_history = train_copy_model(first, task, training, evaluation)
    second_history = train_copy_model(second, task, training, evaluation)

    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert first_history == second_history
    assert [point.step for point in first_history] == [1, 2]
    assert all(len(point.validation.per_position_loss) == 2 for point in first_history)
    assert any(
        not torch.equal(initial, final.detach())
        for initial, final in zip(initial_parameters, first.parameters(), strict=True)
    )
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_training_rejects_nonfinite_loss() -> None:
    task = CopyTaskSettings(symbol_count=8, memorize_length=2, gap=1)
    model = build_copy_model(
        kind="rnn", initialization="default", seed=42, hidden_size=4, input_size=10
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    training = CopyTrainingSettings(
        updates=1,
        learning_rate=0.001,
        batch_size=2,
        data_seed=1000,
        history_interval=1,
    )
    evaluation = CopyEvaluationSettings(batch_size=2, batches=1, data_seed=2000)
    with pytest.raises(FloatingPointError, match="update 1"):
        train_copy_model(model, task, training, evaluation)
