import numpy as np
import pytest
from torch.nn import functional as F
from torch.utils.data import DataLoader

from spbunn.optimization.mini_batch import (
    evaluate_mini_batch,
    produce_mini_batch_inputs,
)
from tests.optimization_helpers import compose_optimization_config


def test_mini_batch_producer_preserves_seed_and_separate_intervals() -> None:
    cfg = compose_optimization_config("config_test_simple_loss_optimization")

    first = produce_mini_batch_inputs(cfg)[0]
    second = produce_mini_batch_inputs(cfg)[0]

    training_x = np.array(
        [float(first.training_dataset[index][0]) for index in range(16)]
    )
    repeated_x = np.array(
        [float(second.training_dataset[index][0]) for index in range(16)]
    )
    np.testing.assert_array_equal(training_x, repeated_x)
    expected_grid = np.linspace(cfg.training_x_min, cfg.training_x_max, 16)
    expected_permutation = np.random.default_rng(seed=42).choice(
        expected_grid, size=16, replace=False
    )
    np.testing.assert_array_equal(training_x, expected_permutation)
    np.testing.assert_allclose(np.sort(training_x), expected_grid)
    assert not np.array_equal(
        training_x, np.linspace(cfg.training_x_min, cfg.training_x_max, 16)
    )
    np.testing.assert_allclose(first.landscape.data.x.numpy(), np.linspace(6, 10, 16))


def test_mini_batch_evaluator_uses_exact_update_count_and_pre_update_loss() -> None:
    cfg = compose_optimization_config("config_test_simple_loss_optimization")
    experiment = produce_mini_batch_inputs(cfg)[0]

    result = evaluate_mini_batch(experiment)

    assert result.theta_history.shape == (4, 2)
    assert result.loss_history.shape == (3,)
    assert result.loss_values.shape == (5, 5)
    np.testing.assert_allclose(result.theta_history[0], [-2.0, 0.0])
    initial_prediction = experiment.landscape.func_factory(
        theta_0=experiment.landscape.theta_0_initial,
        theta_1=experiment.landscape.theta_1_initial,
    )
    x_batch, y_batch = next(
        iter(DataLoader(experiment.training_dataset, batch_size=experiment.batch_size))
    )
    expected_first_loss = F.mse_loss(initial_prediction(x_batch), y_batch).item()
    assert result.loss_history[0] == pytest.approx(expected_first_loss)


def test_production_simple_configuration_preserves_25_updates() -> None:
    cfg = compose_optimization_config("config_simple_loss_optimization")

    assert cfg.n_optimization_steps == 25


def test_non_positive_update_count_is_rejected() -> None:
    cfg = compose_optimization_config(
        "config_test_simple_loss_optimization", ["n_optimization_steps=0"]
    )

    with pytest.raises(ValueError, match="update"):
        produce_mini_batch_inputs(cfg)
