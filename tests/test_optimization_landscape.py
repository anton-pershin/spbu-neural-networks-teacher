import numpy as np
import pytest
import torch

from spbunn.optimization.landscape import (
    LandscapeInput,
    OptimizerSettings,
    RegressionData,
    evaluate_full_batch_trajectory,
    evaluate_landscape,
    evaluate_landscape_experiment,
    produce_landscape_inputs,
)
from spbunn.optimization.objectives import LinearFunc
from tests.optimization_helpers import compose_optimization_config


def small_experiment() -> LandscapeInput:
    return LandscapeInput(
        name="linear",
        func_factory=LinearFunc,
        theta_0_true=1.0,
        theta_1_true=2.0,
        data=RegressionData(
            x=torch.tensor([0.0, 1.0], dtype=torch.float64),
            y=torch.tensor([1.0, 3.0], dtype=torch.float64),
        ),
        theta_0_values=np.array([0.0, 1.0]),
        theta_1_values=np.array([1.0, 2.0]),
        levels=(0.1, 1.0),
        theta_0_initial=0.0,
        theta_1_initial=0.0,
        learning_rate=0.01,
        full_batch_steps=1,
        optimizers=(),
    )


def test_landscape_matches_hand_calculated_mse_and_orientation() -> None:
    losses = evaluate_landscape(small_experiment())

    np.testing.assert_allclose(losses, [[2.5, 1.0], [0.5, 0.0]])
    assert losses.dtype == np.float64


def test_full_batch_trajectory_records_pre_update_loss() -> None:
    experiment = small_experiment()
    settings = OptimizerSettings(
        name="SGD", suffix="sgd", optimizer_name="sgd", learning_rate=0.01
    )

    result = evaluate_full_batch_trajectory(experiment, settings)

    np.testing.assert_allclose(result.trajectory_theta_0, [0.0, 0.04])
    np.testing.assert_allclose(result.trajectory_theta_1, [0.0, 0.03])
    np.testing.assert_allclose(result.losses, [5.0])


def test_landscape_pipeline_preserves_optimizer_order_and_history_sizes() -> None:
    cfg = compose_optimization_config("config_test_loss_landscape")
    experiment = produce_landscape_inputs(cfg)[0]

    result = evaluate_landscape_experiment(experiment)

    assert [
        (
            run.settings.name,
            run.settings.learning_rate,
            run.settings.momentum,
            run.settings.nesterov,
            run.settings.betas,
        )
        for run in result.runs
    ] == [
        ("SGD", 0.01, 0.0, False, (0.9, 0.999)),
        ("SGD + momentum", 0.01, 0.9, False, (0.9, 0.999)),
        ("SGD + Nesterov momentum", 0.01, 0.9, True, (0.9, 0.999)),
        ("RMSprop", 0.01, 0.0, False, (0.9, 0.999)),
        ("Adam", 0.1, 0.0, False, (0.9, 0.999)),
    ]
    assert result.loss_values.shape == (5, 5)
    for run in result.runs:
        assert run.trajectory_theta_0.shape == (3,)
        assert run.trajectory_theta_1.shape == (3,)
        assert run.losses.shape == (2,)
        assert run.trajectory_theta_0[0] == pytest.approx(-2.0)
        assert run.trajectory_theta_1[0] == pytest.approx(0.0)


def test_unknown_optimizer_is_rejected() -> None:
    settings = OptimizerSettings(
        name="Unknown",
        suffix="unknown",
        optimizer_name="unknown",
        learning_rate=0.01,
    )

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        evaluate_full_batch_trajectory(small_experiment(), settings)


def test_empty_landscape_grid_is_rejected() -> None:
    experiment = small_experiment()
    object.__setattr__(experiment, "theta_0_values", np.array([]))

    with pytest.raises(ValueError, match="grid"):
        evaluate_landscape(experiment)


@pytest.mark.parametrize("steps", [0, -1])
def test_per_function_full_batch_step_count_must_be_positive(steps: int) -> None:
    cfg = compose_optimization_config(
        "config_test_loss_landscape", [f"functions.0.n_steps={steps}"]
    )

    with pytest.raises(ValueError, match="step count"):
        produce_landscape_inputs(cfg)


def test_direct_full_batch_step_count_must_be_positive() -> None:
    experiment = small_experiment()
    object.__setattr__(experiment, "full_batch_steps", 0)
    settings = OptimizerSettings("SGD", "sgd", "sgd", 0.01)

    with pytest.raises(ValueError, match="step count"):
        evaluate_full_batch_trajectory(experiment, settings)
