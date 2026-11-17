import numpy as np
import pytest

from spbunn.optimization.schedulers import evaluate_schedules, produce_scheduler_input
from tests.optimization_helpers import compose_optimization_config


def test_scheduler_curves_preserve_legacy_values() -> None:
    experiment = produce_scheduler_input(
        compose_optimization_config("config_test_schedulers")
    )

    curves = {curve.settings.name: curve for curve in evaluate_schedules(experiment)}

    assert set(curves) == {
        "linearlr",
        "steplr",
        "reducelronplateau",
        "cosineannealing",
    }
    linear = np.concatenate(
        [0.1 - 0.003 * np.arange(20), np.full(20, 0.1 - 0.003 * 19)]
    )
    np.testing.assert_allclose(curves["linearlr"].learning_rates, linear)
    step = np.concatenate(
        [
            np.full(5, 0.2),
            np.full(5, 0.1),
            np.full(5, 0.05),
            np.full(5, 0.025),
            np.full(20, 0.0125),
        ]
    )
    np.testing.assert_allclose(curves["steplr"].learning_rates, step)
    plateau = np.concatenate([np.full(10, 0.1), np.full(20, 0.05), np.full(10, 0.025)])
    np.testing.assert_allclose(curves["reducelronplateau"].learning_rates, plateau)
    cosine = curves["cosineannealing"].learning_rates
    expected_quarter = 0.1 + 0.025 * (1 + np.cos(np.arange(10) / 10 * np.pi))
    np.testing.assert_allclose(cosine, np.tile(expected_quarter, 4))


def test_scheduler_input_rejects_unknown_schedule() -> None:
    cfg = compose_optimization_config(
        "config_test_schedulers", ["schedulers.0.schedule_type=unknown"]
    )
    experiment = produce_scheduler_input(cfg)

    with pytest.raises(ValueError, match="Unsupported schedule"):
        evaluate_schedules(experiment)
