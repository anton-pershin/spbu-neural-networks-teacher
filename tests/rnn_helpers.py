"""Shared configuration and result fixtures for recurrent-audit tests."""

from collections.abc import Sequence
from functools import lru_cache

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from spbunn.rnn.gradient_flow_pipeline import (
    GradientFlowExperimentResult,
    evaluate_gradient_flow_experiment,
    produce_gradient_flow_input,
)
from spbunn.rnn.pipeline import (
    BPTTExperimentResult,
    evaluate_bptt_experiment,
    produce_bptt_input,
)
from spbunn.utils.common import get_config_path


def compose_bptt_config(
    config_name: str = "config_test_bptt_instability",
    overrides: Sequence[str] = (),
) -> DictConfig:
    """Compose one BPTT audit configuration from the real config directory."""
    with initialize_config_dir(
        config_dir=str(get_config_path()),
        job_name="bptt_instability_test",
        version_base="1.3",
    ):
        return compose(config_name=config_name, overrides=list(overrides))


@lru_cache(maxsize=1)
def reduced_bptt_result() -> BPTTExperimentResult:
    """Evaluate the reduced example once for independent checks."""
    return evaluate_bptt_experiment(produce_bptt_input(compose_bptt_config()))


def compose_gradient_flow_config(
    config_name: str = "config_test_gradient_flow_audit",
    overrides: Sequence[str] = (),
) -> DictConfig:
    """Compose one gradient-flow configuration from the real config directory."""
    with initialize_config_dir(
        config_dir=str(get_config_path()),
        job_name="gradient_flow_audit_test",
        version_base="1.3",
    ):
        return compose(config_name=config_name, overrides=list(overrides))


@lru_cache(maxsize=1)
def reduced_gradient_flow_result() -> GradientFlowExperimentResult:
    """Evaluate the reduced gradient-flow experiment once for shared checks."""
    return evaluate_gradient_flow_experiment(
        produce_gradient_flow_input(compose_gradient_flow_config())
    )
