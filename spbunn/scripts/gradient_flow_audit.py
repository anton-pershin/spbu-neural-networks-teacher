"""Run the matched vanilla-RNN/LSTM gradient-flow audit."""

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.reporting import ReportWriter
from spbunn.reporting.gradient_flow import build_gradient_flow_report
from spbunn.rnn.gradient_flow_pipeline import (
    evaluate_gradient_flow_experiment,
    produce_gradient_flow_input,
)
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_gradient_flow_audit",
)
def gradient_flow_audit(cfg: DictConfig) -> None:
    """Train and audit copy models and write the educational report."""
    experiment = produce_gradient_flow_input(cfg)
    result = evaluate_gradient_flow_experiment(experiment)
    report = build_gradient_flow_report(
        result, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    gradient_flow_audit()  # pylint: disable=no-value-for-parameter
