import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.optimization.landscape import (
    evaluate_landscape_experiment,
    produce_landscape_inputs,
)
from spbunn.reporting import ReportWriter
from spbunn.reporting.optimization import build_loss_landscape_report
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_loss_landscape",
)
def loss_landscape(cfg: DictConfig) -> None:
    """Run the landscape and full-batch trajectory pipeline."""
    experiments = produce_landscape_inputs(cfg)
    results = tuple(
        evaluate_landscape_experiment(experiment) for experiment in experiments
    )
    report = build_loss_landscape_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    loss_landscape()  # pylint: disable=no-value-for-parameter
