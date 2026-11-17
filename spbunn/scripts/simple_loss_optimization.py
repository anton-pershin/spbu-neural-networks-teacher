import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.optimization.mini_batch import (
    evaluate_mini_batch,
    produce_mini_batch_inputs,
)
from spbunn.reporting import ReportWriter
from spbunn.reporting.optimization import build_simple_optimization_report
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_simple_loss_optimization",
)
def simple_loss_optimization(cfg: DictConfig) -> None:
    """Run the mini-batch loss-optimization pipeline."""
    experiments = produce_mini_batch_inputs(cfg)
    results = tuple(evaluate_mini_batch(experiment) for experiment in experiments)
    report = build_simple_optimization_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    simple_loss_optimization()  # pylint: disable=no-value-for-parameter
