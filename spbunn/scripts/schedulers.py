import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.optimization.schedulers import evaluate_schedules, produce_scheduler_input
from spbunn.reporting import ReportWriter
from spbunn.reporting.optimization import build_scheduler_report
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_schedulers",
)
def schedulers(cfg: DictConfig) -> None:
    """Run the learning-rate scheduler pipeline."""
    experiment = produce_scheduler_input(cfg)
    curves = evaluate_schedules(experiment)
    report = build_scheduler_report(
        curves, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    schedulers()  # pylint: disable=no-value-for-parameter
