import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.mlp.architecture import (
    evaluate_architecture_experiment,
    produce_width_experiment,
)
from spbunn.reporting import ReportWriter
from spbunn.reporting.mlp import build_mlp_width_report
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_mlp_width_experiment",
)
def mlp_width_experiment(cfg: DictConfig) -> None:
    """Run the shallow MLP width and convergence pipeline."""
    experiment = produce_width_experiment(cfg)
    results = evaluate_architecture_experiment(experiment)
    report = build_mlp_width_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    mlp_width_experiment()  # pylint: disable=no-value-for-parameter
