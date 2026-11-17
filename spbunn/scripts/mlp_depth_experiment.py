import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.mlp.architecture import (
    evaluate_architecture_experiment,
    produce_depth_experiment,
)
from spbunn.reporting import ReportWriter
from spbunn.reporting.mlp import build_mlp_depth_report
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_mlp_depth_experiment",
)
def mlp_depth_experiment(cfg: DictConfig) -> None:
    """Run the deep plain MLP depth and convergence pipeline."""
    experiment = produce_depth_experiment(cfg)
    results = evaluate_architecture_experiment(experiment)
    report = build_mlp_depth_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    mlp_depth_experiment()  # pylint: disable=no-value-for-parameter
