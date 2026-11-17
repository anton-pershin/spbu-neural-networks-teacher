import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.mlp.propagation import evaluate_input_gradients, produce_balduzzi_input
from spbunn.reporting import ReportWriter
from spbunn.reporting.mlp import build_balduzzi_report
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_balduzzi_analysis",
)
def balduzzi_analysis(cfg: DictConfig) -> None:
    """Run plain and residual summed-output input-gradient analysis."""
    experiment = produce_balduzzi_input(cfg)
    results = evaluate_input_gradients(experiment)
    report = build_balduzzi_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    balduzzi_analysis()  # pylint: disable=no-value-for-parameter
