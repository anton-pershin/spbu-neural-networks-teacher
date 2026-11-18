import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.convolution import evaluate_convolutions, produce_convolution_inputs
from spbunn.reporting import ReportWriter
from spbunn.reporting.convolution import build_convolution_report
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_simple_convolution",
)
def simple_convolution(cfg: DictConfig) -> None:
    """Run the horizontal, vertical, and diagonal convolution pipeline."""
    experiments = produce_convolution_inputs(cfg)
    results = evaluate_convolutions(experiments)
    report = build_convolution_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    simple_convolution()  # pylint: disable=no-value-for-parameter
