import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.mlp.propagation import evaluate_propagation, produce_propagation_input
from spbunn.reporting import ReportWriter
from spbunn.reporting.mlp import build_hidden_norms_report
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_hidden_norms",
)
def hidden_norms_propagation(cfg: DictConfig) -> None:
    """Run hidden-norm, sensitivity, and input-Jacobian diagnostics."""
    experiment = produce_propagation_input(cfg)
    results = evaluate_propagation(experiment)
    report = build_hidden_norms_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    hidden_norms_propagation()  # pylint: disable=no-value-for-parameter
