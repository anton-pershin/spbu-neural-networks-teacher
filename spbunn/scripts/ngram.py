"""Run the offline n-gram language-model pipeline."""

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.ngram.pipeline import evaluate_ngram_experiment, produce_ngram_input
from spbunn.reporting import ReportWriter
from spbunn.reporting.ngram import build_ngram_report
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_ngram",
)
def ngram(cfg: DictConfig) -> None:
    """Run the n-gram training, evaluation, and reporting pipeline."""
    experiment = produce_ngram_input(cfg)
    result = evaluate_ngram_experiment(experiment)
    report = build_ngram_report(
        result, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    ngram()  # pylint: disable=no-value-for-parameter
