"""Run the vanilla-RNN BPTT instability audit."""

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.reporting import ReportWriter
from spbunn.reporting.rnn import build_bptt_report
from spbunn.rnn.pipeline import evaluate_bptt_experiment, produce_bptt_input
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_bptt_instability",
)
def bptt_instability(cfg: DictConfig) -> None:
    """Run synthetic recurrent audits and write their teaching report."""
    experiment = produce_bptt_input(cfg)
    result = evaluate_bptt_experiment(experiment)
    report = build_bptt_report(
        result, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    bptt_instability()  # pylint: disable=no-value-for-parameter
