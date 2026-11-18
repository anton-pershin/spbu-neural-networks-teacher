import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from spbunn.reporting import ReportWriter
from spbunn.reporting.bpe import build_bpe_report
from spbunn.sequence.pipeline import (
    produce_large_input,
    produce_toy_input,
    run_large_experiment,
    run_toy_experiment,
)
from spbunn.utils.common import get_config_path


@hydra.main(
    version_base="1.3",
    config_path=str(get_config_path()),
    config_name="config_bpe",
)
def bpe(cfg: DictConfig) -> None:
    """Run the toy and larger-corpus BPE teaching pipeline."""
    toy = run_toy_experiment(produce_toy_input(cfg))
    large = run_large_experiment(produce_large_input(cfg))
    report = build_bpe_report(
        toy,
        large,
        title=cfg.report.title,
        artifact_name=cfg.report.artifact_name,
    )
    try:
        ReportWriter(cfg.result_dir).write(report)
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


if __name__ == "__main__":
    bpe()  # pylint: disable=no-value-for-parameter
