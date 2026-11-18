from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spbunn.convolution import evaluate_convolutions, produce_convolution_inputs
from spbunn.reporting.convolution import build_convolution_report
from tests.convolution_helpers import (
    compose_convolution_config,
    write_convolution_input_images,
)


def test_convolution_report_has_legacy_figures_and_explanatory_content(
    tmp_path: Path,
) -> None:
    rgb_path, grayscale_path = write_convolution_input_images(tmp_path / "inputs")
    cfg = compose_convolution_config(
        overrides=[
            f"images.rgb={rgb_path}",
            f"images.grayscale={grayscale_path}",
            "kernel_size=3",
            "n_repetitions=2",
        ]
    )
    results = evaluate_convolutions(produce_convolution_inputs(cfg))

    report = build_convolution_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )

    try:
        assert report.title == cfg.report.title
        assert report.artifact_name == cfg.report.artifact_name
        assert [figure.artifact_name for figure in report.figures] == [
            "stripes_convoluted_h",
            "stripes_convoluted_v",
            "waves_convoluted",
        ]
        assert [len(figure.figure.axes) for figure in report.figures] == [6, 6, 2]
        for figure in report.figures:
            for axis in figure.figure.axes:
                assert len(axis.images) == 1
                assert axis.images[0].get_cmap().name == "gray_r"
        for figure in report.figures[:2]:
            axes = figure.figure.axes
            assert axes[1].get_title() == "Original image"
            assert axes[4].get_title() == "After 2 convolutions"
            assert axes[0].get_title() == ""
            assert axes[2].get_title() == ""
        grayscale_axes = report.figures[2].figure.axes
        assert grayscale_axes[0].get_title() == "Original image"
        assert grayscale_axes[1].get_title() == "After 2 convolutions"

        for result, figure_model in zip(results, report.figures, strict=True):
            axes = figure_model.figure.axes
            channels = result.experiment.image.shape[1]
            for channel in range(channels):
                original = np.asarray(axes[channel].images[0].get_array())
                convolved = np.asarray(axes[channel + channels].images[0].get_array())
                assert np.array_equal(
                    original, result.experiment.image[0, channel].numpy()
                )
                assert np.array_equal(
                    convolved, result.convolved_image[0, channel].numpy()
                )

        assert r"\(\operatorname{conv2d}\)" in report.html_content
        assert r"\(\mu_x\)" in report.html_content
        assert r"\(\sigma_x\)" in report.html_content
        assert "without flipping" in report.html_content
        assert "groups" in report.html_content
        assert "Horizontal kernel" in report.html_content
        assert "Vertical kernel" in report.html_content
        assert "Diagonal kernel" in report.html_content
        assert "grayscale diagonal example is not normalized" in report.html_content
    finally:
        for figure in report.figures:
            plt.close(figure.figure)
