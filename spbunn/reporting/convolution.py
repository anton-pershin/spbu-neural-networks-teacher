import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from spbunn.convolution import ConvolutionResult
from spbunn.reporting.models import FigureModel, ReportModel

_ARTIFACT_NAMES = {
    "horizontal": "stripes_convoluted_h",
    "vertical": "stripes_convoluted_v",
    "diagonal": "waves_convoluted",
}


def _plot_rgb_figure(result: ConvolutionResult) -> Figure:
    figure, axes = plt.subplots(2, 3, figsize=(12, 5.5))
    images = (result.experiment.image, result.convolved_image)
    for row, image in enumerate(images):
        for channel in range(3):
            axes[row, channel].imshow(
                image[0, channel, ...].detach().cpu().numpy(), cmap="gray_r"
            )
    repetitions = result.experiment.repetitions
    axes[0, 1].set_title("Original image", fontsize=12)
    axes[1, 1].set_title(f"After {repetitions} convolutions", fontsize=12)
    figure.tight_layout()
    return figure


def _plot_grayscale_figure(result: ConvolutionResult) -> Figure:
    figure, axes = plt.subplots(2, 1, figsize=(6, 5.5))
    images = (result.experiment.image, result.convolved_image)
    for axis, image in zip(axes, images, strict=True):
        axis.imshow(image[0, 0, ...].detach().cpu().numpy(), cmap="gray_r")
    repetitions = result.experiment.repetitions
    axes[0].set_title("Original image", fontsize=12)
    axes[1].set_title(f"After {repetitions} convolutions", fontsize=12)
    figure.tight_layout()
    return figure


def build_convolution_report(
    results: tuple[ConvolutionResult, ...], title: str, artifact_name: str
) -> ReportModel:
    """Build the simple-convolution teaching report without file output."""
    sections = [
        "<h1>Simple repeated convolution</h1>",
        (
            r"<p>Each example repeatedly applies \(\operatorname{conv2d}\) "
            "with unit stride and same padding. PyTorch computes "
            "cross-correlation without flipping the kernel. The output has "
            "the same spatial dimensions as its input; values beyond the "
            "input boundary contribute zero.</p>"
        ),
        (
            "<p>The horizontal and vertical examples process the three RGB "
            "channels independently with three groups and one two-dimensional "
            "kernel per channel. The diagonal example uses one grayscale "
            "channel and one group. Every experiment repeats its configured "
            "operation the same number of times.</p>"
        ),
        (
            r"<p>For an RGB tensor \(X\), the legacy normalization computes "
            r"a mean \(\mu_x\) and sample standard deviation \(\sigma_x\) "
            "over the channel and vertical dimensions for each horizontal "
            r"index \(x\). It then computes \((X-\mu_x)/\sigma_x\) for the "
            "input and after every horizontal or vertical convolution. The "
            "grayscale diagonal example is not normalized.</p>"
        ),
        (
            r"<section><h2>Horizontal kernel</h2><p>The center row is one "
            r"and every other entry is zero: \(K_h[k/2, x]=1\). Repeated "
            "application mixes values along the horizontal direction in "
            "each RGB channel.</p></section>"
        ),
        (
            r"<section><h2>Vertical kernel</h2><p>The center column is one "
            r"and every other entry is zero: \(K_v[y, k/2]=1\). Repeated "
            "application mixes values along the vertical direction in each "
            "RGB channel.</p></section>"
        ),
        (
            r"<section><h2>Diagonal kernel</h2><p>The identity kernel "
            r"satisfies \(K_d[y,x]=1\) when \(y=x\) and zero otherwise. "
            "It mixes diagonal neighbors in the grayscale image.</p></section>"
        ),
    ]

    figures = []
    for result in results:
        channels = result.experiment.image.shape[1]
        if channels == 3:
            figure = _plot_rgb_figure(result)
        elif channels == 1:
            figure = _plot_grayscale_figure(result)
        else:
            raise ValueError(f"Unsupported image channel count: {channels}")
        try:
            stem = _ARTIFACT_NAMES[result.experiment.name]
        except KeyError as error:
            plt.close(figure)
            raise ValueError(
                f"Unsupported convolution experiment: {result.experiment.name}"
            ) from error
        figures.append(FigureModel(figure=figure, artifact_name=stem))
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=tuple(figures),
        artifact_name=artifact_name,
    )
