from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2.functional as tv_F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@dataclass(frozen=True)
class ConvolutionExperiment:
    """One image and the settings for its repeated grouped convolution."""

    name: str
    image: torch.Tensor
    filters: torch.Tensor
    groups: int
    repetitions: int
    normalize: bool


@dataclass(frozen=True)
class ConvolutionResult:
    """The convolved image and its immutable experiment reference."""

    experiment: ConvolutionExperiment
    convolved_image: torch.Tensor


def _legacy_normalize(image: torch.Tensor) -> torch.Tensor:
    return tv_F.normalize(
        image,
        mean=image.mean(dim=(1, 2)),
        std=image.std(dim=(1, 2)),
    )


def _resolve_input_path(raw_path: object) -> str:
    if raw_path is None or not str(raw_path).strip():
        raise ValueError("Configured image paths must be non-empty")
    return to_absolute_path(str(raw_path).strip())


def _load_image(path: str, mode: torchvision.io.ImageReadMode) -> torch.Tensor:
    image = torchvision.io.read_image(path, mode=mode)
    return image.unsqueeze(0).to(dtype=torch.float64)


def _rgb_filters(kernel: torch.Tensor) -> torch.Tensor:
    return torch.stack((kernel, kernel, kernel), dim=0).unsqueeze(dim=1)


def produce_convolution_inputs(
    cfg: DictConfig,
) -> tuple[ConvolutionExperiment, ...]:
    """Load configured images and construct the three legacy experiments."""
    kernel_size = int(cfg.kernel_size)
    repetitions = int(cfg.n_repetitions)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("The convolution kernel size must be a positive odd integer")
    if repetitions <= 0:
        raise ValueError("The convolution repetition count must be positive")
    rgb_path = _resolve_input_path(cfg.images.rgb)
    grayscale_path = _resolve_input_path(cfg.images.grayscale)

    rgb_image = _load_image(rgb_path, torchvision.io.ImageReadMode.RGB)
    rgb_image = _legacy_normalize(rgb_image)
    grayscale_image = _load_image(grayscale_path, torchvision.io.ImageReadMode.GRAY)

    center = kernel_size // 2
    horizontal_kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float64)
    horizontal_kernel[center, :] = 1
    vertical_kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float64)
    vertical_kernel[:, center] = 1
    diagonal_kernel = torch.eye(kernel_size, dtype=torch.float64)

    return (
        ConvolutionExperiment(
            name="horizontal",
            image=rgb_image,
            filters=_rgb_filters(horizontal_kernel),
            groups=3,
            repetitions=repetitions,
            normalize=True,
        ),
        ConvolutionExperiment(
            name="vertical",
            image=rgb_image,
            filters=_rgb_filters(vertical_kernel),
            groups=3,
            repetitions=repetitions,
            normalize=True,
        ),
        ConvolutionExperiment(
            name="diagonal",
            image=grayscale_image,
            filters=diagonal_kernel.unsqueeze(dim=0).unsqueeze(dim=1),
            groups=1,
            repetitions=repetitions,
            normalize=False,
        ),
    )


def _validate_experiment(experiment: ConvolutionExperiment) -> None:
    image = experiment.image
    filters = experiment.filters
    if experiment.repetitions <= 0:
        raise ValueError("The convolution repetition count must be positive")
    if image.ndim != 4:
        raise ValueError("The input image must have four dimensions")
    if image.shape[0] != 1:
        raise ValueError("The input image must have batch size 1")
    if any(size <= 0 for size in image.shape):
        raise ValueError("The input image dimensions must be non-empty")
    if filters.ndim != 4:
        raise ValueError("The convolution filters must have four dimensions")
    if experiment.groups <= 0:
        raise ValueError("The convolution groups must be positive")
    if filters.shape[0] != image.shape[1]:
        raise ValueError("Filter output channels must equal the input image channels")
    if filters.shape[1] * experiment.groups != image.shape[1]:
        raise ValueError(
            "Filter input channels and convolution groups are incompatible"
        )
    if filters.shape[2] <= 0 or filters.shape[3] <= 0:
        raise ValueError("The convolution kernel dimensions must be non-empty")
    if filters.dtype != image.dtype:
        raise ValueError("The image and convolution filters must have the same dtype")


def evaluate_convolution(experiment: ConvolutionExperiment) -> ConvolutionResult:
    """Evaluate one repeated convolution without rendering or file output."""
    _validate_experiment(experiment)
    output = experiment.image
    with torch.no_grad():
        for _ in range(experiment.repetitions):
            output = F.conv2d(
                output,
                experiment.filters,
                groups=experiment.groups,
                padding="same",
            )
            if experiment.normalize:
                output = _legacy_normalize(output)
    return ConvolutionResult(experiment=experiment, convolved_image=output)


def evaluate_convolutions(
    experiments: tuple[ConvolutionExperiment, ...],
) -> tuple[ConvolutionResult, ...]:
    """Evaluate experiments in producer order."""
    return tuple(evaluate_convolution(experiment) for experiment in experiments)
