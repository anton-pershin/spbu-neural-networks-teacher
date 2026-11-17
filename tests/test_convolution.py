from pathlib import Path

import pytest
import torch
import torchvision
import torchvision.transforms.v2.functional as tv_F

from spbunn.convolution import (
    ConvolutionExperiment,
    evaluate_convolution,
    evaluate_convolutions,
    produce_convolution_inputs,
)
from tests.convolution_helpers import (
    compose_convolution_config,
    write_convolution_input_images,
)


def _experiment(
    image: torch.Tensor,
    filters: torch.Tensor,
    *,
    groups: int = 1,
    repetitions: int = 1,
    normalize: bool = False,
) -> ConvolutionExperiment:
    return ConvolutionExperiment(
        name="test",
        image=image,
        filters=filters,
        groups=groups,
        repetitions=repetitions,
        normalize=normalize,
    )


def _configured_inputs(tmp_path: Path) -> tuple[ConvolutionExperiment, ...]:
    rgb_path, grayscale_path = write_convolution_input_images(tmp_path / "inputs")
    cfg = compose_convolution_config(
        overrides=[
            f"images.rgb={rgb_path}",
            f"images.grayscale={grayscale_path}",
            "kernel_size=3",
            "n_repetitions=2",
        ]
    )
    return produce_convolution_inputs(cfg)


def test_producer_loads_images_and_constructs_legacy_kernels(
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

    experiments = produce_convolution_inputs(cfg)

    assert [experiment.name for experiment in experiments] == [
        "horizontal",
        "vertical",
        "diagonal",
    ]
    horizontal, vertical, diagonal = experiments
    raw_rgba = torchvision.io.read_image(str(rgb_path))
    expected_rgb = raw_rgba[:-1].unsqueeze(0).to(torch.float64)
    expected_rgb = tv_F.normalize(
        expected_rgb,
        mean=expected_rgb.mean(dim=(1, 2)),
        std=expected_rgb.std(dim=(1, 2)),
    )
    assert torch.equal(horizontal.image, expected_rgb)
    assert horizontal.image is vertical.image
    assert horizontal.image.dtype == torch.float64
    assert horizontal.image.shape == (1, 3, 5, 7)
    expected_grayscale = (
        torchvision.io.read_image(str(grayscale_path)).unsqueeze(0).to(torch.float64)
    )
    assert torch.equal(diagonal.image, expected_grayscale)
    assert diagonal.image.shape == (1, 1, 5, 7)

    expected_horizontal = torch.zeros(3, 3, dtype=torch.float64)
    expected_horizontal[1, :] = 1
    expected_vertical = torch.zeros(3, 3, dtype=torch.float64)
    expected_vertical[:, 1] = 1
    expected_diagonal = torch.eye(3, dtype=torch.float64)
    assert horizontal.filters.shape == (3, 1, 3, 3)
    assert vertical.filters.shape == (3, 1, 3, 3)
    assert diagonal.filters.shape == (1, 1, 3, 3)
    for channel in range(3):
        assert torch.equal(horizontal.filters[channel, 0], expected_horizontal)
        assert torch.equal(vertical.filters[channel, 0], expected_vertical)
    assert torch.equal(diagonal.filters[0, 0], expected_diagonal)
    assert (horizontal.groups, vertical.groups, diagonal.groups) == (3, 3, 1)
    assert (horizontal.normalize, vertical.normalize, diagonal.normalize) == (
        True,
        True,
        False,
    )
    assert all(experiment.repetitions == 2 for experiment in experiments)


def test_producer_resolves_relative_paths_from_invocation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_convolution_input_images(tmp_path / "inputs")
    monkeypatch.chdir(tmp_path)
    cfg = compose_convolution_config(
        overrides=[
            "images.rgb=inputs/rgb.png",
            "images.grayscale=inputs/grayscale.png",
        ]
    )

    experiments = produce_convolution_inputs(cfg)

    assert len(experiments) == 3


def test_production_configuration_preserves_legacy_settings() -> None:
    cfg = compose_convolution_config("config_simple_convolution")

    assert cfg.images.rgb == "climate_change.png"
    assert cfg.images.grayscale == "waves.png"
    assert cfg.kernel_size == 7
    assert cfg.n_repetitions == 20
    assert cfg.result_dir == cfg.hydra_dir
    assert cfg.hydra_root
    assert cfg.report.artifact_name == "simple_convolution.html"


@pytest.mark.parametrize("kernel_size", [0, -1, 2, 4])
def test_producer_rejects_non_positive_or_even_kernel_sizes(
    tmp_path: Path, kernel_size: int
) -> None:
    rgb_path, grayscale_path = write_convolution_input_images(tmp_path / "inputs")
    cfg = compose_convolution_config(
        overrides=[
            f"images.rgb={rgb_path}",
            f"images.grayscale={grayscale_path}",
            f"kernel_size={kernel_size}",
        ]
    )

    with pytest.raises(ValueError, match="positive odd"):
        produce_convolution_inputs(cfg)


def test_producer_rejects_non_positive_repetitions(tmp_path: Path) -> None:
    rgb_path, grayscale_path = write_convolution_input_images(tmp_path / "inputs")
    cfg = compose_convolution_config(
        overrides=[
            f"images.rgb={rgb_path}",
            f"images.grayscale={grayscale_path}",
            "n_repetitions=0",
        ]
    )

    with pytest.raises(ValueError, match="repetition count must be positive"):
        produce_convolution_inputs(cfg)


def test_producer_rejects_empty_image_path(tmp_path: Path) -> None:
    rgb_path, grayscale_path = write_convolution_input_images(tmp_path / "inputs")
    cfg = compose_convolution_config(
        overrides=[
            f"images.rgb={rgb_path}",
            f"images.grayscale={grayscale_path}",
        ]
    )
    cfg.images.rgb = "  "

    with pytest.raises(ValueError, match="paths must be non-empty"):
        produce_convolution_inputs(cfg)


def test_same_padding_matches_hand_calculated_single_convolution() -> None:
    image = torch.arange(1, 10, dtype=torch.float64).reshape(1, 1, 3, 3)
    filters = torch.ones(1, 1, 3, 3, dtype=torch.float64)

    result = evaluate_convolution(_experiment(image, filters))

    expected = torch.tensor(
        [[12, 21, 16], [27, 45, 33], [24, 39, 28]], dtype=torch.float64
    )
    assert torch.equal(result.convolved_image[0, 0], expected)
    assert result.experiment.image is image


def test_grouped_repeated_convolution_preserves_channels() -> None:
    image = torch.tensor([[[[1, 2]], [[3, 4]]]], dtype=torch.float64)
    filters = torch.tensor([[[[2.0]]], [[[3.0]]]], dtype=torch.float64)

    result = evaluate_convolution(_experiment(image, filters, groups=2, repetitions=2))

    expected = torch.tensor([[[[4, 8]], [[27, 36]]]], dtype=torch.float64)
    assert torch.equal(result.convolved_image, expected)


def test_rgb_style_normalization_is_applied_after_each_convolution() -> None:
    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
    filters = torch.ones(1, 1, 1, 1, dtype=torch.float64)

    result = evaluate_convolution(_experiment(image, filters, normalize=True))

    scale = 1 / (2**0.5)
    expected = torch.tensor([[[[-scale, -scale], [scale, scale]]]], dtype=torch.float64)
    assert torch.allclose(result.convolved_image, expected)


def test_evaluator_preserves_order_and_original_images(tmp_path: Path) -> None:
    experiments = _configured_inputs(tmp_path)
    original_images = [experiment.image.clone() for experiment in experiments]

    results = evaluate_convolutions(experiments)

    assert [result.experiment.name for result in results] == [
        "horizontal",
        "vertical",
        "diagonal",
    ]
    for result, original in zip(results, original_images, strict=True):
        assert torch.equal(result.experiment.image, original)
        assert result.convolved_image.shape == original.shape


@pytest.mark.parametrize(
    ("image", "filters", "groups", "match"),
    [
        (
            torch.ones(3, 3, dtype=torch.float64),
            torch.ones(1, 1, 1, 1, dtype=torch.float64),
            1,
            "four dimensions",
        ),
        (
            torch.ones(2, 1, 2, 2, dtype=torch.float64),
            torch.ones(1, 1, 1, 1, dtype=torch.float64),
            1,
            "batch size 1",
        ),
        (
            torch.ones(1, 2, 2, 2, dtype=torch.float64),
            torch.ones(1, 1, 1, dtype=torch.float64),
            1,
            "four dimensions",
        ),
        (
            torch.ones(1, 3, 2, 2, dtype=torch.float64),
            torch.ones(3, 1, 1, 1, dtype=torch.float64),
            1,
            "groups",
        ),
        (
            torch.ones(1, 3, 2, 2, dtype=torch.float64),
            torch.ones(2, 1, 1, 1, dtype=torch.float64),
            3,
            "output channels",
        ),
    ],
)
def test_evaluator_rejects_incompatible_shapes(
    image: torch.Tensor, filters: torch.Tensor, groups: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_convolution(_experiment(image, filters, groups=groups))


def test_evaluator_rejects_non_positive_repetitions() -> None:
    image = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    filters = torch.ones(1, 1, 1, 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="repetition count must be positive"):
        evaluate_convolution(_experiment(image, filters, repetitions=0))
