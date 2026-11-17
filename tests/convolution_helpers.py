from collections.abc import Sequence
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image

from spbunn.utils.common import get_config_path


def compose_convolution_config(
    config_name: str = "config_test_simple_convolution",
    overrides: Sequence[str] = (),
) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(get_config_path()),
        job_name="convolution_test",
        version_base="1.3",
    ):
        return compose(config_name=config_name, overrides=list(overrides))


def write_convolution_input_images(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    rgb_path = directory / "rgb.png"
    grayscale_path = directory / "grayscale.png"
    rgba = (
        torch.arange(4 * 5 * 7, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(4, 5, 7)
    )
    rgba[3] = 255
    grayscale = (
        torch.arange(5 * 7, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(1, 5, 7)
    )
    Image.fromarray(rgba.permute(1, 2, 0).numpy()).save(rgb_path)
    Image.fromarray(grayscale[0].numpy()).save(grayscale_path)
    return rgb_path, grayscale_path
