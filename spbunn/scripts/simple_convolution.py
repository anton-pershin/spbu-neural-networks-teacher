import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2.functional as tv_F


def plot_rgb_image_by_channels(fig, axes, im: torch.Tensor, title: str) -> None:
    for i in range(3):
        axes[i].imshow(im[0, i, ...], cmap="gray_r")
    axes[1].set_title(title, fontsize=12)
    fig.tight_layout()


def plot_gray_image_by_channels(fig, ax, im: torch.Tensor, title: str) -> None:
    ax.imshow(im[0, 0, ...], cmap="gray_r")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()


def convolute(im, filters, groups: int, normalize=False) -> torch.Tensor:
    out = im
    for i in range(20):
        out = F.conv2d(out, filters, groups=groups, padding="same")
        if normalize:
            out = tv_F.normalize(out, mean=out.mean(dim=(1, 2)), std=out.std(dim=(1, 2)))
    return out

def convolute_and_plot_rgb_image(im, h_kernel, label: str) -> None:
    # out_channels = 3, in_channels/groups = in_channels = 3, kH = 10, kW = 10
    filters = torch.stack((h_kernel, h_kernel, h_kernel), dim=0).unsqueeze(dim=1)
    out = convolute(im, filters, groups=3, normalize=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5))
    plot_rgb_image_by_channels(fig, axes[0, :], im=im, title="Original image")
    plot_rgb_image_by_channels(fig, axes[1, :], im=out, title="After 20 convolutions")
    fig.savefig(f"stripes_convoluted_{label}.png", dpi=100)


def horizontal_and_vertical_convolution() -> None:
    im = torchvision.io.read_image("climate_change.png")
    im = im[:-1, ...]  # drop the alpha channel
    # Add batch dimension
    im = im.unsqueeze(0)
    im = im.type(torch.float64)
    im = tv_F.normalize(im, mean=im.mean(dim=(1, 2)), std=im.std(dim=(1, 2)))

    h_kernel = torch.zeros((7, 7), dtype=torch.float64)
    h_kernel[3, :] = 1.
    convolute_and_plot_rgb_image(im, h_kernel, label="h")

    v_kernel = torch.zeros((7, 7), dtype=torch.float64)
    v_kernel[:, 3] = 1.
    convolute_and_plot_rgb_image(im, v_kernel, label="v")


def diag_convolution() -> None:
    im = torchvision.io.read_image("waves.png", mode=torchvision.io.ImageReadMode.GRAY)
    im = im.unsqueeze(0)
    im = im.type(torch.float64)
    h_kernel = torch.eye(7, dtype=torch.float64)
    filters = h_kernel.unsqueeze(dim=0).unsqueeze(dim=1)
    out = convolute(im, filters, groups=1, normalize=False)

    fig, axes = plt.subplots(2, 1, figsize=(6, 5.5))
    plot_gray_image_by_channels(fig, axes[0], im=im, title="Original image")
    plot_gray_image_by_channels(fig, axes[1], im=out, title="After 20 convolutions")
    fig.savefig("waves_convoluted.png", dpi=100)


if __name__ == "__main__":
    horizontal_and_vertical_convolution()
    diag_convolution()

#if __name__ == "__main__":
#    hydra.main(
#        version_base=None,
#        config_path="../../config",
#        config_name="config_loss_landscape"
#    )(simple_loss_optimization)()
