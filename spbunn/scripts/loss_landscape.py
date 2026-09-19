import os

import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import ticker
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader 
import tqdm
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from spbunn.optimization.objectives import SamplingFromFunctionDataset
from spbunn.utils.typing import NDArrayFloat

def compute_loss_landscape(
    func_info: DictConfig,
    theta_0_values: NDArrayFloat,
    theta_1_values: NDArrayFloat,
) -> NDArrayFloat:
    f_true = instantiate(func_info.func)(
        theta_0=func_info.theta_0_true,
        theta_1=func_info.theta_1_true,
    )
    n = 1024
    #fresh_rng = np.random.default_rng(seed=42)
    dataset = SamplingFromFunctionDataset(f=f_true, n=n, rng=None)   
    data_loader = DataLoader(dataset, batch_size=len(dataset))

    total_loss_values = np.zeros((len(theta_0_values), len(theta_1_values)))
    found_min_loss = np.inf
    found_argmin = (None, None)

#    if os.path.exists("buf.npy"):
#        total_loss_values = np.load("buf.npy")
#    else: 
    for i in tqdm.trange(len(theta_0_values)):
        for j in range(len(theta_1_values)):
            x_batch, y_batch = next(iter(data_loader))
            f_pred = instantiate(func_info.func)(
                theta_0=theta_0_values[i],
                theta_1=theta_1_values[j],
            )
            y_pred = f_pred(x_batch) 
            total_loss_values[i, j] = F.mse_loss(y_pred, y_batch).numpy()
            if found_min_loss > total_loss_values[i, j]:
                found_min_loss = total_loss_values[i, j]
                found_argmin = (theta_0_values[i], theta_1_values[j])
    return total_loss_values


def plot_loss_landscape(
    ax,
    theta_0_values: NDArrayFloat,
    theta_1_values: NDArrayFloat,
    total_loss_values: NDArrayFloat,
    levels: list[float],
) -> None:
    found_argmin = np.unravel_index(np.argmin(total_loss_values), total_loss_values.shape)
    theta_0_mesh, theta_1_mesh = np.meshgrid(theta_0_values, theta_1_values, indexing="ij")
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'truncated_bone',
        matplotlib.colormaps["bone"](np.linspace(0, 0.8, 100))
    )

    conts = ax.contour(
        theta_0_mesh,
        theta_1_mesh,
        total_loss_values,
        levels=levels,
        cmap=new_cmap,
        norm="log",
        zorder=-10,
    )
    ax.clabel(conts, levels=levels, fmt="%.e")
    ax.scatter(
        [theta_0_values[found_argmin[0]]],
        [theta_1_values[found_argmin[1]]],
        s=64,
        color="tab:orange"
    )
    ax.set_xlabel(r"$\theta_1$", fontsize=12)
    ax.set_ylabel(r"$\theta_2$", fontsize=12)


def loss_landscape(cfg: DictConfig):
    for func_info in cfg.functions:
        print(f"Processing func {func_info.name}...")
        theta_0_values = np.linspace(func_info.theta_0_range[0], func_info.theta_0_range[1], 50)
        theta_1_values = np.linspace(func_info.theta_1_range[0], func_info.theta_1_range[1], 50)
        total_loss_values = compute_loss_landscape(
            func_info=func_info,
            theta_0_values=theta_0_values,
            theta_1_values=theta_1_values,
        )
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_loss_landscape(
            ax=ax,
            theta_0_values=theta_0_values,
            theta_1_values=theta_1_values,
            total_loss_values=total_loss_values,
            levels=func_info.levels,
        )
        plt.tight_layout()
        plt.savefig(f"loss_landscape_func_{func_info.name}.svg")


if __name__ == "__main__":
    hydra.main(
        version_base=None,
        config_path="../../config",
        config_name="config_loss_landscape"
    )(loss_landscape)()

