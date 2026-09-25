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
    dataset = SamplingFromFunctionDataset(
        f=f_true, n=n, rng=None,
        x_min=func_info.get("x_min", -10.0),
        x_max=func_info.get("x_max", 10.0),
    )
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


def compute_trajectory(
    func_info: DictConfig,
    theta_0: float,
    theta_1: float,
    lr: float,
    n_steps: int,
    momentum: float = 0.0,
    optimizer_name: str = "sgd",
    betas: tuple[float, float] = (0.9, 0.999),
) -> tuple[list[float], list[float], list[float]]:
    """Runs full-batch optimization on (theta_0, theta_1) and records the trajectory."""
    f_true = instantiate(func_info.func)(
        theta_0=func_info.theta_0_true,
        theta_1=func_info.theta_1_true,
    )
    n = 1024
    dataset = SamplingFromFunctionDataset(
        f=f_true,
        n=n,
        rng=None,
        x_min=func_info.get("x_min", -10.0),
        x_max=func_info.get("x_max", 10.0),
    )
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    x_batch, y_batch = next(iter(data_loader))

    trainable_func_cls = instantiate(func_info.func).func
    f_pred = trainable_func_cls(theta_0=theta_0, theta_1=theta_1, requires_grad=True)

    parameters = [f_pred.theta_0, f_pred.theta_1]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, lr=lr)
    else:
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)

    trajectory_theta_0: list[float] = [theta_0]
    trajectory_theta_1: list[float] = [theta_1]
    losses: list[float] = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = F.mse_loss(f_pred(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        trajectory_theta_0.append(f_pred.theta_0.item())
        trajectory_theta_1.append(f_pred.theta_1.item())
        losses.append(loss.item())
    return trajectory_theta_0, trajectory_theta_1, losses


def plot_loss_landscape(
    ax,
    theta_0_values: NDArrayFloat,
    theta_1_values: NDArrayFloat,
    total_loss_values: NDArrayFloat,
    levels: list[float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
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
        color="tab:orange",
        zorder=10,
    )
    ax.set_xlabel(r"$\theta_0$", fontsize=18)
    ax.set_ylabel(r"$\theta_1$", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def plot_trajectory(
    ax,
    trajectory_theta_0: list[float],
    trajectory_theta_1: list[float],
    color: str,
    every_nth: int = 10,
) -> None:
    theta_0_trajectory = np.asarray(trajectory_theta_0)
    theta_1_trajectory = np.asarray(trajectory_theta_1)
    ax.plot(
        theta_0_trajectory,
        theta_1_trajectory,
        "o--",
        markersize=4,
        linewidth=1.5,
        color=color,
    )
    ax.scatter(
        theta_0_trajectory[::every_nth],
        theta_1_trajectory[::every_nth],
        s=32,
        color=color,
        zorder=10,
    )
    ax.scatter(
        [theta_0_trajectory[0]],
        [theta_1_trajectory[0]],
        s=64,
        marker="*",
        color="tab:green",
        zorder=10,
    )


def plot_loss_curve(
    ax,
    losses: list[float],
    color: str,
) -> None:
    steps = np.arange(1, len(losses) + 1)
    ax.plot(
        steps,
        losses,
        "o--",
        markersize=4,
        linewidth=1.5,
        color=color,
    )
    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_yscale("log")
    ax.grid(True)


def optimizer_runs(func_info: DictConfig) -> list[dict]:
    """Builds the list of optimizer configurations to run for each function."""
    return [
        {
            "name": "SGD",
            "color": "tab:red",
            "suffix": "sgd",
            "optimizer_name": "sgd",
            "lr": func_info.lr,
            "momentum": 0.0,
        },
        {
            "name": "SGD + momentum",
            "color": "tab:blue",
            "suffix": "sgd_momentum",
            "optimizer_name": "sgd",
            "lr": func_info.lr,
            "momentum": func_info.get("momentum", 0.9),
        },
        {
            "name": "RMSprop",
            "color": "tab:purple",
            "suffix": "rmsprop",
            "optimizer_name": "rmsprop",
            "lr": func_info.lr,
        },
        {
            "name": "Adam",
            "color": "tab:green",
            "suffix": "adam",
            "optimizer_name": "adam",
            "lr": func_info.get("adam_lr", func_info.lr),
            "betas": (0.9, 0.999),
        },
    ]


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

        runs = optimizer_runs(func_info)
        for run in runs:
            run["trajectory_theta_0"], run["trajectory_theta_1"], run["losses"] = compute_trajectory(
                func_info=func_info,
                theta_0=func_info.theta_0_initial_guess,
                theta_1=func_info.theta_1_initial_guess,
                lr=run["lr"],
                n_steps=func_info.get("n_steps", 500),
                momentum=run.get("momentum", 0.0),
                optimizer_name=run["optimizer_name"],
                betas=run.get("betas", (0.9, 0.999)),
            )

        # All figures share identical axis limits so the per-optimizer plots are comparable.
        all_theta_0 = np.concatenate(
            [run["trajectory_theta_0"] for run in runs] + [theta_0_values]
        )
        all_theta_1 = np.concatenate(
            [run["trajectory_theta_1"] for run in runs] + [theta_1_values]
        )
        xlim = (float(all_theta_0.min()), float(all_theta_0.max()))
        ylim = (float(all_theta_1.min()), float(all_theta_1.max()))
        max_steps = max(len(run["losses"]) for run in runs)
        loss_xlim = (1, float(max_steps))

        for run in runs:
            fig, (landscape_ax, loss_ax) = plt.subplots(
                2,
                1,
                figsize=(8, 8),
                height_ratios=[3, 1],
            )
            plot_loss_landscape(
                ax=landscape_ax,
                theta_0_values=theta_0_values,
                theta_1_values=theta_1_values,
                total_loss_values=total_loss_values,
                levels=func_info.levels,
                xlim=xlim,
                ylim=ylim,
            )
            plot_trajectory(
                ax=landscape_ax,
                trajectory_theta_0=run["trajectory_theta_0"],
                trajectory_theta_1=run["trajectory_theta_1"],
                color="red",
            )
            plot_loss_curve(
                ax=loss_ax,
                losses=run["losses"],
                color="tab:blue",
            )
            loss_ax.set_xlim(*loss_xlim)
            plt.tight_layout()
            plt.savefig(f"loss_landscape_func_{func_info.name}_{run['suffix']}.png")
            plt.close(fig)


if __name__ == "__main__":
    hydra.main(
        version_base=None,
        config_path="../../config",
        config_name="config_loss_landscape"
    )(loss_landscape)()
