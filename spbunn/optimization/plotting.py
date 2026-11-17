from collections.abc import Sequence

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def plot_loss_landscape(
    ax: Axes,
    theta_0_values: FloatArray,
    theta_1_values: FloatArray,
    total_loss_values: FloatArray,
    levels: Sequence[float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Render an MSE contour plot and its discrete-grid minimum."""
    found_argmin = np.unravel_index(
        np.argmin(total_loss_values), total_loss_values.shape
    )
    theta_0_mesh, theta_1_mesh = np.meshgrid(
        theta_0_values, theta_1_values, indexing="ij"
    )
    truncated_bone = matplotlib.colors.LinearSegmentedColormap.from_list(
        "truncated_bone",
        matplotlib.colormaps["bone"](np.linspace(0, 0.8, 100)),
    )
    contours = ax.contour(
        theta_0_mesh,
        theta_1_mesh,
        total_loss_values,
        levels=levels,
        cmap=truncated_bone,
        norm="log",
        zorder=-10,
    )
    ax.clabel(contours, levels=levels, fmt="%.e")
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
    ax: Axes,
    trajectory_theta_0: FloatArray,
    trajectory_theta_1: FloatArray,
    color: str,
    every_nth: int = 10,
) -> None:
    """Render a parameter trajectory and its initial point."""
    ax.plot(
        trajectory_theta_0,
        trajectory_theta_1,
        "o--",
        markersize=4,
        linewidth=1.5,
        color=color,
    )
    ax.scatter(
        trajectory_theta_0[::every_nth],
        trajectory_theta_1[::every_nth],
        s=32,
        color=color,
        zorder=10,
    )
    ax.scatter(
        [trajectory_theta_0[0]],
        [trajectory_theta_1[0]],
        s=64,
        marker="*",
        color="tab:green",
        zorder=10,
    )


def plot_loss_curve(
    ax: Axes,
    losses: FloatArray,
    color: str,
    fixed_limits: bool = True,
) -> None:
    """Render a pre-update loss history on a logarithmic axis."""
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
    if fixed_limits:
        ax.set_ylim(1e-2, 1e3)
        loss_ticks = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        ax.set_yticks(loss_ticks)
        ax.set_yticklabels(["1e-2", "1e-1", "1", "10", "100", "1e3"])
    ax.grid(True)
