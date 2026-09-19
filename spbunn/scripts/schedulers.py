import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra.utils import call
from omegaconf import DictConfig

from spbunn.utils.typing import NDArrayFloat


def plot_linearlr(ax, s_info: DictConfig, epochs: NDArrayFloat, starting_lr: float) -> None:
    n_epochs = len(epochs)
    first_part_of_epochs = epochs[:int(n_epochs // 2)]
    second_part_of_epochs = epochs[int(n_epochs // 2) - 1:]
    lrs = starting_lr - 3e-3 * first_part_of_epochs
    ax.plot(first_part_of_epochs, lrs, "o--", color="tab:blue")
    ax.plot(second_part_of_epochs, [lrs[-1]] * len(second_part_of_epochs), "o--", color="tab:blue")


def plot_steplr(ax, s_info: DictConfig, epochs: NDArrayFloat, starting_lr: float) -> None:
    n_epochs = len(epochs)
    n_oct = int(n_epochs // 8)
    last_epoch = 25
    epoch = 0
    cur_lr = starting_lr / s_info.alpha
    epoch_arrays = []
    lr_arrays = []
    while epoch < last_epoch:
        part_of_epochs = epochs[epoch:epoch + n_oct]
        lrs = [cur_lr] * len(part_of_epochs)
        epoch_arrays.append(part_of_epochs)
        lr_arrays.append(lrs)
        epoch += n_oct
        cur_lr *= s_info.alpha
    last_part_of_epochs = epochs[epoch:]
    lrs = [cur_lr / s_info.alpha] * len(last_part_of_epochs)
    epoch_arrays.append(last_part_of_epochs)
    lr_arrays.append(lrs)
    ax.plot(epochs, np.concatenate(lr_arrays), "o--", color="tab:blue")


def plot_reducelronplateau(ax, s_info: DictConfig, epochs: NDArrayFloat, starting_lr: float) -> None:
    n_epochs = len(epochs)
    n_1q = int(n_epochs // 4)
    n_3q = int(3 * n_epochs // 4)
    first_part_of_epochs = epochs[:n_1q]
    second_part_of_epochs = epochs[n_1q:n_3q]
    third_part_of_epochs = epochs[n_3q:]
    epoch_arrays = [
        first_part_of_epochs,
        second_part_of_epochs,
        third_part_of_epochs
    ]
    lr_arrays = [
        [starting_lr] * len(first_part_of_epochs),
        [s_info.alpha * starting_lr] * len(second_part_of_epochs),
        [s_info.alpha**2 * starting_lr] * len(third_part_of_epochs),
    ]
    ax.plot(epochs, np.concatenate(lr_arrays), "o--", color="tab:blue")


def plot_cosineannealing(ax, s_info: DictConfig, epochs: NDArrayFloat, starting_lr: float) -> None:
    n_epochs = len(epochs)
    n_q = int(n_epochs // 4)
    lr_arrays = [starting_lr + 0.5 * (starting_lr - 0.5*starting_lr) * (1 + np.cos(epochs[:n_q] / n_q * np.pi))] * 4
    ax.plot(epochs, np.concatenate(lr_arrays), "o--", color="tab:blue")


def schedulers(cfg: DictConfig) -> None:
    epochs = np.arange(40)
    for s_info in cfg.schedulers:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        plotting_func = call(s_info.plotting_func)
        plotting_func(ax=ax, s_info=s_info, epochs=epochs, starting_lr=cfg.starting_lr)
        if s_info.annotations:
            for annot_info in s_info.annotations:
                ax.annotate(
                    annot_info.text,
                    xy=annot_info.xy,
                    xytext=annot_info.xytext,
                    arrowprops=dict(facecolor='black', width=1, headwidth=10, shrink=0.05)
                )
        ax.grid()
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel(r"$\gamma$", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"scheduler_{s_info.name}.svg")


if __name__ == "__main__":
    hydra.main(
        version_base=None,
        config_path="../../config",
        config_name="config_schedulers"
    )(schedulers)()

