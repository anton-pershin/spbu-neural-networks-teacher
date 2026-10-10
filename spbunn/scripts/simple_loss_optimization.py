import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader 

from spbunn.optimization.objectives import SamplingFromFunctionDataset
from spbunn.utils.typing import NDArrayFloat
from spbunn.scripts.loss_landscape import compute_loss_landscape, plot_loss_landscape


def optimization_loop(f_pred, optimizer, data_loader, n_epoch, total_steps_limit):
    theta_history = [[f_pred.theta_0.detach().item(), f_pred.theta_1.detach().item()]] 
    loss_history = []
    total_steps = 0
    for epoch in range(n_epoch):
        for x_batch, y_batch in data_loader:
            if total_steps > total_steps_limit:
                return theta_history, loss_history

            y_pred = f_pred(x_batch) 
            loss = F.mse_loss(y_pred, y_batch)
            loss_history.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            theta_history.append([f_pred.theta_0.detach().item(), f_pred.theta_1.detach().item()])
            total_steps += 1

    return theta_history, loss_history


def simple_loss_optimization(cfg: DictConfig):
    for func_info in cfg.functions:
        print(f"Optimizing func {func_info.name}...")
        n = 1024
        batch_size = 16  # 8, 64, 1024 (16 for lr experiment)
        f_true = instantiate(func_info.func)(
            theta_0=func_info.theta_0_true,
            theta_1=func_info.theta_1_true,
        )
        f_pred = instantiate(func_info.func)(
            theta_0=func_info.theta_0_initial_guess,
            theta_1=func_info.theta_1_initial_guess,
            requires_grad=True,
        )
        fresh_rng = np.random.default_rng(seed=42)
        dataset = SamplingFromFunctionDataset(f=f_true, n=n, rng=fresh_rng)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        #optimizer = optim.Adam([f_pred.theta_0, f_pred.theta_1], lr=6*1e-1)
        #optimizer = optim.SGD([f_pred.theta_0, f_pred.theta_1], lr=2*1e-2, momentum=0.5, dampening=0.5)
        optimizer = optim.SGD([f_pred.theta_0, f_pred.theta_1], lr=func_info.lr)
        theta_history, loss_history = optimization_loop(
            f_pred=f_pred,
            optimizer=optimizer,
            data_loader=data_loader,
            n_epoch=24,
            total_steps_limit=24,
        )
        theta_history = np.array(theta_history)
        print(f"Plotting landscape for func {func_info.name}...")
        theta_0_values = np.linspace(func_info.theta_0_range[0], func_info.theta_0_range[1], 50)
        theta_1_values = np.linspace(func_info.theta_1_range[0], func_info.theta_1_range[1], 50)
        total_loss_values = compute_loss_landscape(
            func_info=func_info,
            theta_0_values=theta_0_values,
            theta_1_values=theta_1_values,
        )
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
        )
        landscape_ax.plot(theta_history[:, 0], theta_history[:, 1], "o--", color="red")
        landscape_ax.set_xlabel(r"$\theta_1$", fontsize=18)
        landscape_ax.set_ylabel(r"$\theta_2$", fontsize=18)
        landscape_ax.tick_params(axis="both", labelsize=14)
        loss_ax.plot(np.arange(1, len(loss_history) + 1), loss_history, "o--", color="tab:blue")
        loss_ax.set_xlabel("Iteration", fontsize=18)
        loss_ax.set_ylabel("Loss", fontsize=18)
        loss_ax.tick_params(axis="both", labelsize=14)
        loss_ax.set_yscale("log")
        loss_ax.grid(True)
        plt.tight_layout()
        lr_str = f"{float(func_info.lr):.1e}"
        lr_str = (
            lr_str.replace(".0e", "e")
            .replace("e+0", "e")
            .replace("e-0", "e_")
            .replace("e+", "e")
            .replace(".", "_")
            .replace("-", "_")
        )
        plt.savefig(f"sgd_batch_{batch_size}_lr_{lr_str}_{func_info.name}.svg")


if __name__ == "__main__":
    hydra.main(
        version_base=None,
        config_path="../../config",
        config_name="config_loss_landscape"
    )(simple_loss_optimization)()
