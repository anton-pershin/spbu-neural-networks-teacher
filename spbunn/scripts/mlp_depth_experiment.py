import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from spbunn.mlp.simple_relu_nn import (
    DeepReLUPlainModel,
    OutputCollector,
    SmoothedLossCollector,
    Trainer,
)
from spbunn.scripts.mlp_width_experiment import (
    GENERATING_FUNCTIONS,
    GeneratingFunction,
    generate_dataset,
    LR_EVAL_TAIL_FRACTION,
    LR_GAMMA,
    LR_GRID,
    LR_SEARCH_EPOCHS,
    LR_SEARCH_SEEDS,
    LOSS_EMA_BETA,
    OUTPUT_SCHEDULE_BY_EPOCH,
    STEP_SIZE_FRACTIONS,
    step_size_for,
)


HIDDEN_LAYER_DIM = 8
DEPTH_LAYER_GRID = [1, 2, 4, 8, 16, 32]
DEPTH_TO_PLOT_SEQUENCE = 32
DEPTH_CACHE_PATH = "mlp_depth_experiment_lrs.json"
FINAL_N_EPOCHS = 30000


# A width experiment with hidden width D_w has 3*D_w + 1 parameters.
# A depth experiment with width D=8 and L hidden layers has
# 3*8 + 1 + (L - 1)*(8**2 + 8) = 25 + 72*(L - 1) parameters.
# Therefore, an exactly matched shallow width is D_w = 24*L - 16.
# For example, L=1, 2, 6, and 22 match width D_w=8, 32, 128, and 512.
def parameter_count(hidden_layer_dim: int, n_layers: int) -> int:
    return (
        3 * hidden_layer_dim
        + 1
        + (n_layers - 1) * (hidden_layer_dim**2 + hidden_layer_dim)
    )


def matching_width_dim(n_layers: int, hidden_layer_dim: int = HIDDEN_LAYER_DIM) -> int:
    depth_parameter_count = parameter_count(hidden_layer_dim, n_layers)
    assert (depth_parameter_count - 1) % 3 == 0
    return (depth_parameter_count - 1) // 3


def plot_sequence_of_conv_curves_wrt_epochs(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    output_collector: OutputCollector,
    true_func_name: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(train_x.detach().numpy(), train_y.detach().numpy(), color="lightgrey", linewidth=12)
    n_outputs = len(output_collector.xs)
    for i in range(n_outputs):
        x = output_collector.xs[i]
        y = output_collector.ys[i]
        ax.plot(
            x.detach().numpy(),
            y.detach().numpy(),
            color=matplotlib.colormaps["cool"](i / n_outputs),
            linewidth=4,
            alpha=0.5,
            label=f"Epoch = {output_collector.schedule_by_epoch[i]}",
        )
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$f(x)$", fontsize=12)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"mlp_depth_conv_wrt_loss_{true_func_name}.svg")


def add_conv_graph_wrt_depth(
    ax,
    train_y: torch.Tensor,
    output_collector: OutputCollector,
    n_layers: int,
    smoothed_loss_collector: SmoothedLossCollector = None,
):
    mae_errors = torch.zeros((len(output_collector.xs),))
    for i, epoch in enumerate(output_collector.schedule_by_epoch):
        if epoch >= 10e2 and smoothed_loss_collector is not None:
            mae_errors[i] = smoothed_loss_collector.smoothed_losses[i]
        else:
            y = output_collector.ys[i].detach()
            mae_errors[i] = torch.abs(y - train_y.detach()).mean()

    ax.loglog(
        output_collector.schedule_by_epoch,
        mae_errors,
        "o--",
        linewidth=2,
        markersize=8,
        label=rf"$L = {n_layers}$",
    )


def plot_depth_graph(fig, ax, true_func_name: str):
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"mlp_depth_conv_wrt_mae_{true_func_name}.svg")


def print_depth_search_summary(results: dict) -> None:
    for func_name, depths in results.items():
        for n_layers, stats in depths.items():
            other = stats["other_losses"]
            if other["min"] is None:
                other_str = "n/a (all other configurations diverged)"
            else:
                other_str = (
                    f"{other['min']:.3e} / {other['mean']:.3e} / {other['max']:.3e}"
                )
            print(
                f"Optimal configuration for {func_name}, L = {n_layers}: "
                f"lr = {stats['lr']:.3e}, "
                f"step_size = {stats['step_size_fraction']:.2f} of search "
                f"({step_size_for(stats['step_size_fraction'], LR_SEARCH_EPOCHS)} epochs), "
                f"loss = {stats['best_loss']:.3e}, "
                f"other configurations loss (min/mean/max): {other_str}"
            )


def extract_best_hyperparameters(search_results: dict) -> dict:
    return {
        func_name: {
            int(n_layers): {
                "lr": stats["lr"],
                "step_size_fraction": stats["step_size_fraction"],
            }
            for n_layers, stats in depths.items()
        }
        for func_name, depths in search_results.items()
    }


def find_optimal_learning_rates(
    generating_functions: dict,
    depth_layer_grid: list,
    hidden_layer_dim: int,
) -> dict:
    cache_meta = {
        "hidden_layer_dim": hidden_layer_dim,
        "depth_layer_grid": depth_layer_grid,
        "lr_grid": LR_GRID,
        "lr_gamma": LR_GAMMA,
        "step_size_fractions": STEP_SIZE_FRACTIONS,
        "lr_search_epochs": LR_SEARCH_EPOCHS,
        "lr_eval_tail_fraction": LR_EVAL_TAIL_FRACTION,
        "lr_search_seeds": LR_SEARCH_SEEDS,
        "function_names": list(generating_functions.keys()),
    }
    if os.path.exists(DEPTH_CACHE_PATH):
        with open(DEPTH_CACHE_PATH, "r") as f:
            cache = json.load(f)
        if cache.get("meta") == cache_meta:
            print(f"Loaded depth hyperparameters from cache: {DEPTH_CACHE_PATH}")
            print_depth_search_summary(cache["results"])
            return extract_best_hyperparameters(cache["results"])
        print(f"Cache {DEPTH_CACHE_PATH} is outdated, re-running the depth search")

    loss_fn = nn.MSELoss()
    n_tail = max(1, int(LR_SEARCH_EPOCHS * LR_EVAL_TAIL_FRACTION))
    scores = {}
    combos = [
        (func_name, n_layers, lr, step_size_fraction)
        for func_name in generating_functions
        for n_layers in depth_layer_grid
        for lr in LR_GRID
        for step_size_fraction in STEP_SIZE_FRACTIONS
    ]
    for func_name, n_layers, lr, step_size_fraction in tqdm(combos, desc="Depth LR search"):
        func = generating_functions[func_name]
        x, y = generate_dataset(x_lims=torch.Tensor([-2., 2.]), func=func)
        train_x = x.unsqueeze(1)
        train_y = y.unsqueeze(1)

        seed_scores = []
        for seed in LR_SEARCH_SEEDS:
            torch.manual_seed(seed)
            model = DeepReLUPlainModel(
                hidden_layer_dim=hidden_layer_dim,
                n_layers=n_layers,
            )
            losses = []
            trainer = Trainer(
                after_loss_clb=lambda _x, pred_y, _epoch: losses.append(
                    loss_fn(pred_y, train_y).item()
                ),
                after_backward_clb=None,
                tb_writer=None,
                print_losses_at_epochs=False,
            )
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size_for(step_size_fraction, LR_SEARCH_EPOCHS),
                gamma=LR_GAMMA,
            )
            trainer.train(
                n_epochs=LR_SEARCH_EPOCHS,
                optimizer=optimizer,
                scheduler=scheduler,
                model=model,
                loss_fn=loss_fn,
                train_x=train_x,
                train_y=train_y,
            )
            tail_losses = np.asarray(losses[-n_tail:])
            if not np.all(np.isfinite(tail_losses)):
                seed_scores.append(np.inf)
            else:
                seed_scores.append(tail_losses.mean())
        scores.setdefault(func_name, {}).setdefault(n_layers, {})[
            (lr, step_size_fraction)
        ] = float(np.mean(seed_scores))

    results = {}
    for func_name in generating_functions:
        assert func_name in scores, f"All configurations diverged for function: {func_name}"
        for n_layers in depth_layer_grid:
            assert n_layers in scores[func_name], (
                f"All configurations diverged for {func_name}, L = {n_layers}"
            )
            per_configuration_scores = scores[func_name][n_layers]
            finite_scores = {
                configuration: score
                for configuration, score in per_configuration_scores.items()
                if np.isfinite(score)
            }
            assert finite_scores, f"All configurations diverged for {func_name}, L = {n_layers}"
            best_configuration = min(finite_scores, key=finite_scores.get)
            best_lr, best_step_size_fraction = best_configuration
            other_losses = [
                score
                for configuration, score in finite_scores.items()
                if configuration != best_configuration
            ]
            if other_losses:
                other_losses_stats = {
                    "min": float(np.min(other_losses)),
                    "mean": float(np.mean(other_losses)),
                    "max": float(np.max(other_losses)),
                }
            else:
                other_losses_stats = {"min": None, "mean": None, "max": None}
            results.setdefault(func_name, {})[n_layers] = {
                "lr": best_lr,
                "step_size_fraction": best_step_size_fraction,
                "best_loss": finite_scores[best_configuration],
                "other_losses": other_losses_stats,
            }

    print_depth_search_summary(results)
    with open(DEPTH_CACHE_PATH, "w") as f:
        json.dump({"meta": cache_meta, "results": results}, f, indent=4)
    print(f"Saved depth hyperparameters to: {DEPTH_CACHE_PATH}")
    return extract_best_hyperparameters(results)


def select_best_seed_run(
    func_name: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    hidden_layer_dim: int,
    n_layers: int,
    lr: float,
    step_size_fraction: float,
    n_epochs: int,
) -> dict:
    loss_fn = nn.MSELoss()
    runs = []
    for seed in LR_SEARCH_SEEDS:
        torch.manual_seed(seed)
        model = DeepReLUPlainModel(
            hidden_layer_dim=hidden_layer_dim,
            n_layers=n_layers,
        )
        output_collector = OutputCollector(schedule_by_epoch=OUTPUT_SCHEDULE_BY_EPOCH)
        smoothed_loss_collector = SmoothedLossCollector(
            schedule_by_epoch=OUTPUT_SCHEDULE_BY_EPOCH,
            train_y=train_y,
            ema_beta=LOSS_EMA_BETA,
        )
        trainer = Trainer(
            after_loss_clb=lambda x, pred_y, epoch: (
                output_collector(x, pred_y, epoch),
                smoothed_loss_collector(x, pred_y, epoch),
            ),
            after_backward_clb=None,
            tb_writer=None,
            print_losses_at_epochs=False,
        )
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size_for(step_size_fraction, n_epochs),
            gamma=LR_GAMMA,
        )
        trainer.train(
            n_epochs=n_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            loss_fn=loss_fn,
            train_x=train_x,
            train_y=train_y,
        )
        final_loss = loss_fn(model(train_x), train_y).item()
        runs.append(
            {
                "seed": seed,
                "model": model,
                "output_collector": output_collector,
                "smoothed_loss_collector": smoothed_loss_collector,
                "final_loss": final_loss,
            }
        )
    best_run = min(runs, key=lambda run: run["final_loss"])
    losses_str = ", ".join(f"{run['seed']}: {run['final_loss']:.3e}" for run in runs)
    print(
        f"{func_name}, L = {n_layers}, final losses (seed: loss): "
        f"[{losses_str}]. Selected seed {best_run['seed']}"
    )
    return best_run


if __name__ == "__main__":
    optimal_hyperparameters = find_optimal_learning_rates(
        generating_functions=GENERATING_FUNCTIONS,
        depth_layer_grid=DEPTH_LAYER_GRID,
        hidden_layer_dim=HIDDEN_LAYER_DIM,
    )

    for func_name, func in GENERATING_FUNCTIONS.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for n_layers in tqdm(DEPTH_LAYER_GRID, desc="Iteration over depths"):
            x, y = generate_dataset(x_lims=torch.Tensor([-2., 2.]), func=func)
            train_x = x.unsqueeze(1)
            train_y = y.unsqueeze(1)
            hyperparameters = optimal_hyperparameters[func_name][n_layers]
            best_run = select_best_seed_run(
                func_name=func_name,
                train_x=train_x,
                train_y=train_y,
                hidden_layer_dim=HIDDEN_LAYER_DIM,
                n_layers=n_layers,
                lr=hyperparameters["lr"],
                step_size_fraction=hyperparameters["step_size_fraction"],
                n_epochs=FINAL_N_EPOCHS,
            )
            add_conv_graph_wrt_depth(
                ax=ax,
                train_y=train_y,
                output_collector=best_run["output_collector"],
                n_layers=n_layers,
                smoothed_loss_collector=best_run["smoothed_loss_collector"],
            )
            if n_layers == DEPTH_TO_PLOT_SEQUENCE:
                plot_sequence_of_conv_curves_wrt_epochs(
                    train_x=train_x,
                    train_y=train_y,
                    output_collector=best_run["output_collector"],
                    true_func_name=func_name,
                )
        plot_depth_graph(fig=fig, ax=ax, true_func_name=func_name)
