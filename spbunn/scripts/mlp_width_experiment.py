from typing import Union, Protocol
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from spbunn.mlp.simple_relu_nn import ShallowReLUModel, OutputCollector, Trainer

class GeneratingFunction(Protocol):
    def __call__(x: torch.Tensor) -> torch.Tensor:
        ...


def cubic_function(x: torch.Tensor) -> torch.Tensor:
    return x**3 -x


def gaussian_function(x: torch.Tensor, mu=0., sigma_sq=0.1) -> torch.Tensor:
    return torch.exp(-(x - mu)**2 / (2. * sigma_sq))


def discontinuous_function(x: torch.Tensor) -> torch.Tensor:
    mask = x > 0
    y = torch.zeros_like(x)
    y[mask] = x[mask] + 5.
    y[~mask] = cubic_function(x[~mask])
    return y


def periodic_function(
        x: torch.Tensor,
        amps: torch.Tensor = torch.Tensor([1., 1., 1.]),
        freqs: torch.Tensor = torch.Tensor([2., 8., 16.])
    ) -> torch.Tensor:
    assert len(amps.shape) == 1 and len(freqs.shape) == 1, "Amplitude and frequency shapes must be 1D"
    assert amps.shape[0] == freqs.shape[0], "Amplitude and frequency shapes must coincide"
    
    y = torch.zeros_like(x)
    for i in range(len(amps)):
        if i % 2 == 0:
            y += amps[i] * torch.sin(freqs[i] * x)
        else:
            y += amps[i] * torch.cos(freqs[i] * x)
    return y


def generate_dataset(x_lims: torch.Tensor, func: GeneratingFunction) -> tuple[torch.Tensor, torch.Tensor]:
    n = 100
    x = torch.linspace(x_lims[0], x_lims[1], n)
    y = func(x)
    return x, y
    

def plot_sequence_of_conv_curves_wrt_epochs(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    output_collector: OutputCollector,
    true_func_name: str
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
            label=f"Epoch = {output_collector.schedule_by_epoch[i]}"
        )
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$f(x)$", fontsize=12)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"mlp_conv_wrt_loss_{true_func_name}.svg")


def add_conv_graph_wrt_hidden_layer_dimension(
    ax, 
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    output_collector: OutputCollector,
    true_func_name: str,
    hidden_layer_dimension: int,
):
    n_outputs = len(output_collector.xs)
    mae_errors = torch.zeros((n_outputs,))
    for i in range(n_outputs):
        x = output_collector.xs[i]
        y = output_collector.ys[i]
        mae_errors[i] = torch.abs(y.detach() - train_y.detach()).mean()
    ax.loglog(
        output_collector.schedule_by_epoch,
        mae_errors,
        "o--",
        linewidth=2,
        markersize=8,
        label=r"$D = " + str(hidden_layer_dimension) + r"$"
    )


def plot_conv_graph_wrt_hidden_layer_dimension(
    fig,
    ax,
    true_func_name: str
):
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"mlp_conv_wrt_mae_{true_func_name}.svg")


GENERATING_FUNCTIONS = {
    "cubic": cubic_function,
    "gaussian": gaussian_function,
    "discontinuous": discontinuous_function,
    "periodic": periodic_function,
}

LR_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
LR_SEARCH_EPOCHS = 3000
LR_EVAL_TAIL_FRACTION = 0.1
LR_SEARCH_SEEDS = [42, 43, 44]
LR_CACHE_PATH = "mlp_width_experiment_lrs.json"


def print_lr_search_summary(results: dict) -> None:
    for func_name, dims in results.items():
        for d, stats in dims.items():
            other = stats["other_losses"]
            if other["min"] is None:
                other_str = "n/a (all other LRs diverged)"
            else:
                other_str = (
                    f"{other['min']:.3e} / {other['mean']:.3e} / {other['max']:.3e}"
                )
            print(
                f"Optimal LR for {func_name}, D = {d}: "
                f"lr = {stats['lr']:.3e}, loss = {stats['best_loss']:.3e}, "
                f"other LRs loss (min/mean/max): {other_str}"
            )


def extract_lrs(lr_search_results: dict) -> dict:
    return {
        func_name: {int(d): stats["lr"] for d, stats in dims.items()}
        for func_name, dims in lr_search_results.items()
    }


def find_optimal_learning_rates(
    generating_functions: dict,
    hidden_layer_dims: list,
) -> dict:
    cache_meta = {
        "lr_grid": LR_GRID,
        "lr_search_epochs": LR_SEARCH_EPOCHS,
        "lr_eval_tail_fraction": LR_EVAL_TAIL_FRACTION,
        "lr_search_seeds": LR_SEARCH_SEEDS,
        "hidden_layer_dims": hidden_layer_dims,
        "function_names": list(generating_functions.keys()),
    }
    if os.path.exists(LR_CACHE_PATH):
        with open(LR_CACHE_PATH, "r") as f:
            cache = json.load(f)
        if cache.get("meta") == cache_meta:
            print(f"Loaded optimal learning rates from cache: {LR_CACHE_PATH}")
            print_lr_search_summary(cache["results"])
            return extract_lrs(cache["results"])
        print(f"Cache {LR_CACHE_PATH} is outdated, re-running the LR search")

    loss_fn = nn.MSELoss()
    n_tail = max(1, int(LR_SEARCH_EPOCHS * LR_EVAL_TAIL_FRACTION))
    scores = {}
    combos = [
        (func_name, d, lr)
        for func_name in generating_functions
        for d in hidden_layer_dims
        for lr in LR_GRID
    ]
    for func_name, d, lr in tqdm(combos, desc="LR search"):
        func = generating_functions[func_name]
        x, y = generate_dataset(x_lims=torch.Tensor([-2., 2.]), func=func)
        train_x = x.unsqueeze(1)
        train_y = y.unsqueeze(1)

        seed_scores = []
        for seed in LR_SEARCH_SEEDS:
            torch.manual_seed(seed)
            model = ShallowReLUModel(hidden_layer_dim=d)
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
            trainer.train(
                n_epochs=LR_SEARCH_EPOCHS,
                optimizer=optimizer,
                scheduler=None,
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
        scores.setdefault(func_name, {}).setdefault(d, {})[lr] = float(
            np.mean(seed_scores)
        )

    results = {}
    for func_name in generating_functions:
        assert func_name in scores, f"All LRs diverged for function: {func_name}"
        for d in hidden_layer_dims:
            assert d in scores[func_name], f"All LRs diverged for {func_name}, D = {d}"
            per_lr_scores = scores[func_name][d]
            finite_scores = {
                lr: score for lr, score in per_lr_scores.items() if np.isfinite(score)
            }
            best_lr = min(finite_scores, key=finite_scores.get)
            other_losses = [
                score for lr, score in finite_scores.items() if lr != best_lr
            ]
            if other_losses:
                other_losses_stats = {
                    "min": float(np.min(other_losses)),
                    "mean": float(np.mean(other_losses)),
                    "max": float(np.max(other_losses)),
                }
            else:
                other_losses_stats = {"min": None, "mean": None, "max": None}
            results.setdefault(func_name, {})[d] = {
                "lr": best_lr,
                "best_loss": finite_scores[best_lr],
                "other_losses": other_losses_stats,
            }

    print_lr_search_summary(results)

    with open(LR_CACHE_PATH, "w") as f:
        json.dump({"meta": cache_meta, "results": results}, f, indent=4)
    print(f"Saved optimal learning rates to: {LR_CACHE_PATH}")
    return extract_lrs(results)


def select_best_seed_run(
    func_name: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    hidden_layer_dim: int,
    lr: float,
    n_epochs: int,
) -> dict:
    loss_fn = nn.MSELoss()
    runs = []
    for seed in LR_SEARCH_SEEDS:
        torch.manual_seed(seed)
        model = ShallowReLUModel(hidden_layer_dim=hidden_layer_dim)
        output_collector = OutputCollector(
            schedule_by_epoch=(1, 10, 50, 100, 1000, 10000)
        )  # (1, 10, 50, 100, 200, 500)
        trainer = Trainer(
            after_loss_clb=output_collector,
            after_backward_clb=None,
            tb_writer=None,
            print_losses_at_epochs=False,
        )
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)
        #scheduler = None
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
                "final_loss": final_loss,
            }
        )
    best_run = min(runs, key=lambda run: run["final_loss"])
    losses_str = ", ".join(f"{run['seed']}: {run['final_loss']:.3e}" for run in runs)
    print(
        f"{func_name}, D = {hidden_layer_dim}, final losses (seed: loss): "
        f"[{losses_str}]. Selected seed {best_run['seed']}"
    )
    return best_run


if __name__ == "__main__":
    hidden_layer_dims = [8, 16, 32, 64, 128, 256, 512, 1024]
    # hidden_layer_dims = [1024]
    hidden_layer_dim_to_plot_sequence_of_conv_curves = 1024
    LEARNING_RATES = find_optimal_learning_rates(
        generating_functions=GENERATING_FUNCTIONS,
        hidden_layer_dims=hidden_layer_dims,
    )

    for func_name, func in GENERATING_FUNCTIONS.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for d in tqdm(hidden_layer_dims, desc="Iteration over hidden layer dimensions"):
            #  Build the training dataset only
            x, y = generate_dataset(
                x_lims=torch.Tensor([-2., 2.]),
                func=func
            )
            train_x = x.unsqueeze(1)
            train_y = y.unsqueeze(1)

            # Train the model with several seeds and keep the best run
            lr = LEARNING_RATES[func_name][d]
            n_epochs = 30000
            best_run = select_best_seed_run(
                func_name=func_name,
                train_x=train_x,
                train_y=train_y,
                hidden_layer_dim=d,
                lr=lr,
                n_epochs=n_epochs,
            )
            output_collector = best_run["output_collector"]

            add_conv_graph_wrt_hidden_layer_dimension(
                ax=ax,
                train_x=train_x,
                train_y=train_y,
                output_collector=output_collector,
                true_func_name=func_name,
                hidden_layer_dimension=d, 
            )

            if d == hidden_layer_dim_to_plot_sequence_of_conv_curves:
                # Plot outputs at selected epochs to show the convergence
                plot_sequence_of_conv_curves_wrt_epochs(
                    train_x=train_x,
                    train_y=train_y,
                    output_collector=output_collector,
                    true_func_name=func_name,
                )

        plot_conv_graph_wrt_hidden_layer_dimension(
            fig=fig,
            ax=ax,
            true_func_name=func_name,
        )

