from typing import Union, Protocol

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

# 32 neurons
#LEARNING_RATES = {
#    "cubic": 1e-2,
#    "gaussian": 1e-1,
#    "periodic": 1e-1,  # try 1e-1 because 1e-2 and 1e-3 seem to converge too slowly seems to be too large
#    "discontinuous": 1e-1,
#}


# 1024 neurons
LEARNING_RATES = {
    "cubic": 1e-2,
    "gaussian": 1e-2,
    "periodic": 1e-2,  # try 1e-1 because 1e-2 and 1e-3 seem to converge too slowly seems to be too large
    "discontinuous": 1e-2,
}


if __name__ == "__main__":
    hidden_layer_dims = [8, 16, 32, 64, 128, 256, 512, 1024]
    # hidden_layer_dims = [1024]
    hidden_layer_dim_to_plot_sequence_of_conv_curves = 1024
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
            
            # Train the model and collect outputs on the training dataset at the same time
            model = ShallowReLUModel(hidden_layer_dim=d)
            output_collector = OutputCollector(schedule_by_epoch=(1, 10, 50, 100, 1000, 10000))  # (1, 10, 50, 100, 200, 500)
            trainer = Trainer(
                after_loss_clb=output_collector,
                after_backward_clb=None,
                tb_writer=None,
                print_losses_at_epochs=False,
            )
            loss_fn = nn.MSELoss()
            lr = LEARNING_RATES[func_name]
            n_epochs = 30000 
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

