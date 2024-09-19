import os
import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib


WRITE_TB_LOGS = False


class DeepReLUModel(nn.Module):
    def __init__(self, hidden_layer_dim=8, n_layers=1):
        super().__init__()
        assert n_layers > 0, "Number of layers must be greater than 0"

        sequential_layers = [nn.Linear(1, hidden_layer_dim)]
        for i in range(n_layers - 1):
            hidden_layers += [
                nn.Linear(hidden_layer_dim, hidden_layer_dim),
                nn.ReLU(),
            ]
        sequential_layers.append(nn.Linear(hidden_layer_dim, 1))
        self.sequential = nn.Sequential(*sequential_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sequential(input)


class ShallowReLUModel(nn.Module):
    def __init__(self, hidden_layer_dim=8):
        super().__init__()
        self.hidden_linear = nn.Linear(1, hidden_layer_dim)
        self.hidden_activation = nn.ReLU()
        self.output_linear = nn.Linear(hidden_layer_dim, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h = self.hidden_linear(input)
        h = self.hidden_activation(h)
        h = self.output_linear(h)
        return h


class Trainer:
    def __init__(
        self,
        after_loss_clb=None,
        after_backward_clb=None,
        tb_writer=None,
        print_losses_at_epochs=True,
    ):
        self.after_loss_clb = after_loss_clb
        self.after_backward_clb = after_backward_clb
        self.tb_writer = tb_writer
        self.print_losses_at_epochs = print_losses_at_epochs

    def train(
        self,
        n_epochs: int,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        model,
        loss_fn,
        train_x,
        train_y
    ) -> None:
        for epoch in range(1, n_epochs + 1):
            pred_y = model(train_x)
            train_loss = loss_fn(pred_y, train_y)
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tb_writer.add_histogram(
                    "Weights/hidden_linear_histogram",
                    model.hidden_linear.weight.detach(),
                    epoch,
                )

            if self.after_loss_clb:
                self.after_loss_clb(train_x, pred_y, epoch)
            
            if epoch % 100 == 0:
                if self.print_losses_at_epochs:
                    print(f"Epoch #{epoch}. Training loss = {train_loss}")

                if self.tb_writer is not None:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                    ax.imshow(model.hidden_linear.weight.detach())
                    self.tb_writer.add_figure("Weights/hidden_linear_tensor", fig, epoch)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(train_loss)


class OutputCollector:
    def __init__(self, schedule_by_epoch=None):
        self.xs = []
        self.ys = []
        self.schedule_by_epoch = schedule_by_epoch 

    def __call__(self, x, y, epoch):
        if epoch in self.schedule_by_epoch:
            self.xs.append(x)
            self.ys.append(y)


def groundtruth(x):
    return x**3 - x


if __name__ == "__main__":
    n = 100
    train_x = torch.linspace(-2, 2, n).unsqueeze(1)  # append shape with 1
    train_y = groundtruth(train_x)
    # Visualization of the initial state of the network
    model = ShallowReLUModel(hidden_layer_dim=32)
    output_collector = OutputCollector(schedule_by_epoch=(1, 10, 50, 100, 1000, 10000))  # (1, 10, 50, 100, 200, 500)
    trainer = Trainer(
        after_loss_clb=output_collector, after_backward_clb=None, tb_writer=None
    )
    loss_fn = nn.MSELoss()
    lr = 1e-2
    n_epochs = 30000 
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = None
    trainer.train(
        n_epochs=n_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        loss_fn=loss_fn,
        train_x=train_x,
        train_y=train_y,
    )

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
    fig.savefig("mlp_conv_wrt_loss.svg")

    tb_writer = None
    if WRITE_TB_LOGS:
        tb_writer = SummaryWriter(
            log_dir=os.path.join(
                "tb_log_dir", datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
            )
        )
        tb_writer.add_graph(model, input)

