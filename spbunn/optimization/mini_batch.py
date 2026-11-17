from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader

from spbunn.optimization.landscape import (
    LandscapeInput,
    evaluate_landscape,
    produce_landscape_inputs,
)
from spbunn.optimization.objectives import SamplingFromFunctionDataset

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MiniBatchInput:
    landscape: LandscapeInput
    training_dataset: SamplingFromFunctionDataset
    batch_size: int
    n_epochs: int
    n_optimization_steps: int


@dataclass(frozen=True)
class MiniBatchResult:
    experiment: MiniBatchInput
    loss_values: FloatArray
    theta_history: FloatArray
    loss_history: FloatArray


def produce_mini_batch_inputs(cfg: DictConfig) -> tuple[MiniBatchInput, ...]:
    """Create seeded training data and separate ordered landscape inputs."""
    batch_size = int(cfg.batch_size)
    n_epochs = int(cfg.n_epochs)
    n_optimization_steps = int(cfg.n_optimization_steps)
    seed = int(cfg.seed)
    if batch_size <= 0:
        raise ValueError("The mini-batch size must be positive")
    if n_epochs <= 0:
        raise ValueError("The epoch count must be positive")
    if n_optimization_steps <= 0:
        raise ValueError("The update count must be positive")

    landscape_inputs = produce_landscape_inputs(cfg, include_optimizers=False)
    experiments = []
    for landscape_input in landscape_inputs:
        f_true = landscape_input.func_factory(
            theta_0=landscape_input.theta_0_true,
            theta_1=landscape_input.theta_1_true,
        )
        training_dataset = SamplingFromFunctionDataset(
            f=f_true,
            n=int(cfg.n_samples),
            rng=np.random.default_rng(seed=seed),
            x_min=float(cfg.get("training_x_min", -10.0)),
            x_max=float(cfg.get("training_x_max", 10.0)),
        )
        experiments.append(
            MiniBatchInput(
                landscape=landscape_input,
                training_dataset=training_dataset,
                batch_size=batch_size,
                n_epochs=n_epochs,
                n_optimization_steps=n_optimization_steps,
            )
        )
    return tuple(experiments)


def evaluate_mini_batch(experiment: MiniBatchInput) -> MiniBatchResult:
    """Run exact-step mini-batch SGD and compute its loss landscape."""
    landscape_input = experiment.landscape
    f_pred = landscape_input.func_factory(
        theta_0=landscape_input.theta_0_initial,
        theta_1=landscape_input.theta_1_initial,
        requires_grad=True,
    )
    optimizer = torch.optim.SGD(
        [f_pred.theta_0, f_pred.theta_1], lr=landscape_input.learning_rate
    )
    data_loader = DataLoader(
        experiment.training_dataset,
        batch_size=experiment.batch_size,
        shuffle=False,
    )
    theta_history = [[landscape_input.theta_0_initial, landscape_input.theta_1_initial]]
    loss_history = []
    completed_steps = 0
    stop = False
    for _ in range(experiment.n_epochs):
        for x_batch, y_batch in data_loader:
            if completed_steps >= experiment.n_optimization_steps:
                stop = True
                break
            loss = F.mse_loss(f_pred(x_batch), y_batch)
            loss_history.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            theta_history.append([f_pred.theta_0.item(), f_pred.theta_1.item()])
            completed_steps += 1
        if stop:
            break
    if completed_steps != experiment.n_optimization_steps:
        raise ValueError(
            "The configured epochs do not provide enough batches for the update count"
        )

    return MiniBatchResult(
        experiment=experiment,
        loss_values=evaluate_landscape(landscape_input),
        theta_history=np.asarray(theta_history, dtype=np.float64),
        loss_history=np.asarray(loss_history, dtype=np.float64),
    )
