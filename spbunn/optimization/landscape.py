from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import torch
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader

from spbunn.optimization.objectives import SamplingFromFunctionDataset

FloatArray = NDArray[np.float64]


class ParametricFunc(Protocol):
    theta_0: torch.Tensor
    theta_1: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


FuncFactory = Callable[..., ParametricFunc]


@dataclass(frozen=True)
class RegressionData:
    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class OptimizerSettings:
    name: str
    suffix: str
    optimizer_name: str
    learning_rate: float
    momentum: float = 0.0
    nesterov: bool = False
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True)
class LandscapeInput:
    name: str
    func_factory: FuncFactory
    theta_0_true: float
    theta_1_true: float
    data: RegressionData
    theta_0_values: FloatArray
    theta_1_values: FloatArray
    levels: tuple[float, ...]
    theta_0_initial: float
    theta_1_initial: float
    learning_rate: float
    full_batch_steps: int
    optimizers: tuple[OptimizerSettings, ...]


@dataclass(frozen=True)
class OptimizationRunResult:
    settings: OptimizerSettings
    trajectory_theta_0: FloatArray
    trajectory_theta_1: FloatArray
    losses: FloatArray


@dataclass(frozen=True)
class LandscapeResult:
    experiment: LandscapeInput
    loss_values: FloatArray
    runs: tuple[OptimizationRunResult, ...]


def _optimizer_settings(func_info: DictConfig) -> tuple[OptimizerSettings, ...]:
    learning_rate = float(func_info.lr)
    momentum = float(func_info.get("momentum", 0.9))
    return (
        OptimizerSettings("SGD", "sgd", "sgd", learning_rate),
        OptimizerSettings(
            "SGD + momentum",
            "sgd_momentum",
            "sgd",
            learning_rate,
            momentum=momentum,
        ),
        OptimizerSettings(
            "SGD + Nesterov momentum",
            "sgd_nesterov",
            "sgd",
            learning_rate,
            momentum=momentum,
            nesterov=True,
        ),
        OptimizerSettings("RMSprop", "rmsprop", "rmsprop", learning_rate),
        OptimizerSettings(
            "Adam",
            "adam",
            "adam",
            float(func_info.get("adam_lr", learning_rate)),
        ),
    )


def produce_landscape_inputs(
    cfg: DictConfig, include_optimizers: bool = True
) -> tuple[LandscapeInput, ...]:
    """Create deterministic regression data, grids, and optimizer settings."""
    n_samples = int(cfg.n_samples)
    grid_size = int(cfg.grid_size)
    full_batch_steps = int(cfg.full_batch_steps)
    if n_samples < 2:
        raise ValueError("The sample count must be at least two")
    if grid_size < 2:
        raise ValueError("The landscape grid size must be at least two")
    if full_batch_steps <= 0:
        raise ValueError("The full-batch step count must be positive")

    experiments = []
    for func_info in cfg.functions:
        func_factory = cast(FuncFactory, instantiate(func_info.func))
        f_true = func_factory(
            theta_0=float(func_info.theta_0_true),
            theta_1=float(func_info.theta_1_true),
        )
        dataset = SamplingFromFunctionDataset(
            f=f_true,
            n=n_samples,
            rng=None,
            x_min=float(func_info.get("x_min", -10.0)),
            x_max=float(func_info.get("x_max", 10.0)),
        )
        x_batch, y_batch = next(
            iter(DataLoader(dataset, batch_size=n_samples, shuffle=False))
        )
        function_steps = int(func_info.get("n_steps", full_batch_steps))
        if function_steps <= 0:
            raise ValueError(
                f"The full-batch step count for {func_info.name} must be positive"
            )
        experiments.append(
            LandscapeInput(
                name=str(func_info.name),
                func_factory=func_factory,
                theta_0_true=float(func_info.theta_0_true),
                theta_1_true=float(func_info.theta_1_true),
                data=RegressionData(
                    x=x_batch.to(dtype=torch.float64),
                    y=y_batch.to(dtype=torch.float64),
                ),
                theta_0_values=np.linspace(
                    float(func_info.theta_0_range[0]),
                    float(func_info.theta_0_range[1]),
                    grid_size,
                    dtype=np.float64,
                ),
                theta_1_values=np.linspace(
                    float(func_info.theta_1_range[0]),
                    float(func_info.theta_1_range[1]),
                    grid_size,
                    dtype=np.float64,
                ),
                levels=tuple(float(level) for level in func_info.levels),
                theta_0_initial=float(func_info.theta_0_initial_guess),
                theta_1_initial=float(func_info.theta_1_initial_guess),
                learning_rate=float(func_info.lr),
                full_batch_steps=function_steps,
                optimizers=(
                    _optimizer_settings(func_info) if include_optimizers else ()
                ),
            )
        )
    return tuple(experiments)


def evaluate_landscape(experiment: LandscapeInput) -> FloatArray:
    """Evaluate MSE on the parameter grid with no file output."""
    if experiment.theta_0_values.size == 0 or experiment.theta_1_values.size == 0:
        raise ValueError("The landscape grid must not be empty")
    loss_values = np.zeros(
        (len(experiment.theta_0_values), len(experiment.theta_1_values)),
        dtype=np.float64,
    )
    with torch.no_grad():
        for i, theta_0 in enumerate(experiment.theta_0_values):
            for j, theta_1 in enumerate(experiment.theta_1_values):
                f_pred = experiment.func_factory(
                    theta_0=float(theta_0), theta_1=float(theta_1)
                )
                loss = F.mse_loss(f_pred(experiment.data.x), experiment.data.y)
                loss_values[i, j] = loss.item()
    return loss_values


def evaluate_full_batch_trajectory(
    experiment: LandscapeInput, settings: OptimizerSettings
) -> OptimizationRunResult:
    """Run one full-batch optimizer and retain pre-update losses."""
    if experiment.full_batch_steps <= 0:
        raise ValueError("The full-batch step count must be positive")
    f_pred = experiment.func_factory(
        theta_0=experiment.theta_0_initial,
        theta_1=experiment.theta_1_initial,
        requires_grad=True,
    )
    parameters = [f_pred.theta_0, f_pred.theta_1]
    optimizer: torch.optim.Optimizer
    if settings.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=settings.learning_rate,
            momentum=settings.momentum,
            nesterov=settings.nesterov,
        )
    elif settings.optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, lr=settings.learning_rate)
    elif settings.optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            parameters, lr=settings.learning_rate, betas=settings.betas
        )
    else:
        raise ValueError(f"Unsupported optimizer: {settings.optimizer_name}")

    trajectory_theta_0 = [experiment.theta_0_initial]
    trajectory_theta_1 = [experiment.theta_1_initial]
    losses = []
    for _ in range(experiment.full_batch_steps):
        optimizer.zero_grad()
        loss = F.mse_loss(f_pred(experiment.data.x), experiment.data.y)
        loss.backward()
        optimizer.step()
        trajectory_theta_0.append(f_pred.theta_0.item())
        trajectory_theta_1.append(f_pred.theta_1.item())
        losses.append(loss.item())
    return OptimizationRunResult(
        settings=settings,
        trajectory_theta_0=np.asarray(trajectory_theta_0, dtype=np.float64),
        trajectory_theta_1=np.asarray(trajectory_theta_1, dtype=np.float64),
        losses=np.asarray(losses, dtype=np.float64),
    )


def evaluate_landscape_experiment(experiment: LandscapeInput) -> LandscapeResult:
    """Compute a landscape and all configured full-batch trajectories."""
    loss_values = evaluate_landscape(experiment)
    runs = tuple(
        evaluate_full_batch_trajectory(experiment, settings)
        for settings in experiment.optimizers
    )
    return LandscapeResult(experiment=experiment, loss_values=loss_values, runs=runs)
