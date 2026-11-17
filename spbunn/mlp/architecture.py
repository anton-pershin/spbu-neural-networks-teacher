from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from spbunn.mlp.simple_relu_nn import DeepReLUPlainModel, ShallowReLUModel

ArchitectureKind = Literal["width", "depth"]
GeneratingFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrainingSettings:
    lr_grid: tuple[float, ...]
    lr_gamma: float
    step_size_fractions: tuple[float, ...]
    search_epochs: int
    eval_tail_fraction: float
    seeds: tuple[int, ...]
    output_schedule: tuple[int, ...]
    ema_beta: float
    smoothed_loss_min_epoch: int
    final_epochs: int


@dataclass(frozen=True)
class ArchitectureCase:
    function_name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    hidden_layer_dim: int
    n_layers: int
    architecture_size: int


@dataclass(frozen=True)
class ArchitectureExperimentInput:
    kind: ArchitectureKind
    cases: tuple[ArchitectureCase, ...]
    settings: TrainingSettings
    sequence_size: int


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    step_size_fraction: float


@dataclass(frozen=True)
class OtherLossStats:
    minimum: float | None
    mean: float | None
    maximum: float | None


@dataclass(frozen=True)
class ArchitectureResult:
    case: ArchitectureCase
    hyperparameters: Hyperparameters
    search_loss: float
    other_losses: OtherLossStats
    selected_seed: int
    seed_final_losses: tuple[tuple[int, float], ...]
    output_epochs: tuple[int, ...]
    output_xs: tuple[torch.Tensor, ...]
    predictions: tuple[torch.Tensor, ...]
    smoothed_losses: tuple[float, ...]
    mae_errors: torch.Tensor


@dataclass(frozen=True)
class ArchitectureExperimentResult:
    kind: ArchitectureKind
    settings: TrainingSettings
    sequence_size: int
    results: tuple[ArchitectureResult, ...]


@dataclass(frozen=True)
class _SearchOutcome:
    hyperparameters: Hyperparameters
    best_loss: float
    other_losses: OtherLossStats


@dataclass(frozen=True)
class _TrainingRun:
    losses: tuple[float, ...]
    final_loss: float | None
    output_xs: tuple[torch.Tensor, ...]
    predictions: tuple[torch.Tensor, ...]
    direct_mae: tuple[torch.Tensor, ...]
    smoothed_losses: tuple[float, ...]


def cubic_function(x: torch.Tensor) -> torch.Tensor:
    return x**3 - x


def gaussian_function(
    x: torch.Tensor, mu: float = 0.0, sigma_sq: float = 0.1
) -> torch.Tensor:
    return torch.exp(-((x - mu) ** 2) / (2.0 * sigma_sq))


def discontinuous_function(x: torch.Tensor) -> torch.Tensor:
    mask = x > 0
    y = torch.zeros_like(x)
    y[mask] = x[mask] + 5.0
    y[~mask] = cubic_function(x[~mask])
    return y


def periodic_function(
    x: torch.Tensor,
    amps: torch.Tensor | None = None,
    freqs: torch.Tensor | None = None,
) -> torch.Tensor:
    if amps is None:
        amps = torch.tensor([1.0, 1.0, 1.0])
    if freqs is None:
        freqs = torch.tensor([2.0, 8.0, 16.0])
    if amps.ndim != 1 or freqs.ndim != 1 or amps.shape[0] != freqs.shape[0]:
        raise ValueError("Amplitudes and frequencies must be equally sized 1D tensors")
    y = torch.zeros_like(x)
    for index, (amplitude, frequency) in enumerate(zip(amps, freqs, strict=True)):
        if index % 2 == 0:
            y = y + amplitude * torch.sin(frequency * x)
        else:
            y = y + amplitude * torch.cos(frequency * x)
    return y


GENERATING_FUNCTIONS: dict[str, GeneratingFunction] = {
    "cubic": cubic_function,
    "gaussian": gaussian_function,
    "discontinuous": discontinuous_function,
    "periodic": periodic_function,
}


def generate_dataset(
    x_lims: tuple[float, float], func: GeneratingFunction, n_samples: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the legacy ordered, noiseless function-regression data."""
    if n_samples <= 0:
        raise ValueError("The regression sample count must be positive")
    lower, upper = x_lims
    if not lower < upper:
        raise ValueError("The regression interval must have increasing endpoints")
    x = torch.linspace(lower, upper, n_samples)
    return x, func(x)


def step_size_for(fraction: float, n_epochs: int) -> int:
    return max(1, round(fraction * n_epochs))


def parameter_count(hidden_layer_dim: int, n_layers: int) -> int:
    if hidden_layer_dim <= 0 or n_layers <= 0:
        raise ValueError("Width and depth must be positive")
    return (
        3 * hidden_layer_dim
        + 1
        + (n_layers - 1) * (hidden_layer_dim**2 + hidden_layer_dim)
    )


def matching_width_dim(n_layers: int, hidden_layer_dim: int = 8) -> int:
    depth_parameter_count = parameter_count(hidden_layer_dim, n_layers)
    if (depth_parameter_count - 1) % 3:
        raise ValueError("The parameter count does not define an integer width")
    return (depth_parameter_count - 1) // 3


def _positive_ints(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must be a non-empty sequence of positive integers")
    return result


def _training_settings(cfg: DictConfig) -> TrainingSettings:
    lr_grid = tuple(float(value) for value in cfg.lr_grid)
    fractions = tuple(float(value) for value in cfg.step_size_fractions)
    seeds = tuple(int(value) for value in cfg.lr_search_seeds)
    schedule = _positive_ints(cfg.output_schedule_by_epoch, "The output schedule")
    search_epochs = int(cfg.lr_search_epochs)
    final_epochs = int(cfg.final_n_epochs)
    tail_fraction = float(cfg.lr_eval_tail_fraction)
    ema_beta = float(cfg.loss_ema_beta)
    smoothed_min_epoch = int(cfg.smoothed_loss_min_epoch)
    if not lr_grid or any(not np.isfinite(value) or value < 0 for value in lr_grid):
        raise ValueError(
            "The learning-rate grid must contain finite non-negative values"
        )
    if float(cfg.lr_gamma) <= 0:
        raise ValueError("The learning-rate decay must be positive")
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("Step-size fractions must be in the interval (0, 1]")
    if search_epochs <= 0 or final_epochs <= 0:
        raise ValueError("Search and final epoch counts must be positive")
    if not 0 < tail_fraction <= 1:
        raise ValueError("The evaluation tail fraction must be in the interval (0, 1]")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("The seed sequence must contain non-negative integers")
    if len(set(schedule)) != len(schedule) or tuple(sorted(schedule)) != schedule:
        raise ValueError("The output schedule must be strictly increasing")
    if schedule[-1] > final_epochs:
        raise ValueError("The output schedule cannot exceed the final epoch count")
    if not 0 <= ema_beta < 1:
        raise ValueError("The MAE EMA beta must be in the interval [0, 1)")
    if smoothed_min_epoch <= 0:
        raise ValueError("The smoothed-loss threshold must be positive")
    return TrainingSettings(
        lr_grid=lr_grid,
        lr_gamma=float(cfg.lr_gamma),
        step_size_fractions=fractions,
        search_epochs=search_epochs,
        eval_tail_fraction=tail_fraction,
        seeds=seeds,
        output_schedule=schedule,
        ema_beta=ema_beta,
        smoothed_loss_min_epoch=smoothed_min_epoch,
        final_epochs=final_epochs,
    )


def _regression_datasets(
    cfg: DictConfig,
) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    function_names = tuple(str(name) for name in cfg.function_names)
    if not function_names:
        raise ValueError("At least one generating function must be selected")
    unknown = set(function_names) - set(GENERATING_FUNCTIONS)
    if unknown:
        raise ValueError(f"Unsupported generating functions: {sorted(unknown)}")
    n_samples = int(cfg.n_samples)
    limits = tuple(float(value) for value in cfg.x_lims)
    if len(limits) != 2:
        raise ValueError("The regression interval must contain two endpoints")
    x_lims = (limits[0], limits[1])
    datasets = []
    for function_name in function_names:
        x, y = generate_dataset(
            x_lims,
            GENERATING_FUNCTIONS[function_name],
            n_samples=n_samples,
        )
        datasets.append((function_name, x.unsqueeze(1), y.unsqueeze(1)))
    return tuple(datasets)


def produce_width_experiment(cfg: DictConfig) -> ArchitectureExperimentInput:
    """Produce function data and shallow-width sweep settings."""
    widths = _positive_ints(cfg.hidden_layer_dims, "The width sweep")
    sequence_size = int(cfg.sequence_hidden_layer_dim)
    if sequence_size not in widths:
        raise ValueError("The prediction-sequence width must belong to the width sweep")
    datasets = _regression_datasets(cfg)
    # Function-major, width-minor ordering matches the legacy experiment.
    cases = tuple(
        ArchitectureCase(
            function_name=function_name,
            train_x=train_x,
            train_y=train_y,
            hidden_layer_dim=width,
            n_layers=1,
            architecture_size=width,
        )
        for function_name, train_x, train_y in datasets
        for width in widths
    )
    return ArchitectureExperimentInput(
        kind="width",
        cases=cases,
        settings=_training_settings(cfg),
        sequence_size=sequence_size,
    )


def produce_depth_experiment(cfg: DictConfig) -> ArchitectureExperimentInput:
    """Produce function data and plain-deep-network sweep settings."""
    depths = _positive_ints(cfg.depth_layer_grid, "The depth sweep")
    hidden_layer_dim = int(cfg.hidden_layer_dim)
    if hidden_layer_dim <= 0:
        raise ValueError("The depth experiment's hidden width must be positive")
    sequence_size = int(cfg.depth_to_plot_sequence)
    if sequence_size not in depths:
        raise ValueError("The prediction-sequence depth must belong to the depth sweep")
    datasets = _regression_datasets(cfg)
    cases = tuple(
        ArchitectureCase(
            function_name=function_name,
            train_x=train_x,
            train_y=train_y,
            hidden_layer_dim=hidden_layer_dim,
            n_layers=depth,
            architecture_size=depth,
        )
        for function_name, train_x, train_y in datasets
        for depth in depths
    )
    return ArchitectureExperimentInput(
        kind="depth",
        cases=cases,
        settings=_training_settings(cfg),
        sequence_size=sequence_size,
    )


def build_architecture_model(
    kind: ArchitectureKind, hidden_layer_dim: int, n_layers: int = 1
) -> nn.Module:
    if hidden_layer_dim <= 0 or n_layers <= 0:
        raise ValueError("Model width and depth must be positive")
    if kind == "width":
        return ShallowReLUModel(hidden_layer_dim=hidden_layer_dim)
    if kind == "depth":
        return DeepReLUPlainModel(hidden_layer_dim=hidden_layer_dim, n_layers=n_layers)
    raise ValueError(f"Unsupported architecture kind: {kind}")


def _train_run(
    experiment: ArchitectureExperimentInput,
    case: ArchitectureCase,
    hyperparameters: Hyperparameters,
    seed: int,
    n_epochs: int,
    collect_outputs: bool,
) -> _TrainingRun:
    settings = experiment.settings
    torch.manual_seed(seed)
    model = build_architecture_model(
        experiment.kind,
        hidden_layer_dim=case.hidden_layer_dim,
        n_layers=case.n_layers,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size_for(hyperparameters.step_size_fraction, n_epochs),
        gamma=settings.lr_gamma,
    )
    losses = []
    output_xs = []
    predictions = []
    direct_mae = []
    smoothed_losses = []
    ema: torch.Tensor | None = None
    for epoch in range(1, n_epochs + 1):
        prediction = model(case.train_x)
        loss = loss_fn(prediction, case.train_y)
        losses.append(loss.item())
        if collect_outputs:
            mae = torch.abs(prediction.detach() - case.train_y).mean()
            ema = (
                mae
                if ema is None
                else settings.ema_beta * ema + (1.0 - settings.ema_beta) * mae
            )
            if epoch in settings.output_schedule:
                output_xs.append(case.train_x.detach().clone())
                predictions.append(prediction.detach().clone())
                direct_mae.append(mae.detach().clone())
                smoothed_losses.append(ema.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    final_loss = None
    if collect_outputs:
        final_loss = loss_fn(model(case.train_x), case.train_y).item()
    return _TrainingRun(
        losses=tuple(losses),
        final_loss=final_loss,
        output_xs=tuple(output_xs),
        predictions=tuple(predictions),
        direct_mae=tuple(direct_mae),
        smoothed_losses=tuple(smoothed_losses),
    )


def _search_hyperparameters(
    experiment: ArchitectureExperimentInput, case: ArchitectureCase
) -> _SearchOutcome:
    settings = experiment.settings
    n_tail = max(1, int(settings.search_epochs * settings.eval_tail_fraction))
    scores: dict[Hyperparameters, float] = {}
    for learning_rate in settings.lr_grid:
        for step_size_fraction in settings.step_size_fractions:
            hyperparameters = Hyperparameters(learning_rate, step_size_fraction)
            seed_scores = []
            for seed in settings.seeds:
                run = _train_run(
                    experiment,
                    case,
                    hyperparameters,
                    seed,
                    settings.search_epochs,
                    collect_outputs=False,
                )
                tail_losses = np.asarray(run.losses[-n_tail:])
                seed_scores.append(
                    float(tail_losses.mean())
                    if np.all(np.isfinite(tail_losses))
                    else np.inf
                )
            scores[hyperparameters] = float(np.mean(seed_scores))
    finite_scores = {
        hyperparameters: score
        for hyperparameters, score in scores.items()
        if np.isfinite(score)
    }
    if not finite_scores:
        raise ValueError(
            f"All configurations diverged for {case.function_name}, "
            f"size {case.architecture_size}"
        )
    best = min(finite_scores, key=lambda item: finite_scores[item])
    other_losses = [score for item, score in finite_scores.items() if item != best]
    stats = OtherLossStats(
        minimum=float(np.min(other_losses)) if other_losses else None,
        mean=float(np.mean(other_losses)) if other_losses else None,
        maximum=float(np.max(other_losses)) if other_losses else None,
    )
    return _SearchOutcome(best, finite_scores[best], stats)


def _select_final_run(
    experiment: ArchitectureExperimentInput,
    case: ArchitectureCase,
    search: _SearchOutcome,
) -> ArchitectureResult:
    runs = [
        _train_run(
            experiment,
            case,
            search.hyperparameters,
            seed,
            experiment.settings.final_epochs,
            collect_outputs=True,
        )
        for seed in experiment.settings.seeds
    ]
    final_losses = tuple(
        (seed, run.final_loss)
        for seed, run in zip(experiment.settings.seeds, runs, strict=True)
        if run.final_loss is not None
    )
    best_index = min(range(len(runs)), key=lambda index: final_losses[index][1])
    best_run = runs[best_index]
    mae_errors = torch.zeros(len(best_run.predictions))
    for index, epoch in enumerate(experiment.settings.output_schedule):
        if epoch >= experiment.settings.smoothed_loss_min_epoch:
            mae_errors[index] = best_run.smoothed_losses[index]
        else:
            mae_errors[index] = best_run.direct_mae[index]
    return ArchitectureResult(
        case=case,
        hyperparameters=search.hyperparameters,
        search_loss=search.best_loss,
        other_losses=search.other_losses,
        selected_seed=final_losses[best_index][0],
        seed_final_losses=final_losses,
        output_epochs=experiment.settings.output_schedule,
        output_xs=best_run.output_xs,
        predictions=best_run.predictions,
        smoothed_losses=best_run.smoothed_losses,
        mae_errors=mae_errors,
    )


def evaluate_architecture_experiment(
    experiment: ArchitectureExperimentInput,
) -> ArchitectureExperimentResult:
    """Search hyperparameters, train every seed, and retain the best run."""
    if not experiment.cases:
        raise ValueError("The architecture experiment needs at least one case")
    search_results = [
        _search_hyperparameters(experiment, case) for case in experiment.cases
    ]
    results = tuple(
        _select_final_run(experiment, case, search)
        for case, search in zip(experiment.cases, search_results, strict=True)
    )
    return ArchitectureExperimentResult(
        kind=experiment.kind,
        settings=experiment.settings,
        sequence_size=experiment.sequence_size,
        results=results,
    )
