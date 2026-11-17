from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import nn

from spbunn.mlp.simple_relu_nn import DeepReLUModel

FloatArray = NDArray[np.float64]

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}
NORM_KEYS = (
    "hidden_pre_norm",
    "grad_pre_norm",
    "grad_input_norm",
    "grad_weight_norm",
)
CONDITIONING_METRICS = (
    "empirical_sensitivity",
    "sigma_max",
    "sigma_min",
    "condition_number",
    "jacobian_frobenius",
)


@dataclass(frozen=True)
class PropagationExperimentInput:
    input_dim: int
    hidden_dim: int
    batch_size: int
    seeds: tuple[int, ...]
    activation: str
    experiments: tuple[str, ...]
    depth_grid: tuple[int, ...]
    weight_scale: float
    init_n_layers: int
    init_weight_scales: tuple[tuple[str, float], ...]
    conditioning_inputs: int
    conditioning_perturbations: int
    perturbation_norm: float


@dataclass(frozen=True)
class NormCurve:
    name: str
    label: str
    weight_scale: float
    norms: dict[str, FloatArray]


@dataclass(frozen=True)
class ConditioningVariant:
    label: str
    residual: bool
    branch_scale: float
    weight_scale: float | None


@dataclass(frozen=True)
class ConditioningResult:
    variants: tuple[ConditioningVariant, ...]
    values: dict[tuple[str, str], FloatArray]


@dataclass(frozen=True)
class PropagationExperimentResult:
    experiment: PropagationExperimentInput
    init_curves: tuple[NormCurve, ...]
    plain_conditioning: ConditioningResult | None
    residual_conditioning: ConditioningResult | None


@dataclass(frozen=True)
class BalduzziInput:
    seed: int
    hidden_layer_dim: int
    n_layers: int
    residual_branch_scale: float
    plain_network: nn.Module
    residual_network: nn.Module
    inputs: torch.Tensor


@dataclass(frozen=True)
class InputGradientCurve:
    label: str
    artifact_name: str
    inputs: torch.Tensor
    gradient: torch.Tensor


@dataclass(frozen=True)
class BalduzziResult:
    seed: int
    curves: tuple[InputGradientCurve, ...]


class RandomMLP(nn.Module):
    """Hidden Linear/activation blocks with optional square residual skips."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        weight_scale: float,
        activation_name: str,
        residual: bool = False,
        branch_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or n_layers <= 0:
            raise ValueError("MLP input dimension, width, and depth must be positive")
        if not np.isfinite(weight_scale) or weight_scale <= 0:
            raise ValueError("The MLP weight scale must be finite and positive")
        if not np.isfinite(branch_scale) or branch_scale <= 0:
            raise ValueError("The residual branch scale must be finite and positive")
        if activation_name not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation_name}")
        self.residual = residual
        self.branch_scale = branch_scale
        activation_class = ACTIVATIONS[activation_name]
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            linear = nn.Linear(in_dim, hidden_dim, bias=False)
            nn.init.normal_(linear.weight, mean=0.0, std=weight_scale)
            layers.append(nn.Sequential(linear, activation_class()))
            in_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)

    def block_forward(self, index: int, h: torch.Tensor) -> torch.Tensor:
        layer = cast(nn.Sequential, self.hidden_layers[index])
        linear = cast(nn.Linear, layer[0])
        activation = layer[1]
        branch = activation(linear(h))
        if self.residual and linear.in_features == linear.out_features:
            return h + self.branch_scale * branch
        return branch

    def forward_hidden(
        self, x: torch.Tensor, n_layers: int | None = None
    ) -> torch.Tensor:
        selected_layers = len(self.hidden_layers) if n_layers is None else n_layers
        if not 1 <= selected_layers <= len(self.hidden_layers):
            raise ValueError("The hidden-layer prefix must be within the model depth")
        h = x
        for index in range(selected_layers):
            h = self.block_forward(index, h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_hidden(x)

    def forward_with_norms(
        self, x: torch.Tensor, output_grad: torch.Tensor
    ) -> list[dict[str, float]]:
        """Collect legacy per-layer forward and manual backward norm values."""
        norms: list[dict[str, float]] = []
        pre_activations: list[torch.Tensor] = []
        layer_inputs: list[torch.Tensor] = []
        h = x
        for index, layer in enumerate(self.hidden_layers):
            sequential = cast(nn.Sequential, layer)
            linear = cast(nn.Linear, sequential[0])
            layer_inputs.append(h)
            pre_activation = linear(h)
            pre_activations.append(pre_activation)
            norms.append({"hidden_pre_norm": pre_activation.norm(dim=1).mean().item()})
            h = self.block_forward(index, h)
        current_grad = output_grad
        for index in reversed(range(len(self.hidden_layers))):
            layer = cast(nn.Sequential, self.hidden_layers[index])
            linear = cast(nn.Linear, layer[0])
            pre_activation = pre_activations[index]
            mask = (pre_activation > 0).to(current_grad.dtype)
            is_square_residual = (
                self.residual and linear.in_features == linear.out_features
            )
            effective_scale = self.branch_scale if is_square_residual else 1.0
            grad_weight = (
                effective_scale * (current_grad * mask).T @ layer_inputs[index]
            )
            grad_input = effective_scale * (current_grad * mask) @ linear.weight
            if is_square_residual:
                grad_input = grad_input + current_grad
            norms[index]["grad_pre_norm"] = (
                (current_grad * mask).norm(dim=1).mean().item()
            )
            norms[index]["grad_input_norm"] = grad_input.norm(dim=1).mean().item()
            norms[index]["grad_weight_norm"] = grad_weight.norm().item()
            current_grad = grad_input
        return norms


def run_single_experiment(
    input_dim: int,
    hidden_dim: int,
    n_layers: int,
    weight_scale: float,
    activation_name: str,
    batch_size: int,
    seed: int,
) -> dict[str, FloatArray]:
    """Build one default-dtype model, convert it, and measure its norms."""
    if batch_size <= 0 or seed < 0:
        raise ValueError("The batch size must be positive and the seed non-negative")
    torch.manual_seed(seed)
    model = RandomMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        weight_scale=weight_scale,
        activation_name=activation_name,
    ).to(dtype=torch.float64)
    x = torch.randn(batch_size, input_dim, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed + 10_000)
    output_grad = torch.randn(
        batch_size, hidden_dim, generator=generator, dtype=torch.float64
    )
    norms = model.forward_with_norms(x=x, output_grad=output_grad)
    return {
        key: np.asarray([layer_norms[key] for layer_norms in norms])
        for key in NORM_KEYS
    }


def run_averaged_experiment(
    n_layers: int,
    weight_scale: float,
    input_dim: int,
    hidden_dim: int,
    activation_name: str,
    batch_size: int,
    seeds: list[int] | tuple[int, ...],
) -> dict[str, FloatArray]:
    if not seeds:
        raise ValueError("At least one propagation seed is required")
    per_seed_results = [
        run_single_experiment(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            weight_scale=weight_scale,
            activation_name=activation_name,
            batch_size=batch_size,
            seed=seed,
        )
        for seed in seeds
    ]
    result = {}
    for key in NORM_KEYS:
        values = np.stack([per_seed[key] for per_seed in per_seed_results])
        result[f"{key}_mean"] = values.mean(axis=0)
        result[f"{key}_std"] = values.std(axis=0)
    return result


def _perturbation_sensitivities(
    model: RandomMLP,
    n_layers: int,
    x: torch.Tensor,
    generator: torch.Generator,
    n_perturbations: int,
    perturbation_norm: float,
) -> list[float]:
    if n_perturbations <= 0:
        raise ValueError("The perturbation count must be positive")
    if not np.isfinite(perturbation_norm) or perturbation_norm <= 0:
        raise ValueError("The perturbation norm must be finite and positive")
    sensitivities = []
    with torch.no_grad():
        for _ in range(n_perturbations):
            dx = torch.randn(x.shape[-1], dtype=torch.float64, generator=generator)
            dx = dx * (perturbation_norm / dx.norm())
            dy = model.forward_hidden(x + dx, n_layers) - model.forward_hidden(
                x, n_layers
            )
            sensitivities.append((dy.norm() / perturbation_norm).item())
    return sensitivities


def compute_empirical_sensitivity(
    model: RandomMLP,
    n_layers: int,
    x: torch.Tensor,
    generator: torch.Generator,
    n_perturbations: int,
    perturbation_norm: float,
) -> float:
    values = _perturbation_sensitivities(
        model,
        n_layers,
        x,
        generator,
        n_perturbations,
        perturbation_norm,
    )
    return float(np.mean(values))


def compute_jacobian_metrics(
    model: RandomMLP, n_layers: int, x: torch.Tensor
) -> dict[str, float]:
    """Compute exact singular-value diagnostics for one hidden-layer prefix."""

    def forward_n_layers(inputs: torch.Tensor) -> torch.Tensor:
        return model.forward_hidden(inputs, n_layers=n_layers)

    jacobian = torch.autograd.functional.jacobian(forward_n_layers, x)
    singular_values = torch.linalg.svdvals(jacobian)
    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]
    return {
        "sigma_max": sigma_max.item(),
        "sigma_min": sigma_min.item(),
        "condition_number": (sigma_max / sigma_min).item(),
        "jacobian_frobenius": jacobian.norm().item(),
    }


def run_conditioning_experiment(
    input_dim: int,
    hidden_dim: int,
    activation_name: str,
    weight_scale: float,
    depth_grid: list[int] | tuple[int, ...],
    n_inputs: int,
    n_perturbations: int,
    perturbation_norm: float,
    seeds: list[int] | tuple[int, ...],
    variants: tuple[ConditioningVariant, ...] | None = None,
) -> ConditioningResult:
    """Measure finite-difference sensitivity and exact Jacobian spectra."""
    if not depth_grid or any(depth <= 0 for depth in depth_grid):
        raise ValueError("The conditioning depth grid must contain positive depths")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("Conditioning seeds must be non-negative and non-empty")
    if n_inputs <= 0:
        raise ValueError("The conditioning input count must be positive")
    selected_variants = variants or (ConditioningVariant("plain", False, 1.0, None),)
    if not selected_variants:
        raise ValueError("At least one conditioning variant is required")
    labels = [variant.label for variant in selected_variants]
    if len(set(labels)) != len(labels):
        raise ValueError("Conditioning variant labels must be unique")
    metrics: dict[str, dict[str, list[float]]] = {
        variant.label: {key: [] for key in CONDITIONING_METRICS}
        for variant in selected_variants
    }
    for seed in seeds:
        torch.manual_seed(seed)
        static_models = {
            variant.label: RandomMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_layers=max(depth_grid),
                weight_scale=(
                    variant.weight_scale
                    if variant.weight_scale is not None
                    else weight_scale
                ),
                activation_name=activation_name,
                residual=variant.residual,
                branch_scale=variant.branch_scale,
            ).to(dtype=torch.float64)
            for variant in selected_variants
            if variant.branch_scale > 0
        }

        generator = torch.Generator().manual_seed(seed + 20_000)
        xs = torch.randn(n_inputs, input_dim, dtype=torch.float64, generator=generator)
        for n_layers in depth_grid:
            for variant in selected_variants:
                if variant.branch_scale > 0:
                    model = static_models[variant.label]
                else:
                    resolved_scale = {
                        -1.0: 1.0 / np.sqrt(n_layers),
                        -2.0: 1.0 / n_layers,
                    }.get(variant.branch_scale)
                    if resolved_scale is None:
                        raise ValueError(
                            "Unsupported residual scale sentinel: "
                            f"{variant.branch_scale}"
                        )
                    torch.manual_seed(seed)
                    model = RandomMLP(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        n_layers=n_layers,
                        weight_scale=(
                            variant.weight_scale
                            if variant.weight_scale is not None
                            else weight_scale
                        ),
                        activation_name=activation_name,
                        residual=variant.residual,
                        branch_scale=resolved_scale,
                    ).to(dtype=torch.float64)
                sensitivities: list[float] = []
                sigma_max_values: list[float] = []
                sigma_min_values: list[float] = []
                condition_values: list[float] = []
                frobenius_values: list[float] = []
                for input_value in xs:
                    x = input_value.detach().requires_grad_(True)
                    sensitivities.extend(
                        _perturbation_sensitivities(
                            model,
                            n_layers,
                            x,
                            generator,
                            n_perturbations,
                            perturbation_norm,
                        )
                    )
                    jacobian_metrics = compute_jacobian_metrics(
                        model, n_layers=n_layers, x=x
                    )
                    sigma_max_values.append(jacobian_metrics["sigma_max"])
                    sigma_min_values.append(jacobian_metrics["sigma_min"])
                    condition_values.append(jacobian_metrics["condition_number"])
                    frobenius_values.append(jacobian_metrics["jacobian_frobenius"])
                variant_metrics = metrics[variant.label]
                variant_metrics["empirical_sensitivity"].append(
                    float(np.mean(sensitivities))
                )
                variant_metrics["sigma_max"].append(float(np.mean(sigma_max_values)))
                variant_metrics["sigma_min"].append(float(np.mean(sigma_min_values)))
                variant_metrics["condition_number"].append(
                    float(np.mean(condition_values))
                )
                variant_metrics["jacobian_frobenius"].append(
                    float(np.mean(frobenius_values))
                )
    values = {
        (variant.label, key): np.asarray(metric_values)
        .reshape(len(seeds), len(depth_grid))
        .mean(axis=0)
        for variant in selected_variants
        for key, metric_values in metrics[variant.label].items()
    }
    return ConditioningResult(variants=selected_variants, values=values)


def produce_propagation_input(cfg: DictConfig) -> PropagationExperimentInput:
    """Convert Hydra settings into an explicit propagation input pipe."""
    input_dim = int(cfg.input_dim)
    hidden_dim = int(cfg.hidden_dim)
    batch_size = int(cfg.batch_size)
    n_seeds = int(cfg.n_seeds)
    init_n_layers = int(cfg.init_sweep.n_layers)
    conditioning_inputs = int(cfg.conditioning.n_inputs)
    conditioning_perturbations = int(cfg.conditioning.n_perturbations)
    perturbation_norm = float(cfg.conditioning.perturbation_norm)
    if min(input_dim, hidden_dim, batch_size, n_seeds, init_n_layers) <= 0:
        raise ValueError(
            "Propagation dimensions, batch, depth, and seeds must be positive"
        )
    activation = str(cfg.activation)
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation}")
    experiments = tuple(str(value) for value in cfg.experiments)
    if not experiments or set(experiments) - {"init", "conditioning"}:
        raise ValueError("Experiments must select init, conditioning, or both")
    depth_grid = tuple(int(value) for value in cfg.depth_grid)
    if not depth_grid or any(value <= 0 for value in depth_grid):
        raise ValueError("The depth grid must contain positive integers")
    weight_scale = float(cfg.weight_scale)
    if not np.isfinite(weight_scale) or weight_scale <= 0:
        raise ValueError("The critical weight scale must be finite and positive")
    init_weight_scales = tuple(
        (str(label), float(scale))
        for label, scale in cfg.init_sweep.weight_scales.items()
    )
    if not init_weight_scales or any(
        not np.isfinite(scale) or scale <= 0 for _, scale in init_weight_scales
    ):
        raise ValueError("Initialization weight scales must be finite and positive")
    if conditioning_inputs <= 0 or conditioning_perturbations <= 0:
        raise ValueError("Conditioning input and perturbation counts must be positive")
    if not np.isfinite(perturbation_norm) or perturbation_norm <= 0:
        raise ValueError("The perturbation norm must be finite and positive")
    return PropagationExperimentInput(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        seeds=tuple(range(n_seeds)),
        activation=activation,
        experiments=experiments,
        depth_grid=depth_grid,
        weight_scale=weight_scale,
        init_n_layers=init_n_layers,
        init_weight_scales=init_weight_scales,
        conditioning_inputs=conditioning_inputs,
        conditioning_perturbations=conditioning_perturbations,
        perturbation_norm=perturbation_norm,
    )


def evaluate_propagation(
    experiment: PropagationExperimentInput,
) -> PropagationExperimentResult:
    """Run the configured initialization and conditioning diagnostics."""
    init_curves = []
    if "init" in experiment.experiments:
        init_labels = {
            "1.0/sqrt(N)": r"$\sigma^2 = 1/N$",
            "sqrt(2/N)": r"$\sigma^2 = 2/N$ (Kaiming)",
            "1.0 (unscaled)": r"$\sigma^2 = 1$",
        }
        for name, scale in experiment.init_weight_scales:
            try:
                label = init_labels[name]
            except KeyError as error:
                raise ValueError(f"Unsupported initialization label: {name}") from error
            norms = run_averaged_experiment(
                n_layers=experiment.init_n_layers,
                weight_scale=scale,
                input_dim=experiment.input_dim,
                hidden_dim=experiment.hidden_dim,
                activation_name=experiment.activation,
                batch_size=experiment.batch_size,
                seeds=experiment.seeds,
            )
            init_curves.append(NormCurve(name, label, scale, norms))
    plain_conditioning = None
    residual_conditioning = None
    if "conditioning" in experiment.experiments:
        plain_variants = (
            ConditioningVariant(
                "1.0/sqrt(N)",
                False,
                1.0,
                1.0 / np.sqrt(experiment.hidden_dim),
            ),
            ConditioningVariant("1.0 (unscaled)", False, 1.0, 1.0),
            ConditioningVariant(
                "Kaiming sqrt(2/N)", False, 1.0, experiment.weight_scale
            ),
        )
        plain_conditioning = run_conditioning_experiment(
            input_dim=experiment.input_dim,
            hidden_dim=experiment.hidden_dim,
            activation_name=experiment.activation,
            weight_scale=experiment.weight_scale,
            depth_grid=experiment.depth_grid,
            n_inputs=experiment.conditioning_inputs,
            n_perturbations=experiment.conditioning_perturbations,
            perturbation_norm=experiment.perturbation_norm,
            seeds=experiment.seeds,
            variants=plain_variants,
        )
        residual_variants = (
            ConditioningVariant("plain", False, 1.0, None),
            ConditioningVariant("residual (branch x1)", True, 1.0, None),
            ConditioningVariant("residual (branch x1/sqrt(L))", True, -1.0, None),
            ConditioningVariant("residual (branch x1/L)", True, -2.0, None),
        )
        residual_conditioning = run_conditioning_experiment(
            input_dim=experiment.input_dim,
            hidden_dim=experiment.hidden_dim,
            activation_name=experiment.activation,
            weight_scale=experiment.weight_scale,
            depth_grid=experiment.depth_grid,
            n_inputs=experiment.conditioning_inputs,
            n_perturbations=experiment.conditioning_perturbations,
            perturbation_norm=experiment.perturbation_norm,
            seeds=experiment.seeds,
            variants=residual_variants,
        )
    return PropagationExperimentResult(
        experiment=experiment,
        init_curves=tuple(init_curves),
        plain_conditioning=plain_conditioning,
        residual_conditioning=residual_conditioning,
    )


class DeepReLUModelWithResidualConnections(nn.Module):
    """Balduzzi network with additive skips around square hidden blocks."""

    def __init__(
        self,
        hidden_layer_dim: int = 8,
        n_layers: int = 1,
        branch_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_layer_dim <= 0 or n_layers <= 0:
            raise ValueError("Balduzzi network width and depth must be positive")
        if not np.isfinite(branch_scale) or branch_scale <= 0:
            raise ValueError("The Balduzzi residual branch scale must be positive")
        self.branch_scale = branch_scale
        self.input_linear = nn.Linear(1, hidden_layer_dim)
        self.hidden_blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU())
                for _ in range(n_layers - 1)
            ]
        )
        self.output_linear = nn.Linear(hidden_layer_dim, 1)

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )

        self.apply(init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_linear(inputs)
        for block in self.hidden_blocks:
            hidden = hidden + self.branch_scale * block(hidden)
        return self.output_linear(hidden)


def produce_balduzzi_input(cfg: DictConfig) -> BalduzziInput:
    """Seed and construct the plain/residual networks and their input grid."""
    seed = int(cfg.seed)
    hidden_layer_dim = int(cfg.hidden_layer_dim)
    n_layers = int(cfg.n_layers)
    residual_branch_scale = float(cfg.residual_branch_scale)
    n_grid_points = int(cfg.n_grid_points)
    limits = tuple(float(value) for value in cfg.x_lims)
    if seed < 0:
        raise ValueError("The Balduzzi seed must be non-negative")
    if hidden_layer_dim <= 0 or n_layers <= 0:
        raise ValueError("Balduzzi network width and depth must be positive")
    if n_grid_points < 2:
        raise ValueError("The Balduzzi input grid needs at least two points")
    if len(limits) != 2 or not limits[0] < limits[1]:
        raise ValueError("The Balduzzi input interval must have increasing endpoints")
    torch.manual_seed(seed)
    plain_network = DeepReLUModel(hidden_layer_dim=hidden_layer_dim, n_layers=n_layers)
    residual_network = DeepReLUModelWithResidualConnections(
        hidden_layer_dim=hidden_layer_dim,
        n_layers=n_layers,
        branch_scale=residual_branch_scale,
    )
    inputs = torch.linspace(limits[0], limits[1], n_grid_points, dtype=torch.float32)
    return BalduzziInput(
        seed=seed,
        hidden_layer_dim=hidden_layer_dim,
        n_layers=n_layers,
        residual_branch_scale=residual_branch_scale,
        plain_network=plain_network,
        residual_network=residual_network,
        inputs=inputs,
    )


def evaluate_input_gradients(experiment: BalduzziInput) -> BalduzziResult:
    """Evaluate detached gradients of each summed network output."""
    curves = []
    network_entries = (
        ("plain", "balduzzi_analysis_plain_network", experiment.plain_network),
        (
            "residual",
            "balduzzi_analysis_res_network",
            experiment.residual_network,
        ),
    )
    for label, artifact_name, network in network_entries:
        inputs = experiment.inputs.detach().clone().requires_grad_(True)
        output = network(inputs.unsqueeze(dim=1)).sum()
        output.backward()
        if inputs.grad is None:
            raise RuntimeError("The network did not produce an input gradient")
        curves.append(
            InputGradientCurve(
                label=label,
                artifact_name=artifact_name,
                inputs=inputs.detach().clone(),
                gradient=inputs.grad.detach().clone(),
            )
        )
    return BalduzziResult(seed=experiment.seed, curves=tuple(curves))
