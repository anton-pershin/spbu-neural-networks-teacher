import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

"""Single-pass diagnostics of signal and gradient propagation in a random MLP.

For a freshly initialized deep ReLU network we record, for each hidden layer:
  - the L2 norm of the pre-activations (forward signal);
  - the L2 norm of the gradient w.r.t. the layer pre-activation (backward signal);
  - the Frobenius norm of the gradient w.r.t. the layer weights.

Inputs are iid N(0, 1) of configurable dimension and the output gradient is
isotropic N(0, 1) noise (equivalent to fitting random labels at
initialization): the backward norms then measure the Jacobian chain itself,
not any particular target function (cf. Poole et al. 2016,
Schoenholz et al. 2017).

For ReLU with zero biases and weights ~ N(0, sigma^2) the theory lines are
  E ||h_l||^2 = d * (H * sigma^2 / 2)^l                              (forward)
  E ||dL/dh_l||^2 = H * (sigma^2 * (H + 1) / 2)^(L - l + 1)          (backward)
where d is the input dimension and H the hidden width. The forward factor is
H * sigma^2 / 2 (exactly, in expectation); the backward one carries the +1
finite-width correction from the weight-gradient term. Note the backward
signal compounds over the REMAINING layers, hence its much steeper depth
dependence. Kaiming's sigma^2 = 2 / H makes the forward factor exactly 1.

These closed forms are ZERO-MEAN approximations: relu outputs have positive
means, and near criticality the accumulated mean shift moves the empirical
fixed point slightly above the prediction (forward curves settle a bit above
sqrt(d) instead of exactly sqrt(d); the backward per-layer factor deviates
from 1 by a few percent). The decay/explode regimes match the theory over
many orders of magnitude regardless.
"""

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

NORM_KEYS = ("hidden_pre_norm", "grad_pre_norm", "grad_input_norm", "grad_weight_norm")


class RandomMLP(nn.Module):
    """MLP with n_layers hidden linear+activation layers and zero biases."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        weight_scale: float,
        activation_name: str,
    ):
        super().__init__()
        act_cls = ACTIVATIONS[activation_name]
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            linear = nn.Linear(in_dim, hidden_dim, bias=False)
            nn.init.normal_(linear.weight, mean=0.0, std=weight_scale)
            layers.append(nn.Sequential(linear, act_cls()))
            in_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)

    def forward_with_norms(
        self,
        x: torch.Tensor,
        output_grad: torch.Tensor,
    ) -> list[dict[str, float]]:
        """One forward and one backward pass, collecting per-layer norms.

        The backward pass computes gradients of <output, output_grad> where
        output_grad is an isotropic random vector, so the backward norms probe
        the Jacobian chain rather than a particular target function.
        """
        norms: list[dict[str, float]] = []
        # Forward pass, remembering pre-activations and layer inputs.
        pre_activations: list[torch.Tensor] = []
        layer_inputs: list[torch.Tensor] = []
        h = x
        for layer in self.hidden_layers:
            linear: nn.Linear = layer[0]
            act = layer[1]
            layer_inputs.append(h)
            pre_activation = linear(h)
            h = act(pre_activation)
            pre_activations.append(pre_activation)
            norms.append({"hidden_pre_norm": pre_activation.norm(dim=1).mean().item()})
        # Backward pass: gradients of <output, output_grad>.
        current_grad = output_grad
        for index in reversed(range(len(self.hidden_layers))):
            layer = self.hidden_layers[index]
            linear = layer[0]
            grad_pre = current_grad * (pre_activations[index] > 0).to(current_grad.dtype)
            grad_input = grad_pre @ linear.weight
            grad_weight = grad_pre.T @ layer_inputs[index]
            norms[index]["grad_pre_norm"] = grad_pre.norm(dim=1).mean().item()
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
) -> dict[str, np.ndarray]:
    """Builds a random MLP and returns per-layer norms averaged over the batch."""
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
    output_grad = torch.randn(batch_size, hidden_dim, generator=generator, dtype=torch.float64)
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
    seeds: list[int],
) -> dict[str, np.ndarray]:
    """Averages per-layer norms over several random initializations."""
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
        result[key + "_mean"] = values.mean(axis=0)
        result[key + "_std"] = values.std(axis=0)
    return result


def add_theory_lines(
    ax,
    n_layers: int,
    weight_scale: float,
    input_dim: int,
    hidden_dim: int,
    mode: str,
    with_label: bool = True,
) -> None:
    """Dashed theoretical predictions for the mean-squared norms (ReLU)."""
    layer_indices = np.arange(1, n_layers + 1)
    if mode == "forward":
        # E ||h_l||^2 = d * (H sigma^2 / 2)^l for ReLU with zero biases.
        theory = input_dim * (weight_scale**2 * hidden_dim / 2.0) ** layer_indices
        label = "zero-mean theory " + r"$\sqrt{d (H\sigma^2/2)^l}$" if with_label else None
    else:
        # E ||dL/dh_l||^2 = H * (sigma^2 (H + 1) / 2)^(L - l + 1) for l >= 2:
        # the backward signal starts at E||g||^2 = H (isotropic output
        # gradient) and is multiplied by the expected squared Frobenius norm
        # of each layer Jacobian as it travels towards earlier layers. The
        # FIRST layer (l = 1) instead contracts into the d-dimensional input
        # space, replacing its last factor by sigma^2 * d / 2.
        theory = hidden_dim * (weight_scale**2 * (hidden_dim + 1) / 2.0) ** (
            n_layers - layer_indices + 1
        )
        theory[0] = (
            hidden_dim
            * (weight_scale**2 * (hidden_dim + 1) / 2.0) ** (n_layers - 1)
            * (weight_scale**2 * input_dim / 2.0)
        )
        label = "zero-mean theory " + r"$\sqrt{H (\sigma^2 (H+1)/2)^{L-l+1}}$" if with_label else None
    ax.loglog(
        layer_indices,
        np.sqrt(theory),
        "--",
        marker=".",
        markersize=4,
        color="grey",
        linewidth=1.2,
        alpha=0.8,
        label=label,
    )


def plot_norm_curves(
    ax,
    curves: dict,
    norm_key: str,
    title: str,
    ylabel: str,
    with_theory: bool,
    input_dim: int,
    hidden_dim: int,
) -> None:
    """curves: mapping of any key -> {label, weight_scale, norms}."""
    cmap = matplotlib.colormaps["viridis"]
    for i, (key, entry) in enumerate(curves.items()):
        norms = entry["norms"]
        weight_scale = entry["weight_scale"]
        n_layers = len(norms[norm_key + "_mean"])
        xs = np.arange(1, n_layers + 1)
        mean = norms[norm_key + "_mean"]
        std = norms[norm_key + "_std"]
        color = cmap(i / max(len(curves) - 1, 1))
        ax.plot(xs, mean, "-o", markersize=3, linewidth=1.5, color=color, label=entry["label"])
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)
        if with_theory and norm_key in ("hidden_pre_norm", "grad_input_norm"):
            add_theory_lines(
                ax=ax,
                n_layers=n_layers,
                weight_scale=weight_scale,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                mode="forward" if norm_key == "hidden_pre_norm" else "backward",
                with_label=(i == 0),
            )
    ax.set_xlabel("Layer index $l$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)


def plot_figure(
    curves: dict,
    title_suffix: str,
    filename_base: str,
    input_dim: int,
    hidden_dim: int,
    with_theory: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_norm_curves(
        ax=axes[0],
        curves=curves,
        norm_key="hidden_pre_norm",
        title="Forward: hidden pre-activations",
        ylabel=r"$\|h_l\|_2$ (batch mean)",
        with_theory=with_theory,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )
    plot_norm_curves(
        ax=axes[1],
        curves=curves,
        norm_key="grad_input_norm",
        title="Backward: gradients w.r.t. layer pre-activations",
        ylabel=r"$\|dL/dh_l\|_2$ (batch mean)",
        with_theory=with_theory,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )
    fig.suptitle(title_suffix, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{filename_base}.svg")
    fig.savefig(f"{filename_base}.png", dpi=150)
    plt.close(fig)


def run_depth_sweep(cfg: DictConfig, seeds: list[int]) -> None:
    curves = {}
    for n_layers in cfg.depth_grid:
        curves[n_layers] = {
            "label": rf"$L = {n_layers}$",
            "weight_scale": cfg.weight_scale,
            "norms": run_averaged_experiment(
                n_layers=n_layers,
                weight_scale=cfg.weight_scale,
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                activation_name=cfg.activation,
                batch_size=cfg.batch_size,
                seeds=seeds,
            ),
        }
    plot_figure(
        curves=curves,
        title_suffix=f" ({cfg.activation}, $\\sigma = {cfg.weight_scale:.3f}$, varied depth $L$)",
        filename_base=f"hidden_norms_depth_{cfg.activation}",
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        with_theory=cfg.with_theory,
    )


def run_init_sweep(cfg: DictConfig, seeds: list[int]) -> None:
    curves = {}
    n_layers = cfg.init_sweep.n_layers
    for label, weight_scale in cfg.init_sweep.weight_scales.items():
        curves[label] = {
            "label": rf"$\sigma = {weight_scale:.4f}$ ({label})",
            "weight_scale": weight_scale,
            "norms": run_averaged_experiment(
                n_layers=n_layers,
                weight_scale=weight_scale,
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                activation_name=cfg.activation,
                batch_size=cfg.batch_size,
                seeds=seeds,
            ),
        }
    plot_figure(
        curves=curves,
        title_suffix=f" ({cfg.activation}, $L = {n_layers}$, varied init $\\sigma$)",
        filename_base=f"hidden_norms_init_{cfg.activation}",
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        with_theory=cfg.with_theory,
    )


def run_conditioning_experiment(
    input_dim: int,
    hidden_dim: int,
    activation_name: str,
    weight_scale: float,
    depth_grid: list[int],
    n_inputs: int,
    n_perturbations: int,
    perturbation_norm: float,
    seeds: list[int],
) -> dict[str, np.ndarray]:
    """Measures input sensitivity and Jacobian conditioning vs depth.

    For each random input x we compute:
      - the empirical sensitivity: ||f(x + dx) - f(x)|| / ||dx|| for n_perturbations
        random directions dx with ||dx|| = perturbation_norm;
      - the exact input Jacobian J = dh_L/dx (via autograd), from which we take
        the largest and smallest singular values and their ratio (condition
        number). The Frobenius norm corresponds to the RMS sensitivity over
        all directions and is expected to stay flat at the Kaiming scale,
        while the condition number grows ~exponentially: the average hides
        the ill-conditioning, the spectrum does not.
    """
    metrics: dict[str, list[float]] = {
        "empirical_sensitivity": [],
        "sigma_max": [],
        "sigma_min": [],
        "condition_number": [],
        "jacobian_frobenius": [],
    }
    for seed in seeds:
        torch.manual_seed(seed)
        model = RandomMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=max(depth_grid),
            weight_scale=weight_scale,
            activation_name=activation_name,
        ).to(dtype=torch.float64)

        def forward_n_layers(x: torch.Tensor, n_layers: int) -> torch.Tensor:
            h = x
            for layer in model.hidden_layers[:n_layers]:
                h = layer(h)
            return h

        generator = torch.Generator().manual_seed(seed + 20_000)
        xs = torch.randn(n_inputs, input_dim, dtype=torch.float64, generator=generator)
        for n_layers in depth_grid:
            sensitivities: list[float] = []
            sigma_max_list: list[float] = []
            sigma_min_list: list[float] = []
            frobenius_list: list[float] = []
            for x in xs:
                x = x.detach().requires_grad_(True)
                # Empirical perturbation sensitivity (finite differences).
                for _ in range(n_perturbations):
                    dx = torch.randn(input_dim, dtype=torch.float64, generator=generator)
                    dx = dx * (perturbation_norm / dx.norm())
                    dy = forward_n_layers(x + dx, n_layers) - forward_n_layers(x, n_layers)
                    sensitivities.append((dy.norm() / perturbation_norm).item())
                # Exact Jacobian spectrum.
                jacobian = torch.autograd.functional.jacobian(
                    lambda xx: forward_n_layers(xx, n_layers), x
                )
                singular_values = torch.linalg.svdvals(jacobian)
                sigma_max_list.append(singular_values[0].item())
                sigma_min_list.append(singular_values[-1].item())
                frobenius_list.append(jacobian.norm().item())
            metrics["empirical_sensitivity"].append(np.mean(sensitivities))
            metrics["sigma_max"].append(np.mean(sigma_max_list))
            metrics["sigma_min"].append(np.mean(sigma_min_list))
            metrics["condition_number"].append(
                np.mean(np.asarray(sigma_max_list) / np.asarray(sigma_min_list))
            )
            metrics["jacobian_frobenius"].append(np.mean(frobenius_list))

    # Reorganize into per-metric arrays of shape (n_seeds, len(depth_grid)).
    n_depths = len(depth_grid)
    result = {
        key: np.asarray(values).reshape(len(seeds), n_depths).mean(axis=0)
        for key, values in metrics.items()
    }
    return result


def run_conditioning_sweep(cfg: DictConfig, seeds: list[int]) -> None:
    sweep_cfg = cfg.conditioning
    result = run_conditioning_experiment(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        activation_name=cfg.activation,
        weight_scale=cfg.weight_scale,
        depth_grid=list(cfg.depth_grid),
        n_inputs=sweep_cfg.n_inputs,
        n_perturbations=sweep_cfg.n_perturbations,
        perturbation_norm=sweep_cfg.perturbation_norm,
        seeds=seeds,
    )
    depth_grid = np.asarray(cfg.depth_grid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Left: empirical sensitivity (mean over random inputs and directions).
    axes[0].loglog(
        depth_grid,
        result["empirical_sensitivity"],
        "o-",
        color="tab:blue",
        label=r"empirical $\|dy/dx\|$ (random $dx$)",
    )
    axes[0].loglog(
        depth_grid,
        result["jacobian_frobenius"] / np.sqrt(cfg.input_dim),
        "s--",
        color="tab:orange",
        label=r"$\|J\|_F / \sqrt{d}$ (RMS over directions)",
    )
    axes[0].set_xlabel("Number of layers $L$", fontsize=12)
    axes[0].set_ylabel(r"sensitivity of $h_L$ to the input", fontsize=12)
    axes[0].set_ylim(0.5, 2.0)
    axes[0].set_title("Average sensitivity", fontsize=13)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=9)
    # Right: Jacobian spectrum spread.
    axes[1].loglog(depth_grid, result["sigma_max"], "o-", color="tab:red", label=r"$\sigma_{\max}$")
    axes[1].loglog(depth_grid, result["sigma_min"], "o-", color="tab:green", label=r"$\sigma_{\min}$")
    axes[1].loglog(
        depth_grid,
        result["condition_number"],
        "o-",
        color="tab:purple",
        label=r"$\sigma_{\max} / \sigma_{\min}$",
    )
    axes[1].set_xlabel("Number of layers $L$", fontsize=12)
    axes[1].set_ylabel("input-Jacobian singular values", fontsize=12)
    axes[1].set_title("Jacobian conditioning", fontsize=13)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.suptitle(
        f"({cfg.activation}, $\\sigma = {cfg.weight_scale:.3f}$, "
        f"$||dx|| = {sweep_cfg.perturbation_norm:.0e}$, {sweep_cfg.n_inputs} inputs)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"hidden_norms_conditioning_{cfg.activation}.svg")
    fig.savefig(f"hidden_norms_conditioning_{cfg.activation}.png", dpi=150)
    plt.close(fig)
    for i, n_layers in enumerate(depth_grid):
        print(
            f"L = {n_layers:3d}: empirical = {result['empirical_sensitivity'][i]:.3f}, "
            f"sigma_max = {result['sigma_max'][i]:.3e}, "
            f"sigma_min = {result['sigma_min'][i]:.3e}, "
            f"cond = {result['condition_number'][i]:.3e}"
        )


def main(cfg: DictConfig) -> None:
    seeds = list(range(cfg.n_seeds))
    if "depth" in cfg.experiments:
        run_depth_sweep(cfg, seeds)
        print("Saved hidden_norms_depth_*.svg/.png")
    if "init" in cfg.experiments:
        run_init_sweep(cfg, seeds)
        print("Saved hidden_norms_init_*.svg/.png")
    if "conditioning" in cfg.experiments:
        run_conditioning_sweep(cfg, seeds)
        print("Saved hidden_norms_conditioning_*.svg/.png")


if __name__ == "__main__":
    hydra.main(
        version_base=None,
        config_path="../../config",
        config_name="config_hidden_norms",
    )(main)()
