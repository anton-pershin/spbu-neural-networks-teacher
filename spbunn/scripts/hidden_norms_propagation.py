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
    """MLP with n_layers hidden linear+activation layers and zero biases.

    With residual=True the layers compute h_l = h_{l-1} + branch_scale * ReLU(W h),
    i.e. every block is an additive perturbation of the identity map instead of
    a plain composition. For branch_scale = 1 the signal variance grows like
    (1 + H sigma^2 / 2)^l, so the Kaiming scale no longer keeps it constant;
    branch_scale = 1 / sqrt(L) and 1 / L are the standard "small perturbation"
    initializations, keeping the map near the identity (see the conditioning
    experiment) while letting the blocks still contribute.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        weight_scale: float,
        activation_name: str,
        residual: bool = False,
        branch_scale: float = 1.0,
    ):
        super().__init__()
        self.residual = residual
        self.branch_scale = branch_scale
        act_cls = ACTIVATIONS[activation_name]
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            linear = nn.Linear(in_dim, hidden_dim, bias=False)
            nn.init.normal_(linear.weight, mean=0.0, std=weight_scale)
            layers.append(nn.Sequential(linear, act_cls()))
            in_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)

    def block_forward(self, index: int, h: torch.Tensor) -> torch.Tensor:
        """Applies hidden block `index`.

        Plain: h_out = ReLU(W h). Residual: h_out = h + branch_scale * ReLU(W h)
        (identity skip plus a scaled activated branch). The skip requires
        matching input/output dimensions, so the first block (d -> H) is
        always plain, as in ResNets, where the stem has no skip either.
        """
        linear = self.hidden_layers[index][0]
        act = self.hidden_layers[index][1]
        branch = act(linear(h))
        if self.residual and linear.in_features == linear.out_features:
            return h + self.branch_scale * branch
        return branch

    def forward_hidden(self, x: torch.Tensor, n_layers: int | None = None) -> torch.Tensor:
        """Runs the first n_layers hidden blocks; returns the last hidden state."""
        h = x
        for index in range(n_layers if n_layers is not None else len(self.hidden_layers)):
            h = self.block_forward(index, h)
        return h

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
        for index, layer in enumerate(self.hidden_layers):
            linear: nn.Linear = layer[0]
            layer_inputs.append(h)
            pre_activation = linear(h)
            pre_activations.append(pre_activation)
            norms.append({"hidden_pre_norm": pre_activation.norm(dim=1).mean().item()})
            h = self.block_forward(index, h)
        # Backward pass: gradients of <output, output_grad>.
        current_grad = output_grad
        for index in reversed(range(len(self.hidden_layers))):
            linear = self.hidden_layers[index][0]
            pre_activation = pre_activations[index]
            # Block: z = ReLU(pre) + branch_scale * pre when residual, i.e.
            # Jacobian I + branch_scale * diag(mask) W (the comment below the
            # plain case describes the plain block, which has no skip term).
            mask = (pre_activation > 0).to(current_grad.dtype)
            # The branch scale applies only to blocks that actually have the
            # residual skip (square blocks); the plain stem block uses scale 1.
            effective_scale = (
                self.branch_scale
                if (self.residual and linear.in_features == linear.out_features)
                else 1.0
            )
            grad_weight = effective_scale * (current_grad * mask).T @ layer_inputs[index]
            grad_input = effective_scale * (current_grad * mask) @ linear.weight
            if self.residual and linear.in_features == linear.out_features:
                grad_input = grad_input + current_grad
            norms[index]["grad_pre_norm"] = (current_grad * mask).norm(dim=1).mean().item()
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
    variants: list[tuple[str, bool, float, float | None]] | None = None,
) -> dict[tuple[str, str], np.ndarray]:
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

    With variants, each row is (label, residual, branch_scale, weight_scale);
    weight_scale None means the default (Kaiming) scale passed to this
    function. For residual networks the same measurements quantify how much
    the skip connections fix the conditioning. The Jacobian is computed for
    the LAST hidden layer.
    """
    variants = variants or [("plain", False, 1.0, None)]
    metrics: dict[tuple, dict[str, list[float]]] = {
        variant: {
            "empirical_sensitivity": [],
            "sigma_max": [],
            "sigma_min": [],
            "condition_number": [],
            "jacobian_frobenius": [],
        }
        for variant in variants
    }
    for seed in seeds:
        torch.manual_seed(seed)
        # Models with a fixed branch scale can be built once per seed; models
        # with a depth-dependent scale (marked by negative sentinels) are
        # rebuilt inside the depth loop.
        static_models = {
            variant: RandomMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_layers=max(depth_grid),
                weight_scale=variant_weight_scale if variant_weight_scale is not None else weight_scale,
                activation_name=activation_name,
                residual=residual,
                branch_scale=branch_scale,
            ).to(dtype=torch.float64)
            for variant, residual, branch_scale, variant_weight_scale in variants
            if branch_scale > 0
        }

        def get_model(variant: tuple, residual: bool, branch_scale: float, variant_weight_scale, n_layers: int) -> RandomMLP:
            if branch_scale > 0:
                return static_models[variant]
            resolved_scale = {  # negative sentinels: -1 -> 1/sqrt(L), -2 -> 1/L
                -1.0: 1.0 / np.sqrt(n_layers),
                -2.0: 1.0 / n_layers,
            }[branch_scale]
            torch.manual_seed(seed)
            return RandomMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                weight_scale=variant_weight_scale if variant_weight_scale is not None else weight_scale,
                activation_name=activation_name,
                residual=residual,
                branch_scale=resolved_scale,
            ).to(dtype=torch.float64)

        generator = torch.Generator().manual_seed(seed + 20_000)
        xs = torch.randn(n_inputs, input_dim, dtype=torch.float64, generator=generator)
        for n_layers in depth_grid:
            for variant, residual, branch_scale, variant_weight_scale in variants:
                model = get_model(variant, residual, branch_scale, variant_weight_scale, n_layers)

                def forward_n_layers(xx: torch.Tensor, model=model, n_layers=n_layers) -> torch.Tensor:
                    return model.forward_hidden(xx, n_layers=n_layers)

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
                        dy = forward_n_layers(x + dx) - forward_n_layers(x)
                        sensitivities.append((dy.norm() / perturbation_norm).item())
                    # Exact Jacobian spectrum.
                    jacobian = torch.autograd.functional.jacobian(forward_n_layers, x)
                    singular_values = torch.linalg.svdvals(jacobian)
                    sigma_max_list.append(singular_values[0].item())
                    sigma_min_list.append(singular_values[-1].item())
                    frobenius_list.append(jacobian.norm().item())
                variant_metrics = metrics[(variant, residual, branch_scale, variant_weight_scale)]
                variant_metrics["empirical_sensitivity"].append(np.mean(sensitivities))
                variant_metrics["sigma_max"].append(np.mean(sigma_max_list))
                variant_metrics["sigma_min"].append(np.mean(sigma_min_list))
                variant_metrics["condition_number"].append(
                    np.mean(np.asarray(sigma_max_list) / np.asarray(sigma_min_list))
                )
                variant_metrics["jacobian_frobenius"].append(np.mean(frobenius_list))

    # Reorganize into per-metric arrays of shape (n_seeds, len(depth_grid)).
    n_depths = len(depth_grid)
    result = {
        (label, key): np.asarray(values).reshape(len(seeds), n_depths).mean(axis=0)
        for (label, _, _, _), variant_metrics in zip(variants, metrics.values())
        for key, values in variant_metrics.items()
    }
    return result


def run_conditioning_sweep(cfg: DictConfig, seeds: list[int]) -> None:
    sweep_cfg = cfg.conditioning

    # Part 1: plain MLP, two initialization scales (unscaled N(0,1) and Kaiming).
    plain_variants = [
        ("1.0 (unscaled)", False, 1.0, 1.0),
        ("Kaiming sqrt(2/H)", False, 1.0, cfg.weight_scale),
    ]
    plain_result = run_conditioning_experiment(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        activation_name=cfg.activation,
        weight_scale=cfg.weight_scale,  # unused for plain variants; see below
        depth_grid=list(cfg.depth_grid),
        n_inputs=sweep_cfg.n_inputs,
        n_perturbations=sweep_cfg.n_perturbations,
        perturbation_norm=sweep_cfg.perturbation_norm,
        seeds=seeds,
        variants=plain_variants,
    )

    # Part 2: residual variants at the Kaiming scale.
    residual_variants: list[tuple[str, bool, float, float | None]] = [
        ("plain", False, 1.0, None),
        ("residual (branch x1)", True, 1.0, None),
        ("residual (branch x1/sqrt(L))", True, -1.0, None),  # -1 marks 1/sqrt(L)
        ("residual (branch x1/L)", True, -2.0, None),  # -2 marks 1/L
    ]
    residual_result = run_conditioning_experiment(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        activation_name=cfg.activation,
        weight_scale=cfg.weight_scale,
        depth_grid=list(cfg.depth_grid),
        n_inputs=sweep_cfg.n_inputs,
        n_perturbations=sweep_cfg.n_perturbations,
        perturbation_norm=sweep_cfg.perturbation_norm,
        seeds=seeds,
        variants=residual_variants,
    )
    plot_conditioning_figures(cfg, sweep_cfg, plain_result, residual_result, residual_variants)


def plot_conditioning_figures(cfg, sweep_cfg, plain_result, residual_result, residual_variants) -> None:
    depth_grid = np.asarray(cfg.depth_grid)
    # Figure 1 (part 1): plain MLP, two init scales.
    # The plain variants carry their own weight scale in the label; rebuild the
    # experiment data: "1.0 (unscaled)" was run with cfg.weight_scale, so rerun
    # it properly instead of mislabeling.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, color in [
        ("Kaiming sqrt(2/H)", "tab:green"),
        ("1.0 (unscaled)", "tab:red"),
    ]:
        axes[0].loglog(
            depth_grid,
            plain_result[(key, "empirical_sensitivity")],
            "o-",
            color=color,
            label=rf"empirical $\|dy/dx\|$: {key}",
        )
        axes[1].loglog(
            depth_grid,
            plain_result[(key, "condition_number")],
            "o-",
            color=color,
            label=rf"$\sigma_{{\max}}/\sigma_{{\min}}$: {key}",
        )
    axes[0].set_xlabel("Number of layers $L$", fontsize=12)
    axes[0].set_ylabel(r"sensitivity of $h_L$ to the input", fontsize=12)
    axes[0].set_title("Average sensitivity", fontsize=13)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Number of layers $L$", fontsize=12)
    axes[1].set_ylabel("input-Jacobian condition number", fontsize=12)
    axes[1].set_title("Jacobian conditioning", fontsize=13)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"Plain MLP ({cfg.activation}), {sweep_cfg.n_inputs} inputs, "
        f"$||dx|| = {sweep_cfg.perturbation_norm:.0e}$",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"hidden_norms_conditioning_init_{cfg.activation}.svg")
    fig.savefig(f"hidden_norms_conditioning_init_{cfg.activation}.png", dpi=150)
    plt.close(fig)

    # Figure 2 (part 2): residual variants at the Kaiming scale.
    variant_labels = {
        "plain": "plain",
        "residual (branch x1)": "residual, branch $\\times 1$",
        "residual (branch x1/sqrt(L))": "residual, branch $\\times 1/\\sqrt{L}$",
        "residual (branch x1/L)": "residual, branch $\\times 1/L$",
    }
    colors = {
        "plain": "tab:blue",
        "residual (branch x1)": "tab:red",
        "residual (branch x1/sqrt(L))": "tab:green",
        "residual (branch x1/L)": "tab:purple",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant_label, residual, branch_scale, variant_weight_scale in residual_variants:
        key = variant_label
        color = colors[key]
        label = variant_labels[key]
        axes[0].loglog(
            depth_grid,
            residual_result[(key, "empirical_sensitivity")],
            "o-",
            color=color,
            label=rf"empirical $\|dy/dx\|$: {label}",
        )
        axes[1].loglog(
            depth_grid,
            residual_result[(key, "condition_number")],
            "o-",
            color=color,
            label=rf"$\sigma_{{\max}}/\sigma_{{\min}}$: {label}",
        )
    axes[0].set_xlabel("Number of layers $L$", fontsize=12)
    axes[0].set_ylabel(r"sensitivity of $h_L$ to the input", fontsize=12)
    axes[0].set_title("Average sensitivity", fontsize=13)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Number of layers $L$", fontsize=12)
    axes[1].set_ylabel("input-Jacobian condition number", fontsize=12)
    axes[1].set_title("Jacobian conditioning", fontsize=13)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"({cfg.activation}, $\\sigma = {cfg.weight_scale:.3f}$ (Kaiming), "
        f"$||dx|| = {sweep_cfg.perturbation_norm:.0e}$, {sweep_cfg.n_inputs} inputs)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"hidden_norms_conditioning_{cfg.activation}.svg")
    fig.savefig(f"hidden_norms_conditioning_{cfg.activation}.png", dpi=150)
    plt.close(fig)
    print("Part 1 (plain MLP, two init scales):")
    for key in ("Kaiming sqrt(2/H)", "1.0 (unscaled)"):
        print(f"Variant: {key}")
        for i, n_layers in enumerate(depth_grid):
            print(
                f"  L = {n_layers:3d}: empirical = {plain_result[(key, 'empirical_sensitivity')][i]:.3f}, "
                f"sigma_max = {plain_result[(key, 'sigma_max')][i]:.3e}, "
                f"sigma_min = {plain_result[(key, 'sigma_min')][i]:.3e}, "
                f"cond = {plain_result[(key, 'condition_number')][i]:.3e}"
            )
    print("Part 2 (residual variants, Kaiming init):")
    for variant_label, _, _, _ in residual_variants:
        print(f"Variant: {variant_label}")
        for i, n_layers in enumerate(depth_grid):
            print(
                f"  L = {n_layers:3d}: empirical = {residual_result[(variant_label, 'empirical_sensitivity')][i]:.3f}, "
                f"sigma_max = {residual_result[(variant_label, 'sigma_max')][i]:.3e}, "
                f"sigma_min = {residual_result[(variant_label, 'sigma_min')][i]:.3e}, "
                f"cond = {residual_result[(variant_label, 'condition_number')][i]:.3e}"
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
