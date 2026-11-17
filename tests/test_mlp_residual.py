from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

from spbunn.mlp.propagation import (
    ConditioningVariant,
    DeepReLUModelWithResidualConnections,
    RandomMLP,
    compute_jacobian_metrics,
    resolve_branch_scale,
    run_conditioning_experiment,
)


def test_residual_jacobian_spectrum_matches_fixed_weight_formula() -> None:
    model = RandomMLP(
        input_dim=1,
        hidden_dim=1,
        n_layers=1,
        weight_scale=1.0,
        activation_name="relu",
        residual=True,
        branch_scale=0.5,
    ).to(dtype=torch.float64)
    layer = model.hidden_layers[0]
    assert isinstance(layer, nn.Sequential)
    linear = layer[0]
    assert isinstance(linear, nn.Linear)
    with torch.no_grad():
        linear.weight.fill_(4.0)

    metrics = compute_jacobian_metrics(
        model, n_layers=1, x=torch.tensor([3.0], dtype=torch.float64)
    )

    # J = I + alpha * diag(ReLU'(W x)) * W = 1 + 0.5 * 4 = 3.
    assert metrics == {
        "sigma_max": pytest.approx(3.0),
        "sigma_min": pytest.approx(3.0),
        "condition_number": pytest.approx(1.0),
        "jacobian_frobenius": pytest.approx(3.0),
    }


def test_depth_dependent_residual_scale_resolution() -> None:
    assert resolve_branch_scale(1.0, 4) == 1.0
    assert resolve_branch_scale(-1.0, 4) == 0.5
    assert resolve_branch_scale(-2.0, 4) == 0.25
    assert resolve_branch_scale(-1.0, 9) == pytest.approx(1.0 / 3.0)
    assert resolve_branch_scale(-2.0, 9) == pytest.approx(1.0 / 9.0)
    with pytest.raises(ValueError, match="Unsupported residual scale sentinel"):
        resolve_branch_scale(-3.0, 4)
    with pytest.raises(ValueError, match="depth must be positive"):
        resolve_branch_scale(-1.0, 0)


def test_conditioning_aggregation_matches_independent_multi_seed_oracle() -> None:
    depths = (1, 2)
    seeds = (0, 1)
    variant = ConditioningVariant("residual", True, 0.5, 0.2)
    expected = {
        key: np.zeros((len(seeds), len(depths)))
        for key in (
            "empirical_sensitivity",
            "sigma_max",
            "sigma_min",
            "condition_number",
            "jacobian_frobenius",
        )
    }
    for seed_index, seed in enumerate(seeds):
        torch.manual_seed(seed)
        model = RandomMLP(
            input_dim=2,
            hidden_dim=2,
            n_layers=max(depths),
            weight_scale=0.2,
            activation_name="relu",
            residual=True,
            branch_scale=0.5,
        ).to(dtype=torch.float64)
        generator = torch.Generator().manual_seed(seed + 20_000)
        inputs = torch.randn(2, 2, dtype=torch.float64, generator=generator)
        for depth_index, depth in enumerate(depths):
            sensitivities = []
            sigma_max_values = []
            sigma_min_values = []
            condition_values = []
            frobenius_values = []
            forward = partial(model.forward_hidden, n_layers=depth)
            for input_value in inputs:
                x = input_value.detach().requires_grad_(True)
                dx = torch.randn(2, dtype=torch.float64, generator=generator)
                dx = dx * (0.01 / dx.norm())
                with torch.no_grad():
                    dy = forward(x + dx) - forward(x)
                sensitivities.append((dy.norm() / 0.01).item())
                jacobian = torch.autograd.functional.jacobian(forward, x)
                singular_values = torch.linalg.svdvals(jacobian)
                sigma_max_values.append(singular_values[0].item())
                sigma_min_values.append(singular_values[-1].item())
                condition_values.append(
                    (singular_values[0] / singular_values[-1]).item()
                )
                frobenius_values.append(jacobian.norm().item())
            expected["empirical_sensitivity"][seed_index, depth_index] = np.mean(
                sensitivities
            )
            expected["sigma_max"][seed_index, depth_index] = np.mean(sigma_max_values)
            expected["sigma_min"][seed_index, depth_index] = np.mean(sigma_min_values)
            expected["condition_number"][seed_index, depth_index] = np.mean(
                condition_values
            )
            expected["jacobian_frobenius"][seed_index, depth_index] = np.mean(
                frobenius_values
            )

    result = run_conditioning_experiment(
        input_dim=2,
        hidden_dim=2,
        activation_name="relu",
        weight_scale=0.2,
        depth_grid=depths,
        n_inputs=2,
        n_perturbations=1,
        perturbation_norm=0.01,
        seeds=seeds,
        variants=(variant,),
    )

    for metric, values in expected.items():
        assert np.allclose(
            result.values[("residual", metric)],
            values.mean(axis=0),
            rtol=1e-12,
            atol=1e-12,
        )


def test_explicitly_empty_conditioning_variants_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one conditioning variant"):
        run_conditioning_experiment(
            input_dim=2,
            hidden_dim=2,
            activation_name="relu",
            weight_scale=0.2,
            depth_grid=(1,),
            n_inputs=1,
            n_perturbations=1,
            perturbation_norm=0.01,
            seeds=(0,),
            variants=(),
        )


def test_balduzzi_residual_forward_and_gradient_match_hand_calculation() -> None:
    model = DeepReLUModelWithResidualConnections(
        hidden_layer_dim=1, n_layers=2, branch_scale=0.5
    ).to(dtype=torch.float64)
    hidden_block = model.hidden_blocks[0]
    assert isinstance(hidden_block, nn.Sequential)
    hidden_linear = hidden_block[0]
    assert isinstance(hidden_linear, nn.Linear)
    with torch.no_grad():
        model.input_linear.weight.fill_(2.0)
        model.input_linear.bias.fill_(1.0)
        hidden_linear.weight.fill_(3.0)
        hidden_linear.bias.fill_(1.0)
        model.output_linear.weight.fill_(4.0)
        model.output_linear.bias.fill_(2.0)
    inputs = torch.tensor([1.0, -4.0], dtype=torch.float64, requires_grad=True)

    outputs = model(inputs.unsqueeze(1)).squeeze(1)
    gradients = torch.autograd.grad(outputs.sum(), inputs)[0]

    # At x=1: h0=3, h1=3+0.5*ReLU(10)=8, y=34, dy/dx=20.
    # At x=-4: h0=-7, ReLU(-20)=0, y=-26, and the skip gives dy/dx=8.
    assert torch.allclose(outputs, torch.tensor([34.0, -26.0], dtype=torch.float64))
    assert torch.allclose(gradients, torch.tensor([20.0, 8.0], dtype=torch.float64))
