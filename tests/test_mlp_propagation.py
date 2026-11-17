from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from spbunn.mlp.propagation import (
    ConditioningVariant,
    DeepReLUModelWithResidualConnections,
    RandomMLP,
    compute_empirical_sensitivity,
    compute_jacobian_metrics,
    evaluate_input_gradients,
    evaluate_propagation,
    produce_balduzzi_input,
    produce_propagation_input,
    run_averaged_experiment,
    run_single_experiment,
)
from spbunn.mlp.simple_relu_nn import DeepReLUModel
from tests.mlp_helpers import compose_mlp_config


def _one_dimensional_model(residual: bool = False) -> RandomMLP:
    model = RandomMLP(
        input_dim=1,
        hidden_dim=1,
        n_layers=1,
        weight_scale=1.0,
        activation_name="relu",
        residual=residual,
        branch_scale=0.5,
    ).to(dtype=torch.float64)
    layer = cast(nn.Sequential, model.hidden_layers[0])
    linear = layer[0]
    assert isinstance(linear, nn.Linear)
    with torch.no_grad():
        linear.weight.fill_(2.0)
    return model


def test_forward_and_manual_backward_norms_match_hand_calculation() -> None:
    model = _one_dimensional_model()
    x = torch.tensor([[3.0]], dtype=torch.float64)
    output_grad = torch.tensor([[5.0]], dtype=torch.float64)

    norms = model.forward_with_norms(x, output_grad)

    assert norms == [
        {
            "hidden_pre_norm": 6.0,
            "grad_pre_norm": 5.0,
            "grad_input_norm": 10.0,
            "grad_weight_norm": 15.0,
        }
    ]


def test_residual_norms_include_scaled_branch_and_identity_skip() -> None:
    model = _one_dimensional_model(residual=True)
    x = torch.tensor([[3.0]], dtype=torch.float64)
    output_grad = torch.tensor([[5.0]], dtype=torch.float64)

    norms = model.forward_with_norms(x, output_grad)

    assert model.forward_hidden(x).item() == 6.0
    assert norms[0]["hidden_pre_norm"] == 6.0
    assert norms[0]["grad_pre_norm"] == 5.0
    assert norms[0]["grad_weight_norm"] == 7.5
    assert norms[0]["grad_input_norm"] == 10.0


def test_averaged_norms_align_layers_and_seeds() -> None:
    per_seed = [
        run_single_experiment(
            input_dim=2,
            hidden_dim=3,
            n_layers=2,
            weight_scale=0.2,
            activation_name="relu",
            batch_size=2,
            seed=seed,
        )
        for seed in (0, 1)
    ]

    result = run_averaged_experiment(
        input_dim=2,
        hidden_dim=3,
        n_layers=2,
        weight_scale=0.2,
        activation_name="relu",
        batch_size=2,
        seeds=[0, 1],
    )

    for key, values in per_seed[0].items():
        stacked = np.stack([values, per_seed[1][key]])
        assert stacked.shape == (2, 2)
        assert np.array_equal(result[f"{key}_mean"], stacked.mean(axis=0))
        assert np.array_equal(result[f"{key}_std"], stacked.std(axis=0))


def test_finite_difference_and_jacobian_metrics_match_linear_hand_case() -> None:
    model = _one_dimensional_model()
    x = torch.tensor([3.0], dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)

    sensitivity = compute_empirical_sensitivity(
        model=model,
        n_layers=1,
        x=x,
        generator=generator,
        n_perturbations=2,
        perturbation_norm=0.1,
    )
    metrics = compute_jacobian_metrics(model, n_layers=1, x=x)

    assert sensitivity == pytest.approx(2.0)
    assert metrics == {
        "sigma_max": pytest.approx(2.0),
        "sigma_min": pytest.approx(2.0),
        "condition_number": pytest.approx(1.0),
        "jacobian_frobenius": pytest.approx(2.0),
    }


def test_small_propagation_evaluation_has_complete_ordered_results() -> None:
    experiment = produce_propagation_input(
        compose_mlp_config("config_test_hidden_norms")
    )

    first = evaluate_propagation(experiment)
    second = evaluate_propagation(experiment)

    assert experiment.seeds == (0,)
    assert experiment.depth_grid == (1, 2)
    assert [curve.name for curve in first.init_curves] == [
        "1.0/sqrt(N)",
        "sqrt(2/N)",
        "1.0 (unscaled)",
    ]
    assert first.init_curves[0].norms["hidden_pre_norm_mean"].shape == (2,)
    assert first.plain_conditioning is not None
    assert first.residual_conditioning is not None
    assert second.residual_conditioning is not None
    assert [variant.label for variant in first.plain_conditioning.variants] == [
        "1.0/sqrt(N)",
        "1.0 (unscaled)",
        "Kaiming sqrt(2/N)",
    ]
    assert [variant.label for variant in first.residual_conditioning.variants] == [
        "plain",
        "residual (branch x1)",
        "residual (branch x1/sqrt(L))",
        "residual (branch x1/L)",
    ]
    assert [
        variant.branch_scale for variant in first.residual_conditioning.variants
    ] == [1.0, 1.0, -1.0, -2.0]
    for metric in (
        "empirical_sensitivity",
        "sigma_max",
        "sigma_min",
        "condition_number",
        "jacobian_frobenius",
    ):
        assert first.plain_conditioning.values[("1.0/sqrt(N)", metric)].shape == (2,)
        assert np.allclose(
            first.residual_conditioning.values[("plain", metric)],
            second.residual_conditioning.values[("plain", metric)],
            equal_nan=True,
        )


def test_production_propagation_settings_retain_legacy_values() -> None:
    experiment = produce_propagation_input(compose_mlp_config("config_hidden_norms"))

    assert experiment.input_dim == 16
    assert experiment.hidden_dim == 256
    assert experiment.batch_size == 256
    assert experiment.seeds == (0, 1, 2, 3, 4)
    assert experiment.activation == "relu"
    assert experiment.experiments == ("init", "conditioning")
    assert experiment.depth_grid == (1, 2, 4, 8, 16, 32, 64)
    assert experiment.weight_scale == 0.08838834764831845
    assert experiment.init_n_layers == 32
    assert experiment.init_weight_scales == (
        ("1.0/sqrt(N)", 0.0625),
        ("sqrt(2/N)", 0.08838834764831845),
        ("1.0 (unscaled)", 1.0),
    )
    assert experiment.conditioning_inputs == 16
    assert experiment.conditioning_perturbations == 8
    assert experiment.perturbation_norm == 1.0e-3
    assert experiment.plain_variants == (
        ConditioningVariant("1.0/sqrt(N)", False, 1.0, 0.0625),
        ConditioningVariant("1.0 (unscaled)", False, 1.0, 1.0),
        ConditioningVariant("Kaiming sqrt(2/N)", False, 1.0, 0.08838834764831845),
    )
    assert experiment.residual_variants == (
        ConditioningVariant("plain", False, 1.0, None),
        ConditioningVariant("residual (branch x1)", True, 1.0, None),
        ConditioningVariant("residual (branch x1/sqrt(L))", True, -1.0, None),
        ConditioningVariant("residual (branch x1/L)", True, -2.0, None),
    )


@pytest.mark.parametrize(
    "override",
    [
        "input_dim=0",
        "depth_grid=[]",
        "n_seeds=0",
        "conditioning.n_inputs=0",
        "conditioning.n_perturbations=0",
        "conditioning.perturbation_norm=0",
        "conditioning.residual_variants=[]",
        "activation=unsupported",
    ],
)
def test_propagation_input_validation(override: str) -> None:
    cfg = compose_mlp_config("config_test_hidden_norms", [override])
    with pytest.raises(ValueError):
        produce_propagation_input(cfg)


def test_balduzzi_producer_is_seeded_and_preserves_input_grid() -> None:
    first = produce_balduzzi_input(compose_mlp_config("config_test_balduzzi_analysis"))
    second = produce_balduzzi_input(compose_mlp_config("config_test_balduzzi_analysis"))

    assert first.seed == 42
    assert first.inputs.dtype == torch.float32
    assert torch.equal(first.inputs, torch.linspace(-1, 1, 9))
    assert isinstance(first.residual_network, DeepReLUModelWithResidualConnections)
    for first_parameter, second_parameter in zip(
        first.plain_network.parameters(), second.plain_network.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)
    for first_parameter, second_parameter in zip(
        first.residual_network.parameters(),
        second.residual_network.parameters(),
        strict=True,
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_balduzzi_gradients_match_independent_autograd() -> None:
    experiment = produce_balduzzi_input(
        compose_mlp_config("config_test_balduzzi_analysis")
    )

    result = evaluate_input_gradients(experiment)

    assert [curve.label for curve in result.curves] == ["plain", "residual"]
    for curve, network in zip(
        result.curves,
        (experiment.plain_network, experiment.residual_network),
        strict=True,
    ):
        x = experiment.inputs.detach().clone().requires_grad_(True)
        expected = torch.autograd.grad(network(x.unsqueeze(1)).sum(), x)[0]
        assert torch.equal(curve.inputs, experiment.inputs)
        assert not curve.inputs.requires_grad
        assert not curve.gradient.requires_grad
        assert torch.allclose(curve.gradient, expected)


def test_production_balduzzi_settings_define_missing_legacy_choices() -> None:
    experiment = produce_balduzzi_input(compose_mlp_config("config_balduzzi_analysis"))

    assert experiment.seed == 42
    assert experiment.inputs.shape == (256,)
    assert torch.equal(experiment.inputs, torch.linspace(-1, 1, 256))
    assert experiment.hidden_layer_dim == 200
    assert experiment.n_layers == 24
    assert experiment.residual_branch_scale == 1.0
    plain = experiment.plain_network
    residual = experiment.residual_network
    assert isinstance(plain, DeepReLUModel)
    plain_layers = list(plain.sequential)
    assert len(plain_layers) == 48
    assert isinstance(plain_layers[0], nn.Linear)
    assert plain_layers[0].in_features == 1
    assert plain_layers[0].out_features == 200
    assert isinstance(plain_layers[1], nn.Linear)
    assert isinstance(plain_layers[-1], nn.Linear)
    assert plain_layers[-1].in_features == 200
    assert plain_layers[-1].out_features == 1
    assert sum(isinstance(layer, nn.ReLU) for layer in plain_layers) == 23
    assert isinstance(residual, DeepReLUModelWithResidualConnections)
    assert residual.branch_scale == 1.0
    assert residual.input_linear.in_features == 1
    assert residual.input_linear.out_features == 200
    assert residual.output_linear.in_features == 200
    assert residual.output_linear.out_features == 1
    assert len(residual.hidden_blocks) == 23
    for block in residual.hidden_blocks:
        assert isinstance(block, nn.Sequential)
        assert isinstance(block[0], nn.Linear)
        assert block[0].in_features == block[0].out_features == 200
        assert isinstance(block[1], nn.ReLU)
