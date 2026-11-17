import numpy as np
import pytest
import torch
from torch import nn

from spbunn.mlp.architecture import (
    ArchitectureKind,
    build_architecture_model,
    cubic_function,
    discontinuous_function,
    evaluate_architecture_experiment,
    gaussian_function,
    generate_dataset,
    matching_width_dim,
    parameter_count,
    periodic_function,
    produce_depth_experiment,
    produce_width_experiment,
    step_size_for,
)
from spbunn.mlp.simple_relu_nn import DeepReLUPlainModel, ShallowReLUModel
from tests.mlp_helpers import compose_mlp_config


def test_function_regression_data_matches_legacy_grid() -> None:
    x, y = generate_dataset((-2.0, 2.0), cubic_function, n_samples=5)

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert torch.equal(x, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert torch.equal(y, x**3 - x)


def test_all_legacy_generating_function_values_are_preserved() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0])

    assert torch.equal(gaussian_function(x), torch.exp(-(x**2) / 0.2))
    assert torch.equal(discontinuous_function(x), torch.tensor([0.0, 0.0, 6.0]))
    expected_periodic = torch.sin(2 * x) + torch.cos(8 * x) + torch.sin(16 * x)
    assert torch.equal(periodic_function(x), expected_periodic)
    with pytest.raises(ValueError, match="equally sized 1D tensors"):
        periodic_function(x, amps=torch.ones(2), freqs=torch.ones(2, 1))


def test_depth_parameter_matching_examples_are_preserved() -> None:
    assert parameter_count(hidden_layer_dim=8, n_layers=1) == 25
    assert parameter_count(hidden_layer_dim=8, n_layers=2) == 97
    assert [matching_width_dim(depth) for depth in (1, 2, 6, 22)] == [
        8,
        32,
        128,
        512,
    ]
    assert step_size_for(0.1, 3000) == 300
    assert step_size_for(0.01, 2) == 1


@pytest.mark.parametrize(
    ("kind", "n_layers"),
    [("width", 1), ("depth", 1), ("depth", 3)],
)
def test_model_construction_has_expected_shape(
    kind: ArchitectureKind, n_layers: int
) -> None:
    model = build_architecture_model(kind=kind, hidden_layer_dim=3, n_layers=n_layers)

    output = model(torch.ones(2, 1))
    assert output.shape == (2, 1)
    if kind == "width":
        assert isinstance(model, ShallowReLUModel)
        assert isinstance(model.hidden_linear, nn.Linear)
        assert model.hidden_linear.out_features == 3
    else:
        assert isinstance(model, DeepReLUPlainModel)
        relu_count = sum(isinstance(layer, nn.ReLU) for layer in model.sequential)
        assert relu_count == n_layers
        assert len(model.sequential) == 2 * n_layers + 1


def test_architecture_producers_preserve_function_and_sweep_order() -> None:
    width = produce_width_experiment(
        compose_mlp_config("config_test_mlp_width_experiment")
    )
    depth = produce_depth_experiment(
        compose_mlp_config("config_test_mlp_depth_experiment")
    )

    assert [case.function_name for case in width.cases[:2]] == ["cubic", "cubic"]
    assert [case.function_name for case in width.cases[::2]] == [
        "cubic",
        "gaussian",
        "discontinuous",
        "periodic",
    ]
    assert [case.architecture_size for case in width.cases[:2]] == [2, 4]
    assert all(case.n_layers == 1 for case in width.cases)
    assert width.cases[0].train_x.shape == (8, 1)
    assert width.sequence_size == 4
    assert [case.architecture_size for case in depth.cases[:2]] == [1, 2]
    assert [case.n_layers for case in depth.cases[:2]] == [1, 2]
    assert all(case.hidden_layer_dim == 2 for case in depth.cases)
    assert depth.sequence_size == 2
    assert width.settings.seeds == (42, 43)
    assert width.settings.output_schedule == (1, 2, 3)


@pytest.mark.parametrize("kind", ["width", "depth"])
def test_one_epoch_training_matches_manual_pre_update_and_post_update_values(
    kind: ArchitectureKind,
) -> None:
    config_name = f"config_test_mlp_{kind}_experiment"
    overrides = [
        "function_names=[cubic]",
        "lr_search_seeds=[42]",
        "lr_grid=[0.01]",
        "step_size_fractions=[0.5]",
        "lr_search_epochs=1",
        "lr_eval_tail_fraction=1.0",
        "final_n_epochs=1",
        "output_schedule_by_epoch=[1]",
        "smoothed_loss_min_epoch=2",
    ]
    if kind == "width":
        overrides.extend(["hidden_layer_dims=[2]", "sequence_hidden_layer_dim=2"])
    else:
        overrides.extend(["depth_layer_grid=[2]", "depth_to_plot_sequence=2"])
    producer = produce_width_experiment if kind == "width" else produce_depth_experiment
    experiment = producer(compose_mlp_config(config_name, overrides))

    result = evaluate_architecture_experiment(experiment).results[0]
    case = result.case
    torch.manual_seed(42)
    model = build_architecture_model(
        kind=kind,
        hidden_layer_dim=case.hidden_layer_dim,
        n_layers=case.n_layers,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    prediction = model(case.train_x)
    loss = loss_fn(prediction, case.train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_final_loss = loss_fn(model(case.train_x), case.train_y).item()

    assert result.hyperparameters.learning_rate == 0.01
    assert result.hyperparameters.step_size_fraction == 0.5
    assert result.selected_seed == 42
    assert result.seed_final_losses == ((42, expected_final_loss),)
    assert result.output_epochs == (1,)
    assert torch.equal(result.predictions[0], prediction.detach())
    assert torch.equal(result.output_xs[0], case.train_x)
    expected_mae = torch.abs(prediction.detach() - case.train_y).mean()
    assert torch.allclose(result.mae_errors, expected_mae.reshape(1))
    assert result.smoothed_losses == (expected_mae.item(),)


def test_search_selects_the_lowest_independently_calculated_tail_score() -> None:
    cfg = compose_mlp_config(
        "config_test_mlp_width_experiment",
        [
            "function_names=[cubic]",
            "hidden_layer_dims=[2]",
            "sequence_hidden_layer_dim=2",
            "lr_search_seeds=[42]",
            "lr_grid=[0.0, 0.01]",
            "step_size_fractions=[0.5]",
            "lr_search_epochs=2",
            "lr_eval_tail_fraction=0.5",
            "final_n_epochs=1",
            "output_schedule_by_epoch=[1]",
        ],
    )
    experiment = produce_width_experiment(cfg)
    case = experiment.cases[0]
    expected_scores = {}
    for learning_rate in experiment.settings.lr_grid:
        torch.manual_seed(42)
        model = build_architecture_model("width", hidden_layer_dim=2)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        losses = []
        for _ in range(2):
            prediction = model(case.train_x)
            loss = loss_fn(prediction, case.train_y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        expected_scores[learning_rate] = losses[-1]
    expected_lr = min(expected_scores, key=lambda lr: expected_scores[lr])

    result = evaluate_architecture_experiment(experiment).results[0]

    assert result.hyperparameters.learning_rate == expected_lr
    assert result.search_loss == pytest.approx(expected_scores[expected_lr])
    other_score = expected_scores[0.01 if expected_lr == 0.0 else 0.0]
    assert result.other_losses.minimum == pytest.approx(other_score)
    assert result.other_losses.mean == pytest.approx(other_score)
    assert result.other_losses.maximum == pytest.approx(other_score)


@pytest.mark.parametrize("kind", ["width", "depth"])
def test_seeded_architecture_evaluation_is_reproducible(kind: str) -> None:
    config_name = f"config_test_mlp_{kind}_experiment"
    overrides = ["function_names=[cubic]"]
    if kind == "width":
        overrides.extend(["hidden_layer_dims=[2]", "sequence_hidden_layer_dim=2"])
    else:
        overrides.extend(["depth_layer_grid=[2]", "depth_to_plot_sequence=2"])
    producer = produce_width_experiment if kind == "width" else produce_depth_experiment
    experiment = producer(compose_mlp_config(config_name, overrides))

    first = evaluate_architecture_experiment(experiment).results[0]
    second = evaluate_architecture_experiment(experiment).results[0]

    assert first.selected_seed == second.selected_seed
    assert first.seed_final_losses == second.seed_final_losses
    assert first.hyperparameters == second.hyperparameters
    expected_raw_mae = torch.abs(first.predictions[0] - first.case.train_y).mean()
    assert torch.allclose(first.mae_errors[0], expected_raw_mae)
    assert first.mae_errors[1].item() == pytest.approx(first.smoothed_losses[1])
    assert first.mae_errors[2].item() == pytest.approx(first.smoothed_losses[2])
    assert np.array_equal(first.mae_errors.numpy(), second.mae_errors.numpy())
    for first_prediction, second_prediction in zip(
        first.predictions, second.predictions, strict=True
    ):
        assert torch.equal(first_prediction, second_prediction)


@pytest.mark.parametrize(
    ("config_name", "override"),
    [
        ("config_test_mlp_width_experiment", "n_samples=0"),
        ("config_test_mlp_width_experiment", "hidden_layer_dims=[]"),
        ("config_test_mlp_depth_experiment", "depth_layer_grid=[]"),
        ("config_test_mlp_depth_experiment", "lr_search_epochs=0"),
        ("config_test_mlp_width_experiment", "output_schedule_by_epoch=[4]"),
    ],
)
def test_architecture_input_validation(config_name: str, override: str) -> None:
    cfg = compose_mlp_config(config_name, [override])
    producer = (
        produce_width_experiment if "width" in config_name else produce_depth_experiment
    )

    with pytest.raises(ValueError):
        producer(cfg)


def test_production_architecture_settings_retain_legacy_values() -> None:
    width = produce_width_experiment(compose_mlp_config("config_mlp_width_experiment"))
    depth = produce_depth_experiment(compose_mlp_config("config_mlp_depth_experiment"))

    assert [case.architecture_size for case in width.cases[:8]] == [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
    ]
    assert [case.architecture_size for case in depth.cases[:6]] == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]
    assert width.settings.lr_grid == (
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
    )
    assert width.settings.step_size_fractions == (0.1, 0.25, 0.5)
    assert width.settings.search_epochs == 3000
    assert width.settings.final_epochs == 30000
    assert width.settings.seeds == (42, 43, 44)
    assert width.settings.output_schedule == (1, 10, 50, 100, 1000, 10000)
    assert width.settings.eval_tail_fraction == 0.1
    assert width.settings.ema_beta == 0.9
    assert width.settings.smoothed_loss_min_epoch == 1000
    assert depth.settings == width.settings
