import matplotlib.pyplot as plt
import numpy as np
import pytest

from spbunn.mlp.architecture import (
    evaluate_architecture_experiment,
    produce_depth_experiment,
    produce_width_experiment,
)
from spbunn.mlp.propagation import (
    evaluate_input_gradients,
    evaluate_propagation,
    produce_balduzzi_input,
    produce_propagation_input,
)
from spbunn.reporting.mlp import (
    build_balduzzi_report,
    build_hidden_norms_report,
    build_mlp_depth_report,
    build_mlp_width_report,
)
from tests.mlp_helpers import compose_mlp_config


def _close_report(report: object) -> None:
    for figure in report.figures:  # type: ignore[attr-defined]
        plt.close(figure.figure)


@pytest.mark.parametrize("kind", ["width", "depth"])
def test_architecture_report_has_legacy_stems_labels_and_explanations(
    kind: str,
) -> None:
    config_name = f"config_test_mlp_{kind}_experiment"
    overrides = ["function_names=[cubic]"]
    if kind == "width":
        overrides.extend(["hidden_layer_dims=[2]", "sequence_hidden_layer_dim=2"])
    producer = produce_width_experiment
    builder = build_mlp_width_report
    prefix = "mlp_conv"
    architecture_label = "$D = 2$"
    if kind == "depth":
        overrides.extend(["depth_layer_grid=[2]", "depth_to_plot_sequence=2"])
        producer = produce_depth_experiment
        builder = build_mlp_depth_report
        prefix = "mlp_depth_conv"
        architecture_label = "$L = 2$"
    cfg = compose_mlp_config(config_name, overrides)
    result = evaluate_architecture_experiment(producer(cfg))

    report = builder(
        result, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )

    try:
        assert report.title == cfg.report.title
        assert report.artifact_name == cfg.report.artifact_name
        assert [figure.artifact_name for figure in report.figures] == [
            f"{prefix}_wrt_loss_cubic",
            f"{prefix}_wrt_mae_cubic",
        ]
        prediction_axis = report.figures[0].figure.axes[0]
        convergence_axis = report.figures[1].figure.axes[0]
        assert prediction_axis.get_xlabel() == "$x$"
        assert prediction_axis.get_ylabel() == "$f(x)$"
        assert len(prediction_axis.lines) == 4
        assert [line.get_label() for line in prediction_axis.lines[1:]] == [
            "Epoch = 1",
            "Epoch = 2",
            "Epoch = 3",
        ]
        assert convergence_axis.get_xlabel() == "Epochs"
        assert convergence_axis.get_ylabel() == "MAE"
        assert convergence_axis.get_xscale() == "log"
        assert convergence_axis.get_yscale() == "log"
        assert convergence_axis.lines[0].get_label() == architecture_label
        assert "pre-update" in report.html_content
        assert "post-training MSE" in report.html_content
        assert "EMA" in report.html_content
        assert r"\operatorname{ReLU}" in report.html_content
        assert "AdamW" in report.html_content
    finally:
        _close_report(report)


def test_hidden_norms_report_has_six_ordered_diagnostic_figures() -> None:
    cfg = compose_mlp_config("config_test_hidden_norms")
    result = evaluate_propagation(produce_propagation_input(cfg))

    report = build_hidden_norms_report(
        result, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )

    try:
        assert report.title == cfg.report.title
        assert report.artifact_name == cfg.report.artifact_name
        assert [figure.artifact_name for figure in report.figures] == [
            "hidden_norms_init_forward_relu",
            "hidden_norms_init_backward_relu",
            "hidden_norms_conditioning_init_sensitivity_relu",
            "hidden_norms_conditioning_init_jacobian_relu",
            "hidden_norms_conditioning_sensitivity_relu",
            "hidden_norms_conditioning_jacobian_relu",
        ]
        axes = [figure.figure.axes[0] for figure in report.figures]
        assert all(axis.get_yscale() == "log" for axis in axes)
        assert axes[0].get_xlabel() == "Layer index $l$"
        assert axes[0].get_ylabel() == r"$\|\xi_l\|_2$ (batch mean)"
        assert axes[1].get_ylabel() == r"$\|dL/d\xi_l\|_2$ (batch mean)"
        assert axes[2].get_xlabel() == "Number of layers $M$"
        assert axes[3].get_ylabel() == "input-Jacobian condition number"
        assert len(axes[0].lines) == 3
        assert len(axes[4].lines) == 4
        assert r"H\sigma^2/2" in report.html_content
        assert "zero-mean approximations" in report.html_content
        assert "mean of per-input ratios" in report.html_content
        assert r"1/\sqrt{M}" in report.html_content
    finally:
        _close_report(report)


def test_balduzzi_report_has_plain_and_residual_gradient_curves() -> None:
    cfg = compose_mlp_config("config_test_balduzzi_analysis")
    result = evaluate_input_gradients(produce_balduzzi_input(cfg))

    report = build_balduzzi_report(
        result, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )

    try:
        assert report.title == cfg.report.title
        assert report.artifact_name == cfg.report.artifact_name
        assert [figure.artifact_name for figure in report.figures] == [
            "balduzzi_analysis_plain_network",
            "balduzzi_analysis_res_network",
        ]
        for curve, figure in zip(result.curves, report.figures, strict=True):
            axis = figure.figure.axes[0]
            assert axis.get_xlabel() == "$x$"
            assert axis.get_ylabel() == "$dh / dx$"
            assert len(axis.lines) == 1
            assert np.asarray(axis.lines[0].get_xydata()).shape == (9, 2)
            assert axis.get_autoscale_on()
            assert axis.get_autoscalex_on()
            assert axis.get_autoscaley_on()
            assert axis.get_xlim() == pytest.approx((-1.1, 1.1))
            gradient = curve.gradient.numpy()
            margin = 0.05 * (float(gradient.max()) - float(gradient.min()))
            expected_ylim = (
                float(gradient.min()) - margin,
                float(gradient.max()) + margin,
            )
            assert axis.get_ylim() == pytest.approx(expected_ylim)
        assert "summed network output" in report.html_content
        assert r"h + \alpha\operatorname{ReLU}(Wh+b)" in report.html_content
        assert r"\alpha = 1" in report.html_content
        assert "seed 42" in report.html_content
    finally:
        _close_report(report)


def test_balduzzi_report_uses_configured_residual_branch_scale() -> None:
    cfg = compose_mlp_config(
        "config_test_balduzzi_analysis", ["residual_branch_scale=0.5"]
    )
    result = evaluate_input_gradients(produce_balduzzi_input(cfg))

    report = build_balduzzi_report(
        result, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )

    try:
        assert result.residual_branch_scale == 0.5
        assert r"\alpha = 0.5" in report.html_content
        assert "with unit branch scale" not in report.html_content
    finally:
        _close_report(report)
