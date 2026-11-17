import matplotlib.pyplot as plt
from matplotlib.text import Annotation

from spbunn.optimization.landscape import (
    evaluate_landscape_experiment,
    produce_landscape_inputs,
)
from spbunn.optimization.mini_batch import (
    evaluate_mini_batch,
    produce_mini_batch_inputs,
)
from spbunn.optimization.schedulers import evaluate_schedules, produce_scheduler_input
from spbunn.reporting.optimization import (
    build_loss_landscape_report,
    build_scheduler_report,
    build_simple_optimization_report,
)
from tests.optimization_helpers import compose_optimization_config


def test_scheduler_report_has_ordered_annotations_and_latex() -> None:
    cfg = compose_optimization_config("config_test_schedulers")
    curves = evaluate_schedules(produce_scheduler_input(cfg))

    report = build_scheduler_report(
        curves, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )

    try:
        assert report.title == cfg.report.title
        assert report.artifact_name == cfg.report.artifact_name
        assert len(report.figures) == 4
        assert [figure.artifact_name for figure in report.figures] == [
            "scheduler_linearlr",
            "scheduler_steplr",
            "scheduler_reducelronplateau",
            "scheduler_cosineannealing",
        ]
        axes = [figure.figure.axes[0] for figure in report.figures]
        assert all(axis.get_xlabel() == "Epochs" for axis in axes)
        assert all(axis.get_ylabel() == r"$\gamma$" for axis in axes)
        assert [[text.get_text() for text in axis.texts] for axis in axes] == [
            ["Last epoch"],
            ["Last epoch"],
            ["Loss stagnation", "Loss st."],
            [],
        ]
        linear_annotation = axes[0].texts[0]
        step_annotation = axes[1].texts[0]
        assert isinstance(linear_annotation, Annotation)
        assert isinstance(step_annotation, Annotation)
        assert linear_annotation.xy == (19, 0.045)
        assert linear_annotation.get_position() == (25, 0.052)
        assert step_annotation.xy == (20, 0.02)
        assert axes[2].texts[1].get_position() == (32, 0.045)
        assert r"\(\gamma_e\)" in report.html_content
        assert "synthetic teaching curves" in report.html_content
        assert "The initial rate is 0.2" in report.html_content
        assert "does not read a measured loss" in report.html_content
        assert r"\cos" in report.html_content
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


def test_loss_landscape_report_has_shared_limits_and_stable_stems() -> None:
    cfg = compose_optimization_config("config_test_loss_landscape")
    results = tuple(
        evaluate_landscape_experiment(experiment)
        for experiment in produce_landscape_inputs(cfg)
    )

    report = build_loss_landscape_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )

    try:
        assert report.title == cfg.report.title
        assert report.artifact_name == cfg.report.artifact_name
        assert len(report.figures) == 5
        assert [figure.artifact_name for figure in report.figures] == [
            "loss_landscape_func_linear_sgd",
            "loss_landscape_func_linear_sgd_momentum",
            "loss_landscape_func_linear_sgd_nesterov",
            "loss_landscape_func_linear_rmsprop",
            "loss_landscape_func_linear_adam",
        ]
        x_limits = [figure.figure.axes[0].get_xlim() for figure in report.figures]
        y_limits = [figure.figure.axes[0].get_ylim() for figure in report.figures]
        assert all(item == x_limits[0] for item in x_limits)
        assert all(item == y_limits[0] for item in y_limits)
        assert all(
            figure.figure.axes[1].get_yscale() == "log" for figure in report.figures
        )
        assert r"\(\theta_0\)" in report.html_content
        assert "SGD + Nesterov momentum" in report.html_content
        assert r"\nabla_\theta" in report.html_content
        assert r"\beta_1 = 0.9" in report.html_content
        assert "bias correction" in report.html_content
    finally:
        for figure in report.figures:
            plt.close(figure.figure)


def test_simple_report_has_correct_parameter_labels() -> None:
    cfg = compose_optimization_config("config_test_simple_loss_optimization")
    results = tuple(
        evaluate_mini_batch(experiment) for experiment in produce_mini_batch_inputs(cfg)
    )

    report = build_simple_optimization_report(
        results, title=cfg.report.title, artifact_name=cfg.report.artifact_name
    )

    try:
        assert report.title == cfg.report.title
        assert report.artifact_name == cfg.report.artifact_name
        assert len(report.figures) == 1
        assert report.figures[0].artifact_name == "sgd_batch_4_lr_1e_2_linear"
        landscape_axis = report.figures[0].figure.axes[0]
        assert landscape_axis.get_xlabel() == r"$\theta_0$"
        assert landscape_axis.get_ylabel() == r"$\theta_1$"
        assert report.figures[0].figure.axes[1].get_yscale() == "log"
        assert "25" not in report.html_content
        assert "3 updates" in report.html_content
        assert r"\nabla_\theta" in report.html_content
        assert r"\(\theta_0\)" in report.html_content
        assert "configured NumPy seed permutes" in report.html_content
        assert "separate ordered regression data set" in report.html_content
    finally:
        for figure in report.figures:
            plt.close(figure.figure)
