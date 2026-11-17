from html import escape

import matplotlib.pyplot as plt
import numpy as np

from spbunn.optimization.landscape import LandscapeResult, OptimizerSettings
from spbunn.optimization.mini_batch import MiniBatchResult
from spbunn.optimization.plotting import (
    plot_loss_curve,
    plot_loss_landscape,
    plot_trajectory,
)
from spbunn.optimization.schedulers import ScheduleCurve
from spbunn.reporting.models import FigureModel, ReportModel


def _schedule_description(curve: ScheduleCurve) -> str:
    settings = curve.settings
    n_epochs = len(curve.epochs)
    if settings.schedule_type == "linear":
        transition = n_epochs // 2 - 1
        starting_lr = curve.learning_rates[0]
        plateau = curve.learning_rates[-1]
        return (
            f"<p>Through epoch {transition}, the line follows "
            rf"\(\gamma_e = {starting_lr:g} - {settings.decrement:g}e\). "
            f"It then stays at {plateau:g}. The plateau is an imposed "
            "teaching transition.</p>"
        )
    if settings.schedule_type == "step":
        initial_lr = curve.learning_rates[0]
        starting_lr = initial_lr * settings.alpha
        plateau_start = (
            (settings.last_epoch - 1) // settings.step_size
        ) * settings.step_size
        return (
            f"<p>The initial rate is {initial_lr:g}, because the curve starts "
            f"with the configured starting rate {starting_lr:g} divided by "
            rf"\(\alpha = {settings.alpha:g}\). The rate is multiplied by "
            rf"\(\alpha\) every {settings.step_size} epochs until epoch "
            f"{plateau_start}. It then stays at {curve.learning_rates[-1]:g}. "
            "These steps are imposed by the teaching configuration.</p>"
        )
    if settings.schedule_type == "reduce_on_plateau":
        first_end = n_epochs // 4
        second_end = 3 * n_epochs // 4
        return (
            f"<p>The rate changes at epochs {first_end} and {second_end}. "
            r"The three levels are \(\gamma_0\), \(\alpha\gamma_0\), and "
            rf"\(\alpha^2\gamma_0\) with \(\alpha = {settings.alpha:g}\). "
            "This synthetic curve does not read a measured loss. Its "
            "transitions only illustrate a reduce-on-plateau pattern.</p>"
        )
    if settings.schedule_type == "cosine_annealing":
        quarter_length = n_epochs // settings.repetitions
        starting_lr = curve.learning_rates[0] / 1.5
        return (
            "<p>One quarter uses "
            rf"\(\gamma_e = \gamma_0 + \frac{{\gamma_0}}{{4}}"
            rf"(1 + \cos(\pi r/{quarter_length}))\), where \(r\) is the "
            rf"epoch within the quarter and \(\gamma_0 = {starting_lr:g}\). "
            f"The same {quarter_length}-point quarter repeats "
            f"{settings.repetitions} times, so the jumps are intentional.</p>"
        )
    raise ValueError(f"Unsupported schedule: {settings.schedule_type}")


def build_scheduler_report(
    curves: tuple[ScheduleCurve, ...], title: str, artifact_name: str
) -> ReportModel:
    """Build the scheduler teaching report without writing artifacts."""
    sections = [
        "<h1>Learning-rate schedules</h1>",
        (
            "<p>These synthetic teaching curves show the learning rate "
            r"\(\gamma_e\) at epoch \(e\). They illustrate schedule shapes; "
            "they are not measurements from a training run or direct traces "
            "of PyTorch scheduler calls.</p>"
        ),
    ]
    for curve in curves:
        sections.append(f"<section><h2>{escape(curve.settings.title)}</h2>")
        sections.append(_schedule_description(curve))
        sections.append("</section>")

    figures = []
    for curve in curves:
        figure, axis = plt.subplots(1, 1, figsize=(4, 3))
        axis.plot(
            curve.epochs,
            curve.learning_rates,
            "o--",
            color="tab:blue",
        )
        for annotation in curve.settings.annotations:
            axis.annotate(
                annotation.text,
                xy=annotation.xy,
                xytext=annotation.xytext,
                arrowprops={
                    "facecolor": "black",
                    "width": 1,
                    "headwidth": 10,
                    "shrink": 0.05,
                },
            )
        axis.grid(True)
        axis.set_title(curve.settings.title)
        axis.set_xlabel("Epochs", fontsize=12)
        axis.set_ylabel(r"$\gamma$", fontsize=12)
        figure.tight_layout()
        figures.append(
            FigureModel(
                figure=figure,
                artifact_name=f"scheduler_{curve.settings.name}",
            )
        )
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=tuple(figures),
        artifact_name=artifact_name,
    )


def _combined_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate(arrays)
    return float(values.min()), float(values.max())


def _optimizer_description(settings: OptimizerSettings) -> str:
    learning_rate = settings.learning_rate
    if settings.optimizer_name == "sgd" and settings.nesterov:
        return (
            rf"<p>With \(\eta = {learning_rate:g}\) and "
            rf"\(\mu = {settings.momentum:g}\), Nesterov momentum uses "
            r"\(b_t = \mu b_{t-1} + g_t\) and "
            r"\(\theta_{t+1} = \theta_t - \eta(g_t + \mu b_t)\).</p>"
        )
    if settings.optimizer_name == "sgd" and settings.momentum:
        return (
            rf"<p>With \(\eta = {learning_rate:g}\) and "
            rf"\(\mu = {settings.momentum:g}\), classical momentum uses "
            r"\(b_t = \mu b_{t-1} + g_t\) and "
            r"\(\theta_{t+1} = \theta_t - \eta b_t\).</p>"
        )
    if settings.optimizer_name == "sgd":
        return (
            rf"<p>Plain SGD uses \(\theta_{{t+1}} = \theta_t - \eta g_t\) "
            rf"with \(\eta = {learning_rate:g}\) and no momentum.</p>"
        )
    if settings.optimizer_name == "rmsprop":
        return (
            rf"<p>RMSprop uses \(\eta = {learning_rate:g}\), "
            r"\(v_t = 0.99v_{t-1} + 0.01g_t^2\), and "
            r"\(\theta_{t+1} = \theta_t - \eta g_t/"
            r"(\sqrt{v_t} + 10^{-8})\). It uses no momentum or centering.</p>"
        )
    if settings.optimizer_name == "adam":
        beta_1, beta_2 = settings.betas
        return (
            rf"<p>Adam uses \(\eta = {learning_rate:g}\), "
            rf"\(\beta_1 = {beta_1:g}\), \(\beta_2 = {beta_2:g}\), and "
            r"\(\epsilon = 10^{-8}\). It applies bias correction to the "
            "first and second gradient moments before the parameter update.</p>"
        )
    raise ValueError(f"Unsupported optimizer: {settings.optimizer_name}")


def build_loss_landscape_report(
    results: tuple[LandscapeResult, ...], title: str, artifact_name: str
) -> ReportModel:
    """Build full-batch landscape and trajectory figures."""
    sections = [
        "<h1>Loss landscapes and full-batch optimization</h1>",
        (
            r"<p>For each grid cell, the mean squared error is \(L(\theta) = "
            r"\frac{1}{N}\sum_{i=1}^{N}(f_\theta(x_i) - y_i)^2\). "
            r"The gradient \(g_t = \nabla_\theta L(\theta_t)\) uses the "
            r"complete ordered data set. Contours span \(\theta_0\) and "
            r"\(\theta_1\). The orange point is the best sampled "
            "grid cell, and the green star is the initial point. Loss values "
            "are measured before each update.</p>"
        ),
    ]
    for result in results:
        for run in result.runs:
            sections.append(
                f"<section><h2>{escape(result.experiment.name)}: "
                f"{escape(run.settings.name)}</h2>"
            )
            sections.append(_optimizer_description(run.settings))
            sections.append("</section>")

    figures = []
    for result in results:
        experiment = result.experiment
        if not result.runs:
            continue
        xlim = _combined_limits(
            [run.trajectory_theta_0 for run in result.runs]
            + [experiment.theta_0_values]
        )
        ylim = _combined_limits(
            [run.trajectory_theta_1 for run in result.runs]
            + [experiment.theta_1_values]
        )
        max_steps = max(len(run.losses) for run in result.runs)
        for run in result.runs:
            figure, (landscape_axis, loss_axis) = plt.subplots(
                2,
                1,
                figsize=(8, 8),
                height_ratios=[3, 1],
            )
            plot_loss_landscape(
                landscape_axis,
                experiment.theta_0_values,
                experiment.theta_1_values,
                result.loss_values,
                experiment.levels,
                xlim,
                ylim,
            )
            plot_trajectory(
                landscape_axis,
                run.trajectory_theta_0,
                run.trajectory_theta_1,
                color="red",
            )
            landscape_axis.set_title(f"{experiment.name}: {run.settings.name}")
            plot_loss_curve(loss_axis, run.losses, color="tab:blue")
            loss_axis.set_xlim(1, float(max_steps))
            figure.tight_layout()
            figures.append(
                FigureModel(
                    figure=figure,
                    artifact_name=(
                        f"loss_landscape_func_{experiment.name}_"
                        f"{run.settings.suffix}"
                    ),
                )
            )
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=tuple(figures),
        artifact_name=artifact_name,
    )


def _learning_rate_stem(learning_rate: float) -> str:
    return (
        f"{learning_rate:.1e}".replace(".0e", "e")
        .replace("e+0", "e")
        .replace("e-0", "e_")
        .replace("e+", "e")
        .replace(".", "_")
        .replace("-", "_")
    )


def build_simple_optimization_report(
    results: tuple[MiniBatchResult, ...], title: str, artifact_name: str
) -> ReportModel:
    """Build mini-batch optimization histories and landscape figures."""
    sections = [
        "<h1>Simple mini-batch loss optimization</h1>",
        (
            r"<p>For each mini-batch \(B\), the loss is \(L_B(\theta) = "
            r"\frac{1}{|B|}\sum_{i \in B}(f_\theta(x_i) - y_i)^2\). "
            r"SGD applies \(\theta_{t+1} = \theta_t - "
            r"\eta\nabla_\theta L_B(\theta_t)\). The lower panel records "
            "the loss before each update.</p>"
        ),
        (
            "<p>The configured NumPy seed permutes the training grid before "
            "mini-batches are read. DataLoader shuffling is disabled. The "
            "landscape uses a separate ordered regression data set on its "
            "configured input interval. The red path shows the parameter "
            r"history over \(\theta_0\) and \(\theta_1\).</p>"
        ),
    ]
    for result in results:
        mini_batch = result.experiment
        sections.append(
            f"<p><strong>{escape(mini_batch.landscape.name)}</strong>: "
            f"{mini_batch.n_optimization_steps} updates, batch size "
            f"{mini_batch.batch_size}, learning rate "
            f"{mini_batch.landscape.learning_rate}.</p>"
        )

    figures = []
    for result in results:
        landscape_input = result.experiment.landscape
        figure, (landscape_axis, loss_axis) = plt.subplots(
            2,
            1,
            figsize=(8, 8),
            height_ratios=[3, 1],
        )
        xlim = _combined_limits(
            [landscape_input.theta_0_values, result.theta_history[:, 0]]
        )
        ylim = _combined_limits(
            [landscape_input.theta_1_values, result.theta_history[:, 1]]
        )
        plot_loss_landscape(
            landscape_axis,
            landscape_input.theta_0_values,
            landscape_input.theta_1_values,
            result.loss_values,
            landscape_input.levels,
            xlim,
            ylim,
        )
        landscape_axis.plot(
            result.theta_history[:, 0],
            result.theta_history[:, 1],
            "o--",
            color="red",
        )
        landscape_axis.set_title(f"{landscape_input.name}: mini-batch SGD")
        plot_loss_curve(
            loss_axis,
            result.loss_history,
            color="tab:blue",
            fixed_limits=False,
        )
        figure.tight_layout()
        artifact_stem = (
            f"sgd_batch_{result.experiment.batch_size}_lr_"
            f"{_learning_rate_stem(landscape_input.learning_rate)}_"
            f"{landscape_input.name}"
        )
        figures.append(FigureModel(figure=figure, artifact_name=artifact_stem))
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=tuple(figures),
        artifact_name=artifact_name,
    )
