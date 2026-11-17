from html import escape

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from spbunn.mlp.architecture import (
    ArchitectureExperimentResult,
    ArchitectureResult,
)
from spbunn.mlp.propagation import (
    BalduzziResult,
    ConditioningResult,
    InputGradientCurve,
    NormCurve,
    PropagationExperimentResult,
)
from spbunn.reporting.models import FigureModel, ReportModel

LABEL_FONTSIZE = 18
TICK_FONTSIZE = 14


def _architecture_groups(
    result: ArchitectureExperimentResult,
) -> tuple[tuple[str, tuple[ArchitectureResult, ...]], ...]:
    groups: dict[str, list[ArchitectureResult]] = {}
    for item in result.results:
        groups.setdefault(item.case.function_name, []).append(item)
    return tuple((name, tuple(items)) for name, items in groups.items())


def _prediction_figure(
    function_results: tuple[ArchitectureResult, ...], sequence_size: int, stem: str
) -> FigureModel:
    matching = [
        item
        for item in function_results
        if item.case.architecture_size == sequence_size
    ]
    if len(matching) != 1:
        raise ValueError("Each function needs one prediction-sequence result")
    result = matching[0]
    figure, axis = plt.subplots(1, 1, figsize=(6, 4))
    axis.plot(
        result.case.train_x.detach().numpy(),
        result.case.train_y.detach().numpy(),
        color="lightgrey",
        linewidth=12,
    )
    n_outputs = len(result.predictions)
    if n_outputs != len(result.output_epochs) or not n_outputs:
        raise ValueError(
            "Scheduled predictions and epochs must be non-empty and aligned"
        )
    for index, (x, y, epoch) in enumerate(
        zip(
            result.output_xs,
            result.predictions,
            result.output_epochs,
            strict=True,
        )
    ):
        axis.plot(
            x.detach().numpy(),
            y.detach().numpy(),
            color=matplotlib.colormaps["cool"](index / n_outputs),
            linewidth=4,
            alpha=0.5,
            label=f"Epoch = {epoch}",
        )
    axis.set_xlabel(r"$x$", fontsize=12)
    axis.set_ylabel(r"$f(x)$", fontsize=12)
    axis.grid()
    axis.legend()
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=stem)


def _convergence_figure(
    kind: str, function_results: tuple[ArchitectureResult, ...], stem: str
) -> FigureModel:
    figure, axis = plt.subplots(1, 1, figsize=(6, 4))
    for result in function_results:
        symbol = "D" if kind == "width" else "L"
        axis.loglog(
            result.output_epochs,
            result.mae_errors.detach().numpy(),
            "o--",
            linewidth=2,
            markersize=8,
            label=rf"${symbol} = {result.case.architecture_size}$",
        )
    axis.set_xlabel("Epochs", fontsize=12)
    axis.set_ylabel("MAE", fontsize=12)
    axis.grid()
    axis.legend()
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=stem)


def _build_architecture_report(
    result: ArchitectureExperimentResult, title: str, artifact_name: str
) -> ReportModel:
    if result.kind not in {"width", "depth"}:
        raise ValueError(f"Unsupported architecture report: {result.kind}")
    is_width = result.kind == "width"
    heading = "MLP width and convergence" if is_width else "MLP depth and convergence"
    architecture_equation = (
        r"\(\hat y = W_2\operatorname{ReLU}(W_1x+b_1)+b_2\). "
        "The shallow model has one hidden layer."
        if is_width
        else r"\(h_l = \operatorname{ReLU}(W_lh_{l-1}+b_l)\). "
        "Every configured hidden layer is followed by ReLU."
    )
    settings = result.settings
    sections = [
        f"<h1>{heading}</h1>",
        (
            "<p>Each ordered, noiseless regression set is evaluated by the "
            f"same {architecture_equation}</p>"
        ),
        (
            "<p>The search uses full-batch AdamW and StepLR. A configuration "
            "is scored by the mean pre-update MSE over the final "
            f"{settings.eval_tail_fraction:g} fraction of "
            f"{settings.search_epochs} search epochs, averaged over seeds "
            f"{settings.seeds}. Step sizes are "
            r"\(\max(1,\operatorname{round}(fE))\).</p>"
        ),
        (
            f"<p>Final training runs for {settings.final_epochs} epochs at "
            "every seed. The selected run has the lowest post-training MSE. "
            "Prediction curves contain pre-update outputs at scheduled "
            "epochs. Convergence uses raw MAE below epoch "
            f"{settings.smoothed_loss_min_epoch} and the MAE exponential "
            f"moving average (EMA, beta {settings.ema_beta:g}) from that "
            "epoch onward.</p>"
        ),
    ]
    figures = []
    prefix = "mlp_conv" if is_width else "mlp_depth_conv"
    symbol = "D" if is_width else "L"
    for function_name, function_results in _architecture_groups(result):
        sequence_result = next(
            item
            for item in function_results
            if item.case.architecture_size == result.sequence_size
        )
        sections.extend(
            [
                f"<section><h2>{escape(function_name)}</h2>",
                (
                    rf"<p>The prediction panel uses \({symbol} = "
                    rf"{result.sequence_size}\), seed "
                    rf"{sequence_result.selected_seed}, learning rate "
                    rf"{sequence_result.hyperparameters.learning_rate:g}, "
                    "and the configured step-size fraction "
                    rf"{sequence_result.hyperparameters.step_size_fraction:g}. "
                    "The convergence panel overlays the complete ordered "
                    "sweep.</p></section>"
                ),
            ]
        )
        figures.append(
            _prediction_figure(
                function_results,
                result.sequence_size,
                f"{prefix}_wrt_loss_{function_name}",
            )
        )
        figures.append(
            _convergence_figure(
                result.kind,
                function_results,
                f"{prefix}_wrt_mae_{function_name}",
            )
        )
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=tuple(figures),
        artifact_name=artifact_name,
    )


def build_mlp_width_report(
    result: ArchitectureExperimentResult, title: str, artifact_name: str
) -> ReportModel:
    """Build the shallow-width experiment report without writing files."""
    if result.kind != "width":
        raise ValueError("The width report requires width experiment results")
    return _build_architecture_report(result, title, artifact_name)


def build_mlp_depth_report(
    result: ArchitectureExperimentResult, title: str, artifact_name: str
) -> ReportModel:
    """Build the deep-plain-network report without writing files."""
    if result.kind != "depth":
        raise ValueError("The depth report requires depth experiment results")
    return _build_architecture_report(result, title, artifact_name)


def _style_norm_axis(axis: Axes, ylabel: str) -> None:
    axis.set_xlabel("Layer index $l$", fontsize=LABEL_FONTSIZE)
    axis.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=LABEL_FONTSIZE)
    axis.tick_params(labelsize=TICK_FONTSIZE)


def _norm_figure(
    curves: tuple[NormCurve, ...], norm_key: str, ylabel: str, stem: str
) -> FigureModel:
    figure, axis = plt.subplots(figsize=(7, 5))
    figure.dpi = 150
    for index, curve in enumerate(curves):
        mean = curve.norms[f"{norm_key}_mean"]
        std = curve.norms[f"{norm_key}_std"]
        xs = np.arange(1, len(mean) + 1)
        color = f"C{index}"
        axis.plot(
            xs,
            mean,
            "-o",
            markersize=6,
            linewidth=2.5,
            color=color,
            label=curve.label,
        )
        axis.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)
    _style_norm_axis(axis, ylabel)
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=stem)


def _conditioning_figure(
    result: ConditioningResult,
    entries: tuple[tuple[str, str], ...],
    depth_grid: tuple[int, ...],
    metric: str,
    ylabel: str,
    stem: str,
    use_explicit_colors: bool = False,
) -> FigureModel:
    figure, axis = plt.subplots(figsize=(7, 5))
    figure.dpi = 150
    for index, (key, label) in enumerate(entries):
        options = {"color": f"C{index}"} if use_explicit_colors else {}
        axis.semilogy(
            np.asarray(depth_grid),
            result.values[(key, metric)],
            "o-",
            markersize=6,
            linewidth=2.5,
            label=label,
            **options,
        )
    axis.set_xlabel("Number of layers $M$", fontsize=LABEL_FONTSIZE)
    axis.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=LABEL_FONTSIZE)
    axis.tick_params(labelsize=TICK_FONTSIZE)
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=stem)


def build_hidden_norms_report(
    result: PropagationExperimentResult, title: str, artifact_name: str
) -> ReportModel:
    """Build initialization, sensitivity, and Jacobian diagnostic figures."""
    experiment = result.experiment
    sections = [
        "<h1>Signal, gradient, and input-Jacobian propagation</h1>",
        (
            r"<p>Inputs are independent standard normal samples. The output "
            r"adjoint is also isotropic normal noise, so the backward norms "
            r"probe the Jacobian chain instead of a supervised target. "
            r"Curves are averaged over the configured seeds "
            rf"{experiment.seeds}.</p>"
        ),
        "<h2>Initialization and norm propagation</h2>",
        (
            r"<p>For zero-bias ReLU networks, the documented theory gives "
            r"\(E\|h_l\|^2 = d(H\sigma^2/2)^l\). The forward factor is "
            r"\(H\sigma^2/2\), so Kaiming initialization "
            r"\(\sigma^2=2/H\) makes it one. The backward factor contains "
            r"the finite-width \(H+1\) correction and compounds over the "
            r"remaining layers. These are zero-mean approximations: positive "
            r"ReLU means shift empirical curves near criticality. Bands show "
            r"the population standard deviation across seeds.</p>"
        ),
        "<h2>Empirical sensitivity and exact conditioning</h2>",
        (
            r"<p>Finite differences measure "
            r"\(\|f_M(x+\Delta x)-f_M(x)\|_2/\|\Delta x\|_2\). "
            rf"The perturbation norm is {experiment.perturbation_norm:g}. "
            r"Autograd separately computes the exact input Jacobian "
            r"\(J_M=dh_M/dx\). The condition number is "
            r"\(\sigma_{\max}(J_M)/\sigma_{\min}(J_M)\); the report averages "
            r"the mean of per-input ratios, not the ratio of mean singular "
            r"values. Residual branches use scales \(1\), \(1/\sqrt{M}\), "
            r"and \(1/M\).</p>"
        ),
    ]
    figures = []
    activation = experiment.activation
    if result.init_curves:
        figures.extend(
            [
                _norm_figure(
                    result.init_curves,
                    "hidden_pre_norm",
                    r"$\|\xi_l\|_2$ (batch mean)",
                    f"hidden_norms_init_forward_{activation}",
                ),
                _norm_figure(
                    result.init_curves,
                    "grad_input_norm",
                    r"$\|dL/d\xi_l\|_2$ (batch mean)",
                    f"hidden_norms_init_backward_{activation}",
                ),
            ]
        )
    if result.plain_conditioning is not None:
        plain_entries = (
            ("1.0/sqrt(N)", r"$\sigma^2 = 1/N$"),
            ("Kaiming sqrt(2/N)", r"$\sigma^2 = 2/N$ (Kaiming)"),
            ("1.0 (unscaled)", r"$\sigma^2 = 1$"),
        )
        figures.extend(
            [
                _conditioning_figure(
                    result.plain_conditioning,
                    plain_entries,
                    experiment.depth_grid,
                    "empirical_sensitivity",
                    r"sensitivity of $\xi_M$ to the input",
                    f"hidden_norms_conditioning_init_sensitivity_{activation}",
                ),
                _conditioning_figure(
                    result.plain_conditioning,
                    plain_entries,
                    experiment.depth_grid,
                    "condition_number",
                    "input-Jacobian condition number",
                    f"hidden_norms_conditioning_init_jacobian_{activation}",
                ),
            ]
        )
    if result.residual_conditioning is not None:
        residual_entries = (
            ("plain", "plain"),
            ("residual (branch x1)", r"residual, branch $\times 1$"),
            (
                "residual (branch x1/sqrt(L))",
                r"residual, branch $\times 1/\sqrt{M}$",
            ),
            ("residual (branch x1/L)", r"residual, branch $\times 1/M$"),
        )
        figures.extend(
            [
                _conditioning_figure(
                    result.residual_conditioning,
                    residual_entries,
                    experiment.depth_grid,
                    "empirical_sensitivity",
                    r"sensitivity of $\xi_M$ to the input",
                    f"hidden_norms_conditioning_sensitivity_{activation}",
                    use_explicit_colors=True,
                ),
                _conditioning_figure(
                    result.residual_conditioning,
                    residual_entries,
                    experiment.depth_grid,
                    "condition_number",
                    "input-Jacobian condition number",
                    f"hidden_norms_conditioning_jacobian_{activation}",
                    use_explicit_colors=True,
                ),
            ]
        )
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=tuple(figures),
        artifact_name=artifact_name,
    )


def _gradient_figure(curve: InputGradientCurve, artifact_name: str) -> FigureModel:
    figure, axis = plt.subplots(1, 1, figsize=(5, 3))
    axis.plot(
        curve.inputs.detach().numpy(),
        curve.gradient.detach().numpy(),
        "o--",
        markersize=4,
    )
    axis.grid()
    axis.set_xlabel(r"$x$", fontsize=12)
    axis.set_ylabel(r"$dh / dx$", fontsize=12)
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=artifact_name)


def build_balduzzi_report(
    result: BalduzziResult, title: str, artifact_name: str
) -> ReportModel:
    """Build plain and residual input-gradient figures."""
    sections = [
        "<h1>Balduzzi input-gradient analysis</h1>",
        (
            "<p>For every scalar grid point, the curve is the derivative of "
            "that point's network output with respect to its input. Autograd "
            "obtains all pointwise derivatives by differentiating the summed "
            r"network output \(\sum_i h(x_i)\). Deep plain ReLU networks can "
            "exhibit rapidly changing input-gradient structure; residual "
            "connections provide an identity path through square hidden "
            "blocks.</p>"
        ),
        (
            r"<p>The residual blocks use \(h + \operatorname{ReLU}(Wh+b)\) "
            "with unit branch scale. Both networks have the configured "
            f"same width and depth. The producer uses seed {result.seed} "
            "before constructing the plain network and then the residual "
            "network. Axes retain Matplotlib autoscaling.</p>"
        ),
    ]
    figures = tuple(
        _gradient_figure(curve, curve.artifact_name) for curve in result.curves
    )
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=figures,
        artifact_name=artifact_name,
    )
