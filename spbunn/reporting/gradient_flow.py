"""HTML and figures for the matched RNN/LSTM gradient-flow audit."""

import math
from collections.abc import Sequence
from html import escape
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator

from spbunn.reporting.models import FigureModel, ReportModel
from spbunn.rnn.gradient_audit import LSTMGradientAudit, RNNGradientAudit
from spbunn.rnn.gradient_flow_pipeline import (
    AnyGradientAudit,
    GradientFlowCaseResult,
    GradientFlowExperimentResult,
)
from spbunn.rnn.tasks import copy_memoryless_loss

FIGURE_STEMS = (
    "gradient_flow_training",
    "gradient_flow_per_position",
    "gradient_flow_states",
    "gradient_flow_cell_path",
    "gradient_flow_forget_gates",
)
DISPLAY_ZERO_FLOOR = 1e-160
ARM_COLORS = {
    "default": "#e76f51",
    "orthogonal": "#9b5de5",
    "lstm": "#129c8c",
}
DIRECT_COLOR = "#1668b4"
SCALAR_COLOR = "#c78b24"
INITIAL_COLOR = "#7b8794"


class _SvgTextFigure(Figure):
    """Serialize labels as SVG text through a local plotting context."""

    def savefig(self, *args: Any, **kwargs: Any) -> Any:
        with plt.rc_context({"svg.fonttype": "none"}):
            return super().savefig(*args, **kwargs)


def build_gradient_flow_report(
    result: GradientFlowExperimentResult, title: str, artifact_name: str
) -> ReportModel:
    """Build the complete report without writing artifacts."""
    sections = [
        "<style>section{margin:0 0 2rem}table{border-collapse:collapse;margin:1rem 0}"
        "th,td{padding:.4rem .65rem;border-bottom:1px solid #dce4ef;text-align:right}"
        "th:first-child,td:first-child{text-align:left}figcaption{max-width:70rem}"
        ".note{padding:.8rem 1rem;border-left:4px solid #1668b4;background:#edf5fb}"
        "</style>",
        f"<h1>{escape(title)}</h1>",
        _overview(result),
        _mathematics_section(),
        _training_section(result),
        _audit_section(result),
        _verification_section(result),
        _limitations_section(),
    ]
    with plt.rc_context(
        {
            "font.size": 18.0,
            "axes.labelsize": 20.0,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "legend.fontsize": 14.0,
        }
    ):
        figures = (
            _training_figure(result),
            _per_position_figure(result),
            _state_gradient_figure(result),
            _cell_path_figure(result),
            _forget_gate_figure(result),
        )
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=figures,
        artifact_name=artifact_name,
    )


def _overview(result: GradientFlowExperimentResult) -> str:
    settings = result.settings
    return (
        "<section><h2>Matched copy-memory experiment</h2>"
        f"<p>Each sequence contains {settings.memorize_length} symbols, a delimiter, "
        "a fixed blank gap, and blank reproduction positions. Loss applies only to "
        f"reproduction. Separate runs use gaps {settings.gaps}. The optimal "
        "memoryless uniform predictor has cross-entropy "
        f"{copy_memoryless_loss(settings.symbol_count):.6f}.</p>"
        "<p>The arms are a default vanilla <code>tanh</code> RNN, an RNN with "
        "orthogonal recurrent initialization, and an LSTM with both forget biases "
        "set to zero. All arms receive identical training-data schedules and Adam "
        f"updates. Production uses {settings.training.updates} updates per run, "
        f"batch size {settings.batch_size}, hidden size {settings.hidden_size}, "
        f"learning rate {settings.training.learning_rate}, and seed "
        f"{settings.model_seed}.</p>"
        "<p>Equal hidden size does not mean equal parameter count. The RNN arms "
        "have 5,514 parameters and the LSTM has 20,106 at production size. Packed "
        "PyTorch modules make training efficient. Explicit equivalent recurrences "
        "retain each separate state for the audits.</p></section>"
    )


def _mathematics_section() -> str:
    return (
        "<section><h2>Measurements and GetSomeMath notation</h2>"
        r"<p>The vanilla RNN is \(\bm{h}_t=a(\bm{b}_h+\bm{W}_h\bm{h}_{t-1}+"
        r"\bm{W}_x\bm{x}_t)\), with \(a=\tanh\). The LSTM uses</p>"
        r"\[\bm{f}_t=\sigma(\bm{b}_f+\bm{W}_{xf}\bm{x}_t+\bm{W}_{hf}\bm{h}_{t-1}),\]"
        r"\[\bm{c}_t=\bm{f}_t\odot\bm{c}_{t-1}+\bm{i}_t\odot\tilde{\bm{c}}_t,"
        r"\qquad \bm{h}_t=\bm{o}_t\odot\tanh(\bm{c}_t).\]"
        r"<p>Let \(t_*\) be the last blank state before reproduction and let "
        r"\(L^{(b)}\) be the loss for sequence \(b\). The <strong>relative "
        r"state-gradient norm</strong> is</p>"
        r"\[\frac1B\sum_{b=1}^{B}\frac{\left\|\nabla_{\bm{h}_{t_*-\tau}^{(b)}}"
        r"L^{(b)}\right\|_2}{\left\|\nabla_{\bm{h}_{t_*}^{(b)}}L^{(b)}"
        r"\right\|_2}.\]"
        r"<p>The <strong>relative cell-gradient norm</strong> replaces "
        r"\(\bm{h}\) with \(\bm{c}\):</p>"
        r"\[\frac1B\sum_{b=1}^{B}\frac{\left\|\nabla_{\bm{c}_{t_*-\tau}^{(b)}}"
        r"L^{(b)}\right\|_2}{\left\|\nabla_{\bm{c}_{t_*}^{(b)}}L^{(b)}"
        r"\right\|_2}.\]"
        "<p>Each sequence is normalized before the batch average. The reference "
        "point avoids direct readout-loss terms at reproduction positions. One "
        "backward pass exposes every retained state.</p>"
        r"<p>One direct cell-path contribution to \(\nabla_{\bm{c}_{t-1}}L\) is "
        r"\(\bm{f}_t\odot\nabla_{\bm{c}_t}L\). After \(\tau\) direct transitions, "
        "the vector-valued term is</p>"
        r"\[\bm{f}_{t+1}\odot\cdots\odot\bm{f}_{t+\tau}\odot"
        r"\nabla_{\bm{c}_{t+\tau}}L.\]"
        "<p>The full cell gradient also includes hidden-state and other gate "
        "paths. The direct term and full gradient are therefore plotted as "
        "different quantities. The direct calculation uses the same retained "
        "gates and backward pass.</p>"
        r"<p>The <strong>mean forget-gate activation</strong> at time \(t\) is</p>"
        r"\[\overline{f}_t=\frac1{BN_h}\sum_{b=1}^{B}\sum_{j=1}^{N_h}"
        r"(\bm{f}_t)_{b,j}.\]"
        "<p>The scalar product of these means and the curve "
        r"\(0.5^\tau\) are references only. Neither measured vector norm must "
        "follow that reference.</p></section>"
    )


def _training_section(result: GradientFlowExperimentResult) -> str:
    rows = "".join(_training_row(case) for case in result.cases)
    return (
        "<section><h2>Training and per-position loss</h2>"
        "<p>The first figure shows validation history for every fixed gap. The "
        "second shows each reproduction position after training. The dashed "
        "horizontal line is the memoryless reference.</p>"
        "<table><thead><tr><th>Gap</th><th>Arm</th><th>Parameters</th>"
        "<th>Initial loss</th><th>Trained loss</th><th>Trained accuracy</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></section>"
    )


def _training_row(case: GradientFlowCaseResult) -> str:
    return (
        f"<tr><td>{case.gap}</td><td>{escape(_arm_name(case))}</td>"
        f"<td>{case.parameter_count}</td><td>{case.initial_validation.loss:.6f}</td>"
        f"<td>{case.trained_validation.loss:.6f}</td>"
        f"<td>{100 * case.trained_validation.accuracy:.2f}%</td></tr>"
    )


def _audit_section(result: GradientFlowExperimentResult) -> str:
    rows = "".join(_audit_row(case) for case in result.cases)
    return (
        "<section><h2>Gradient and gate audit</h2>"
        "<p>The full-state figure contains all gaps. The cell-path figure gives "
        "the longest-gap before/after comparison. The forget-gate figure uses "
        "the per-time mean defined above; step zero is the delimiter. Displayed "
        "long curves are uniformly pruned with preserved endpoints. Stored "
        "numerical arrays remain complete.</p>"
        f"<p>Only the log figures replace exact zeros with {DISPLAY_ZERO_FLOOR:.0e}. "
        "Tables retain numerical outcomes without that plotting floor.</p>"
        "<table><thead><tr><th>Gap</th><th>Arm</th>"
        "<th>Initial full gradient at lag=gap</th>"
        "<th>Trained full gradient</th><th>Trained direct cell path</th>"
        "<th>Float64 trained full gradient</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def _audit_row(case: GradientFlowCaseResult) -> str:
    lag = case.gap
    initial = _full_curve(case.initial_audit)[lag]
    trained = _full_curve(case.trained_audit)[lag]
    direct = (
        _format_number(case.trained_audit.direct_cell_path_gradients[lag])
        if isinstance(case.trained_audit, LSTMGradientAudit)
        else "—"
    )
    verification = (
        _format_number(_full_curve(case.trained_verification)[lag])
        if case.trained_verification is not None
        else "not run"
    )
    return (
        f"<tr><td>{case.gap}</td><td>{escape(_arm_name(case))}</td>"
        f"<td>{_format_number(initial)}</td><td>{_format_number(trained)}</td>"
        f"<td>{direct}</td><td>{verification}</td></tr>"
    )


def _verification_section(result: GradientFlowExperimentResult) -> str:
    enabled = result.settings.verification_dtype is not None
    return (
        "<section><h2>Numerical verification</h2>"
        "<p>Longest-gap initial and trained float64 verification: "
        f"<strong>{enabled}</strong>. "
        "The same float32 model and generated batch are converted to float64. "
        "Autograd uses the selected dtype; all norm calculations use float64. "
        "This distinguishes numerical underflow from the underlying trend."
        "</p></section>"
    )


def _limitations_section() -> str:
    return (
        "<section><h2>Interpretation limits</h2><div class='note'>"
        "<p>At gap 200, every investigated arm remained near the memoryless "
        "loss after 5,000 updates. The LSTM nevertheless retained much more "
        "gradient. Better gradient flow does not guarantee task learning.</p>"
        "<p>Orthogonal initialization applies only at initialization. Training "
        "can change both the matrix and the direction of the gradient trend. "
        "Near-one forget gates and a delimiter dip are not universal properties. "
        "These observations are specific to the configured task, seed, and "
        "budget; they are not model-performance acceptance thresholds.</p>"
        "<p>The approved visual investigation is retained at "
        "<code>.internal/investigations/gradient-flow-audit/index.html</code>. "
        "It contains controlled animations and the same measurement definitions."
        "</p></div></section>"
    )


def _training_figure(result: GradientFlowExperimentResult) -> FigureModel:
    figure, axes = _panel_figure(result.settings.gaps)
    for axis, gap in zip(axes, result.settings.gaps, strict=True):
        for case in _gap_cases(result, gap):
            steps = [point.step for point in case.history]
            losses = [point.validation.loss for point in case.history]
            x, y = _prune(steps, losses, result.settings.figure_max_points)
            axis.plot(x, y, linewidth=2, label=_arm_name(case), color=_case_color(case))
        axis.axhline(
            copy_memoryless_loss(result.settings.symbol_count),
            color="#4b5563",
            linestyle="--",
            linewidth=2,
        )
        axis.set(xlabel=f"Optimizer update — gap {gap}", ylabel="Validation loss")
        axis.grid(alpha=0.2)
    _add_legend(figure, axes[-1])
    return FigureModel(
        figure=figure,
        artifact_name=FIGURE_STEMS[0],
        caption=(
            "Figure 1 — Matched training histories. Each panel is one fixed gap; "
            "lines show validation loss for the three arms and the dashed line "
            "is ln(8). No internal titles are used; the gap appears in each x label."
        ),
    )


def _per_position_figure(result: GradientFlowExperimentResult) -> FigureModel:
    figure, axes = _panel_figure(result.settings.gaps)
    positions = list(range(1, result.settings.memorize_length + 1))
    for axis, gap in zip(axes, result.settings.gaps, strict=True):
        for case in _gap_cases(result, gap):
            axis.plot(
                positions,
                case.trained_validation.per_position_loss,
                linewidth=2,
                markersize=7,
                marker="o",
                label=_arm_name(case),
                color=_case_color(case),
            )
        axis.axhline(
            copy_memoryless_loss(result.settings.symbol_count),
            color="#4b5563",
            linestyle="--",
            linewidth=2,
        )
        axis.set(
            xlabel=f"Reproduction position — gap {gap}",
            xticks=positions,
        )
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Trained cross-entropy")
    _add_legend(figure, axes[-1])
    return FigureModel(
        figure=figure,
        artifact_name=FIGURE_STEMS[1],
        caption=(
            "Figure 2 — Per-position validation loss after training. Each panel "
            "uses one fixed gap. The dashed line is the closed-form memoryless "
            "loss ln(8); all ten reproduction positions are shown."
        ),
    )


def _state_gradient_figure(result: GradientFlowExperimentResult) -> FigureModel:
    figure, axes = _panel_figure(result.settings.gaps)
    for axis, gap in zip(axes, result.settings.gaps, strict=True):
        for case in _gap_cases(result, gap):
            if case.kind == "lstm":
                _plot_curve(
                    axis,
                    _full_curve(case.initial_audit),
                    label="LSTM before training",
                    color=INITIAL_COLOR,
                    limit=result.settings.figure_max_points,
                )
            _plot_curve(
                axis,
                _full_curve(case.trained_audit),
                label=_arm_name(case),
                color=_case_color(case),
                limit=result.settings.figure_max_points,
            )
        _style_log_axis(axis, f"Lag τ — gap {gap}")
    _add_legend(figure, axes[-1])
    return FigureModel(
        figure=figure,
        artifact_name=FIGURE_STEMS[2],
        caption=(
            "Figure 3 — Full relative state- and cell-gradient norms on a log "
            "scale. RNNs are after training; training does not preserve "
            "orthogonality. LSTM before/after curves use the same fixed batch. "
            "Lag zero is the last blank state before reproduction. Only display "
            "values use the disclosed zero floor."
        ),
    )


def _cell_path_figure(result: GradientFlowExperimentResult) -> FigureModel:
    gap = max(result.settings.gaps)
    case = next(item for item in _gap_cases(result, gap) if item.kind == "lstm")
    figure = plt.figure(figsize=(12, 6), FigureClass=_SvgTextFigure)
    axes = figure.subplots(1, 2, sharey=True)
    for axis, audit, phase in (
        (axes[0], case.initial_audit, "before training"),
        (axes[1], case.trained_audit, "after training"),
    ):
        assert isinstance(audit, LSTMGradientAudit)
        _plot_curve(
            axis,
            audit.relative_cell_gradients,
            label="Full cell gradient",
            color=ARM_COLORS["lstm"],
            limit=result.settings.figure_max_points,
        )
        _plot_curve(
            axis,
            audit.direct_cell_path_gradients,
            label="Direct cell-path term",
            color=DIRECT_COLOR,
            linestyle="--",
            limit=result.settings.figure_max_points,
        )
        _plot_curve(
            axis,
            audit.mean_gate_products,
            label="Product of mean gates",
            color=SCALAR_COLOR,
            linestyle="-.",
            limit=result.settings.figure_max_points,
        )
        lags = list(range(len(audit.relative_cell_gradients)))
        reference = [0.5**lag for lag in lags]
        _plot_curve(
            axis,
            reference,
            label="0.5^tau reference",
            color="#4b5563",
            linestyle=":",
            limit=result.settings.figure_max_points,
        )
        _style_log_axis(axis, f"Lag τ — {phase}")
    _add_legend(figure, axes[-1])
    return FigureModel(
        figure=figure,
        artifact_name=FIGURE_STEMS[3],
        caption=(
            f"Figure 4 — Longest-gap ({gap}) LSTM cell audit before and after "
            "training. The full cell gradient and vector-valued direct cell-path "
            "term are distinct. The scalar product of mean gates and 0.5^tau "
            "are references, not required agreement conditions."
        ),
    )


def _forget_gate_figure(result: GradientFlowExperimentResult) -> FigureModel:
    figure, axes = _panel_figure(result.settings.gaps)
    memorize_length = result.settings.memorize_length
    for axis, gap in zip(axes, result.settings.gaps, strict=True):
        case = next(item for item in _gap_cases(result, gap) if item.kind == "lstm")
        for audit, label, color, style in (
            (case.initial_audit, "Before training", INITIAL_COLOR, "-"),
            (case.trained_audit, "After training", ARM_COLORS["lstm"], "-"),
        ):
            assert isinstance(audit, LSTMGradientAudit)
            values = audit.mean_forget_gates
            steps = list(range(gap + 1))
            gates = [values[memorize_length]] + list(
                values[memorize_length + 1 : memorize_length + gap + 1]
            )
            x, y = _prune(steps, gates, result.settings.figure_max_points)
            axis.plot(x, y, linewidth=2, linestyle=style, label=label, color=color)
        axis.set(
            xlabel=f"Blank-gap step (0 is delimiter) — gap {gap}",
            ylim=(0, 1.03),
        )
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean forget activation")
    _add_legend(figure, axes[-1])
    return FigureModel(
        figure=figure,
        artifact_name=FIGURE_STEMS[4],
        caption=(
            "Figure 5 — Mean forget-gate activation at each time step, averaged "
            "over batch and hidden components. Each panel is one fixed gap; "
            "step zero is the delimiter. Near-one gates and a delimiter dip "
            "are observations rather than required outcomes."
        ),
    )


def _panel_figure(gaps: tuple[int, ...]) -> tuple[Figure, tuple[Axes, ...]]:
    rows = math.ceil(len(gaps) / 2)
    figure = plt.figure(figsize=(12, 4.7 * rows), FigureClass=_SvgTextFigure)
    grid = figure.subplots(rows, 2, squeeze=False)
    axes = tuple(grid.flat[: len(gaps)])
    for unused in grid.flat[len(gaps) :]:
        unused.remove()
    figure.subplots_adjust(hspace=0.42, wspace=0.2, bottom=0.12)
    return figure, axes


def _plot_curve(
    axis: Axes,
    curve: Sequence[float],
    *,
    label: str,
    color: str,
    limit: int,
    linestyle: str = "-",
) -> None:
    lags = list(range(len(curve)))
    displayed = [
        value if value > 0 or math.isnan(value) else DISPLAY_ZERO_FLOOR
        for value in curve
    ]
    x, y = _prune(lags, displayed, limit)
    axis.plot(
        x,
        y,
        linewidth=2,
        linestyle=linestyle,
        color=color,
        label=label,
    )


def _style_log_axis(axis: Axes, xlabel: str) -> None:
    axis.set_yscale("log")
    axis.set(xlabel=xlabel, ylim=(DISPLAY_ZERO_FLOOR, 10))
    axis.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    axis.grid(alpha=0.2, which="major")
    axis.set_ylabel("Relative gradient norm")


def _add_legend(figure: Figure, axis: Axes) -> None:
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.2)


def _prune(
    x: list[float] | list[int], y: list[float] | list[int], limit: int
) -> tuple[list[float] | list[int], list[float] | list[int]]:
    if len(x) <= limit:
        return x, y
    indices = sorted(
        {round(index * (len(x) - 1) / (limit - 1)) for index in range(limit)}
    )
    return [x[index] for index in indices], [y[index] for index in indices]


def _gap_cases(
    result: GradientFlowExperimentResult, gap: int
) -> tuple[GradientFlowCaseResult, ...]:
    return tuple(case for case in result.cases if case.gap == gap)


def _full_curve(audit: AnyGradientAudit) -> tuple[float, ...]:
    if isinstance(audit, RNNGradientAudit):
        return audit.relative_state_gradients
    return audit.relative_cell_gradients


def _case_color(case: GradientFlowCaseResult) -> str:
    if case.kind == "lstm":
        return ARM_COLORS["lstm"]
    return ARM_COLORS[case.initialization]


def _arm_name(case: GradientFlowCaseResult) -> str:
    if case.kind == "lstm":
        return "LSTM"
    return "Default RNN" if case.initialization == "default" else "Orthogonal RNN"


def _format_number(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.4e}"
