"""HTML and figures for the vanilla-RNN BPTT instability report."""

import math
from html import escape
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator

from spbunn.reporting.models import FigureModel, ReportModel
from spbunn.reporting.rnn_animation import (
    build_adding_prediction_example,
    build_animation_script,
    build_copy_prediction_example,
)
from spbunn.rnn.audit import GradientAudit, Initialization, TaskKind
from spbunn.rnn.pipeline import BPTTExperimentResult
from spbunn.rnn.tasks import adding_mean_baseline_mse, copy_memoryless_loss

BPTT_FIGURE_STEMS = (
    "bptt_copy_gradient_flow",
    "bptt_adding_gradient_flow",
)
DISPLAY_ZERO_FLOOR = 1e-45


class _SvgTextFigure(Figure):
    """Save SVG labels as text without changing process-global rcParams."""

    def savefig(self, *args: Any, **kwargs: Any) -> Any:
        with plt.rc_context({"svg.fonttype": "none"}):
            return super().savefig(*args, **kwargs)


def build_bptt_report(
    result: BPTTExperimentResult, title: str, artifact_name: str
) -> ReportModel:
    """Build the complete educational report without writing artifacts."""
    sections = [
        "<h1>BPTT instability in vanilla RNNs</h1>",
        _introduction(result),
        _method_section(result),
        _copy_section(result),
        build_copy_prediction_example(result.prediction_examples.copy),
        _adding_section(result),
        build_adding_prediction_example(result.prediction_examples.adding),
        _verification_section(result),
        _limitations_section(),
        build_animation_script(
            result.prediction_examples,
            interval_ms=result.settings.examples.animation_interval_ms,
        ),
    ]
    with plt.rc_context(
        {
            "font.size": 14.0,
            "axes.titlesize": 15.0,
            "axes.labelsize": 15.0,
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
            "legend.fontsize": 13.0,
        }
    ):
        figures = (
            _copy_figure(result),
            _adding_figure(result),
        )
    return ReportModel(
        title=title,
        html_content="\n".join(sections),
        figures=figures,
        artifact_name=artifact_name,
    )


def _introduction(result: BPTTExperimentResult) -> str:
    return (
        "<section><h2>Recurrent adjoints</h2>"
        r"<p>A one-layer vanilla RNN applies the same parameters at every "
        r"step:</p>"
        r"\[h_t=\tanh(W_xx_t+W_hh_{t-1}+b).\]"
        r"<p>For loss \(L\), let \(g_t=\partial L/\partial h_t\). If "
        r"\(D_t=\operatorname{diag}(1-h_t^2)\), then the local Jacobian is "
        r"\(J_t=D_tW_h\), and backpropagation through time gives</p>"
        r"\[g_{t-1}=J_t^Tg_t=W_h^TD_tg_t.\]"
        r"<p>Long products of these Jacobians can make the adjoint norm decay "
        "or grow exponentially. The two experiments below measure both "
        f"directions with {result.settings.batch_size} sequences per case and "
        f"{result.settings.hidden_size} hidden units. The models are not "
        "trained.</p></section>"
    )


def _method_section(result: BPTTExperimentResult) -> str:
    verification = (
        "Enabled for every longest case."
        if result.settings.verification_dtype is not None
        else "Disabled."
    )
    return (
        "<section><h2>Audit method</h2>"
        "<p>The implementation unrolls <code>torch.nn.RNNCell</code> and "
        "retains every separate hidden state. One backward pass then gives "
        "the true adjoint at every time step. Retaining only the single packed "
        "output tensor of <code>nn.RNN</code> would record the direct decoder "
        "gradient at that tensor boundary, not every internal recurrent "
        "adjoint.</p>"
        r"<p>For sequence \(i\), the plotted quantity is</p>"
        r"\[r_i(\tau)=\frac{\|\partial L_i/\partial h_{T-1-\tau}\|_2}"
        r"{\|\partial L_i/\partial h_{T-1}\|_2},\qquad r(\tau)=\frac1B\sum_i "
        r"r_i(\tau).\]"
        "<p>Each sequence is normalized before the batch average. Autograd "
        "uses the configured model dtype. Norms and ratios are calculated in "
        "float64 afterward. Tables preserve exact zeros and non-finite values. "
        f"Only the log-scaled figures replace exact zeros with the disclosed "
        f"display floor {DISPLAY_ZERO_FLOOR:.0e}. Matched-weight float64 "
        f"verification: {verification} The same float32 weights and generated "
        "data are cast to float64 for these checks.</p>"
        f"<p>Longest-case figure curves retain at most "
        f"{result.settings.figure_max_points} uniformly spaced points, including "
        "both endpoints. Pruning affects only the display, not the stored "
        "numerical results. It reduces the serialized SVG size so the shared "
        "writer can apply its normal SVG/PNG threshold.</p></section>"
    )


def _copy_section(result: BPTTExperimentResult) -> str:
    settings = result.settings.copy
    lags = settings.gaps
    rows = "".join(
        _summary_row(
            _primary(
                result,
                task="copy",
                initialization=initialization,
                seed=seed,
                task_length=max(settings.gaps),
            ),
            lags=lags,
        )
        for initialization in settings.initializations
        for seed in settings.seeds
    )
    return (
        "<section><h2>Copy memory task: gradient decay</h2>"
        f"<p>Each sequence contains {settings.memorize_length} random symbols, "
        "a delimiter, a blank gap, and reproduction positions. There are "
        f"{settings.symbol_count} ordinary symbols. Cross-entropy applies only "
        f"to the {settings.memorize_length} reproduction positions. The optimal "
        "memoryless uniform predictor has loss "
        f"{copy_memoryless_loss(settings.symbol_count):.6f}; this is a reference, "
        "not a fitted-model result.</p>"
        "<p><strong>Default</strong> leaves the ordinary PyTorch RNNCell "
        "initialization unchanged. <strong>Orthogonal</strong> applies "
        "<code>torch.nn.init.orthogonal_</code> to the recurrent matrix without "
        "a scale factor. Neither initialization is selected by spectral radius."
        "</p>"
        "<p>Orthogonal initialization is motivated by norm preservation in a "
        "linear map: <code>Q<sup>T</sup>Q = I</code> and "
        "<code>||Qx||<sub>2</sub> = ||x||<sub>2</sub></code>. It avoids a "
        "systematic initial shrinkage or amplification from the recurrent "
        "matrix itself, which is useful when the same matrix is applied many "
        "times. It does not make the full recurrent Jacobian orthogonal: the "
        "<code>tanh</code> derivative factors still shrink or redirect the "
        "adjoint. Orthogonal initialization therefore delays decay but cannot "
        "guarantee long-range gradient flow.</p>"
        f"<p>The table contains every longest case (gap {max(settings.gaps)}). "
        "The first figure contains all gaps and seeds. Exact zeros in the table "
        "are numerical underflow outcomes, not mathematical claims that the "
        "real-valued derivative is exactly zero.</p>"
        "<table><thead><tr><th>Initialization</th><th>Seed</th><th>Loss</th>"
        "<th>Max |h|</th><th>Parameter gradient</th>"
        f"{_ratio_headers(lags)}"
        "<th>Diagnostics</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def _adding_section(result: BPTTExperimentResult) -> str:
    settings = result.settings.adding
    longest = max(settings.lengths)
    lags = (*settings.lengths[:-1], longest - 1)
    rows = "".join(
        _summary_row(
            _primary(
                result,
                task="adding",
                initialization=settings.initialization,
                seed=seed,
                task_length=longest,
            ),
            lags=lags,
        )
        for seed in settings.seeds
    )
    return (
        "<section><h2>Adding problem: gradient growth</h2>"
        "<p>Each input has two channels. The first contains an independent "
        "uniform value in [0, 1]. The second marks two distinct random "
        "positions. The final scalar output must predict the sum of the two "
        "marked values, with mean squared error at that final position only. "
        "Predicting the known target mean of one has MSE "
        f"{adding_mean_baseline_mse():.6f}; this baseline is not a trained model."
        "</p>"
        "<p>Input and recurrent weights use Xavier normal initialization with "
        "the documented PyTorch <code>tanh</code> gain "
        '<code>nn.init.calculate_gain("tanh") = 5/3</code>. Recurrent biases '
        "are zero. The linear readout keeps its ordinary initialization. No "
        "recurrent spectral radius or additional scale is selected.</p>"
        f"<p>The table contains every length-{longest} case. Hidden states stay "
        "bounded by one and losses stay finite, while backward norms grow by "
        "several orders of magnitude. The last ratio column uses lag "
        f"{longest - 1}, the first input state.</p>"
        "<table><thead><tr><th>Initialization</th><th>Seed</th><th>Loss</th>"
        "<th>Max |h|</th><th>Parameter gradient</th>"
        f"{_ratio_headers(lags)}"
        "<th>Diagnostics</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def _ratio_headers(lags: tuple[int, ...]) -> str:
    return "".join(f"<th>g({lag})/g(0)</th>" for lag in lags)


def _summary_row(result: GradientAudit, *, lags: tuple[int, ...]) -> str:
    ratio_cells = "".join(
        f"<td>{_format_float(_at_lag(result, lag))}</td>" for lag in lags
    )
    diagnostics = (
        f"forward finite={result.forward_finite}; "
        f"backward finite={result.backward_finite}; "
        f"zero={result.zero_state_gradient_count}; "
        f"non-finite={result.nonfinite_state_gradient_count}; "
        f"valid={result.valid_sequence_count}/{result.batch_size}"
    )
    return (
        "<tr>"
        f"<td>{escape(result.initialization)}</td><td>{result.seed}</td>"
        f"<td>{_format_float(result.loss)}</td>"
        f"<td>{_format_float(result.maximum_hidden_state)}</td>"
        f"<td>{_format_float(result.parameter_gradient_norm)}</td>"
        f"{ratio_cells}<td>{escape(diagnostics)}</td>"
        "</tr>"
    )


def _verification_section(result: BPTTExperimentResult) -> str:
    if not result.verification_results:
        return (
            "<section><h2>Matched-weight float64 verification</h2>"
            "<p>Verification is disabled in this configuration.</p></section>"
        )
    rows = "".join(
        _verification_row(result, verification)
        for verification in result.verification_results
    )
    return (
        "<section><h2>Matched-weight float64 verification</h2>"
        "<p>Each row compares the first-state ratio for identical initial "
        "weights and data. Small nonzero float64 values confirm decay where "
        "float32 has already underflowed. Finite float64 values also check "
        "that the primary growth curves are not only a plotting artifact.</p>"
        "<table><thead><tr><th>Task</th><th>Initialization</th><th>Seed</th>"
        "<th>First-state lag</th><th>Float32 ratio</th><th>Float64 ratio</th>"
        "<th>Float64 diagnostics</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def _verification_row(
    experiment: BPTTExperimentResult, verification: GradientAudit
) -> str:
    primary = _primary(
        experiment,
        task=verification.task,
        initialization=verification.initialization,
        seed=verification.seed,
        task_length=verification.task_length,
    )
    diagnostics = (
        f"forward finite={verification.forward_finite}; "
        f"backward finite={verification.backward_finite}; "
        f"zero={verification.zero_state_gradient_count}; "
        f"non-finite={verification.nonfinite_state_gradient_count}"
    )
    return (
        "<tr>"
        f"<td>{escape(verification.task)}</td>"
        f"<td>{escape(verification.initialization)}</td>"
        f"<td>{verification.seed}</td>"
        f"<td>{len(verification.relative_state_gradients) - 1}</td>"
        f"<td>{_format_float(primary.relative_state_gradients[0])}</td>"
        f"<td>{_format_float(verification.relative_state_gradients[0])}</td>"
        f"<td>{escape(diagnostics)}</td></tr>"
    )


def _limitations_section() -> str:
    return (
        "<section><h2>Interpretation and limits</h2>"
        "<p>The copy task shows exponential decay under the two configured "
        "initializations. Orthogonal initialization delays decay but does not "
        "remove it. The adding task shows backward growth under the documented "
        "Xavier <code>tanh</code> initialization while forward states remain "
        "bounded. These are measured properties of these tasks, dtypes, and "
        "initializations, not universal rankings.</p>"
        "<p>The report does not claim that every reasonable initialization "
        "explodes, that every copy-task sequence underflows at the same lag, or "
        "that either model has learned its task. It does not evaluate Alice or "
        "n-gram perplexity, truncated BPTT, gradient clipping, LSTMs, or model "
        "training.</p></section>"
    )


def _copy_figure(result: BPTTExperimentResult) -> FigureModel:
    settings = result.settings.copy
    figure = plt.figure(FigureClass=_SvgTextFigure, figsize=(16, 6))
    curve_axis, length_axis = figure.subplots(1, 2)
    longest = max(settings.gaps)
    for initialization in settings.initializations:
        line_style = "-" if initialization == "default" else "--"
        for seed in settings.seeds:
            label = f"{initialization}, seed {seed}"
            longest_result = _primary(
                result,
                task="copy",
                initialization=initialization,
                seed=seed,
                task_length=longest,
            )
            _plot_longest_curve(
                curve_axis,
                longest_result,
                result.settings.figure_max_points,
                label=label,
                line_style=line_style,
            )
            cases = [
                _primary(
                    result,
                    task="copy",
                    initialization=initialization,
                    seed=seed,
                    task_length=gap,
                )
                for gap in settings.gaps
            ]
            _plot_first_state(
                length_axis,
                cases,
                settings.gaps,
                label=label,
                line_style=line_style,
            )
    _style_curve_axis(curve_axis, f"Copy: longest gap {longest}")
    _style_length_axis(length_axis, "Copy: first input state")
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=BPTT_FIGURE_STEMS[0])


def _adding_figure(result: BPTTExperimentResult) -> FigureModel:
    settings = result.settings.adding
    figure = plt.figure(FigureClass=_SvgTextFigure, figsize=(16, 5.5))
    curve_axis, length_axis = figure.subplots(1, 2)
    longest = max(settings.lengths)
    for seed in settings.seeds:
        longest_result = _primary(
            result,
            task="adding",
            initialization=settings.initialization,
            seed=seed,
            task_length=longest,
        )
        _plot_longest_curve(
            curve_axis, longest_result, result.settings.figure_max_points
        )
        cases = [
            _primary(
                result,
                task="adding",
                initialization=settings.initialization,
                seed=seed,
                task_length=length,
            )
            for length in settings.lengths
        ]
        _plot_first_state(length_axis, cases, settings.lengths)
    _style_curve_axis(curve_axis, f"Xavier tanh: longest length {longest}")
    _style_length_axis(length_axis, "Xavier tanh: first input state")
    figure.tight_layout()
    return FigureModel(figure=figure, artifact_name=BPTT_FIGURE_STEMS[1])


def _plot_longest_curve(
    axis: Axes,
    result: GradientAudit,
    max_points: int,
    *,
    label: str | None = None,
    line_style: str = "-",
) -> None:
    values = tuple(reversed(result.relative_state_gradients))
    indices = _pruned_indices(len(values), max_points)
    axis.plot(
        indices,
        _display_values(tuple(values[index] for index in indices)),
        label=label or f"seed {result.seed}",
        linestyle=line_style,
    )


def _pruned_indices(length: int, max_points: int) -> tuple[int, ...]:
    if length <= max_points:
        return tuple(range(length))
    return tuple(
        sorted(
            {
                round(index * (length - 1) / (max_points - 1))
                for index in range(max_points)
            }
        )
    )


def _plot_first_state(
    axis: Axes,
    cases: list[GradientAudit],
    lengths: tuple[int, ...],
    *,
    label: str | None = None,
    line_style: str = "-",
) -> None:
    values = tuple(case.relative_state_gradients[0] for case in cases)
    axis.plot(
        lengths,
        _display_values(values),
        marker="o",
        label=label or f"seed {cases[0].seed}",
        linestyle=line_style,
    )


def _style_curve_axis(axis: Axes, title: str) -> None:
    axis.set_yscale("log")
    axis.set_xlabel(r"Lag $\tau$")
    axis.set_ylabel(r"Mean relative gradient $r(\tau)$")
    axis.set_title(f"{title}\n0 is the final state; the largest lag is the first state")
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis.minorticks_off()
    axis.grid(True, which="major", alpha=0.25)
    axis.legend()
    axis.text(
        0.01,
        0.02,
        f"Zeros, if present, are shown at {DISPLAY_ZERO_FLOOR:.0e}",
        transform=axis.transAxes,
        fontsize=11,
    )


def _style_length_axis(axis: Axes, title: str) -> None:
    axis.set_yscale("log")
    axis.set_xlabel("Task length or blank gap")
    axis.set_ylabel("First-state relative gradient")
    axis.set_title(title)
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis.minorticks_off()
    axis.grid(True, which="major", alpha=0.25)
    axis.legend()


def _display_values(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(DISPLAY_ZERO_FLOOR if value == 0 else value for value in values)


def _primary(
    result: BPTTExperimentResult,
    *,
    task: TaskKind,
    initialization: Initialization,
    seed: int,
    task_length: int,
) -> GradientAudit:
    return next(
        item
        for item in result.primary_results
        if item.task == task
        and item.initialization == initialization
        and item.seed == seed
        and item.task_length == task_length
    )


def _at_lag(result: GradientAudit, lag: int) -> float:
    index = len(result.relative_state_gradients) - 1 - lag
    if index < 0:
        return float("nan")
    return result.relative_state_gradients[index]


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if value == 0:
        return "0"
    return f"{value:.3e}"
