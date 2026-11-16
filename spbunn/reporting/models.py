from dataclasses import dataclass

from matplotlib.figure import Figure


@dataclass(frozen=True)
class FigureModel:
    """A Matplotlib figure and the stem of its output artifact name."""

    figure: Figure
    artifact_name: str


@dataclass(frozen=True)
class ReportModel:
    """The report content and artifacts consumed by ``ReportWriter``."""

    html_content: str
    figures: tuple[FigureModel, ...] = ()
    title: str = "SpbU Neural Networks"
    artifact_name: str = "report.html"
