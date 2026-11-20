from dataclasses import dataclass

from matplotlib.figure import Figure


@dataclass(frozen=True)
class FigureModel:
    """A Matplotlib figure, artifact stem, and optional visible caption."""

    figure: Figure
    artifact_name: str
    caption: str = ""


@dataclass(frozen=True)
class ReportModel:
    """The report content and artifacts consumed by ``ReportWriter``."""

    html_content: str
    figures: tuple[FigureModel, ...] = ()
    title: str = "SpbU Neural Networks"
    artifact_name: str = "report.html"
