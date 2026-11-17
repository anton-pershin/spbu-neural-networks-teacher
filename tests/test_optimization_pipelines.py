import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from omegaconf import DictConfig

from spbunn.reporting import ReportWriter
from spbunn.scripts.loss_landscape import loss_landscape
from spbunn.scripts.schedulers import schedulers
from spbunn.scripts.simple_loss_optimization import simple_loss_optimization
from spbunn.utils.common import get_project_path
from tests.optimization_helpers import compose_optimization_config


@pytest.mark.parametrize(
    ("module", "config_name", "report_name", "figure_count"),
    [
        (
            "spbunn.scripts.schedulers",
            "config_test_schedulers",
            "test_schedulers.html",
            4,
        ),
        (
            "spbunn.scripts.loss_landscape",
            "config_test_loss_landscape",
            "test_loss_landscape.html",
            5,
        ),
        (
            "spbunn.scripts.simple_loss_optimization",
            "config_test_simple_loss_optimization",
            "test_simple_loss_optimization.html",
            1,
        ),
    ],
)
def test_executable_pipelines_write_complete_reports_in_run_directory(
    tmp_path: Path,
    module: str,
    config_name: str,
    report_name: str,
    figure_count: int,
) -> None:
    run_directory = tmp_path / "run"
    working_directory = tmp_path / "cwd"
    working_directory.mkdir()
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [str(get_project_path()), environment.get("PYTHONPATH", "")],
        )
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--config-name",
            config_name,
            f"hydra.run.dir={run_directory}",
            f"result_dir={run_directory}",
            "hydra.job.chdir=False",
        ],
        cwd=working_directory,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    report_path = run_directory / report_name
    html = report_path.read_text(encoding="utf-8")
    assert 'id="mathjax-runtime"' in html
    assert re.search(r"<script\b[^>]*\bsrc=", html) is None
    image_names = re.findall(r'<img src="([^"]+)"', html)
    assert len(image_names) == figure_count
    for image_name in image_names:
        image_path = run_directory / image_name
        assert image_path.is_file()
        assert image_path.suffix in {".svg", ".png"}
        alternative_suffix = ".png" if image_path.suffix == ".svg" else ".svg"
        assert not image_path.with_suffix(alternative_suffix).exists()
    assert not list(working_directory.glob("*.svg"))
    assert not list(working_directory.glob("*.png"))


@pytest.mark.parametrize(
    ("entry_point", "config_name"),
    [
        (schedulers, "config_test_schedulers"),
        (loss_landscape, "config_test_loss_landscape"),
        (simple_loss_optimization, "config_test_simple_loss_optimization"),
    ],
)
def test_orchestrators_close_figures_after_success(
    tmp_path: Path, entry_point: object, config_name: str
) -> None:
    cfg = compose_optimization_config(
        config_name,
        [f"result_dir={tmp_path / 'run'}"],
    )

    assert callable(entry_point)
    entry_point(cfg)
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    ("entry_point", "config_name"),
    [
        (schedulers, "config_test_schedulers"),
        (loss_landscape, "config_test_loss_landscape"),
        (simple_loss_optimization, "config_test_simple_loss_optimization"),
    ],
)
def test_orchestrators_close_figures_after_writer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: object,
    config_name: str,
) -> None:
    cfg = compose_optimization_config(
        config_name,
        [f"result_dir={tmp_path / 'run'}"],
    )

    def fail_write(self: ReportWriter, report: object) -> Path:
        del self, report
        raise RuntimeError("controlled writer failure")

    monkeypatch.setattr(ReportWriter, "write", fail_write)
    assert callable(entry_point)
    with pytest.raises(RuntimeError, match="controlled writer failure"):
        entry_point(cfg)
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    "config_name",
    [
        "config_schedulers",
        "config_loss_landscape",
        "config_simple_loss_optimization",
    ],
)
def test_production_configs_compose_common_run_directory(config_name: str) -> None:
    cfg = compose_optimization_config(
        config_name,
        [],
    )

    assert cfg.result_dir == cfg.hydra_dir
    assert cfg.hydra_root
    assert cfg.report.artifact_name.endswith(".html")


def test_pipeline_entry_point_type_is_dict_config() -> None:
    cfg: DictConfig = compose_optimization_config("config_test_schedulers")
    assert isinstance(cfg, DictConfig)
