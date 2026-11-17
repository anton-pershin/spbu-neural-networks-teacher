import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from omegaconf import DictConfig

from spbunn.reporting import ReportWriter
from spbunn.scripts.balduzzi_analysis import balduzzi_analysis
from spbunn.scripts.hidden_norms_propagation import hidden_norms_propagation
from spbunn.scripts.mlp_depth_experiment import mlp_depth_experiment
from spbunn.scripts.mlp_width_experiment import mlp_width_experiment
from spbunn.utils.common import get_project_path
from tests.mlp_helpers import compose_mlp_config

PIPELINE_CASES = [
    (
        "spbunn.scripts.mlp_width_experiment",
        "config_test_mlp_width_experiment",
        "test_mlp_width_experiment.html",
        8,
    ),
    (
        "spbunn.scripts.mlp_depth_experiment",
        "config_test_mlp_depth_experiment",
        "test_mlp_depth_experiment.html",
        8,
    ),
    (
        "spbunn.scripts.hidden_norms_propagation",
        "config_test_hidden_norms",
        "test_hidden_norms_propagation.html",
        6,
    ),
    (
        "spbunn.scripts.balduzzi_analysis",
        "config_test_balduzzi_analysis",
        "test_balduzzi_analysis.html",
        2,
    ),
]

ENTRY_POINT_CASES = [
    (mlp_width_experiment, "config_test_mlp_width_experiment"),
    (mlp_depth_experiment, "config_test_mlp_depth_experiment"),
    (hidden_norms_propagation, "config_test_hidden_norms"),
    (balduzzi_analysis, "config_test_balduzzi_analysis"),
]


@pytest.mark.parametrize(
    ("module", "config_name", "report_name", "figure_count"), PIPELINE_CASES
)
def test_executable_mlp_pipelines_write_complete_reports_in_run_directory(
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
    environment.update({"MPLBACKEND": "Agg", "PYTHONPATH": str(get_project_path())})
    command = [sys.executable, "-m", module, "--config-name", config_name]
    output_overrides = [
        f"hydra.run.dir={run_directory}",
        f"result_dir={run_directory}",
        "hydra.job.chdir=False",
    ]
    subprocess.run(
        command + output_overrides,
        cwd=working_directory,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    html = (run_directory / report_name).read_text(encoding="utf-8")
    image_paths = [
        run_directory / name for name in re.findall(r'<img src="([^"]+)"', html)
    ]
    assert 'id="mathjax-runtime"' in html
    assert re.search(r"<script\b[^>]*\bsrc=", html) is None
    assert len(image_paths) == figure_count
    assert all(path.is_file() for path in image_paths)
    assert all(path.suffix in {".svg", ".png"} for path in image_paths)
    for path in image_paths:
        other_suffix = ".png" if path.suffix == ".svg" else ".svg"
        assert not path.with_suffix(other_suffix).exists()
    unexpected_patterns = ("*.svg", "*.png", "mlp_*_lrs.json")
    assert all(
        not list(working_directory.glob(pattern)) for pattern in unexpected_patterns
    )


def test_hidden_pipeline_reports_custom_conditioning_variants(tmp_path: Path) -> None:
    cfg = compose_mlp_config(
        "config_test_hidden_norms",
        [
            "experiments=[conditioning]",
            (
                "conditioning.plain_variants=[{label:custom_plain,"
                "residual:false,branch_scale:1.0,weight_scale:0.2}]"
            ),
            (
                "conditioning.residual_variants=[{label:custom_residual,"
                "residual:true,branch_scale:0.5,weight_scale:null}]"
            ),
            f"result_dir={tmp_path}",
        ],
    )

    hidden_norms_propagation(cfg)

    html = (tmp_path / "test_hidden_norms_propagation.html").read_text(encoding="utf-8")
    assert len(re.findall(r'<img src="([^"]+)"', html)) == 4
    assert "custom_plain" in html
    assert "custom_residual" in html
    assert "branch scale=0.5" in html
    assert r"Configured residual branch scales are \(0.5\)." in html
    assert "Initialization and norm propagation" not in html
    assert not plt.get_fignums()


@pytest.mark.parametrize(("entry_point", "config_name"), ENTRY_POINT_CASES)
def test_mlp_orchestrators_close_figures_after_success(
    tmp_path: Path, entry_point: object, config_name: str
) -> None:
    cfg = compose_mlp_config(
        config_name,
        [f"result_dir={tmp_path / 'run'}"],
    )

    assert callable(entry_point)
    entry_point(cfg)
    assert not plt.get_fignums()


@pytest.mark.parametrize(("entry_point", "config_name"), ENTRY_POINT_CASES)
def test_mlp_orchestrators_close_figures_after_writer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: object,
    config_name: str,
) -> None:
    cfg = compose_mlp_config(
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
        "config_mlp_width_experiment",
        "config_mlp_depth_experiment",
        "config_hidden_norms",
        "config_balduzzi_analysis",
    ],
)
def test_mlp_production_configs_compose_common_run_directory(
    config_name: str,
) -> None:
    cfg = compose_mlp_config(config_name)

    assert cfg.result_dir == cfg.hydra_dir
    assert cfg.hydra_root
    assert cfg.report.artifact_name.endswith(".html")


def test_mlp_pipeline_entry_point_type_is_dict_config() -> None:
    cfg: DictConfig = compose_mlp_config("config_test_balduzzi_analysis")
    assert isinstance(cfg, DictConfig)
