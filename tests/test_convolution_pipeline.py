import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from spbunn.reporting import ReportWriter
from spbunn.scripts.simple_convolution import simple_convolution
from spbunn.utils.common import get_project_path
from tests.convolution_helpers import (
    compose_convolution_config,
    write_convolution_input_images,
)


def _image_overrides(tmp_path: Path) -> list[str]:
    rgb_path, grayscale_path = write_convolution_input_images(tmp_path / "inputs")
    return [
        f"images.rgb={rgb_path}",
        f"images.grayscale={grayscale_path}",
    ]


def test_executable_pipeline_writes_complete_report_in_run_directory(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    working_directory = tmp_path / "cwd"
    write_convolution_input_images(working_directory / "inputs")
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
            "spbunn.scripts.simple_convolution",
            "--config-name",
            "config_test_simple_convolution",
            "images.rgb=inputs/rgb.png",
            "images.grayscale=inputs/grayscale.png",
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

    report_path = run_directory / "test_simple_convolution.html"
    html = report_path.read_text(encoding="utf-8")
    assert 'id="mathjax-runtime"' in html
    assert re.search(r"<script\b[^>]*\bsrc=", html) is None
    image_names = re.findall(r'<img src="([^"]+)"', html)
    assert len(image_names) == 3
    assert {Path(name).stem for name in image_names} == {
        "stripes_convoluted_h",
        "stripes_convoluted_v",
        "waves_convoluted",
    }
    for image_name in image_names:
        image_path = run_directory / image_name
        assert image_path.is_file()
        assert image_path.suffix in {".svg", ".png"}
        alternative_suffix = ".png" if image_path.suffix == ".svg" else ".svg"
        assert not image_path.with_suffix(alternative_suffix).exists()
    assert not list(working_directory.glob("*.svg"))
    assert not list(working_directory.glob("*.png"))


def test_orchestrator_closes_figures_after_success(tmp_path: Path) -> None:
    cfg = compose_convolution_config(
        overrides=[
            *_image_overrides(tmp_path),
            f"result_dir={tmp_path / 'run'}",
        ]
    )

    simple_convolution(cfg)

    assert not plt.get_fignums()
    assert (tmp_path / "run" / cfg.report.artifact_name).is_file()


def test_orchestrator_closes_figures_after_writer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = compose_convolution_config(
        overrides=[
            *_image_overrides(tmp_path),
            f"result_dir={tmp_path / 'run'}",
        ]
    )

    def fail_write(self: ReportWriter, report: object) -> Path:
        del self, report
        raise RuntimeError("controlled writer failure")

    monkeypatch.setattr(ReportWriter, "write", fail_write)
    with pytest.raises(RuntimeError, match="controlled writer failure"):
        simple_convolution(cfg)
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    "config_name",
    ["config_simple_convolution", "config_test_simple_convolution"],
)
def test_pipeline_configs_compose_without_input_files(config_name: str) -> None:
    cfg = compose_convolution_config(config_name)

    assert cfg.result_dir == cfg.hydra_dir
    assert cfg.hydra_root
    assert cfg.images.rgb
    assert cfg.images.grayscale
    assert cfg.report.artifact_name.endswith(".html")
