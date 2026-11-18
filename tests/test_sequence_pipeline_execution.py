import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from omegaconf import DictConfig

from spbunn.reporting import ReportWriter
from spbunn.scripts.bpe import bpe
from spbunn.utils.common import get_project_path
from tests.sequence_helpers import compose_sequence_config


def test_executable_pipeline_writes_complete_two_part_report(
    tmp_path: Path,
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
            "spbunn.scripts.bpe",
            "--config-name",
            "config_test_bpe",
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

    report_path = run_directory / "test_bpe.html"
    html = report_path.read_text(encoding="utf-8")
    assert 'id="mathjax-runtime"' in html
    assert re.search(r"<script\b[^>]*\bsrc=", html) is None
    assert "1. How BPE works" in html
    assert "2. What BPE achieves on a larger corpus" in html
    image_names = re.findall(r'<img src="([^"]+)"', html)
    assert len(image_names) == 2
    assert {Path(name).stem for name in image_names} == {
        "bpe_toy_dynamics",
        "bpe_large_corpus_compression",
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
    cfg = compose_sequence_config("config_test_bpe", [f"result_dir={tmp_path / 'run'}"])

    bpe(cfg)

    assert not plt.get_fignums()


def test_orchestrator_closes_figures_after_writer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = compose_sequence_config("config_test_bpe", [f"result_dir={tmp_path / 'run'}"])

    def fail_write(self: ReportWriter, report: object) -> Path:
        del self, report
        raise RuntimeError("controlled writer failure")

    monkeypatch.setattr(ReportWriter, "write", fail_write)
    with pytest.raises(RuntimeError, match="controlled writer failure"):
        bpe(cfg)
    assert not plt.get_fignums()


def test_pipeline_entry_point_receives_dict_config() -> None:
    cfg: DictConfig = compose_sequence_config("config_test_bpe")
    assert isinstance(cfg, DictConfig)
