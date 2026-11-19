"""Configuration, scientific regression, and executable BPTT pipeline checks."""

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from omegaconf import DictConfig

from spbunn.reporting import ReportWriter
from spbunn.rnn.audit import audit_gradients
from spbunn.rnn.pipeline import evaluate_bptt_experiment, produce_bptt_input
from spbunn.rnn.tasks import generate_adding_batch, generate_copy_batch
from spbunn.scripts.bptt_instability import bptt_instability
from spbunn.utils.common import get_project_path
from tests.rnn_helpers import compose_bptt_config, reduced_bptt_result


def test_production_seed42_scientific_regressions() -> None:
    default32 = _production_copy("default", torch.float32)
    default64 = _production_copy("default", torch.float64)
    orthogonal32 = _production_copy("orthogonal", torch.float32)
    adding32 = audit_gradients(
        task="adding",
        batch=generate_adding_batch(
            length=200, batch_size=128, seed=1400, dtype=torch.float32
        ),
        initialization="xavier_tanh",
        seed=42,
        task_length=200,
        hidden_size=64,
        dtype=torch.float32,
    )

    assert _at_lag(default32, 200) == 0
    assert default32.zero_state_gradient_count > 0
    assert _at_lag(default64, 200) == pytest.approx(9.5502577e-46, rel=1e-5)
    assert _at_lag(default64, 200) > 0
    assert default64.zero_state_gradient_count == 0
    assert _at_lag(orthogonal32, 20) == pytest.approx(0.47348896, rel=1e-5)
    assert _at_lag(orthogonal32, 50) == pytest.approx(0.011396109, rel=1e-5)
    assert _at_lag(orthogonal32, 100) == pytest.approx(3.2837714e-5, rel=1e-5)
    assert _at_lag(orthogonal32, 200) == pytest.approx(7.5506447e-10, rel=1e-5)
    assert (
        _at_lag(orthogonal32, 200)
        < _at_lag(orthogonal32, 100)
        < _at_lag(orthogonal32, 50)
        < _at_lag(orthogonal32, 20)
    )

    assert adding32.forward_finite and adding32.backward_finite
    assert adding32.loss == pytest.approx(1.19612336, rel=1e-5)
    assert 0 <= adding32.maximum_hidden_state <= 1
    assert adding32.parameter_gradient_norm == pytest.approx(1.28149256e7, rel=1e-4)
    assert _at_lag(adding32, 20) == pytest.approx(3.7833356, rel=1e-5)
    assert _at_lag(adding32, 50) == pytest.approx(39.020494, rel=1e-5)
    assert _at_lag(adding32, 100) == pytest.approx(1454.8081, rel=1e-4)
    assert _at_lag(adding32, 199) == pytest.approx(1.9496104e7, rel=1e-4)
    assert (
        _at_lag(adding32, 199)
        > _at_lag(adding32, 100)
        > _at_lag(adding32, 50)
        > _at_lag(adding32, 20)
        > 1
    )


def _production_copy(initialization: str, dtype: torch.dtype):
    return audit_gradients(
        task="copy",
        batch=generate_copy_batch(
            symbol_count=8,
            memorize_length=10,
            gap=200,
            batch_size=128,
            seed=1400,
            dtype=dtype,
        ),
        initialization=initialization,  # type: ignore[arg-type]
        seed=42,
        task_length=200,
        hidden_size=64,
        dtype=dtype,
    )


def _at_lag(result, lag: int) -> float:
    return result.relative_state_gradients[-1 - lag]


@pytest.mark.parametrize(
    ("config_name", "primary_count", "verification_count"),
    [
        ("config_bptt_instability", 36, 9),
        ("config_test_bptt_instability", 12, 6),
    ],
)
def test_configs_compose_and_produce_expected_case_counts(
    config_name: str, primary_count: int, verification_count: int
) -> None:
    cfg = compose_bptt_config(config_name)
    experiment = produce_bptt_input(cfg)

    assert isinstance(cfg, DictConfig)
    assert cfg.result_dir == cfg.hydra_dir
    assert cfg.hydra_root
    assert cfg.report.artifact_name.endswith(".html")
    assert experiment.settings.batch_size == cfg.batch_size
    assert len(experiment.settings.copy.gaps) == len(cfg.copy_task.gaps)
    expected_primary = len(experiment.settings.copy.initializations) * len(
        experiment.settings.copy.seeds
    ) * len(experiment.settings.copy.gaps) + len(
        experiment.settings.adding.seeds
    ) * len(
        experiment.settings.adding.lengths
    )
    expected_verification = len(experiment.settings.copy.initializations) * len(
        experiment.settings.copy.seeds
    ) + len(experiment.settings.adding.seeds)
    assert expected_primary == primary_count
    assert expected_verification == verification_count
    if config_name == "config_bptt_instability":
        assert experiment.settings.batch_size == 128
        assert experiment.settings.hidden_size == 64
        assert experiment.settings.copy.seeds == (7, 42, 123)
        assert experiment.settings.adding.lengths == (20, 50, 100, 200)
    if config_name == "config_test_bptt_instability":
        result = reduced_bptt_result()
        assert len(result.primary_results) == primary_count
        assert len(result.verification_results) == verification_count


def test_reduced_experiment_is_deterministic_and_verification_can_be_disabled() -> None:
    first = reduced_bptt_result()
    second = evaluate_bptt_experiment(produce_bptt_input(compose_bptt_config()))
    without_verification = evaluate_bptt_experiment(
        produce_bptt_input(
            compose_bptt_config(overrides=["verification.enabled=false"])
        )
    )

    assert first == second
    assert not without_verification.verification_results
    assert len(without_verification.primary_results) == len(first.primary_results)
    for primary in first.primary_results:
        matching = [
            item
            for item in first.verification_results
            if item.task == primary.task
            and item.initialization == primary.initialization
            and item.seed == primary.seed
            and item.task_length == primary.task_length
        ]
        if primary.task_length == 20:
            assert len(matching) == 1
            assert matching[0].dtype == "float64"
            assert primary.dtype == "float32"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["batch_size=0"], "batch size"),
        (["hidden_size=1.5"], "finite integers"),
        (["primary_dtype=float64"], "primary audit dtype"),
        (["verification.dtype=float32"], "verification audit dtype"),
        (["verification.enabled=1"], "must be a boolean"),
        (["copy_task.gaps=[20,20]"], "unique"),
        (["copy_task.gaps=[50,20]"], "sorted"),
        (["copy_task.initializations=[default,default]"], "unique"),
        (["copy_task.initializations=[xavier_tanh]"], "default or orthogonal"),
        (["copy_task.symbol_count=1"], "at least two"),
        (["adding_task.lengths=[1,20]"], "at least two"),
        (["adding_task.seeds=[42,7]"], "sorted"),
        (["adding_task.initialization=default"], "requires xavier_tanh"),
    ],
)
def test_invalid_pipeline_settings_are_rejected(
    overrides: list[str], message: str
) -> None:
    cfg = compose_bptt_config(overrides=overrides)
    with pytest.raises(ValueError, match=message):
        produce_bptt_input(cfg)


def test_executable_pipeline_writes_complete_offline_report(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    working_directory = tmp_path / "cwd"
    working_directory.mkdir()
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(get_project_path()), environment.get("PYTHONPATH", "")])
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "spbunn.scripts.bptt_instability",
            "--config-name",
            "config_test_bptt_instability",
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

    report_path = run_directory / "test_bptt_instability.html"
    html = report_path.read_text(encoding="utf-8")
    assert 'id="mathjax-runtime"' in html
    assert re.search(r"<script\b[^>]*\bsrc=", html) is None
    assert "Copy memory task: gradient decay" in html
    assert "Adding problem: gradient growth" in html
    image_names = re.findall(r'<img src="([^"]+)"', html)
    assert len(image_names) == 2
    assert {Path(name).stem for name in image_names} == {
        "bptt_copy_gradient_flow",
        "bptt_adding_gradient_flow",
    }
    for image_name in image_names:
        image_path = run_directory / image_name
        assert image_path.is_file()
        assert image_path.suffix in {".svg", ".png"}
        alternative = ".png" if image_path.suffix == ".svg" else ".svg"
        assert not image_path.with_suffix(alternative).exists()
    assert not list(working_directory.glob("*.svg"))
    assert not list(working_directory.glob("*.png"))


def test_orchestrator_closes_figures_after_success(tmp_path: Path) -> None:
    cfg = compose_bptt_config(overrides=[f"result_dir={tmp_path / 'run'}"])

    bptt_instability(cfg)

    assert not plt.get_fignums()


def test_orchestrator_closes_figures_after_writer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = compose_bptt_config(overrides=[f"result_dir={tmp_path / 'run'}"])

    def fail_write(self: ReportWriter, report: object) -> Path:
        del self, report
        raise RuntimeError("controlled writer failure")

    monkeypatch.setattr(ReportWriter, "write", fail_write)
    with pytest.raises(RuntimeError, match="controlled writer failure"):
        bptt_instability(cfg)
    assert not plt.get_fignums()
