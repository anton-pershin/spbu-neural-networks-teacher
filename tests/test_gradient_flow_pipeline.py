"""Configuration, deterministic pipeline, and executable report checks."""

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
from spbunn.rnn import copy_training
from spbunn.rnn import gradient_flow_pipeline as gradient_pipeline
from spbunn.rnn.gradient_audit import LSTMGradientAudit
from spbunn.rnn.gradient_flow_pipeline import (
    evaluate_gradient_flow_experiment,
    produce_gradient_flow_input,
)
from spbunn.rnn.models import build_copy_model
from spbunn.scripts.gradient_flow_audit import gradient_flow_audit
from spbunn.utils.common import get_project_path
from tests.rnn_helpers import (
    compose_gradient_flow_config,
    reduced_gradient_flow_result,
)


@pytest.mark.parametrize(
    ("config_name", "case_count"),
    [("config_gradient_flow_audit", 12), ("config_test_gradient_flow_audit", 6)],
)
def test_configs_compose_with_expected_case_counts(
    config_name: str, case_count: int
) -> None:
    cfg = compose_gradient_flow_config(config_name)
    experiment = produce_gradient_flow_input(cfg)

    assert isinstance(cfg, DictConfig)
    assert cfg.result_dir == cfg.hydra_dir
    assert experiment.settings.batch_size == cfg.batch_size
    assert experiment.settings.gaps == tuple(cfg.copy_task.gaps)
    if config_name == "config_gradient_flow_audit":
        assert experiment.settings.audit_batch_size == 128
        assert experiment.settings.hidden_size == 64
        assert experiment.settings.training.updates == 5000
        assert experiment.settings.gaps == (20, 50, 100, 200)
        assert experiment.settings.evaluation.batches == 4
    else:
        result = reduced_gradient_flow_result()
        assert len(result.cases) == case_count


def test_reduced_experiment_is_deterministic_and_preserves_global_rng() -> None:
    first = reduced_gradient_flow_result()
    torch.manual_seed(12345)
    before = torch.random.get_rng_state()

    second = evaluate_gradient_flow_experiment(
        produce_gradient_flow_input(compose_gradient_flow_config())
    )

    assert torch.equal(before, torch.random.get_rng_state())
    assert first == second
    assert [(case.gap, case.kind, case.initialization) for case in first.cases] == [
        (2, "rnn", "default"),
        (2, "rnn", "orthogonal"),
        (2, "lstm", "default"),
        (4, "rnn", "default"),
        (4, "rnn", "orthogonal"),
        (4, "lstm", "default"),
    ]
    for case in first.cases:
        assert [point.step for point in case.history] == [1, 2]
        assert len(case.trained_validation.per_position_loss) == 10
        assert case.initial_audit.dtype == "float32"
        assert case.trained_audit.dtype == "float32"
        if case.gap == 4:
            assert case.initial_verification is not None
            assert case.trained_verification is not None
            assert case.initial_verification.dtype == "float64"
            assert case.trained_verification.dtype == "float64"
        else:
            assert case.initial_verification is None
            assert case.trained_verification is None
        if case.kind == "lstm":
            assert isinstance(case.initial_audit, LSTMGradientAudit)
            assert len(case.initial_audit.mean_forget_gates) == 2 * 10 + case.gap + 1
            assert len(case.initial_audit.relative_cell_gradients) == 10 + case.gap + 1


def test_verification_can_be_disabled() -> None:
    result = evaluate_gradient_flow_experiment(
        produce_gradient_flow_input(
            compose_gradient_flow_config(overrides=["verification.enabled=false"])
        )
    )
    assert len(result.cases) == 6
    assert all(
        case.initial_verification is None and case.trained_verification is None
        for case in result.cases
    )


def test_matched_float64_audit_preserves_weights_and_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = copy_training.CopyTaskSettings(symbol_count=2, memorize_length=2, gap=2)
    batch = copy_training.make_copy_batch(
        task, batch_size=3, seed=77, dtype=torch.float32
    )
    model = build_copy_model(
        kind="lstm", initialization="default", seed=42, hidden_size=3, input_size=4
    )
    captured = {}
    original_audit = gradient_pipeline._audit_model  # pylint: disable=protected-access

    def capture_audit(converted_model, converted_batch, converted_task):
        captured["model"] = converted_model
        captured["batch"] = converted_batch
        return original_audit(converted_model, converted_batch, converted_task)

    monkeypatch.setattr(gradient_pipeline, "_audit_model", capture_audit)

    result = (
        gradient_pipeline._matched_float64_audit(  # pylint: disable=protected-access
            model, batch, task
        )
    )

    converted_model = captured["model"]
    converted_batch = captured["batch"]
    assert converted_model is not model
    for original, converted in zip(
        model.parameters(), converted_model.parameters(), strict=True
    ):
        assert converted.dtype == torch.float64
        torch.testing.assert_close(converted, original.detach().to(torch.float64))
    assert converted_batch.inputs.dtype == torch.float64
    assert torch.equal(converted_batch.inputs, batch.inputs.to(torch.float64))
    assert torch.equal(converted_batch.targets, batch.targets)
    assert result.dtype == "float64"


def test_every_arm_receives_the_same_data_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original = gradient_pipeline.make_copy_batch

    def record_batch(*args, **kwargs):
        task = kwargs["task"] if "task" in kwargs else args[0]
        calls.append((task.gap, kwargs["batch_size"], kwargs["seed"], kwargs["dtype"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(gradient_pipeline, "make_copy_batch", record_batch)
    monkeypatch.setattr(copy_training, "make_copy_batch", record_batch)
    cfg = compose_gradient_flow_config(
        overrides=["copy_task.gaps=[2]", "training_updates=1", "history_interval=1"]
    )
    evaluate_gradient_flow_experiment(produce_gradient_flow_input(cfg))

    arm_calls = calls[1:]
    assert len(arm_calls) == 12
    arm_length = len(arm_calls) // 3
    assert arm_calls[:arm_length] == arm_calls[arm_length : 2 * arm_length]
    assert arm_calls[:arm_length] == arm_calls[2 * arm_length :]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["batch_size=0"], "training batch size"),
        (["audit_batch_size=0"], "audit batch size"),
        (["validation_batch_size=0"], "validation batch size"),
        (["validation_batches=0"], "validation batch count"),
        (["hidden_size=1.5"], "finite integers"),
        (["training_updates=0"], "training updates"),
        (["learning_rate=0"], "learning rate"),
        (["learning_rate=nan"], "learning rate"),
        (["history_interval=0"], "history interval"),
        (["history_interval=3"], "must not exceed"),
        (["model_seed=-1"], "model seed"),
        (["training_data_seed=-1"], "training data seed"),
        (["validation_data_seed=-1"], "validation data seed"),
        (["audit_data_seed=-1"], "audit data seed"),
        (["copy_task.gaps=[]"], "must not be empty"),
        (["copy_task.gaps=[2,2]"], "unique"),
        (["copy_task.gaps=[4,2]"], "sorted"),
        (["copy_task.gaps=[0,2]"], "positive"),
        (["copy_task.symbol_count=1"], "at least two"),
        (["copy_task.memorize_length=0"], "memorized-symbol count"),
        (["verification.enabled=1"], "must be a boolean"),
        (["verification.dtype=float32"], "must be float64"),
        (["report.figure_max_points=1"], "at least two"),
    ],
)
def test_invalid_gradient_flow_settings_are_rejected(
    overrides: list[str], message: str
) -> None:
    cfg = compose_gradient_flow_config(overrides=overrides)
    with pytest.raises(ValueError, match=message):
        produce_gradient_flow_input(cfg)


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
            "spbunn.scripts.gradient_flow_audit",
            "--config-name",
            "config_test_gradient_flow_audit",
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

    report_path = run_directory / "test_gradient_flow_audit.html"
    html = report_path.read_text(encoding="utf-8")
    assert 'id="mathjax-runtime"' in html
    assert 'macros: {bm: ["\\\\boldsymbol{#1}", 1]}' in html
    assert re.search(r"<script\b[^>]*\bsrc=", html) is None
    assert "Measurements and GetSomeMath notation" in html
    assert "Training and per-position loss" in html
    assert "Gradient and gate audit" in html
    image_names = re.findall(r'<img src="([^"]+)"', html)
    assert len(image_names) == 5
    assert {Path(name).stem for name in image_names} == {
        "gradient_flow_training",
        "gradient_flow_per_position",
        "gradient_flow_states",
        "gradient_flow_cell_path",
        "gradient_flow_forget_gates",
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
    cfg = compose_gradient_flow_config(overrides=[f"result_dir={tmp_path / 'run'}"])

    gradient_flow_audit(cfg)

    assert not plt.get_fignums()


def test_orchestrator_closes_figures_after_writer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = compose_gradient_flow_config(overrides=[f"result_dir={tmp_path / 'run'}"])

    def fail_write(self: ReportWriter, report: object) -> Path:
        del self, report
        raise RuntimeError("controlled writer failure")

    monkeypatch.setattr(ReportWriter, "write", fail_write)
    with pytest.raises(RuntimeError, match="controlled writer failure"):
        gradient_flow_audit(cfg)
    assert not plt.get_fignums()
