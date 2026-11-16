from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from spbunn.utils.common import get_config_path, get_project_path


def test_project_paths_point_to_repository() -> None:
    assert get_project_path() == Path(__file__).parent.parent
    assert get_config_path() == get_project_path() / "config"
    assert get_config_path().is_dir()


def test_compose_common_hydra_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hydra_root = tmp_path / "hydra"
    monkeypatch.setenv("SPBUNN_HYDRA_PATH", str(hydra_root))

    with initialize_config_dir(
        config_dir=str(get_config_path()),
        job_name="foundation_test",
        version_base="1.3",
    ):
        cfg = compose(config_name="config_test", return_hydra_config=True)

    assert str(cfg.result_dir).startswith(f"{hydra_root}/")
    assert cfg.hydra.run.dir == cfg.result_dir
    assert cfg.hydra.job_logging.root.handlers == ["file", "console"]
