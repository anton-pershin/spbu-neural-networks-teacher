from collections.abc import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from spbunn.utils.common import get_config_path


def compose_sequence_config(
    config_name: str, overrides: Sequence[str] = ()
) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(get_config_path()),
        job_name="sequence_test",
        version_base="1.3",
    ):
        return compose(config_name=config_name, overrides=list(overrides))
