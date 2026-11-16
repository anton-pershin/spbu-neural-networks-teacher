from pathlib import Path


def get_project_path() -> Path:
    """Return the repository root path."""
    return Path(__file__).parent.parent.parent


def get_config_path() -> Path:
    """Return the absolute path of the Hydra configuration directory."""
    return get_project_path() / "config"
