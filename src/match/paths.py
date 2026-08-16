from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "notebooks" / "configs"


def resolve_project_path(path: Union[str, Path]) -> Path:
    """Resolve a relative config path from the project root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
