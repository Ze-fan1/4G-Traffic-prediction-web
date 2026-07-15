"""Centralized paths for the single supported project checkout."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TSLIB_ROOT = PROJECT_ROOT / "Time-Series-Library-main"
DATA_DIR = TSLIB_ROOT / "data_provider" / "4g_traffic"
RESULTS_DIR = TSLIB_ROOT / "results"
CHECKPOINTS_DIR = TSLIB_ROOT / "checkpoints"


def require_project_layout() -> None:
    """Fail early when the backend is started from an incomplete deployment."""
    required = (TSLIB_ROOT, DATA_DIR)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing project directories: " + ", ".join(missing))
