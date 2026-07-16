"""Read only reproducible 4G benchmark artifacts.

Legacy result folders were generated with flattened rows that crossed cell and
time boundaries. They deliberately have no manifest and are never served.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from four_g_protocol import FEATURE_COLS, build_windows, fit_training_scaler, load_observations
from model_registry import MODEL_REGISTRY

PROTOCOL_VERSION = "4g-panel-v1"
MANIFEST_NAME = "benchmark_manifest.json"
DEFAULT_WINDOW = 0


def _validated_artifact(result_dir: Path) -> tuple[np.ndarray, np.ndarray, dict] | None:
    manifest_path = result_dir / MANIFEST_NAME
    pred_path = result_dir / "pred.npy"
    true_path = result_dir / "true.npy"
    if not (manifest_path.exists() and pred_path.exists() and true_path.exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pred = np.load(pred_path)
        truth = np.load(true_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if manifest.get("protocol") != PROTOCOL_VERSION:
        return None
    if pred.ndim != 3 or pred.shape != truth.shape:
        return None
    if tuple(manifest.get("shape", ())) != tuple(pred.shape):
        return None
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        return None
    return pred, truth, manifest


def load_benchmark_manifest(model_name: str) -> dict | None:
    """Read only a model manifest; avoid loading large NumPy arrays for lists."""
    info = MODEL_REGISTRY.get(model_name)
    if not info:
        return None
    result_dir = Path(info.get("result_dir", ""))
    manifest_path = result_dir / MANIFEST_NAME
    if not (manifest_path.exists() and (result_dir / "pred.npy").exists() and (result_dir / "true.npy").exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("protocol") != PROTOCOL_VERSION:
        return None
    return manifest


def load_all_benchmark_manifests() -> dict[str, dict]:
    """Return lightweight manifests for every currently validated model."""
    manifests = {}
    for model_name in MODEL_REGISTRY:
        manifest = load_benchmark_manifest(model_name)
        if manifest is not None:
            manifests[model_name] = manifest
    return manifests


def has_benchmark_artifact(model_name: str) -> bool:
    return load_benchmark_manifest(model_name) is not None


def load_preset_curves(model_name: str, window_idx: int | None = DEFAULT_WINDOW) -> dict | None:
    """Load a model's own validated prediction and truth arrays in scaled space."""
    info = MODEL_REGISTRY.get(model_name)
    if not info:
        return None
    artifact = _validated_artifact(Path(info.get("result_dir", "")))
    if artifact is None:
        return None
    pred, truth, manifest = artifact
    idx = 0 if window_idx is None else max(0, min(int(window_idx), len(pred) - 1))
    curves = {
        name: {
            "pred": [round(float(value), 4) for value in pred[idx, :, channel]],
            "truth": [round(float(value), 4) for value in truth[idx, :, channel]],
        }
        for channel, name in enumerate(FEATURE_COLS)
    }
    metrics = manifest.get("metrics")
    if metrics is None:
        errors = truth - pred
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        metrics = {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))}
    return {
        "curves": curves,
        "window_idx": idx,
        "total_windows": int(len(pred)),
        "mode": "single",
        "space": "standardized",
        "metrics_summary": {
            key: round(float(value), 4) if value is not None else None
            for key, value in metrics.items()
        },
        "source": "reproducible_4g_benchmark",
        "protocol": PROTOCOL_VERSION,
        "experiment_id": manifest.get("experiment_id"),
        "window_ref": manifest.get("window_refs", [])[idx] if manifest.get("window_refs") else None,
    }


def get_available_preset_models() -> list[str]:
    return [
        name for name, info in MODEL_REGISTRY.items()
        if has_benchmark_artifact(name)
    ]
