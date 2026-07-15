"""Backfill original-scale metrics for existing trusted benchmark artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark_artifacts import _custom_acc
from four_g_protocol import FEATURE_COLS, fit_training_scaler, load_observations
from model_registry import MODEL_REGISTRY
from preset_curves import MANIFEST_NAME, PROTOCOL_VERSION


def refresh(model_name: str) -> dict:
    result_dir = Path(MODEL_REGISTRY[model_name]["result_dir"])
    manifest_path = result_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL_VERSION:
        raise ValueError(f"{model_name} does not use {PROTOCOL_VERSION}")
    prediction = np.load(result_dir / "pred.npy")
    truth = np.load(result_dir / "true.npy")
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError(f"{model_name} has invalid prediction arrays")
    scaler = fit_training_scaler(load_observations("train"))
    prediction_raw = scaler.inverse_transform(prediction.reshape(-1, len(FEATURE_COLS))).reshape(prediction.shape)
    truth_raw = scaler.inverse_transform(truth.reshape(-1, len(FEATURE_COLS))).reshape(truth.shape)
    valid = truth_raw > 1e-5
    relative = np.zeros_like(truth_raw, dtype=np.float64)
    relative[valid] = np.abs(
        (prediction_raw[valid] - truth_raw[valid]) / truth_raw[valid]
    )
    errors = truth - prediction
    manifest["metrics"] = {
        "mse": float(np.mean(errors ** 2)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": float(np.mean(relative[valid])) if np.any(valid) else None,
        "mspe": float(np.mean(relative[valid] ** 2)) if np.any(valid) else None,
        "custom_acc": float(_custom_acc(prediction_raw, truth_raw)),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest["metrics"]


if __name__ == "__main__":
    for name in MODEL_REGISTRY:
        manifest = Path(MODEL_REGISTRY[name].get("result_dir", "")) / MANIFEST_NAME
        if manifest.exists():
            print(name, refresh(name))
