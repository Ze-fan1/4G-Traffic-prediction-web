"""Persist auditable benchmark outputs for the public curve API."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from four_g_protocol import FEATURE_COLS
from preset_curves import MANIFEST_NAME, PROTOCOL_VERSION


def write_benchmark_artifact(
    result_dir: str | Path,
    model_name: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    refs,
    run_kind: str,
    scaler: StandardScaler | None = None,
    extra: dict | None = None,
) -> dict:
    """Write a prediction/truth pair and its exact panel-window provenance."""
    result_dir = Path(result_dir)
    prediction = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    if prediction.ndim != 3 or prediction.shape != truth.shape:
        raise ValueError("Benchmark prediction and truth must be equal 3D arrays")
    if len(refs) != len(prediction):
        raise ValueError("Every benchmark window needs one provenance reference")
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "pred.npy", prediction)
    np.save(result_dir / "true.npy", truth)
    errors = truth - prediction
    metrics = {
        "mse": float(np.mean(errors ** 2)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }
    if scaler is not None:
        prediction_raw = scaler.inverse_transform(prediction.reshape(-1, prediction.shape[-1])).reshape(prediction.shape)
        truth_raw = scaler.inverse_transform(truth.reshape(-1, truth.shape[-1])).reshape(truth.shape)
        valid = truth_raw > 1e-5
        relative_error = np.abs((prediction_raw[valid] - truth_raw[valid]) / truth_raw[valid])
        metrics["mape"] = float(np.mean(relative_error)) if relative_error.size else None
        metrics["mspe"] = float(np.mean(relative_error ** 2)) if relative_error.size else None
        metrics["custom_acc"] = float(_custom_acc(prediction_raw, truth_raw))
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "model": model_name,
        "run_kind": run_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shape": list(prediction.shape),
        "feature_columns": list(FEATURE_COLS),
        "metrics": metrics,
        "window_refs": [
            {"cell_id": ref.cell_id, "start": ref.start.isoformat()}
            for ref in refs
        ],
        "experiment_id": f"{model_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    }
    if extra:
        manifest["extra"] = extra
    (result_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _custom_acc(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Compute the project's high-value accuracy on original-scale observations."""
    pred_flat = prediction.reshape(-1, prediction.shape[-1])
    truth_flat = truth.reshape(-1, truth.shape[-1])
    scores = []
    for channel in range(truth_flat.shape[-1]):
        actual = truth_flat[:, channel]
        forecast = pred_flat[:, channel]
        high = actual >= actual.mean()
        if not np.any(high):
            continue
        high_mean = actual[high].mean()
        if high_mean == 0:
            continue
        high_mae = np.abs(actual[high] - forecast[high]).mean()
        scores.append(float(np.clip(abs(high_mean - high_mae) / high_mean, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0
