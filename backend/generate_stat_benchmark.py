"""Generate reproducible panel-aware artifacts for statistical 4G baselines."""
from __future__ import annotations

import argparse

import numpy as np

from benchmark_artifacts import write_benchmark_artifact
from four_g_protocol import FEATURE_COLS, build_windows, fit_training_scaler, load_observations
from inference_engine import _infer_statistical
from model_registry import MODEL_REGISTRY


METHODS = {
    "Naive": "naive",
    "Persistent 24h": "persistent",
    "AutoARIMA": "autoarima",
    "AutoAR": "autoar",
    "LinearRegression": "linear_regression",
}


def generate(model_name: str) -> dict:
    if model_name not in METHODS:
        raise ValueError(f"Unsupported statistical model: {model_name}")
    train = load_observations("train")
    test = load_observations("test")
    x, truth, refs = build_windows(test, fit_training_scaler(train))
    prediction = np.asarray([
        _infer_statistical(METHODS[model_name], window[None, :, :], truth.shape[1])[0]
        for window in x
    ], dtype=np.float32)
    return write_benchmark_artifact(
        MODEL_REGISTRY[model_name]["result_dir"], model_name, prediction, truth, refs,
        run_kind="statistical_benchmark",
        extra={"method": METHODS[model_name]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", choices=tuple(METHODS), default=("Naive", "Persistent 24h"))
    args = parser.parse_args()
    for model in args.models:
        artifact = generate(model)
        print(f"{model}: {artifact['metrics_standardized']}")
