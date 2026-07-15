"""Evaluate the supplied external baseline forecast against the 4G test truth.

The base parquet is not an observation/training split.  It contains one
``forecast_<feature>`` value per test row, keyed by date and cell ID.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark_artifacts import write_benchmark_artifact
from four_g_protocol import (
    CELL_ID_COL,
    DATE_COL,
    FEATURE_COLS,
    build_windows,
    dataset_path,
    fit_training_scaler,
    load_observations,
)
from model_registry import MODEL_REGISTRY


def evaluate() -> dict:
    """Write an auditable artifact using test truth and external forecasts."""
    train = load_observations("train")
    test = load_observations("test").copy()
    base = pd.read_parquet(dataset_path("base"))
    forecast_cols = [f"forecast_{column}" for column in FEATURE_COLS]
    required = {DATE_COL, CELL_ID_COL, *forecast_cols}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"External base forecast is missing columns: {missing}")

    keys = [DATE_COL, CELL_ID_COL]
    if base.duplicated(keys).any():
        raise ValueError("External base forecast has duplicate date/cell keys")
    forecast = base.loc[:, keys + forecast_cols].copy()
    forecast[DATE_COL] = pd.to_datetime(forecast[DATE_COL])
    test[DATE_COL] = pd.to_datetime(test[DATE_COL])
    aligned = test.loc[:, keys].merge(forecast, on=keys, how="left", validate="one_to_one")
    if len(aligned) != len(test) or aligned[forecast_cols].isna().any().any():
        raise ValueError("External base forecast does not cover every test observation")

    predicted_rows = test.copy()
    predicted_rows.loc[:, FEATURE_COLS] = aligned.loc[:, forecast_cols].to_numpy()
    scaler = fit_training_scaler(train)
    _, truth, refs = build_windows(test, scaler)
    _, prediction, prediction_refs = build_windows(predicted_rows, scaler)
    if refs != prediction_refs:
        raise AssertionError("Forecast and truth windows do not have identical provenance")

    result_dir = Path(MODEL_REGISTRY["★ BaseModel"]["result_dir"])
    return write_benchmark_artifact(
        result_dir=result_dir,
        model_name="★ BaseModel",
        prediction=prediction,
        truth=truth,
        refs=refs,
        run_kind="external_base_forecast",
        scaler=scaler,
        extra={
            "prediction_source": dataset_path("base").name,
            "truth_source": dataset_path("test").name,
            "alignment": "date + ID",
        },
    )


if __name__ == "__main__":
    manifest = evaluate()
    print(f"METRICS:{manifest['metrics']}", flush=True)
    print("EXTERNAL_BASE_DONE", flush=True)
