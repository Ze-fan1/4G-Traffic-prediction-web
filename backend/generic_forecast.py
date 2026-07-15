"""Forecast arbitrary uploaded numeric series without pretending it is 4G data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

MIN_CONTEXT = 12
MAX_UPLOAD_ROWS = 50_000
SUPPORTED_METHODS = ("naive", "linear_trend", "autoar", "xgboost")


@dataclass(frozen=True)
class GenericForecastResult:
    prediction: np.ndarray
    validation_mae: float
    validation_rmse: float
    train_rows: int
    validation_rows: int


def validate_series(frame: pd.DataFrame, target_col: str, pred_len: int) -> np.ndarray:
    if target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' does not exist")
    if not pd.api.types.is_numeric_dtype(frame[target_col]):
        raise ValueError(f"Target column '{target_col}' must be numeric")
    if not 1 <= pred_len <= 168:
        raise ValueError("Forecast horizon must be between 1 and 168 steps")
    values = frame[target_col].dropna().to_numpy(dtype=np.float64)
    if len(values) > MAX_UPLOAD_ROWS:
        raise ValueError(f"The uploaded series exceeds the {MAX_UPLOAD_ROWS:,}-row limit")
    if len(values) < MIN_CONTEXT + pred_len:
        raise ValueError(f"At least {MIN_CONTEXT + pred_len} non-empty values are required")
    if not np.isfinite(values).all():
        raise ValueError("The target column contains non-finite values")
    return values


def _predict(method: str, history: np.ndarray, horizon: int) -> np.ndarray:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported generic method '{method}'")
    if method == "naive":
        return np.repeat(history[-1], horizon)
    if method == "linear_trend":
        x = np.arange(len(history), dtype=np.float64).reshape(-1, 1)
        model = LinearRegression().fit(x, history)
        return model.predict(np.arange(len(history), len(history) + horizon).reshape(-1, 1))
    if method == "autoar":
        from statsmodels.tsa.ar_model import AutoReg, ar_select_order
        try:
            selected = ar_select_order(history, maxlag=min(24, len(history) // 3), ic="bic", old_names=False)
            if selected.ar_lags is None or len(selected.ar_lags) == 0:
                raise ValueError("No autoregressive lags selected")
            return AutoReg(history, lags=selected.ar_lags, old_names=False).fit().predict(
                start=len(history), end=len(history) + horizon - 1, dynamic=False
            )
        except Exception:
            return np.repeat(history[-1], horizon)
    lags = min(24, max(3, len(history) // 8))
    features = np.array([history[i - lags:i] for i in range(lags, len(history))])
    labels = history[lags:]
    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_leaf_nodes=15, random_state=42)
    model.fit(features, labels)
    working = list(history)
    for _ in range(horizon):
        working.append(float(model.predict(np.asarray(working[-lags:]).reshape(1, -1))[0]))
    return np.asarray(working[-horizon:])


def forecast(frame: pd.DataFrame, target_col: str, pred_len: int, method: str) -> GenericForecastResult:
    """Backtest on the final horizon, then forecast beyond the uploaded history."""
    values = validate_series(frame, target_col, pred_len)
    train = values[:-pred_len]
    truth = values[-pred_len:]
    validation = _predict(method, train, pred_len)
    prediction = _predict(method, values, pred_len)
    errors = truth - validation
    return GenericForecastResult(
        prediction=prediction,
        validation_mae=float(np.mean(np.abs(errors))),
        validation_rmse=float(np.sqrt(np.mean(errors ** 2))),
        train_rows=len(train), validation_rows=len(truth),
    )
