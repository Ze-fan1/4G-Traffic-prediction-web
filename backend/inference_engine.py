"""
统一推理引擎 — 按模型类型分发推理逻辑
"""
import sys
from project_paths import TSLIB_ROOT
sys.path.insert(0, str(TSLIB_ROOT))
sys.path.insert(0, str(TSLIB_ROOT / "models"))  # utils/ 在 models/utils/ 中

import numpy as np
from model_registry import get_model_info

# 重型依赖延迟导入（避免统计推理时加载 PyTorch）
_torch = None
_load_fn = None
_get_device_fn = None


def _lazy_import_torch():
    global _torch
    if _torch is None:
        import torch as _torch_module
        _torch = _torch_module
    return _torch


def _lazy_load():
    global _load_fn, _get_device_fn
    if _load_fn is None:
        from model_loader import load as _l
        from model_loader import get_device as _gd
        _load_fn = _l
        _get_device_fn = _gd
    return _load_fn, _get_device_fn


def infer(model_name: str, X: np.ndarray, pred_len: int = 24) -> np.ndarray:
    """
    对输入 X 执行推理

    Args:
        model_name: 模型名（如 "PatchTST"）
        X: 输入数据 (seq_len, n_channels) 或 (batch, seq_len, n_channels)
        pred_len: 预测长度（仅统计/HF 模型可动态；PyTorch 固定 24 后切片）

    Returns:
        predictions: (pred_len, n_channels)
    """
    info = get_model_info(model_name)
    if info is None:
        raise ValueError(f"未知模型: {model_name}")

    mtype = info["type"]

    # 确保 X 是 (batch, seq_len, n_channels)
    if X.ndim == 2:
        X_batch = X[np.newaxis, :, :]
    else:
        X_batch = X

    if mtype == "statistical":
        result = _infer_statistical(info["method"], X_batch, pred_len)

    elif mtype == "pytorch":
        # PyTorch 固定 24h 输出 → 切片
        result = _infer_pytorch(model_name, X_batch)
        result = result[:, :pred_len, :]

    elif mtype == "huggingface":
        result = _infer_huggingface(model_name, X_batch, pred_len)

    elif mtype == "xgboost":
        result = _infer_xgboost(model_name, X_batch, pred_len)

    elif mtype in ("mamba", "timellm"):
        result = _infer_checkpoint_model(model_name, X_batch, pred_len)

    else:
        raise ValueError(f"未知模型类型: {mtype}")

    # 去掉 batch 维度（如果输入是单样本）
    if result.shape[0] == 1 and X.ndim == 2:
        result = result[0]

    return result


def _infer_statistical(method: str, X: np.ndarray, pred_len: int) -> np.ndarray:
    """
    统计模型推理
    X: (batch, seq_len, n_channels)
    Returns: (batch, pred_len, n_channels)
    """
    batch, seq_len, n_channels = X.shape
    preds = np.zeros((batch, pred_len, n_channels))

    for b in range(batch):
        for c in range(n_channels):
            series = X[b, :, c]

            if method == "naive":
                preds[b, :, c] = series[-1]

            elif method == "persistent":
                history = series[-min(seq_len, pred_len):]
                preds[b, :, c] = np.resize(history, pred_len)

            elif method == "historical_avg":
                avg = np.mean(series)
                preds[b, :, c] = avg

            elif method == "autoarima":
                preds[b, :, c] = _forecast_autoarima(series, pred_len)

            elif method == "autoar":
                preds[b, :, c] = _forecast_autoar(series, pred_len)

            elif method == "linear_regression":
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                t = np.arange(seq_len).reshape(-1, 1)
                lr.fit(t, series)
                t_future = np.arange(seq_len, seq_len + pred_len).reshape(-1, 1)
                preds[b, :, c] = lr.predict(t_future)

    return preds


def _infer_pytorch(model_name: str, X: np.ndarray) -> np.ndarray:
    """
    PyTorch 模型推理
    X: (batch, seq_len, n_channels)
    Returns: (batch, 24, n_channels)
    """
    torch = _lazy_import_torch()
    load, get_device = _lazy_load()

    model_inst, info = load(model_name)
    device = get_device()
    exp = model_inst  # Exp_Long_Term_Forecast 实例

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # 构建 decoder 输入（label_len + pred_len）
    label_len = 12
    dec_inp = torch.zeros((X_tensor.shape[0], label_len + 24, X_tensor.shape[2]),
                          dtype=torch.float32).to(device)
    dec_inp[:, :label_len, :] = X_tensor[:, -label_len:, :]

    with torch.no_grad():
        outputs = exp.model(X_tensor, None, dec_inp, None)

    if isinstance(outputs, tuple):
        outputs = outputs[0]

    # SCINet returns [zero-padded encoder, forecast] with length 48; the
    # forecast is the final pred_len segment. Other models already return 24.
    if outputs.shape[1] > 24:
        outputs = outputs[:, -24:, :]
    return outputs.cpu().numpy()


def _infer_huggingface(model_name: str, X: np.ndarray, pred_len: int) -> np.ndarray:
    """
    HuggingFace 预训练模型推理
    X: (batch, seq_len, n_channels)
    Returns: (batch, pred_len, n_channels)
    """
    torch = _lazy_import_torch()
    load, get_device = _lazy_load()

    model_inst, info = load(model_name)
    model_id = info["model_id"]
    device = get_device()
    batch, seq_len, n_channels = X.shape
    preds = np.zeros((batch, pred_len, n_channels))

    for b in range(batch):
        if "chronos-2" in model_id.lower():
            # Chronos2: 多变量输入 (n_series=1, n_variates=n_channels, history_length=seq_len)
            context = torch.tensor(X[b], dtype=torch.float32)  # CPU
            context = context.T.unsqueeze(0)  # (1, n_channels, seq_len)
            forecast = model_inst.predict(
                context,
                prediction_length=pred_len,
                limit_prediction_length=False,
            )
            # forecast: list of (n_variates, n_quantiles, pred_len) ndarrays
            arr = forecast[0] if isinstance(forecast, list) else forecast
            arr = np.asarray(arr)
            # Shape handling: (n_variates, n_quantiles, pred_len) → take median
            if arr.ndim == 3:
                # arr: (n_variates, n_quantiles, pred_len) or (pred_len, n_quantiles, n_variates)
                # Find quantile dim (should be 21 for Chronos2)
                if arr.shape[1] > arr.shape[0] and arr.shape[1] > arr.shape[2]:
                    # (n_variates, n_quantiles, pred_len) — quantile on axis 1
                    median_idx = arr.shape[1] // 2
                    result = arr[:, median_idx, :]  # (n_variates, pred_len)
                elif arr.shape[0] > arr.shape[1] and arr.shape[0] > arr.shape[2]:
                    # (pred_len, n_quantiles, n_variates) — quantile on axis 1
                    median_idx = arr.shape[1] // 2
                    result = arr[:, median_idx, :]  # (pred_len, n_variates)
                else:
                    result = arr.mean(axis=1)  # fallback: average quantiles
                preds[b] = result.T if result.shape[0] == n_channels else result  # ensure (pred_len, n_channels)
            else:
                preds[b] = arr.T if arr.shape[0] == n_channels else arr  # ensure (pred_len, n_channels)
        elif "chronos" in model_id.lower():
            # Chronos v1: 输入 (1, seq_len) 单变量
            for c in range(n_channels):
                context = torch.tensor(X[b, :, c], dtype=torch.float32).to(device)
                forecast = model_inst.predict(
                    context,
                    prediction_length=pred_len,
                    limit_prediction_length=False,
                )
                preds[b, :, c] = forecast[0].cpu().numpy()

        elif "ttm" in model_id.lower():
            # IBM TTM uses up to 512 real observations. Only short benchmark
            # windows are left-padded; uploaded series can supply full context.
            context_len = info.get("context_len", 512)
            past = X[b, -context_len:]
            if len(past) < context_len:
                pad_len = context_len - len(past)
                past = np.pad(past, ((pad_len, 0), (0, 0)), mode="edge")
            past_tensor = torch.tensor(past, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model_inst(past_values=past_tensor)
                if hasattr(outputs, "prediction_outputs"):
                    p = outputs.prediction_outputs.squeeze(0).cpu().numpy()
                elif hasattr(outputs, "logits"):
                    p = outputs.logits.squeeze(0).cpu().numpy()
                else:
                    p = outputs[0].squeeze(0).cpu().numpy()
            preds[b, :, :] = p[:pred_len, :n_channels]

    return preds


def _infer_xgboost(model_name: str, X: np.ndarray, pred_len: int) -> np.ndarray:
    """
    XGBoost 推理
    X: (batch, seq_len, n_channels)
    Returns: (batch, pred_len, n_channels)
    """
    load, _ = _lazy_load()
    model_inst, info = load(model_name)
    batch, seq_len, n_channels = X.shape
    preds = np.zeros((batch, pred_len, n_channels))

    if not isinstance(model_inst, dict) or "models" not in model_inst:
        raise ValueError("XGBoost weights use an unsupported format; train XGBoost again")
    models = model_inst["models"]
    trained_seq_len = int(model_inst.get("seq_len", seq_len))
    trained_pred_len = int(model_inst.get("pred_len", 24))
    trained_channels = int(model_inst.get("num_channels", n_channels))
    if seq_len != trained_seq_len or n_channels != trained_channels:
        raise ValueError("Uploaded 4G window does not match the saved XGBoost feature contract")
    if len(models) != trained_pred_len * trained_channels:
        raise ValueError("Saved XGBoost regressor count is incomplete")

    for b in range(batch):
        features = X[b].reshape(1, -1)
        flat = np.asarray([regressor.predict(features)[0] for regressor in models])
        forecast = flat.reshape(trained_pred_len, trained_channels)
        preds[b] = forecast[:pred_len]

    return preds


def _fit_stable_ar(series: np.ndarray, max_lag: int) -> tuple[float, np.ndarray]:
    """Fit the lowest-BIC stable AR(p) model with small-window OLS.

    The benchmark has only 24 context points. Using direct least squares keeps
    every local fit deterministic and avoids the occasional explosive forecast
    generated by iterative maximum-likelihood AutoReg fitting on short series.
    """
    values = np.asarray(series, dtype=np.float64)
    if len(values) < 4 or not np.isfinite(values).all():
        return float(values[-1]) if len(values) else 0.0, np.empty(0)

    best = None
    upper_lag = min(max_lag, len(values) // 3)
    for lag in range(0, upper_lag + 1):
        if lag == 0:
            intercept = float(np.mean(values))
            coefficients = np.empty(0)
            residual = values - intercept
        else:
            target = values[lag:]
            design = np.column_stack([
                np.ones(len(target)),
                *[values[lag - step:len(values) - step] for step in range(1, lag + 1)],
            ])
            try:
                solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            except np.linalg.LinAlgError:
                continue
            intercept = float(solution[0])
            coefficients = solution[1:]
            residual = target - design @ solution
            # Stationary AR coefficients prevent runaway recursive forecasts.
            roots = np.roots(np.r_[1.0, -coefficients])
            if roots.size and np.any(np.abs(roots) <= 1.001):
                continue

        variance = max(float(np.mean(residual ** 2)), 1e-12)
        bic = len(residual) * np.log(variance) + (lag + 1) * np.log(len(residual))
        if best is None or bic < best[0]:
            best = (bic, intercept, coefficients)

    if best is None:
        return float(values[-1]), np.empty(0)
    return best[1], np.asarray(best[2], dtype=np.float64)


def _forecast_with_ar(series: np.ndarray, pred_len: int, max_lag: int) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        return np.zeros(pred_len, dtype=np.float64)
    intercept, coefficients = _fit_stable_ar(values, max_lag)
    history = list(values)
    center = float(np.mean(values))
    # A local AR fit must remain within the observed window's scale. This is a
    # numerical guard, not clipping the model output to a global benchmark
    # range: it falls back to the latest local observation when recursion
    # becomes implausible for this specific series.
    observed_std = float(np.std(values))
    local_span = float(np.max(values) - np.min(values))
    limit = max(4.0 * observed_std, 2.0 * local_span, 0.5)
    forecast = []
    for _ in range(pred_len):
        if len(coefficients):
            next_value = intercept + float(np.dot(coefficients, history[-len(coefficients):][::-1]))
        else:
            next_value = intercept
        if not np.isfinite(next_value) or abs(next_value - center) > limit:
            next_value = history[-1]
        history.append(float(next_value))
        forecast.append(float(next_value))
    return np.asarray(forecast, dtype=np.float64)


def _forecast_autoar(series: np.ndarray, pred_len: int) -> np.ndarray:
    """Locally refit a stable BIC-selected autoregression for one series."""
    return _forecast_with_ar(series, pred_len, max_lag=8)


def _forecast_autoarima(series: np.ndarray, pred_len: int) -> np.ndarray:
    """Choose and refit a stable ARIMA(p,d,0) model in local process memory.

    On short 24-step contexts, d in {0, 1} and BIC-selected p in [0, 3] are
    more reliable than an unconstrained ARIMA grid. A small holdout chooses the
    differencing order, then the selected model is refit on all observed rows.
    """
    values = np.asarray(series, dtype=np.float64)
    if len(values) < 6 or not np.isfinite(values).all():
        return np.repeat(values[-1] if len(values) else 0.0, pred_len)

    holdout = min(6, max(2, len(values) // 4))
    train, actual = values[:-holdout], values[-holdout:]
    candidates = []
    for difference_order in (0, 1):
        if difference_order and len(train) < 5:
            continue
        transformed = np.diff(train) if difference_order else train
        projected = _forecast_with_ar(transformed, holdout, max_lag=3)
        if difference_order:
            projected = train[-1] + np.cumsum(projected)
        score = float(np.mean(np.abs(projected - actual)))
        candidates.append((score, difference_order))

    difference_order = min(candidates, default=(0.0, 0))[1]
    transformed = np.diff(values) if difference_order else values
    forecast = _forecast_with_ar(transformed, pred_len, max_lag=3)
    if difference_order:
        # Reintegrating differenced forecasts can magnify one unstable local
        # increment. Validate each reconstructed point against this window's
        # own scale and continue from the last valid observation if needed.
        center = float(np.mean(values))
        limit = max(4.0 * float(np.std(values)), 2.0 * float(np.ptp(values)), 0.5)
        restored = []
        previous = float(values[-1])
        for increment in forecast:
            next_value = previous + float(increment)
            if not np.isfinite(next_value) or abs(next_value - center) > limit:
                next_value = previous
            restored.append(next_value)
            previous = next_value
        forecast = np.asarray(restored, dtype=np.float64)
    return np.asarray(forecast, dtype=np.float64)


def _infer_checkpoint_model(model_name: str, X: np.ndarray, pred_len: int) -> np.ndarray:
    """Run Mamba/TimeLLM models restored from their local training checkpoint."""
    torch = _lazy_import_torch()
    load, get_device = _lazy_load()
    model_inst, _ = load(model_name)
    device = get_device()
    with torch.no_grad():
        prediction = model_inst(torch.tensor(X, dtype=torch.float32, device=device))
    return prediction.detach().cpu().numpy()[:, :pred_len, :]
