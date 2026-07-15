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
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    # Small BIC grid: automatic model order selection without
                    # pretending that a single fixed ARIMA order is automatic.
                    candidates = []
                    for p in range(0, min(4, seq_len - 2)):
                        for q in range(0, min(4, seq_len - 2)):
                            if p == 0 and q == 0:
                                continue
                            try:
                                fitted = ARIMA(series, order=(p, 0, q)).fit()
                                candidates.append((fitted.bic, fitted))
                            except Exception:
                                continue
                    if not candidates:
                        raise ValueError("No ARIMA candidate could be fitted")
                    preds[b, :, c] = min(candidates, key=lambda item: item[0])[1].forecast(steps=pred_len)
                except Exception:
                    preds[b, :, c] = series[-1]

            elif method == "autoar":
                from statsmodels.tsa.ar_model import AutoReg, ar_select_order
                try:
                    selection = ar_select_order(series, maxlag=min(8, seq_len // 3), ic="bic", old_names=False)
                    lags = selection.ar_lags
                    if lags is None or len(lags) == 0:
                        raise ValueError("No autoregressive lags selected")
                    preds[b, :, c] = AutoReg(series, lags=lags, old_names=False).fit().predict(
                        start=seq_len, end=seq_len + pred_len - 1, dynamic=False
                    )
                except Exception:
                    preds[b, :, c] = series[-1]

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
            # IBM TTM: 需要 512 上下文
            context_len = info.get("context_len", 512)
            pad_len = context_len - seq_len
            past = np.pad(X[b], ((pad_len, 0), (0, 0)), mode="reflect")  # 反射填充保留波动特征
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
    import xgboost as xgb
    load, _ = _lazy_load()
    model_inst, info = load(model_name)
    batch, seq_len, n_channels = X.shape
    preds = np.zeros((batch, pred_len, n_channels))

    for b in range(batch):
        for c in range(n_channels):
            last_val = X[b, -1, c]
            if isinstance(model_inst, dict):
                col_model = list(model_inst.values())[min(c, len(model_inst) - 1)]
                for t in range(pred_len):
                    dtest = xgb.DMatrix(np.array([[last_val]]))
                    last_val = col_model.predict(dtest)[0]
                    preds[b, t, c] = last_val
            else:
                for t in range(pred_len):
                    dtest = xgb.DMatrix(np.array([[last_val]]))
                    last_val = model_inst.predict(dtest)[0]
                    preds[b, t, c] = last_val

    return preds
