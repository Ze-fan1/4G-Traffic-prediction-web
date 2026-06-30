"""
FastAPI 后端 — 4G Traffic Model Playground
启动: cd backend && python server.py
"""
import sys, os, time
from pathlib import Path

# 把 Time-Series-Library 加入 Python path
TSLIB = Path(__file__).resolve().parent.parent / ".." / "网络流量预测项目新修改2" / "Time-Series-Library-main"
TSLIB = TSLIB.resolve()
if str(TSLIB) not in sys.path:
    sys.path.insert(0, str(TSLIB))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from model_registry import MODEL_REGISTRY, get_model_info, list_models
from model_loader import load, unload, get_loaded_model_name, get_device
from inference_engine import infer
from data_pipeline import (
    get_test_window, get_all_test_windows,
    parse_csv, build_windows_from_csv,
    inverse_transform_and_clip, FEATURE_COLS,
)

app = FastAPI(title="4G Traffic Playground API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══ Health ═══

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "gpu": str(get_device()),
        "loaded_model": get_loaded_model_name(),
    }


# ═══ Model List ═══

@app.get("/api/models")
async def api_list_models():
    """返回全部 26 模型"""
    result = []
    for name, info in MODEL_REGISTRY.items():
        result.append({
            "name": name,
            "category": info["category"],
            "tier": info["tier"],
            "type": info["type"],
            "available": info["tier"] == 1,
            "tier_reason": info.get("tier2_reason") or info.get("tier3_reason", ""),
        })
    return result


# ═══ Demo: Quick Verification ═══

@app.post("/api/demo/{model_name}/quick")
async def demo_quick(model_name: str, body: dict):
    """
    单窗口快速验证
    Request: {channel_idx: int, window_idx?: int}
    Response: {pred: [24], truth: [24], window: int, mae: float}
    """
    info = get_model_info(model_name)
    if info is None:
        raise HTTPException(404, detail={
            "error": "unknown_model",
            "available": list(MODEL_REGISTRY.keys())[:10]
        })
    if info["tier"] >= 2:
        raise HTTPException(503, detail={
            "error": "model_not_available",
            "tier": info["tier"],
            "reason": info.get("tier2_reason", info.get("tier3_reason", ""))
        })

    channel_idx = body.get("channel_idx", 1)
    window_idx = body.get("window_idx")
    if window_idx is None:
        import random
        window_idx = random.randint(0, 5377)

    # 加载数据
    X, Y, scaler, cols = get_test_window(window_idx, channel_idx)

    try:
        t0 = time.time()
        pred = infer(model_name, X, pred_len=24)
        elapsed = time.time() - t0
    except Exception as e:
        raise HTTPException(500, detail={"error": "inference_failed", "detail": str(e)})

    # 逆标准化
    X_inv = inverse_transform_and_clip(X[np.newaxis, :, :], scaler, cols)[0]
    Y_inv = inverse_transform_and_clip(Y[np.newaxis, :, :], scaler, cols)[0]
    pred_inv = inverse_transform_and_clip(
        pred[np.newaxis, :, :] if pred.ndim == 2 else pred, scaler, cols
    )[0]

    # 通道重排（修复 6/7 交换 bug，如果恰好 8 通道）
    if pred_inv.shape[1] >= 8:
        pred_inv[:, [6, 7]] = pred_inv[:, [7, 6]]
        Y_inv[:, [6, 7]] = Y_inv[:, [7, 6]]

    channel_pred = pred_inv[:, channel_idx].tolist()
    channel_truth = Y_inv[:, channel_idx].tolist()
    mae = float(np.mean(np.abs(np.array(channel_pred) - np.array(channel_truth))))

    return {
        "model": model_name,
        "window": window_idx,
        "channel": FEATURE_COLS[channel_idx] if channel_idx < len(FEATURE_COLS) else f"ch{channel_idx}",
        "channel_idx": channel_idx,
        "pred": channel_pred,
        "truth": channel_truth,
        "mae": round(mae, 6),
        "elapsed_s": round(elapsed, 2),
        "device": str(get_device()),
    }


# ═══ Demo: Full Evaluation ═══

@app.post("/api/demo/{model_name}/full")
async def demo_full(model_name: str, body: dict):
    """
    全量评估
    Request: {channel_idx: int}
    Response: {metrics: {mse, mae, rmse, mape}, n_windows: int, elapsed_s: float}
    """
    info = get_model_info(model_name)
    if info is None:
        raise HTTPException(404, detail={"error": "unknown_model"})
    if info["tier"] >= 2:
        raise HTTPException(503, detail={
            "error": "model_not_available",
            "tier": info["tier"]
        })

    channel_idx = body.get("channel_idx", 1)
    windows, scaler, cols = get_all_test_windows()

    all_preds = []
    all_trues = []
    t0 = time.time()

    for i, (X, Y) in enumerate(windows):
        try:
            pred = infer(model_name, X, pred_len=24)
            all_preds.append(pred[0] if pred.ndim == 3 else pred)
            all_trues.append(Y)
        except Exception as e:
            raise HTTPException(500, detail={
                "error": "inference_failed",
                "window": i,
                "detail": str(e)
            })

    elapsed = time.time() - t0

    all_preds_arr = np.array(all_preds)  # (n_windows, pred_len, n_channels)
    all_trues_arr = np.array(all_trues)  # (n_windows, pred_len, n_channels)

    # Element-wise average across all windows → (pred_len, n_channels)
    avg_pred = np.mean(all_preds_arr, axis=0)
    avg_truth = np.mean(all_trues_arr, axis=0)

    # Inverse transform
    avg_pred_inv = scaler.inverse_transform(avg_pred)
    avg_truth_inv = scaler.inverse_transform(avg_truth)

    # Channel-specific average curves
    avg_pred_ch = avg_pred_inv[:, channel_idx].tolist()
    avg_truth_ch = avg_truth_inv[:, channel_idx].tolist()

    # Global metrics (flattened)
    preds_flat = all_preds_arr.reshape(-1, len(cols))
    trues_flat = all_trues_arr.reshape(-1, len(cols))
    preds_inv = scaler.inverse_transform(preds_flat)
    trues_inv = scaler.inverse_transform(trues_flat)

    mse = float(np.mean((trues_inv - preds_inv) ** 2))
    mae = float(np.mean(np.abs(trues_inv - preds_inv)))
    rmse = float(np.sqrt(mse))

    mask = trues_inv > 1e-5
    mape = float(np.mean(np.abs((trues_inv[mask] - preds_inv[mask]) / trues_inv[mask]))) if np.sum(mask) > 0 else 0.0

    return {
        "model": model_name,
        "metrics": {
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
        },
        "avg_pred": avg_pred_ch,
        "avg_truth": avg_truth_ch,
        "n_windows": len(windows),
        "elapsed_s": round(elapsed, 1),
        "device": str(get_device()),
    }


# ═══ Custom Prediction ═══

@app.post("/api/predict/{model_name}")
async def predict(
    model_name: str,
    csv_file: UploadFile = File(...),
    target_col: str = Form(...),
    pred_len: int = Form(24),
):
    """
    用户上传 CSV 自定义预测
    Response: {predictions: [[...]], meta: {...}}
    """
    info = get_model_info(model_name)
    if info is None:
        raise HTTPException(404, detail={"error": "unknown_model"})
    if info["tier"] >= 2:
        raise HTTPException(503, detail={
            "error": "model_not_available",
            "tier": info["tier"]
        })

    if pred_len not in [6, 12, 18, 24]:
        pred_len = 24

    try:
        file_bytes = await csv_file.read()
        parsed = parse_csv(file_bytes)
    except ValueError as e:
        raise HTTPException(400, detail={
            "error": "csv_parse_error",
            "detail": str(e)
        })

    df = parsed["df"]

    if target_col not in parsed["numeric_cols"]:
        raise HTTPException(400, detail={
            "error": "invalid_target",
            "numeric_cols": parsed["numeric_cols"]
        })

    try:
        windows, scaler, numeric_cols = build_windows_from_csv(df, target_col, pred_len)
    except ValueError as e:
        raise HTTPException(400, detail={"error": str(e)})

    if len(windows) == 0:
        raise HTTPException(400, detail={
            "error": "insufficient_data",
            "min_rows": 30,
            "actual": len(df)
        })

    try:
        X_last = windows[-1]
        pred = infer(model_name, X_last, pred_len=pred_len)
    except Exception as e:
        raise HTTPException(500, detail={
            "error": "inference_failed",
            "detail": str(e)
        })

    # 逆标准化
    pred_flat = pred.reshape(-1, len(numeric_cols)) if pred.ndim == 3 else pred
    if pred_flat.ndim == 1:
        pred_flat = pred_flat.reshape(-1, 1)
    pred_inv = scaler.inverse_transform(pred_flat)
    pred_inv = np.clip(pred_inv, 0, None)

    target_idx = numeric_cols.index(target_col)
    predictions = pred_inv[:, target_idx].tolist()

    return {
        "model": model_name,
        "predictions": predictions,
        "meta": {
            "target_col": target_col,
            "pred_len": pred_len,
            "actual_pred_len": pred.shape[1] if pred.ndim >= 2 else len(pred),
            "input_rows": len(df),
            "n_windows": len(windows),
            "device": str(get_device()),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
