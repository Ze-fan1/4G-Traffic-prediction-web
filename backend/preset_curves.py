"""
预置曲线加载器 — 统一从测试数据计算 σ空间真实值，保证所有模型一致。
支持单窗口（默认 #4394）和所有窗口平均两种模式。
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from model_registry import MODEL_REGISTRY

TSLIB_ROOT = Path(__file__).resolve().parent.parent / ".." / "网络流量预测项目新修改2" / "Time-Series-Library-main"
RESULTS_DIR = TSLIB_ROOT / "results"
DATA_DIR = TSLIB_ROOT / "data_provider" / "4g_traffic"

FEATURE_COLS = [
    "erab流量", "pdcch利用率", "pdsch利用率", "pusch利用率",
    "上行流量", "下行流量", "总流量", "有效连接数"
]

DEFAULT_WINDOW = 4394

# 缓存测试数据标准化结果
_cache = None


def _get_test_data():
    """加载测试数据并标准化，缓存结果。返回 (val_data, scaler, cols, n_windows)"""
    global _cache
    if _cache is not None:
        return _cache

    train_path = DATA_DIR / "df_4g_train_100.parquet"
    test_path  = DATA_DIR / "df_4g_test_100.parquet"
    if not train_path.exists() or not test_path.exists():
        return None, None, None, 0

    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path)

    cols = [c for c in FEATURE_COLS if c in df_test.columns]
    if len(cols) < 8:
        skip = ["ID编号", "厂商", "频段", "场景", "date"]
        cols = [c for c in df_test.columns if c not in skip]

    scaler = StandardScaler()
    scaler.fit(df_train[cols].values)
    val_data = scaler.transform(df_test[cols].values)

    seq_len, pred_len, step = 24, 24, 3
    n_windows = (len(val_data) - seq_len - pred_len) // step + 1

    _cache = (val_data, pred_len, step, cols, n_windows)
    return _cache


def load_preset_curves(model_name: str, window_idx: int | None = DEFAULT_WINDOW) -> dict | None:
    """
    加载预置曲线。window_idx=None 表示所有窗口平均，-1 也表示平均。
    真实值从测试数据统一计算（保证所有模型一致）。
    """
    info = MODEL_REGISTRY.get(model_name)
    if not info:
        return None

    result_dir = info.get("result_dir")
    if not result_dir or not os.path.isdir(result_dir):
        return None

    pred_path = os.path.join(result_dir, "pred.npy")
    true_npy_path = os.path.join(result_dir, "true.npy")
    if not os.path.exists(pred_path):
        pred_path = os.path.join(result_dir, "preds.npy")
        true_npy_path = os.path.join(result_dir, "trues.npy")
    if not os.path.exists(pred_path):
        return None

    try:
        pred_all = np.load(pred_path)
    except Exception:
        return None

    if pred_all.ndim != 3:
        return None

    n_pred_windows, pred_len, n_channels = pred_all.shape
    val_data, _, step, cols, n_windows = _get_test_data()
    if val_data is None:
        return None

    # ─── 判断 pred 的空间并统一到 σ空间 ───
    is_original = np.abs(pred_all.reshape(-1).mean()) > 5.0
    if is_original:
        scaler = None
        train_path = DATA_DIR / "df_4g_train_100.parquet"
        if train_path.exists():
            df_train = pd.read_parquet(train_path)
            c = [c for c in FEATURE_COLS if c in df_train.columns]
            scaler = StandardScaler()
            scaler.fit(df_train[c].values)
        if scaler is not None:
            p_flat = pred_all.reshape(-1, n_channels)
            pred_sigma = scaler.transform(p_flat).reshape(n_pred_windows, pred_len, n_channels)
        else:
            pred_sigma = pred_all
    else:
        pred_sigma = pred_all

    # ─── 单窗口曲线 ───
    idx = min(window_idx, n_windows - 1)
    start = idx * step
    true_sigma = val_data[start + 24:start + 24 + pred_len]  # (pred_len, n_channels)
    pred_idx = min(idx, n_pred_windows - 1)

    curves = {}
    for ch_idx, ch_name in enumerate(cols):
        curves[ch_name] = {
            "pred":  [round(float(v), 4) for v in pred_sigma[pred_idx, :, ch_idx]],
            "truth": [round(float(v), 4) for v in true_sigma[:, ch_idx]],
        }

    # ─── 指标（σ空间，全部窗口）───
    p_flat = pred_sigma.reshape(-1, n_channels)
    limit = min(n_windows, n_pred_windows)
    all_t = np.array([val_data[i*step+24:i*step+24+pred_len].reshape(-1) for i in range(limit)])
    t_flat = all_t.reshape(-1, n_channels)
    common_len = min(len(p_flat), len(t_flat))
    mse  = float(np.mean((t_flat[:common_len] - p_flat[:common_len]) ** 2))
    mae  = float(np.mean(np.abs(t_flat[:common_len] - p_flat[:common_len])))
    rmse = float(np.sqrt(mse))

    return {
        "curves": curves,
        "window_idx": idx,
        "total_windows": n_windows,
        "mode": "single",
        "space": "sigma",
        "metrics_summary": {
            "mse":  round(mse, 4),
            "mae":  round(mae, 4),
            "rmse": round(rmse, 4),
        },
        "source": "benchmark_preset",
    }


def get_available_preset_models() -> list:
    available = []
    for name, info in MODEL_REGISTRY.items():
        result_dir = info.get("result_dir")
        if result_dir and os.path.isdir(result_dir):
            pred_path = os.path.join(result_dir, "pred.npy")
            if not os.path.exists(pred_path):
                pred_path = os.path.join(result_dir, "preds.npy")
            if os.path.exists(pred_path):
                available.append(name)
    return available
