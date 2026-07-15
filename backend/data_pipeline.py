"""
数据预处理管道: CSV/Excel/Parquet 解析 / 标准化 / 滑动窗口 / 测试数据加载
"""
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from four_g_protocol import (
    FEATURE_COLS,
    PRED_LEN,
    SEQ_LEN,
    STEP,
    build_windows,
    fit_training_scaler,
    load_observations,
)

# 确保 Excel 引擎可用
try:
    import openpyxl
except ImportError:
    openpyxl = None

NUM_CHANNELS = 8


def load_test_data():
    """Load benchmark observations and a scaler fitted on training observations."""
    df_train = load_observations("train")
    df_test = load_observations("test")
    return df_train, df_test, fit_training_scaler(df_train), list(FEATURE_COLS)


def get_test_window(window_idx: int, channel_idx: int):
    """
    获取指定测试窗口的输入/输出数据
    返回: X (24, 8), Y (24, 8), scaler, cols
    """
    _, df_test, scaler, cols = load_test_data()
    X, Y, _ = build_windows(df_test, scaler)
    if not 0 <= window_idx < len(X):
        raise IndexError(f"Test window index {window_idx} is out of range")
    return X[window_idx], Y[window_idx], scaler, cols


def get_all_test_windows():
    """Return all valid within-cell, continuous benchmark windows."""
    _, df_test, scaler, cols = load_test_data()
    X, Y, refs = build_windows(df_test, scaler)
    return list(zip(X, Y)), scaler, cols, refs


def parse_csv(file_bytes: bytes) -> dict:
    """
    解析上传的 CSV 文件
    返回: {headers, rows_preview, numeric_cols, df}
    """
    content = file_bytes.decode("utf-8-sig")

    first_line = content.split("\n")[0]
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line:
        sep = ";"
    else:
        sep = ","

    df = pd.read_csv(BytesIO(file_bytes), sep=sep)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_cols) == 0:
        raise ValueError("文件中没有有效的数值列")

    return {
        "headers": list(df.columns),
        "numeric_cols": numeric_cols,
        "rows_preview": df.head(5).to_dict(orient="records"),
        "total_rows": len(df),
        "df": df,
    }


def _parse_excel(file_bytes: bytes) -> dict:
    """
    解析上传的 Excel 文件 (.xlsx/.xls)
    """
    df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl" if openpyxl else None)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_cols) == 0:
        raise ValueError("Excel 文件中没有有效的数值列")

    return {
        "headers": list(df.columns),
        "numeric_cols": numeric_cols,
        "rows_preview": df.head(5).to_dict(orient="records"),
        "total_rows": len(df),
        "df": df,
    }


def _parse_parquet(file_bytes: bytes) -> dict:
    """
    解析上传的 Parquet 文件 (.parquet)
    """
    df = pd.read_parquet(BytesIO(file_bytes))

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_cols) == 0:
        raise ValueError("Parquet 文件中没有有效的数值列")

    return {
        "headers": list(df.columns),
        "numeric_cols": numeric_cols,
        "rows_preview": df.head(5).to_dict(orient="records"),
        "total_rows": len(df),
        "df": df,
    }


def parse_file(file_bytes: bytes, filename: str) -> dict:
    """
    自动检测格式并解析上传文件
    支持: .csv, .tsv, .xlsx, .xls, .parquet

    返回: {headers, rows_preview, numeric_cols, df, format}
    """
    ext = Path(filename).suffix.lower()

    if ext in (".xlsx", ".xls"):
        result = _parse_excel(file_bytes)
        result["format"] = ext[1:]  # "xlsx" or "xls"
    elif ext == ".parquet":
        result = _parse_parquet(file_bytes)
        result["format"] = "parquet"
    else:
        # 默认按 CSV 处理（含 .csv, .tsv, .txt 等）
        result = parse_csv(file_bytes)
        result["format"] = "csv"

    return result


def build_4g_window_from_upload(df: pd.DataFrame, target_col: str):
    """Build a real 4G model input without fabricating missing channels."""
    if target_col not in FEATURE_COLS:
        raise ValueError("The 4G target must be one of the eight benchmark features")
    missing = [column for column in FEATURE_COLS if column not in df.columns]
    if missing:
        raise ValueError("4G prediction requires all eight benchmark features; use generic prediction for other data")
    if len(df) < SEQ_LEN:
        raise ValueError(f"4G prediction requires at least {SEQ_LEN} rows")
    numeric = df.loc[:, FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("All eight 4G features must be numeric")
    scaler = fit_training_scaler(load_observations("train"))
    values = scaler.transform(numeric.to_numpy(dtype=np.float64)).astype(np.float32)
    channel = list(FEATURE_COLS).index(target_col)
    return values[-SEQ_LEN:], scaler, channel


def build_windows_from_csv(df: pd.DataFrame, target_col: str, pred_len: int):
    """
    从上传的 CSV DataFrame 构建推理窗口，自动对齐到模型8通道。
    - 列名匹配 FEATURE_COLS 的自动映射到对应通道
    - 未匹配的通道填零
    - target_col 指定要预测的列
    返回: (X_windows, scaler, target_scaler_idx, numeric_cols)
      - target_scaler_idx: target_col 在用户 scaler 中的列索引，用于逆标准化
    """
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在 CSV 中")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if target_col not in numeric_cols:
        raise ValueError(f"目标列 '{target_col}' 不是数值列，可选: {numeric_cols}")

    data = df[numeric_cols].values.astype(np.float32)

    if len(data) < SEQ_LEN + pred_len:
        raise ValueError(
            f"数据行数不足: 需要至少 {SEQ_LEN + pred_len} 行，实际 {len(data)} 行"
        )

    # 标准化用户数据
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)  # (n_samples, n_user_cols)

    # 找到 target_col 在用户 scaler 中的索引（用于后续逆标准化）
    target_scaler_idx = numeric_cols.index(target_col)

    # ─── 对齐到模型8通道 ───
    n_samples = len(scaled)
    aligned = np.zeros((n_samples, NUM_CHANNELS), dtype=np.float32)
    user_col_to_idx = {name: i for i, name in enumerate(numeric_cols)}
    target_model_channel = 0

    # 第一轮：列名精确匹配
    for model_idx, ch_name in enumerate(FEATURE_COLS):
        if ch_name in user_col_to_idx:
            aligned[:, model_idx] = scaled[:, user_col_to_idx[ch_name]]
            if ch_name == target_col:
                target_model_channel = model_idx

    # 第二轮：剩余空通道用用户列循环 + 微小噪声（不同通道用不同列，保持多样性）
    rng = np.random.RandomState(42)
    unmatched = [i for i, ch in enumerate(FEATURE_COLS) if ch not in user_col_to_idx]
    for j, model_idx in enumerate(unmatched):
        src_idx = j % len(numeric_cols)
        noise = rng.randn(n_samples).astype(np.float32) * 0.005
        aligned[:, model_idx] = scaled[:, src_idx] + noise

    # target_col 的模型通道索引
    if target_col in FEATURE_COLS:
        target_model_channel = FEATURE_COLS.index(target_col)
    else:
        target_model_channel = unmatched[0] if unmatched else 0

    # 构建窗口
    step = max(1, pred_len // 4)
    windows = []
    for i in range(0, len(aligned) - SEQ_LEN - pred_len + 1, step):
        X = aligned[i : i + SEQ_LEN]
        windows.append(X)

    return windows, scaler, target_scaler_idx, target_model_channel, numeric_cols


def inverse_transform_and_clip(predictions: np.ndarray, scaler: StandardScaler, cols: list):
    """逆标准化 + clip 到 >= 0"""
    orig_shape = predictions.shape
    flat = predictions.reshape(-1, len(cols))
    inv = scaler.inverse_transform(flat)
    inv = np.clip(inv, 0, None)
    return inv.reshape(orig_shape)
