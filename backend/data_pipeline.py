"""
数据预处理管道: CSV/Excel/Parquet 解析 / 标准化 / 滑动窗口 / 测试数据加载
"""
import pandas as pd
from io import BytesIO
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from four_g_protocol import (
    FEATURE_COLS,
    SEQ_LEN,
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
        raise ValueError("4G prediction requires all eight benchmark features; choose a statistical model for a single-series upload")
    if len(df) < SEQ_LEN:
        raise ValueError(f"4G prediction requires at least {SEQ_LEN} rows")
    numeric = df.loc[:, FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("All eight 4G features must be numeric")
    scaler = fit_training_scaler(load_observations("train"))
    values = scaler.transform(numeric.to_numpy(dtype=np.float64)).astype(np.float32)
    channel = list(FEATURE_COLS).index(target_col)
    return values[-SEQ_LEN:], scaler, channel
