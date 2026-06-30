"""
数据预处理管道: CSV 解析 / 标准化 / 滑动窗口 / 测试数据加载
"""
import numpy as np
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from pathlib import Path

TSLIB_ROOT = Path(__file__).resolve().parent.parent / ".." / "网络流量预测项目新修改2" / "Time-Series-Library-main"
TSLIB_ROOT = TSLIB_ROOT.resolve()
DATA_DIR = TSLIB_ROOT / "data_provider" / "4g_traffic"

SEQ_LEN = 24
PRED_LEN = 24
STEP = 3
NUM_CHANNELS = 8

FEATURE_COLS = [
    "ERAB流量", "PDCCH利用率", "PDSCH利用率", "PUSCH利用率",
    "上行流量", "下行流量", "有效连接数", "总流量"
]


def load_test_data():
    """加载测试集 + 在训练集上拟合的 Scaler"""
    train_fp = DATA_DIR / "df_4g_train_100.parquet"
    test_fp = DATA_DIR / "df_4g_test_100.parquet"

    if not train_fp.exists() or not test_fp.exists():
        raise FileNotFoundError(f"数据文件不存在: {train_fp} / {test_fp}")

    df_train = pd.read_parquet(train_fp)
    df_test = pd.read_parquet(test_fp)

    train_cols = [c for c in FEATURE_COLS if c in df_train.columns]
    test_cols = [c for c in FEATURE_COLS if c in df_test.columns]

    if len(train_cols) < NUM_CHANNELS:
        skip = ["ID编号", "厂商", "频段", "场景", "date"]
        train_cols = [c for c in df_train.columns if c not in skip]
        test_cols = [c for c in df_test.columns if c not in skip]

    scaler = StandardScaler()
    scaler.fit(df_train[train_cols].values)

    return df_train, df_test, scaler, train_cols


def get_test_window(window_idx: int, channel_idx: int):
    """
    获取指定测试窗口的输入/输出数据
    返回: X (24, 8), Y (24, 8), scaler, cols
    """
    _, df_test, scaler, cols = load_test_data()
    test_data = scaler.transform(df_test[cols].values)

    start = window_idx * STEP
    X = test_data[start : start + SEQ_LEN]
    Y = test_data[start + SEQ_LEN : start + SEQ_LEN + PRED_LEN]

    return X, Y, scaler, cols


def get_all_test_windows():
    """获取全部测试窗口"""
    _, df_test, scaler, cols = load_test_data()
    test_data = scaler.transform(df_test[cols].values)

    n_windows = (len(test_data) - SEQ_LEN - PRED_LEN) // STEP + 1
    windows = []
    for i in range(n_windows):
        start = i * STEP
        X = test_data[start : start + SEQ_LEN]
        Y = test_data[start + SEQ_LEN : start + SEQ_LEN + PRED_LEN]
        windows.append((X, Y))

    return windows, scaler, cols


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
        raise ValueError("CSV 中没有有效的数值列")

    return {
        "headers": list(df.columns),
        "numeric_cols": numeric_cols,
        "rows_preview": df.head(5).to_dict(orient="records"),
        "total_rows": len(df),
        "df": df,
    }


def build_windows_from_csv(df: pd.DataFrame, target_col: str, pred_len: int):
    """
    从上传的 CSV DataFrame 构建推理窗口
    返回: X_windows (list of np.array), scaler, numeric_cols
    """
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在 CSV 中")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    data = df[numeric_cols].values.astype(np.float32)

    if len(data) < SEQ_LEN + pred_len:
        raise ValueError(
            f"数据行数不足: 需要至少 {SEQ_LEN + pred_len} 行，实际 {len(data)} 行"
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    step = max(1, pred_len // 4)
    windows = []
    for i in range(0, len(scaled) - SEQ_LEN - pred_len + 1, step):
        X = scaled[i : i + SEQ_LEN]
        windows.append(X)

    return windows, scaler, numeric_cols


def inverse_transform_and_clip(predictions: np.ndarray, scaler: StandardScaler, cols: list):
    """逆标准化 + clip 到 >= 0"""
    orig_shape = predictions.shape
    flat = predictions.reshape(-1, len(cols))
    inv = scaler.inverse_transform(flat)
    inv = np.clip(inv, 0, None)
    return inv.reshape(orig_shape)
