"""
数据预处理管道: CSV/Excel/Parquet 解析 / 标准化 / 滑动窗口 / 测试数据加载
"""
import numpy as np
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 确保 Excel 引擎可用
try:
    import openpyxl
except ImportError:
    openpyxl = None

TSLIB_ROOT = Path(__file__).resolve().parent.parent / ".." / "网络流量预测项目新修改2" / "Time-Series-Library-main"
TSLIB_ROOT = TSLIB_ROOT.resolve()
DATA_DIR = TSLIB_ROOT / "data_provider" / "4g_traffic"

SEQ_LEN = 24
PRED_LEN = 24
STEP = 3
NUM_CHANNELS = 8

FEATURE_COLS = [
    "erab流量", "pdcch利用率", "pdsch利用率", "pusch利用率",
    "上行流量", "下行流量", "总流量", "有效连接数"
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
