"""
生成 AutoAR 和 LinearRegression 的预置 pred.npy（σ空间）
两个模型都使用 sklearn LinearRegression on time index — 快速向量化
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

TSLIB = r'c:\Users\Admin\Desktop\网络流量预测项目新修改2\Time-Series-Library-main'
DATA_DIR = os.path.join(TSLIB, 'data_provider', '4g_traffic')
RESULTS_DIR = os.path.join(TSLIB, 'results')

FEATURE_COLS = [
    "erab流量", "pdcch利用率", "pdsch利用率", "pusch利用率",
    "上行流量", "下行流量", "总流量", "有效连接数"
]

SEQ_LEN, PRED_LEN, STEP = 24, 24, 3

# ─── Load data ───
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'df_4g_train_100.parquet'))
df_test  = pd.read_parquet(os.path.join(DATA_DIR, 'df_4g_test_100.parquet'))
cols = [c for c in FEATURE_COLS if c in df_test.columns]
n_channels = len(cols)

scaler = StandardScaler()
scaler.fit(df_train[cols].values)
val_data = scaler.transform(df_test[cols].values)  # σ空间

n_windows = (len(val_data) - SEQ_LEN - PRED_LEN) // STEP + 1
print(f"Total windows: {n_windows}, val_data shape: {val_data.shape}")

# Pre-build time index features
t_train = np.arange(SEQ_LEN, dtype=np.float32).reshape(-1, 1)
t_future = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN, dtype=np.float32).reshape(-1, 1)

for method, dirname in [
    ('autoar', 'AutoAR_4G'),
    ('linear_regression', 'LinearRegression_4G'),
]:
    out_dir = os.path.join(RESULTS_DIR, dirname)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGenerating {dirname}...")
    all_pred = np.zeros((n_windows, PRED_LEN, n_channels), dtype=np.float32)

    for i in range(n_windows):
        start = i * STEP
        X = val_data[start:start + SEQ_LEN]  # (24, n_channels)

        for c in range(n_channels):
            series = X[:, c]
            lr = LinearRegression()
            lr.fit(t_train, series)
            all_pred[i, :, c] = lr.predict(t_future)

        if (i + 1) % 2000 == 0 or i == n_windows - 1:
            print(f"  [{dirname}] {i+1}/{n_windows} windows done")

    # Save
    pred_path = os.path.join(out_dir, 'pred.npy')
    np.save(pred_path, all_pred)
    print(f"Saved: {pred_path}  shape={all_pred.shape}  mean={all_pred.mean():.4f}  std={all_pred.std():.4f}")

    # Compute overall metrics in σ space
    all_true = np.zeros_like(all_pred)
    for i in range(n_windows):
        start = i * STEP
        all_true[i] = val_data[start + SEQ_LEN:start + SEQ_LEN + PRED_LEN]

    p_flat = all_pred.reshape(-1, n_channels)
    t_flat = all_true.reshape(-1, n_channels)

    mse = float(np.mean((t_flat - p_flat) ** 2))
    mae = float(np.mean(np.abs(t_flat - p_flat)))
    rmse = float(np.sqrt(mse))
    print(f"Metrics (σ-space): MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

print("\nDone!")
