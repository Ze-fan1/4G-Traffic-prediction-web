"""
Shared utility module — data loading, scaling, metrics computation.
Used by model_chronos.py, model_mamba.py, model_timellm.py, model_xgboost.py.

COLUMN ORDER: Matches Dataset_Custom exactly — drop static cols, reorder target last.
SAVE FORMAT:  Scaled space (same as DL framework), for fair comparison on prediction curves.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = '总流量'


def load_data(root_path='./data'):
    """Load train/test data with Dataset_Custom-compatible column ordering.

    Dataset_Custom does:
      1. Drop ID编号, 厂商, 频段, 场景
      2. Reorder: [date] + cols(no target, no date) + [target]

    Returns: df_train, df_test, feature_order (8 cols, target LAST)
    """
    train_fp = os.path.join(root_path, 'df_4g_train_100.parquet')
    test_fp  = os.path.join(root_path, 'df_4g_test_100.parquet')
    df_train = pd.read_parquet(train_fp)
    df_test  = pd.read_parquet(test_fp)

    # 1. Drop non-numeric / static columns (same as Dataset_Custom)
    for col in ['ID编号', '厂商', '频段', '场景']:
        if col in df_train.columns:
            df_train = df_train.drop(columns=[col])
        if col in df_test.columns:
            df_test = df_test.drop(columns=[col])

    # 2. Reorder: date + cols(not target, not date) + [target]
    cols = list(df_train.columns)
    cols.remove(TARGET)
    cols.remove('date')
    feature_order = cols + [TARGET]  # 8 features, target LAST

    df_train = df_train[['date'] + feature_order]
    df_test  = df_test[['date'] + feature_order]

    return df_train, df_test, feature_order


def fit_scaler(df_train, feature_cols):
    """Fit StandardScaler on training data (column order matches Dataset_Custom)."""
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)
    return scaler


def generate_windows(data_scaled, seq_len=24, pred_len=24, step=3):
    """Create sliding windows matching Dataset_Custom protocol (step=3 for test)."""
    indices = list(range(0, len(data_scaled) - seq_len - pred_len + 1, step))
    trues_list = []
    for i in indices:
        Y = data_scaled[i + seq_len : i + seq_len + pred_len]
        trues_list.append(Y)
    X_windows = [data_scaled[i : i + seq_len] for i in indices]
    return X_windows, trues_list, indices


def calc_custom_acc(preds, trues):
    preds_flat = preds.reshape(-1, preds.shape[-1])
    trues_flat = trues.reshape(-1, trues.shape[-1])
    num_cells = trues_flat.shape[1]
    acc_list = []
    for i in range(num_cells):
        y, y_hat = trues_flat[:, i], preds_flat[:, i]
        y_mean_all = np.mean(y)
        S_mask = y >= y_mean_all
        if not np.any(S_mask):
            continue
        y_S, y_hat_S = y[S_mask], y_hat[S_mask]
        y_mean_S = np.mean(y_S)
        if y_mean_S == 0:
            continue
        MAE_S = np.mean(np.abs(y_S - y_hat_S))
        acc_list.append(max(0.0, min(1.0, np.abs(y_mean_S - MAE_S) / y_mean_S)))
    return np.mean(acc_list) if acc_list else 0.0


def compute_metrics(preds_list, trues_list, scaler, pred_len, num_channels):
    """Compute metrics in scaled space; also return scaled-space arrays for saving."""
    preds_arr = np.array(preds_list).reshape(-1, num_channels)
    trues_arr = np.array(trues_list).reshape(-1, num_channels)

    # Scaled-space metrics (primary)
    mse  = np.mean((trues_arr - preds_arr) ** 2)
    mae  = np.mean(np.abs(trues_arr - preds_arr))
    rmse = float(np.sqrt(mse))

    # Original-space metrics (MAPE, Custom_ACC)
    preds_inv = scaler.inverse_transform(preds_arr)
    trues_inv = scaler.inverse_transform(trues_arr)
    preds_inv_3d = preds_inv.reshape(-1, pred_len, num_channels)
    trues_inv_3d = trues_inv.reshape(-1, pred_len, num_channels)

    avg_acc = calc_custom_acc(preds_inv_3d, trues_inv_3d)
    mask = trues_inv > 1e-5
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((trues_inv[mask] - preds_inv[mask]) / trues_inv[mask]))
        mspe = np.mean(np.square((trues_inv[mask] - preds_inv[mask]) / trues_inv[mask]))
    else:
        mape, mspe = 0.0, 0.0

    return {
        'MSE': mse, 'MAE': mae, 'RMSE': rmse,
        'MAPE': mape, 'MSPE': mspe, 'Custom_ACC': avg_acc,
        'preds_scaled': np.array(preds_list),    # (N, 24, 8) scaled — for pred.npy
        'trues_scaled': np.array(trues_list),    # (N, 24, 8) scaled — for true.npy
        'preds_original': preds_inv_3d,
        'trues_original': trues_inv_3d,
    }


def save_results(model_name, metrics, output_dir='./results'):
    """Save pred.npy and true.npy in SCALED space (matching DL framework convention)."""
    os.makedirs(output_dir, exist_ok=True)

    # Write metrics to result file
    result_line = (f'MSE:{metrics["MSE"]:.4f}, MAE:{metrics["MAE"]:.4f}, '
                   f'RMSE:{metrics["RMSE"]:.4f}, MAPE:{metrics["MAPE"]:.4f}, '
                   f'MSPE:{metrics["MSPE"]:.4f}, Custom_ACC:{metrics["Custom_ACC"]:.4f}')
    with open('result_long_term_forecast.txt', 'a', encoding='utf-8') as f:
        f.write(f'{model_name}\n{result_line}\n\n')

    # Save in SCALED space (same as DL models from Exp_Long_Term_Forecast.test())
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, 'pred.npy'), metrics['preds_scaled'])
    np.save(os.path.join(model_dir, 'true.npy'), metrics['trues_scaled'])
    print(f'Results saved to {output_dir}/{model_name}/  [SCALED space]')
