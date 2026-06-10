"""
共享工具模块 — 数据加载、标准化、指标计算
所有模型脚本共用此文件，保证评估协议一致
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# 数据加载
# ============================================================
def load_data(root_path='./data'):
    """
    加载训练集和测试集
    返回: df_train, df_test, feature_cols
    """
    train_fp = f'{root_path}/df_4g_train_100.parquet'
    test_fp = f'{root_path}/df_4g_test_100.parquet'

    df_train = pd.read_parquet(train_fp)
    df_test = pd.read_parquet(test_fp)

    drop_cols = ['ID编号', '厂商', '频段', '场景', 'date']
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    return df_train, df_test, feature_cols


# ============================================================
# 标准化
# ============================================================
def fit_scaler(df_train, feature_cols):
    """在训练集上拟合 StandardScaler"""
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)
    return scaler


# ============================================================
# 评估协议：滑动窗口生成
# ============================================================
def generate_windows(data_scaled, seq_len=24, pred_len=24, step=3):
    """
    按滑动窗口切分测试数据
    data_scaled: (N, C) 标准化后的测试数据
    返回: preds_list, trues_list — 每个元素形状 (pred_len, C)
    """
    indices = list(range(0, len(data_scaled) - seq_len - pred_len + 1, step))
    trues_list, preds_list = [], []

    # preds_list 由各模型自行填充，这里只返回窗口索引和真实值
    for i in indices:
        Y = data_scaled[i + seq_len : i + seq_len + pred_len]  # (24, C) scaled
        trues_list.append(Y)

    X_windows = [data_scaled[i : i + seq_len] for i in indices]  # (24, C) each

    return X_windows, trues_list, indices


# ============================================================
# 自定义指标 (与 Time-Series-Library 完全一致)
# ============================================================
def calc_custom_acc(preds, trues):
    """
    高值子集准确率
    preds/trues shape: (samples, pred_len, cells) — 原始空间
    """
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


# ============================================================
# 统一指标计算
# ============================================================
def compute_metrics(preds_list, trues_list, scaler, pred_len, num_channels):
    """
    输入:
      preds_list: list of (pred_len, C) arrays — scaled空间
      trues_list: list of (pred_len, C) arrays — scaled空间
      scaler: 拟合好的 StandardScaler
    输出:
      dict with MSE, MAE, RMSE (scaled空间), MAPE, MSPE, Custom_ACC (原始空间)
    """
    preds_arr = np.array(preds_list).reshape(-1, num_channels)
    trues_arr = np.array(trues_list).reshape(-1, num_channels)

    # Scaled空间指标
    mse = np.mean((trues_arr - preds_arr) ** 2)
    mae = np.mean(np.abs(trues_arr - preds_arr))
    rmse = np.sqrt(mse)

    # 反归一化
    preds_inv = scaler.inverse_transform(preds_arr)
    trues_inv = scaler.inverse_transform(trues_arr)

    # 原始空间指标
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
        'preds_original': preds_inv_3d,
        'trues_original': trues_inv_3d,
    }


# ============================================================
# 结果保存
# ============================================================
def save_results(model_name, metrics, output_dir='./results'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 保存指标到文本
    with open(f'{output_dir}/all_results.txt', 'a', encoding='utf-8') as f:
        f.write(f'{model_name}\n')
        f.write(f'MSE:{metrics["MSE"]:.4f}, MAE:{metrics["MAE"]:.4f}, '
                f'RMSE:{metrics["RMSE"]:.4f}, MAPE:{metrics["MAPE"]:.4f}, '
                f'MSPE:{metrics["MSPE"]:.4f}, Custom_ACC:{metrics["Custom_ACC"]:.4f}\n\n')

    # 保存预测数组
    model_dir = f'{output_dir}/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    np.save(f'{model_dir}/preds.npy', metrics['preds_original'])
    np.save(f'{model_dir}/trues.npy', metrics['trues_original'])

    print(f'Results saved to {output_dir}/')
