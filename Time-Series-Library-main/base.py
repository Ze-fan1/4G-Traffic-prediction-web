"""
External Base Model Evaluation — DL-compatible data pipeline.
Matches the framework's Dataset_Custom exactly and saves in SCALED space
so BaseModel can be directly compared with DL models on the same chart.

Usage: python base.py --root_path ./data_provider/4g_traffic/
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.utils.metrics import MAE, MSE, RMSE, MAPE, MSPE, calc_custom_acc

DROP_COLS = ['date', 'ID编号', '厂商', '频段', '场景']
SEQ_LEN, PRED_LEN, STEP = 24, 24, 3
TARGET = '总流量'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate External Base Model')
    parser.add_argument('--root_path', type=str, default='./data_provider/4g_traffic/')
    args = parser.parse_args()

    train_fp = os.path.join(args.root_path, 'df_4g_train_100.parquet')
    test_fp  = os.path.join(args.root_path, 'df_4g_test_100.parquet')
    base_fp  = os.path.join(args.root_path, 'df_4g_base_100.parquet')

    print(">>>>>> Loading data...")
    df_train = pd.read_parquet(train_fp)
    df_test  = pd.read_parquet(test_fp)
    df_base  = pd.read_parquet(base_fp)

    # ── Same column processing as Dataset_Custom ──
    # 1. Drop non-numeric / static columns
    for col in ['ID编号', '厂商', '频段', '场景']:
        if col in df_train.columns:
            df_train = df_train.drop(columns=[col])
        if col in df_test.columns:
            df_test = df_test.drop(columns=[col])
        if col in df_base.columns:
            df_base = df_base.drop(columns=[col])

    # 2. Reorder: date + cols(no target, no date) + [target]
    train_cols = list(df_train.columns)
    train_cols.remove(TARGET)
    train_cols.remove('date')
    feature_order = train_cols + [TARGET]   # 8 features, target last
    df_train = df_train[['date'] + feature_order]

    # Apply same column order to test
    df_test = df_test[['date'] + feature_order]

    num_channels = len(feature_order)
    print(f">>>>>> Features ({num_channels}): {feature_order}")

    # ── Fit scaler on training data (same as DL models) ──
    scaler = StandardScaler()
    scaler.fit(df_train[feature_order].values)

    # ── Scale test data → ground truth in scaled space ──
    test_arr = df_test[feature_order].values          # (16181, 8)
    test_scaled = scaler.transform(test_arr)

    # ── Align base predictions with test data ──
    # test and base are perfectly aligned (same dates, IDs, 16181 rows)
    # Extract forecast columns from base, reorder to match feature_order
    fc_map = {}
    for fc in df_base.columns:
        if fc.startswith('forecast_'):
            base_name = fc[len('forecast_'):]  # strip prefix
            fc_map[base_name] = fc

    pred_vals_list = []
    for feat in feature_order:
        if feat in fc_map:
            pred_vals_list.append(df_base[fc_map[feat]].values)
        else:
            # Fallback: build forecast col name literally
            fc_name = f'forecast_{feat}'
            if fc_name in df_base.columns:
                pred_vals_list.append(df_base[fc_name].values)
            else:
                raise KeyError(f"Cannot find forecast column for feature '{feat}'")

    pred_arr = np.column_stack(pred_vals_list)        # (16181, 8)
    pred_scaled = scaler.transform(pred_arr)

    # Verify alignment
    assert len(test_scaled) == len(pred_scaled) == 16181, \
        f"Row count mismatch: test={len(test_scaled)}, pred={len(pred_scaled)}"

    # ── Create sliding windows (exact Dataset_Custom protocol) ──
    N = len(test_scaled)
    n_expected = (N - SEQ_LEN - PRED_LEN + 1) // STEP  # (16181 - 47) // 3 = 5378
    indices = list(range(0, N - SEQ_LEN - PRED_LEN + 1, STEP))
    assert len(indices) == n_expected == 5378, \
        f"Window count mismatch: got {len(indices)}, expected {n_expected}"
    print(f">>>>>> Sliding windows: {len(indices)} (seq={SEQ_LEN}, pred={PRED_LEN}, step={STEP})")

    trues_list, preds_list = [], []
    for i in indices:
        Y = test_scaled[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]   # (24, 8)
        P = pred_scaled[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]   # (24, 8)
        trues_list.append(Y)
        preds_list.append(P)

    preds_3d = np.array(preds_list)  # (5378, 24, 8)
    trues_3d = np.array(trues_list)  # (5378, 24, 8)

    # ── Metrics in scaled space ──
    mse  = MSE(preds_3d, trues_3d)
    mae  = MAE(preds_3d, trues_3d)
    rmse = float(np.sqrt(mse))

    # ── MAPE / MSPE / Custom_ACC in original space ──
    preds_flat = preds_3d.reshape(-1, num_channels)
    trues_flat = trues_3d.reshape(-1, num_channels)

    preds_orig = scaler.inverse_transform(preds_flat)
    trues_orig = scaler.inverse_transform(trues_flat)

    preds_orig_3d = preds_orig.reshape(preds_3d.shape)
    trues_orig_3d = trues_orig.reshape(trues_3d.shape)

    mask = trues_orig > 1e-5
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((trues_orig[mask] - preds_orig[mask]) / trues_orig[mask]))
        mspe = np.mean(np.square((trues_orig[mask] - preds_orig[mask]) / trues_orig[mask]))
    else:
        mape, mspe = 0.0, 0.0

    avg_acc = calc_custom_acc(preds_orig_3d, trues_orig_3d)

    # ── Report ──
    result_str = (f'MSE:{mse:.4f}, MAE:{mae:.4f}, RMSE:{rmse:.4f}, '
                  f'MAPE:{mape:.4f}, MSPE:{mspe:.4f}, Custom_ACC:{avg_acc:.4f}')

    print('\n' + '=' * 60)
    print('    EXTERNAL BASE MODEL (SCALED SPACE — DL compatible)')
    print('=' * 60)
    print(f'Model: External_BaseModel | {len(indices)} windows')
    print(result_str)
    print('=' * 60)

    setting = 'External_BaseModel_4G_Base_v2_Verify'
    with open('result_long_term_forecast.txt', 'a') as f:
        f.write(f'{setting}\n{result_str}\n\n')

    # ── CRITICAL: Save in SCALED space (same as DL models) ──
    save_dir = f'results/{setting}'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'pred.npy'), preds_3d)   # scaled space ← KEY CHANGE
    np.save(os.path.join(save_dir, 'true.npy'), trues_3d)   # scaled space ← KEY CHANGE
    print(f'Results saved to {save_dir}/  [SCALED space]')
    print('>>>>>> Base model evaluation complete!')


if __name__ == '__main__':
    main()
