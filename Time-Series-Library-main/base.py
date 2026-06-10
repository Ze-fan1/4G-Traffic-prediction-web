"""
External Base Model Evaluation — sliding window protocol.
Matches the framework's 24h→24h, step=3 sliding window evaluation exactly.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.utils.metrics import MAE, MSE, RMSE, MAPE, MSPE, calc_custom_acc

DROP_COLS = ['date', 'ID编号', '厂商', '频段', '场景']
SEQ_LEN, PRED_LEN, STEP = 24, 24, 3


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate External Base Model')
    parser.add_argument('--root_path', type=str, default='./data_provider/4g_traffic/')
    args = parser.parse_args()

    train_fp = os.path.join(args.root_path, 'df_4g_train_100.parquet')
    test_fp = os.path.join(args.root_path, 'df_4g_test_100.parquet')
    base_fp = os.path.join(args.root_path, 'df_4g_base_100.parquet')

    print(">>>>>> Loading data...")
    df_train = pd.read_parquet(train_fp)
    df_test = pd.read_parquet(test_fp)
    df_base = pd.read_parquet(base_fp)

    feature_cols = [c for c in df_train.columns if c not in DROP_COLS]
    num_channels = len(feature_cols)
    print(f">>>>>> Features ({num_channels}): {feature_cols}")

    # Merge base predictions with test data on date/ID
    merge_keys = ['date', 'ID编号']
    for key in ['厂商', '频段', '场景']:
        if key in df_test.columns and key in df_base.columns:
            merge_keys.append(key)

    df_merged = pd.merge(df_test, df_base, on=merge_keys, suffixes=('_true', '_pred'))
    if len(df_merged) == 0:
        raise ValueError("Merge failed! Check date/ID alignment.")
    print(f">>>>>> Merged {len(df_merged)} rows")

    # Extract true and predicted values, sorted by date
    df_merged = df_merged.sort_values('date')
    true_vals = df_merged[feature_cols].values  # (N, 8)
    pred_vals = np.column_stack([
        df_merged[f'forecast_{c}'].values for c in feature_cols
    ])  # (N, 8)

    # Fit scaler on training data (same as DL models)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    true_scaled = scaler.transform(true_vals)
    pred_scaled = scaler.transform(pred_vals)

    # Create sliding windows (same protocol as framework)
    indices = list(range(0, len(true_scaled) - SEQ_LEN - PRED_LEN + 1, STEP))
    print(f">>>>>> Sliding windows: {len(indices)} (seq={SEQ_LEN}, pred={PRED_LEN}, step={STEP})")

    trues_list, preds_list = [], []
    for i in indices:
        Y = true_scaled[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]  # (24, 8)
        P = pred_scaled[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]  # (24, 8)
        trues_list.append(Y)
        preds_list.append(P)

    preds_3d = np.array(preds_list)  # (windows, 24, 8)
    trues_3d = np.array(trues_list)  # (windows, 24, 8)

    # MSE/MAE/RMSE in scaled space
    mse = MSE(preds_3d, trues_3d)
    mae = MAE(preds_3d, trues_3d)
    rmse = np.sqrt(mse)

    # MAPE/MSPE/Custom_ACC in original space (inverse transform)
    preds_flat_scaled = preds_3d.reshape(-1, num_channels)
    trues_flat_scaled = trues_3d.reshape(-1, num_channels)

    preds_orig = scaler.inverse_transform(preds_flat_scaled)
    trues_orig = scaler.inverse_transform(trues_flat_scaled)

    preds_orig_3d = preds_orig.reshape(preds_3d.shape)
    trues_orig_3d = trues_orig.reshape(trues_3d.shape)

    mask = trues_orig > 1e-5
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((trues_orig[mask] - preds_orig[mask]) / trues_orig[mask]))
        mspe = np.mean(np.square((trues_orig[mask] - preds_orig[mask]) / trues_orig[mask]))
    else:
        mape, mspe = 0.0, 0.0

    avg_acc = calc_custom_acc(preds_orig_3d, trues_orig_3d)

    # Save results
    result_str = (f'MSE:{mse:.4f}, MAE:{mae:.4f}, RMSE:{rmse:.4f}, '
                  f'MAPE:{mape:.4f}, MSPE:{mspe:.4f}, Custom_ACC:{avg_acc:.4f}')

    print('\n' + '=' * 60)
    print('    EXTERNAL BASE MODEL EVALUATION RESULTS')
    print('=' * 60)
    print(f'Model: External_BaseModel (sliding window, {len(indices)} windows)')
    print(result_str)
    print('=' * 60)

    setting = f'External_BaseModel_4G_Base_v2_Verify'
    with open('result_long_term_forecast.txt', 'a') as f:
        f.write(f'{setting}\n{result_str}\n\n')

    # Save numpy arrays for plot script
    save_dir = f'results/{setting}'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'pred.npy'), preds_orig_3d)
    np.save(os.path.join(save_dir, 'true.npy'), trues_orig_3d)
    print(f'Results saved to {save_dir}/')

    # Also save a copy with the original folder name for backward compat
    print('>>>>>> Base model evaluation complete!')


if __name__ == '__main__':
    main()
