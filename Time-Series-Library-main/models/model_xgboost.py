"""
XGBoost 时间序列预测
====================
为每个输出维度独立训练一个 XGBoost 回归器（192个模型 = 24小时×8通道）
输入: 过去24小时展平 → 输出: 未来24小时
"""
import sys, os, argparse, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import load_data, fit_scaler, generate_windows, compute_metrics, save_results
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    print('=' * 60)
    print('  XGBoost Baseline')
    print('=' * 60)

    # 加载数据
    df_train, df_test, feature_cols = load_data(args.data_path)
    num_channels = len(feature_cols)
    scaler = fit_scaler(df_train, feature_cols)
    train_scaled = scaler.transform(df_train[feature_cols].values)
    test_scaled = scaler.transform(df_test[feature_cols].values)
    print(f'  Features: {num_channels} | Train: {len(train_scaled)} | Test: {len(test_scaled)}')

    # 生成窗口
    X_windows, trues_list, _ = generate_windows(test_scaled)
    print(f'  Windows: {len(X_windows)}')

    # 构建训练样本
    seq_len, pred_len = 24, 24
    X_train_list, Y_train_list = [], []
    for i in range(0, len(train_scaled) - seq_len - pred_len + 1):
        X_train_list.append(train_scaled[i:i+seq_len])
        Y_train_list.append(train_scaled[i+seq_len:i+seq_len+pred_len])
    X_train = np.array(X_train_list).reshape(len(X_train_list), -1)  # (N, 192)
    Y_train = np.array(Y_train_list).reshape(len(Y_train_list), -1)  # (N, 192)
    print(f'  Training samples: {X_train.shape[0]}, Input dim: {X_train.shape[1]}, Output dim: {Y_train.shape[1]}')

    # 训练192个XGBoost模型
    print(f'  Training {Y_train.shape[1]} XGBoost regressors...')
    models = []
    for out_dim in range(Y_train.shape[1]):
        if (out_dim + 1) % 50 == 0:
            print(f'    Progress: {out_dim+1}/{Y_train.shape[1]}')
        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, objective='reg:squarederror',
            verbosity=0, n_jobs=-1,
        )
        model.fit(X_train, Y_train[:, out_dim])
        models.append(model)
    print(f'  Training done!')

    # Persist all 24 x 8 regressors so the playground can reproduce this run.
    model_name = f'XGBoost_n{args.n_estimators}_d{args.max_depth}_lr{args.learning_rate}'
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump({
            'models': models,
            'seq_len': seq_len,
            'pred_len': pred_len,
            'num_channels': num_channels,
            'feature_order': feature_cols,
        }, f)

    # 预测
    print(f'  Predicting {len(X_windows)} windows...')
    preds_list = []
    for idx, X in enumerate(X_windows):
        X_flat = X.reshape(1, -1)
        pred_flat = np.array([models[d].predict(X_flat)[0] for d in range(len(models))])
        preds_list.append(pred_flat.reshape(pred_len, num_channels))

    # 计算指标
    metrics = compute_metrics(preds_list, trues_list, scaler, pred_len, num_channels)
    print(f'\n  {model_name}')
    print(f'  MSE={metrics["MSE"]:.4f}  MAE={metrics["MAE"]:.4f}  RMSE={metrics["RMSE"]:.4f}')
    print(f'  MAPE={metrics["MAPE"]:.4f}  Custom_ACC={metrics["Custom_ACC"]:.4f}')

    save_results(model_name, metrics, args.output_dir)


if __name__ == '__main__':
    main()
