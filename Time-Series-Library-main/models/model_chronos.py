"""
Amazon Chronos 零样本时间序列预测
=================================
支持三个模型变体:
  --model_name amazon/chronos-t5-tiny   (8M参数, 最快)
  --model_name amazon/chronos-t5-small  (20M参数)
  --model_name amazon/chronos-t5-base   (200M参数, 最慢但理论上最好)

Chronos 是 Amazon 发布的预训练时序模型，无需在目标数据上训练，
直接进行零样本预测。逐通道独立预测，取20次采样中位数。
"""
import sys, os, argparse, numpy as np, torch, warnings
sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import load_data, fit_scaler, generate_windows, compute_metrics, save_results

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--model_name', type=str, default='amazon/chronos-t5-tiny',
                        choices=['amazon/chronos-t5-tiny', 'amazon/chronos-t5-small', 'amazon/chronos-t5-base'])
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    print('=' * 60)
    print(f'  Chronos Zero-Shot: {args.model_name}')
    print('=' * 60)

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'  Device: {device}')

    # 加载 Chronos
    from chronos import ChronosPipeline
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    pipeline = ChronosPipeline.from_pretrained(args.model_name, device_map=device, dtype=dtype)
    print(f'  Model loaded.')

    # 加载数据
    df_train, df_test, feature_cols = load_data(args.data_path)
    num_channels = len(feature_cols)
    scaler = fit_scaler(df_train, feature_cols)
    test_scaled = scaler.transform(df_test[feature_cols].values)
    print(f'  Features: {num_channels} | Test samples: {len(test_scaled)}')

    # 生成窗口
    X_windows, trues_list, _ = generate_windows(test_scaled)
    n_windows = len(X_windows)
    print(f'  Windows: {n_windows}')

    # 预测
    preds_list = []
    print(f'  Predicting (8 channels × {n_windows} windows)...')
    with torch.no_grad():
        for idx, X in enumerate(X_windows):
            if (idx + 1) % 500 == 0:
                print(f'    {idx+1}/{n_windows}')

            pred_24 = np.zeros((24, num_channels), dtype=np.float32)
            for c in range(num_channels):
                ctx = torch.tensor(X[:, c], dtype=torch.float32).to(device)
                samples = pipeline.predict(ctx, prediction_length=24, num_samples=20)
                pred_24[:, c] = np.median(samples[0].cpu().numpy(), axis=0)

            preds_list.append(pred_24)

    # 指标
    metrics = compute_metrics(preds_list, trues_list, scaler, 24, num_channels)
    model_tag = args.model_name.replace('/', '_')
    model_name = f'Chronos_{model_tag}'
    print(f'\n  {model_name}')
    print(f'  MSE={metrics["MSE"]:.4f}  MAE={metrics["MAE"]:.4f}  RMSE={metrics["RMSE"]:.4f}')
    print(f'  MAPE={metrics["MAPE"]:.4f}  Custom_ACC={metrics["Custom_ACC"]:.4f}')

    save_results(model_name, metrics, args.output_dir)


if __name__ == '__main__':
    main()
