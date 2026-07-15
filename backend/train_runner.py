"""训练脚本 — 10 epochs 从零训练 + 生成逆标准化预测曲线（单窗口）"""
import sys, os, json, time

MODEL_NAME = sys.argv[1]
JOB_ID = sys.argv[2]
from project_paths import TSLIB_ROOT

TSLIB = str(TSLIB_ROOT)

sys.path.insert(0, TSLIB)
sys.path.insert(0, os.path.join(TSLIB, 'models'))
os.chdir(TSLIB)

import torch
import numpy as np
from four_g_protocol import FEATURE_COLS, build_windows, fit_training_scaler, load_observations

print(f'JOB:{JOB_ID}:MODEL:{MODEL_NAME}', flush=True)
print(f'GPU:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}', flush=True)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# ─── Args ───
class Args: pass

args = Args()
args.task_name = 'long_term_forecast'; args.is_training = 1
args.model = MODEL_NAME; args.model_id = '4G_Live'
args.data = '4g_panel'; args.root_path = os.path.join(TSLIB, 'data_provider', '4g_traffic')
args.data_path = 'df_4g_train_100.parquet'
args.features = 'M'; args.target = '总流量'; args.freq = 'h'
args.checkpoints = './checkpoints/'
args.seq_len = 24; args.label_len = 12; args.pred_len = 24
args.enc_in = 8; args.dec_in = 8; args.c_out = 8
args.d_model = 512; args.n_heads = 8; args.e_layers = 2; args.d_layers = 1; args.d_ff = 2048
args.moving_avg = 25; args.factor = 1; args.distil = True; args.dropout = 0.1
args.embed = 'timeF'; args.activation = 'gelu'; args.num_workers = 0
args.itr = 1; args.train_epochs = 10; args.batch_size = 32; args.patience = 5
args.learning_rate = 0.0001; args.des = f'Train_{JOB_ID}'
args.loss = 'MSE'; args.lradj = 'type1'; args.use_amp = False
args.use_gpu = True; args.gpu = 0; args.gpu_type = 'cuda'
args.use_multi_gpu = False; args.devices = '0'; args.output_attention = False
args.p_hidden_dims = [128, 128]; args.p_hidden_layers = 2
args.use_dtw = False; args.augmentation_ratio = 0; args.seed = 2
for a in ['jitter','scaling','permutation','randompermutation','magwarp','timewarp',
           'windowslice','windowwarp','rotation','spawner','dtwwarp','shapedtwwarp',
           'wdba','discdtw','discsdtw']:
    setattr(args, a, False)
args.extra_tag = ''; args.patch_len = 16; args.node_dim = 10
args.gcn_depth = 2; args.gcn_dropout = 0.3; args.propalpha = 0.3
args.conv_channel = 32; args.skip_channel = 32; args.individual = False
args.alpha = 0.1; args.top_p = 0.5; args.pos = 1
args.channel_independence = 1; args.decomp_method = 'moving_avg'; args.use_norm = 1
args.down_sampling_layers = 0; args.down_sampling_window = 1
args.down_sampling_method = None
args.seg_len = 96; args.expand = 2; args.d_conv = 4
args.tv_dt = 0; args.tv_B = 0; args.tv_C = 0; args.use_D = 0
args.top_k = 5; args.num_kernels = 6
args.inverse = False; args.mask_rate = 0.25; args.anomaly_ratio = 0.25
args.seasonal_patterns = 'Monthly'

cfgs = {
    'DLinear':      {'d_model': 128, 'individual': False},
    'LightTS':      {'d_model': 128},
    'TSMixer':      {'d_model': 128},
    'PatchTST':     {'d_model': 128, 'd_ff': 256, 'n_heads': 4},
    'Informer':     {'d_model': 128, 'd_ff': 256, 'n_heads': 4},
    'SCINet':       {'d_model': 128},
    'TimesNet':     {'d_model': 64, 'd_ff': 128, 'num_kernels': 3, 'e_layers': 1},
    'SegRNN':       {'seg_len': 24, 'd_model': 256},
    'Autoformer':   {'d_model': 64, 'd_ff': 128, 'n_heads': 8, 'e_layers': 1},
    'Transformer':  {'d_model': 512, 'd_ff': 2048, 'n_heads': 8},
    'iTransformer': {'d_model': 256, 'd_ff': 512, 'n_heads': 8},
    '★ BaseModel': {'d_model': 512, 'd_ff': 2048, 'n_heads': 8},
}
for k, v in cfgs.get(MODEL_NAME, {}).items():
    setattr(args, k, v)

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    args.use_gpu = False

try:
    # ─── 1. 训练 ───
    exp = Exp_Long_Term_Forecast(args)
    setting = f'{MODEL_NAME}_4G_Train_{JOB_ID}'
    print(f'TRAIN_START:{setting}:EPOCHS:{args.train_epochs}', flush=True)

    for ii in range(args.itr):
        exp.train(setting)
        exp.test(setting)
        if args.device.type == 'cuda':
            torch.cuda.empty_cache()

    # ─── 2. Evaluate on the same panel-aware test windows as the benchmark ───
    df_train = load_observations('train')
    df_test = load_observations('test')
    test_x, test_y, test_refs = build_windows(df_test, fit_training_scaler(df_train))
    total_windows = len(test_x)
    display_idx = 0

    print(f'CURVE_START:WINDOWS:{total_windows}', flush=True)

    model = exp.model
    model.eval()
    device = args.device
    n_channels = len(FEATURE_COLS)

    all_pred_sigma = []  # σ空间
    all_true_sigma = []
    display_pred_sigma = None
    display_true_sigma = None

    for i in range(total_windows):
        X = test_x[i]
        Y = test_y[i]

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
        dec_inp = torch.zeros((1, args.label_len + pred_len, n_channels),
                              dtype=torch.float32).to(device)
        dec_inp[:, :args.label_len, :] = X_t[:, -args.label_len:, :]

        with torch.no_grad():
            out = model(X_t, None, dec_inp, None)
        if isinstance(out, tuple):
            out = out[0]
        pred = out[0, -pred_len:, :].cpu().numpy()  # σ空间

        all_pred_sigma.append(pred.reshape(-1))
        all_true_sigma.append(Y.reshape(-1))

        if i == display_idx:
            display_pred_sigma = pred.copy()
            display_true_sigma = Y.copy()

        if (i + 1) % 500 == 0 or i == total_windows - 1:
            print(f'CURVE_PROGRESS:{i+1}:{total_windows}', flush=True)

    print(f'CURVE_DONE:{total_windows}', flush=True)

    # ─── 整体指标（σ空间，全部窗口）───
    all_p = np.array(all_pred_sigma)
    all_t = np.array(all_true_sigma)
    mse_val  = float(np.mean((all_t - all_p) ** 2))
    mae_val  = float(np.mean(np.abs(all_t - all_p)))
    rmse_val = float(np.sqrt(mse_val))

    # ─── 单窗口曲线输出（σ空间）───
    curves = {}
    for ch_idx, channel_name in enumerate(FEATURE_COLS):
        curves[channel_name] = {
            "pred":  [round(float(v), 4) for v in display_pred_sigma[:, ch_idx]],
            "truth": [round(float(v), 4) for v in display_true_sigma[:, ch_idx]],
        }

    curves["_meta"] = {
        "window_idx": display_idx,
        "total_windows": total_windows,
        "cell_id": test_refs[display_idx].cell_id,
        "start": test_refs[display_idx].start.isoformat(),
        "protocol": "4g-panel-v1",
    }
    print(f'CURVES:{json.dumps(curves)}', flush=True)
    # 把指标也一并输出
    metrics_info = {
        "mse":      round(mse_val, 4),
        "mae":      round(mae_val, 4),
        "rmse":     round(rmse_val, 4),
        "train_loss": round(mse_val, 4),  # 兼容前端展示
        "val_loss":   round(mae_val, 4),
        "test_loss":  round(rmse_val, 4),
    }
    print(f'METRICS:{json.dumps(metrics_info)}', flush=True)
    print(f'TRAIN_DONE:{setting}', flush=True)

except Exception as e:
    import traceback
    print(f'TRAIN_ERROR:{e}', flush=True)
    traceback.print_exc()
    sys.exit(1)
