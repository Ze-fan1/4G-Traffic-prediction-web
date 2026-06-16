#!/usr/bin/env python
"""
4G Traffic Prediction — Unified Model Runner
=============================================
一键运行所有模型，交互式菜单选择。

环境: ETP-exp1 (PyTorch 2.10+cu130, RTX 5050)
用法: python run.py
"""
import os, sys, warnings, subprocess
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

# ─── 固定路径 ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data_provider', '4g_traffic')
MODELS_DIR = os.path.join(ROOT, 'models')  # 原 shared_models 已合并到 models/
RESULT_FILE = os.path.join(ROOT, 'result_long_term_forecast.txt')
PYTHON_EXE = sys.executable  # 当前 Python (ETP-exp1)

sys.path.insert(0, ROOT)
# 注意: 不插入 MODELS_DIR 到 sys.path，因为其 utils.py 会覆盖项目的 utils/ 包

from sklearn.preprocessing import StandardScaler


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ═══════════════════════════════════════════════════════════
# 菜单定义
# ═══════════════════════════════════════════════════════════

MODELS = {
    '2': {
        'name': '传统统计模型 (全部7个一键运行)',
        'desc': 'Naive + Persistent_24h + HA + AutoARIMA + AutoAR + LinearRegression + XGBoost',
        'group': 'traditional_all',
    },
    '3':  {'name': 'Naive (重复最后一个值)',              'desc': '简单基线',  'group': 'simple_baseline'},
    '4':  {'name': 'Persistent_24h (24h持续性)',          'desc': '简单基线',  'group': 'simple_baseline'},
    '5':  {'name': 'Historical_Average (HA, 历史均值)',    'desc': '统计基线',  'group': 'simple_baseline'},
    '6':  {'name': 'AutoARIMA',                          'desc': '统计模型',  'group': 'traditional_single'},
    '7':  {'name': 'AutoAR (自动自回归)',                  'desc': '统计模型',  'group': 'traditional_single'},
    '8':  {'name': 'LinearRegression (线性回归)',          'desc': '统计模型',  'group': 'traditional_single'},
    '9':  {'name': 'XGBoost',                            'desc': '机器学习',  'group': 'xgboost'},
    '10': {'name': 'Base [基准对比]',                      'desc': '基准对比',  'group': 'external_base'},
    '11': {'name': 'PatchTST',                           'desc': 'Transformer', 'group': 'dl'},
    '12': {'name': 'iTransformer',                       'desc': 'Transformer', 'group': 'dl'},
    '13': {'name': 'Informer',                           'desc': 'Transformer', 'group': 'dl'},
    '14': {'name': 'Autoformer',                         'desc': 'Transformer', 'group': 'dl'},
    '15': {'name': 'Transformer',                        'desc': 'Transformer', 'group': 'dl'},
    '16': {'name': 'DLinear',                            'desc': 'MLP',        'group': 'dl'},
    '17': {'name': 'LightTS',                            'desc': 'MLP',        'group': 'dl'},
    '18': {'name': 'TSMixer',                            'desc': 'MLP',        'group': 'dl'},
    '19': {'name': 'IBM_TTM (TinyTimeMixer 零样本)',      'desc': 'MLP/预训练', 'group': 'ttm'},
    '20': {'name': 'SCINet',                             'desc': 'CNN',        'group': 'dl'},
    '21': {'name': 'TimesNet',                           'desc': 'CNN',        'group': 'dl'},
    '22': {'name': 'SegRNN',                             'desc': 'RNN',        'group': 'dl'},
    '23': {'name': 'TimeLLM (GPT-2 重编程)',              'desc': 'LLM',        'group': 'timellm'},
    '24': {'name': 'Chronos (Amazon 零样本)',             'desc': 'LLM/预训练', 'group': 'chronos'},
    '25': {'name': 'Mamba (状态空间模型 SSM)',             'desc': 'SSM',        'group': 'mamba'},
}


def print_menu():
    print('\n' + '=' * 70)
    print('    4G 网络流量预测 — 统一模型运行器')
    print('    Python: {} | CUDA: {} | GPU: {}'.format(
        sys.executable.split('envs')[-1].replace('\\', '/').strip('/').split('/')[0] if 'envs' in sys.executable else 'base',
        'Yes' if torch.cuda.is_available() else 'No',
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))
    print('=' * 70)
    print('  ★ 1. 运行所有模型 (全部23个，一键全跑)')
    print('  ★ 2. 传统统计模型一键运行 (7个: Naive→XGBoost)')
    print('-' * 70)
    for key in sorted(MODELS.keys(), key=lambda x: int(x)):
        if key == '2':
            continue
        m = MODELS[key]
        print(f'    {key:>2}. {m["name"]:<40} [{m["desc"]}]')
    print('-' * 70)
    print('    0. 退出')
    print('=' * 70)


# ═══════════════════════════════════════════════════════════
# 简单基线: Naive, Persistent_24h, Historical_Average
# ═══════════════════════════════════════════════════════════

def run_simple_baselines(model_names=None):
    from utils.metrics import calc_custom_acc

    train_fp = os.path.join(DATA_DIR, 'df_4g_train_100.parquet')
    test_fp = os.path.join(DATA_DIR, 'df_4g_test_100.parquet')
    drop_cols = ['date', 'ID编号', '厂商', '频段', '场景']

    df_train = pd.read_parquet(train_fp)
    df_test = pd.read_parquet(test_fp)
    num_cols = [c for c in df_test.columns if c not in drop_cols]
    print(f'特征列 ({len(num_cols)}): {num_cols}')

    scaler = StandardScaler()
    scaler.fit(df_train[num_cols].values)
    test_scaled = scaler.transform(df_test[num_cols].values)

    seq_len, pred_len, step = 24, 24, 3
    indices = list(range(0, len(test_scaled) - seq_len - pred_len + 1, step))
    print(f'测试窗口: {len(indices)}')

    def evaluate(preds_3d, trues_3d, model_name, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        TA = trues_3d.reshape(-1, len(num_cols))
        PA = preds_3d.reshape(-1, len(num_cols))
        mse = np.mean((TA - PA) ** 2)
        mae = np.mean(np.abs(TA - PA))
        rmse = np.sqrt(mse)
        TA_inv = scaler.inverse_transform(TA)
        PA_inv = scaler.inverse_transform(PA)
        mask = TA_inv > 1e-5
        mape = np.mean(np.abs((TA_inv[mask] - PA_inv[mask]) / TA_inv[mask])) if np.sum(mask) > 0 else 0.0
        mspe = np.mean(np.square((TA_inv[mask] - PA_inv[mask]) / TA_inv[mask])) if np.sum(mask) > 0 else 0.0
        preds_inv_3d = PA_inv.reshape(preds_3d.shape)
        trues_inv_3d = TA_inv.reshape(trues_3d.shape)
        avg_acc = calc_custom_acc(preds_inv_3d, trues_inv_3d)
        np.save(os.path.join(save_dir, 'pred.npy'), preds_inv_3d)
        np.save(os.path.join(save_dir, 'true.npy'), trues_inv_3d)
        result_str = f'MSE:{mse:.4f}, MAE:{mae:.4f}, RMSE:{rmse:.4f}, MAPE:{mape:.4f}, MSPE:{mspe:.4f}, Custom_ACC:{avg_acc:.4f}'
        print(f'  [OK] {model_name}: {result_str}')
        with open(RESULT_FILE, 'a') as f:
            f.write(f'{model_name}\n{result_str}\n\n')
        return result_str

    to_run = model_names or ['Naive', 'Persistent_24h', 'Historical_Average']
    results = {}

    if 'Naive' in to_run:
        print('\n--- Naive (Repeat Last Value) ---')
        p = np.zeros((len(indices), pred_len, len(num_cols)))
        t = np.zeros((len(indices), pred_len, len(num_cols)))
        for idx, i in enumerate(indices):
            p[idx] = np.tile(test_scaled[i + seq_len - 1], (pred_len, 1))
            t[idx] = test_scaled[i + seq_len : i + seq_len + pred_len]
        results['Naive'] = evaluate(p, t, 'Naive_4G', 'results/Naive_4G')

    if 'Persistent_24h' in to_run:
        print('\n--- Persistent_24h (Repeat Last 24h) ---')
        p = np.zeros((len(indices), pred_len, len(num_cols)))
        t = np.zeros((len(indices), pred_len, len(num_cols)))
        for idx, i in enumerate(indices):
            p[idx] = test_scaled[i : i + seq_len][-pred_len:]
            t[idx] = test_scaled[i + seq_len : i + seq_len + pred_len]
        results['Persistent_24h'] = evaluate(p, t, 'Persistent_24h_4G', 'results/Persistent24h_4G')

    if 'Historical_Average' in to_run:
        print('\n--- Historical_Average (Hourly Mean) ---')
        df_train_h = pd.read_parquet(train_fp)
        df_test_h = pd.read_parquet(test_fp)
        num_cols_h = [c for c in df_test_h.columns if c not in drop_cols]
        df_train_h['hour'] = pd.to_datetime(df_train_h['date']).dt.hour
        hourly_mean = df_train_h.groupby('hour')[num_cols_h].mean().values
        df_test_h['hour'] = pd.to_datetime(df_test_h['date']).dt.hour
        test_hours = df_test_h['hour'].values
        p = np.zeros((len(indices), pred_len, len(num_cols_h)))
        t = np.zeros((len(indices), pred_len, len(num_cols_h)))
        for idx, i in enumerate(indices):
            future_hours = test_hours[i + seq_len : i + seq_len + pred_len]
            p[idx] = scaler.transform(np.array([hourly_mean[h] for h in future_hours]))
            t[idx] = test_scaled[i + seq_len : i + seq_len + pred_len]
        results['Historical_Average'] = evaluate(p, t, 'Historical_Average_4G', 'results/HistoricalAverage_4G')

    return results


# ═══════════════════════════════════════════════════════════
# 传统统计模型: AutoARIMA, AutoAR, LinearRegression
# ═══════════════════════════════════════════════════════════

def run_traditional_single(model_name):
    from sklearn.linear_model import LinearRegression
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    from statsmodels.tsa.ar_model import ar_select_order, AutoReg

    train_fp = os.path.join(DATA_DIR, 'df_4g_train_100.parquet')
    test_fp = os.path.join(DATA_DIR, 'df_4g_test_100.parquet')
    df_train = pd.read_parquet(train_fp)
    df_test = pd.read_parquet(test_fp)

    feature_cols = [c for c in df_train.columns if c not in ['ID编号', '厂商', '频段', '场景', 'date']]
    scaler = StandardScaler().fit(df_train[feature_cols].values)
    test_scaled = scaler.transform(df_test[feature_cols].values)

    seq_len, pred_len, step = 24, 24, 3
    indices = list(range(0, len(test_scaled) - seq_len - pred_len + 1, step))

    trues, preds = [], []
    print(f'>> 运行 {model_name} ({len(indices)} 个窗口)...')

    for idx, i in enumerate(indices):
        print(f'  窗口 {idx+1}/{len(indices)}', end='\r')
        X = test_scaled[i : i + seq_len]
        Y = test_scaled[i + seq_len : i + seq_len + pred_len]
        trues.append(Y)

        if model_name == 'LinearRegression':
            p = np.array([LinearRegression().fit(
                np.arange(seq_len).reshape(-1, 1), X[:, col]
            ).predict(np.arange(seq_len, seq_len + pred_len).reshape(-1, 1))
                for col in range(X.shape[1])]).T
            preds.append(p)

        elif model_name == 'AutoAR':
            p = np.zeros((pred_len, X.shape[1]))
            for col in range(X.shape[1]):
                try:
                    lags_res = ar_select_order(X[:, col], maxlag=4, ic='aic')
                    if lags_res.ar_lags is not None:
                        p[:, col] = AutoReg(X[:, col], lags=lags_res.ar_lags).fit().predict(
                            start=seq_len, end=seq_len + pred_len - 1)
                    else:
                        p[:, col] = X[-1, col]
                except Exception:
                    p[:, col] = X[-1, col]
            if np.isnan(p).any() or np.isinf(p).any() or np.max(np.abs(p)) > 50.0:
                for col in range(X.shape[1]):
                    if np.isnan(p[:, col]).any() or np.isinf(p[:, col]).any() or np.max(np.abs(p[:, col])) > 50.0:
                        p[:, col] = X[-1, col]
            preds.append(p)

        elif model_name == 'AutoARIMA':
            p = np.zeros((pred_len, X.shape[1]))
            try:
                sf = StatsForecast(models=[AutoARIMA(season_length=1)], freq='h', n_jobs=1)
                sf_df = pd.concat([
                    pd.DataFrame({'unique_id': f'c{col}',
                                  'ds': pd.date_range('2020-01-01', periods=seq_len, freq='H'),
                                  'y': X[:, col]})
                    for col in range(X.shape[1])
                ])
                forecast = sf.forecast(df=sf_df, h=pred_len).reset_index()
                model_col = forecast.columns[2]
                for col in range(X.shape[1]):
                    p[:, col] = forecast[forecast['unique_id'] == f'c{col}'][model_col].values
            except Exception:
                p = np.repeat(X[-1, :][np.newaxis, :], pred_len, axis=0)
            if np.isnan(p).any() or np.isinf(p).any() or np.max(np.abs(p)) > 50.0:
                for col in range(X.shape[1]):
                    if np.isnan(p[:, col]).any() or np.isinf(p[:, col]).any() or np.max(np.abs(p[:, col])) > 50.0:
                        p[:, col] = X[-1, col]
            preds.append(p)

    print()
    preds_arr = np.array(preds).reshape(-1, preds[0].shape[1])
    trues_arr = np.array(trues).reshape(-1, trues[0].shape[1])
    mse = np.mean((trues_arr - preds_arr) ** 2)
    mae = np.mean(np.abs(trues_arr - preds_arr))
    rmse = np.sqrt(mse)

    preds_inv = scaler.inverse_transform(preds_arr)
    trues_inv = scaler.inverse_transform(trues_arr)

    def calc_custom_acc(preds_raw, trues_raw):
        preds_flat = preds_raw.reshape(-1, preds_raw.shape[-1])
        trues_flat = trues_raw.reshape(-1, trues_raw.shape[-1])
        acc_list = []
        for i in range(trues_flat.shape[1]):
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

    acc = calc_custom_acc(preds_inv, trues_inv)
    mask = trues_inv > 1e-5
    mape = np.mean(np.abs((trues_inv[mask] - preds_inv[mask]) / trues_inv[mask])) if np.sum(mask) > 0 else 0.0
    mspe = np.mean(np.square((trues_inv[mask] - preds_inv[mask]) / trues_inv[mask])) if np.sum(mask) > 0 else 0.0

    result_str = f'Model: {model_name} | MSE:{mse:.4f} | MAE:{mae:.4f} | RMSE:{rmse:.4f} | MAPE:{mape:.4f} | MSPE:{mspe:.4f} | ACC:{acc:.4f}'
    print(f'  [OK] {result_str}')
    with open(RESULT_FILE, 'a') as f:
        f.write(result_str + '\n')

    save_dir = f'results/{model_name}_4G'
    os.makedirs(save_dir, exist_ok=True)
    # Save SCALED space (matching DL model convention for generate_web_data.py)
    np.save(f'{save_dir}/pred.npy', preds_arr.reshape(-1, pred_len, preds[0].shape[1]))
    np.save(f'{save_dir}/true.npy', trues_arr.reshape(-1, pred_len, preds[0].shape[1]))

    return mse, mae, rmse, mape, mspe, acc


# ═══════════════════════════════════════════════════════════
# XGBoost
# ═══════════════════════════════════════════════════════════

def run_xgboost():
    print('\n' + '=' * 60)
    print('  XGBoost Baseline')
    print('=' * 60)
    script = os.path.join(MODELS_DIR, 'model_xgboost.py')
    cmd = f'"{PYTHON_EXE}" "{script}" --data_path "{DATA_DIR}" --output_dir "./results"'
    print(f'  CMD: {cmd}')
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════
# Deep Learning 模型
# ═══════════════════════════════════════════════════════════

def run_dl_model(model_name):
    import random
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)

    device = get_device()
    print(f'\n> 使用设备: {device}')

    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    class Args:
        task_name = 'long_term_forecast'
        is_training = 1
        model = model_name
        model_id = '4G'
        data = 'custom'
        root_path = DATA_DIR
        data_path = 'df_4g_base_100.parquet'
        features = 'M'
        target = '总流量'
        freq = 'h'
        checkpoints = './checkpoints/'
        seq_len = 24
        label_len = 12
        pred_len = 24
        enc_in = 8
        dec_in = 8
        c_out = 8
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        moving_avg = 25
        factor = 1
        distil = True
        dropout = 0.1
        embed = 'timeF'
        activation = 'gelu'
        num_workers = 0
        itr = 1
        train_epochs = 10
        batch_size = 32
        patience = 3
        learning_rate = 0.0001
        des = f'{model_name}_v2_Verify'
        loss = 'MSE'
        lradj = 'type1'
        use_amp = False
        use_gpu = (device.type == 'cuda')
        gpu = 0
        gpu_type = 'cuda'
        use_multi_gpu = False
        devices = '0,1,2,3'
        output_attention = False
        p_hidden_dims = [128, 128]
        p_hidden_layers = 2
        use_dtw = False
        augmentation_ratio = 0
        seed = 2
        jitter = False
        scaling = False
        permutation = False
        randompermutation = False
        magwarp = False
        timewarp = False
        windowslice = False
        windowwarp = False
        rotation = False
        spawner = False
        dtwwarp = False
        shapedtwwarp = False
        wdba = False
        discdtw = False
        discsdtw = False
        extra_tag = ''
        patch_len = 16
        node_dim = 10
        gcn_depth = 2
        gcn_dropout = 0.3
        propalpha = 0.3
        conv_channel = 32
        skip_channel = 32
        individual = False
        alpha = 0.1
        top_p = 0.5
        pos = 1
        channel_independence = 1
        decomp_method = 'moving_avg'
        use_norm = 1
        down_sampling_layers = 0
        down_sampling_window = 1
        down_sampling_method = None
        seg_len = 96
        expand = 2
        d_conv = 4
        tv_dt = 0
        tv_B = 0
        tv_C = 0
        use_D = 0
        top_k = 5
        num_kernels = 6
        inverse = False
        mask_rate = 0.25
        anomaly_ratio = 0.25
        seasonal_patterns = 'Monthly'

    args = Args()
    args.device = device

    if not torch.cuda.is_available():
        args.use_gpu = False
        print('> GPU不可用，使用CPU训练')

    # 模型特定轻量化参数
    model_configs = {
        'PatchTST':    dict(d_model=128, d_ff=256, n_heads=4),
        'DLinear':     dict(d_model=128, individual=False),
        'TiDE':        dict(d_model=256, d_ff=512),
        'SegRNN':      dict(seg_len=6, d_model=256),
        'LightTS':     dict(d_model=128),
        'TSMixer':     dict(d_model=128),
        'SCINet':      dict(d_model=128),
        'Informer':    dict(d_model=128, d_ff=256),
        'Autoformer':  dict(d_model=256, d_ff=512),
        'Transformer': dict(d_model=256, d_ff=512),
        'iTransformer': dict(d_model=256, d_ff=512),
    }
    if model_name in model_configs:
        for k, v in model_configs[model_name].items():
            setattr(args, k, v)

    setting = '{}_{}_{}'.format(args.model, args.model_id, args.des)
    print(f'\n{"="*60}')
    print(f'Experiment: {setting}')
    print(f'Model: {model_name} | d_model={args.d_model} | d_ff={args.d_ff}')
    print(f'{"="*60}')

    exp = Exp_Long_Term_Forecast(args)

    if args.is_training:
        for ii in range(args.itr):
            print(f'\n>>>>> Training {setting} >>>>>')
            exp.train(setting)
            print(f'\n>>>>> Testing {setting} >>>>>')
            exp.test(setting)
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f'\n[OK] {model_name} 训练+测试完成！')


# ═══════════════════════════════════════════════════════════
# IBM TTM
# ═══════════════════════════════════════════════════════════

def run_ttm():
    print('\n' + '=' * 60)
    print('  IBM TinyTimeMixer (TTM) Zero-Shot')
    print('=' * 60)

    try:
        from tsfm_public import TinyTimeMixerForPrediction
    except ImportError:
        print('[INFO] tsfm_public 通过 granite-tsfm 已可用')
        from tsfm_public import TinyTimeMixerForPrediction

    from utils.metrics import calc_custom_acc

    train_fp = os.path.join(DATA_DIR, 'df_4g_train_100.parquet')
    test_fp = os.path.join(DATA_DIR, 'df_4g_test_100.parquet')
    df_train = pd.read_parquet(train_fp)
    df_test = pd.read_parquet(test_fp)
    feature_cols = [c for c in df_train.columns if c not in ['ID编号', '厂商', '频段', '场景', 'date']]
    num_channels = len(feature_cols)

    scaler = StandardScaler().fit(df_train[feature_cols].values)
    test_scaled = scaler.transform(df_test[feature_cols].values)

    seq_len, pred_len, step = 24, 24, 3
    indices = list(range(0, len(test_scaled) - seq_len - pred_len + 1, step))
    ttm_context_len = 512

    device = get_device()
    print(f'> 设备: {device} | 窗口数: {len(indices)}')

    model_id = 'ibm-granite/granite-timeseries-ttm-r1'
    print(f'> 加载模型: {model_id}...')
    model = TinyTimeMixerForPrediction.from_pretrained(model_id, revision='main')
    model.to(device)
    model.eval()

    trues_list, preds_list = [], []
    with torch.no_grad():
        for idx, i in enumerate(indices):
            if (idx + 1) % 200 == 0:
                print(f'  进度: {idx+1}/{len(indices)}')
            X = test_scaled[i : i + seq_len]
            Y = test_scaled[i + seq_len : i + seq_len + pred_len]
            trues_list.append(Y)
            pad_len = ttm_context_len - seq_len
            X_padded = np.pad(X, ((pad_len, 0), (0, 0)), mode='constant', constant_values=0)
            past_values = torch.tensor(X_padded, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(past_values=past_values)
            if hasattr(outputs, 'prediction_outputs'):
                p = outputs.prediction_outputs.squeeze(0).cpu().numpy()
            elif hasattr(outputs, 'logits'):
                p = outputs.logits.squeeze(0).cpu().numpy()
            else:
                p = outputs[0].squeeze(0).cpu().numpy()
            preds_list.append(p[:pred_len, :])

    preds_arr = np.array(preds_list).reshape(-1, num_channels)
    trues_arr = np.array(trues_list).reshape(-1, num_channels)
    mse = np.mean((trues_arr - preds_arr) ** 2)
    mae = np.mean(np.abs(trues_arr - preds_arr))
    rmse = np.sqrt(mse)

    preds_inv = scaler.inverse_transform(preds_arr)
    trues_inv = scaler.inverse_transform(trues_arr)
    preds_inv_3d = preds_inv.reshape(-1, pred_len, num_channels)
    trues_inv_3d = trues_inv.reshape(-1, pred_len, num_channels)
    avg_acc = calc_custom_acc(preds_inv_3d, trues_inv_3d)

    mask = trues_inv > 1e-5
    mape = np.mean(np.abs((trues_inv[mask] - preds_inv[mask]) / trues_inv[mask])) if np.sum(mask) > 0 else 0.0
    mspe = np.mean(np.square((trues_inv[mask] - preds_inv[mask]) / trues_inv[mask])) if np.sum(mask) > 0 else 0.0

    setting = f'IBM_TTM_ZeroShot_sl{seq_len}_pl{pred_len}_step{step}'
    result_str = f'MSE:{mse:.4f}, MAE:{mae:.4f}, RMSE:{rmse:.4f}, MAPE:{mape:.4f}, MSPE:{mspe:.4f}, Custom_ACC:{avg_acc:.4f}'
    print(f'\n  {setting}')
    print(f'  {result_str}')
    with open(RESULT_FILE, 'a') as f:
        f.write(f'{setting}\n{result_str}\n\n')

    save_dir = f'results/{setting}'
    os.makedirs(save_dir, exist_ok=True)
    # Save SCALED space (matching DL model convention for generate_web_data.py)
    np.save(f'{save_dir}/pred.npy', preds_arr.reshape(-1, pred_len, num_channels))
    np.save(f'{save_dir}/true.npy', trues_arr.reshape(-1, pred_len, num_channels))
    print(f'  [OK] TTM 完成！')


# ═══════════════════════════════════════════════════════════
# Chronos
# ═══════════════════════════════════════════════════════════

def run_chronos():
    print('\n' + '=' * 60)
    print('  Amazon Chronos Zero-Shot (tiny, 8M params)')
    print('=' * 60)
    device = get_device()
    print(f'> 设备: {device}')
    script = os.path.join(MODELS_DIR, 'model_chronos.py')
    dev_str = 'cuda' if device.type == 'cuda' else 'cpu'
    cmd = f'"{PYTHON_EXE}" "{script}" --model_name amazon/chronos-t5-tiny --data_path "{DATA_DIR}" --output_dir "./results" --device {dev_str}'
    print(f'  CMD: {cmd}')
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print('[FAIL] Chronos 运行失败')
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════
# Mamba
# ═══════════════════════════════════════════════════════════

def run_mamba():
    print('\n' + '=' * 60)
    print('  Mamba (Pure Python SSM, 轻量配置)')
    print('=' * 60)
    script = os.path.join(MODELS_DIR, 'model_mamba.py')
    cmd = f'"{PYTHON_EXE}" "{script}" --data_path "{DATA_DIR}" --output_dir "./results" --d_model 128 --d_ff 32 --epochs 10'
    print(f'  CMD: {cmd}')
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print('[FAIL] Mamba 运行失败')
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════
# TimeLLM
# ═══════════════════════════════════════════════════════════

def run_timellm():
    print('\n' + '=' * 60)
    print('  TimeLLM (GPT-2 frozen backbone + reprogramming)')
    print('=' * 60)
    script = os.path.join(MODELS_DIR, 'model_timellm.py')
    cmd = f'"{PYTHON_EXE}" "{script}" --data_path "{DATA_DIR}" --output_dir "./results" --llm_name gpt2 --epochs 5 --batch_size 4'
    print(f'  CMD: {cmd}')
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print('[FAIL] TimeLLM 运行失败')
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════
# External Base Model
# ═══════════════════════════════════════════════════════════

def run_external_base():
    print('\n' + '=' * 60)
    print('  External Base Model Evaluation')
    print('=' * 60)
    cmd = f'"{PYTHON_EXE}" base.py --root_path "{DATA_DIR}"'
    print(f'  CMD: {cmd}')
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════
# 主菜单
# ═══════════════════════════════════════════════════════════

def run_model(choice):
    """分发到对应的运行函数"""
    if choice == '2':
        print('\n' + '=' * 60)
        print('  传统统计模型 (全部7个) 一键运行')
        print('=' * 60)
        run_simple_baselines()
        run_traditional_single('AutoARIMA')
        run_traditional_single('AutoAR')
        run_traditional_single('LinearRegression')
        run_xgboost()
        print('\n[OK] 传统统计模型全部完成！')
        return

    if choice not in MODELS:
        print(f'无效选项: {choice}')
        return

    model = MODELS[choice]
    group = model['group']
    name = model['name']

    print(f'\n{"="*60}')
    print(f'  运行: {name}')
    print(f'{"="*60}')

    try:
        if group == 'simple_baseline':
            mapping = {'3': ['Naive'], '4': ['Persistent_24h'], '5': ['Historical_Average']}
            model_names = mapping.get(choice)
            if model_names:
                run_simple_baselines(model_names)

        elif group == 'traditional_single':
            mapping = {'6': 'AutoARIMA', '7': 'AutoAR', '8': 'LinearRegression'}
            m = mapping.get(choice)
            if m:
                run_traditional_single(m)

        elif group == 'xgboost':
            run_xgboost()

        elif group == 'external_base':
            run_external_base()

        elif group == 'dl':
            model_name_map = {
                '11': 'PatchTST', '12': 'iTransformer', '13': 'Informer',
                '14': 'Autoformer', '15': 'Transformer', '16': 'DLinear',
                '17': 'LightTS', '18': 'TSMixer', '20': 'SCINet',
                '21': 'TimesNet', '22': 'SegRNN',
            }
            model_name = model_name_map.get(choice)
            if model_name:
                run_dl_model(model_name)

        elif group == 'ttm':
            run_ttm()

        elif group == 'chronos':
            run_chronos()

        elif group == 'mamba':
            run_mamba()

        elif group == 'timellm':
            run_timellm()

        print(f'\n[OK] {name} 完成！')

    except Exception as e:
        print(f'\n[FAIL] {name} 运行出错: {e}')
        import traceback
        traceback.print_exc()


def main():
    while True:
        print_menu()
        try:
            choice = input('请输入选项 (0-25): ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n退出。')
            break

        if choice == '0':
            print('退出。')
            break

        if choice == '1':
            print('\n' + '★' * 35)
            print('  运行所有模型 (选项 2→25) — 将花费很长时间!')
            print('★' * 35)
            confirm = input('确认运行全部? (输入 yes 确认): ').strip().lower()
            if confirm != 'yes':
                print('已取消。')
                continue
            # 按顺序运行: 先传统统计(2), 再逐个运行 3-25
            run_model('2')
            for i in range(3, 26):
                run_model(str(i))
            print('\n' + '★' * 35)
            print('  所有23个模型运行完毕！')
            print('★' * 35)
            # Fall through to the simple continue/exit prompt
        else:
            run_model(choice)

        # After running, show simple continue/exit instead of full menu
        print('\n' + '-' * 40)
        next_choice = input('1. 继续运行下一个模型  2. 退出\n请选择 (1/2): ').strip()
        if next_choice == '2':
            print('退出。')
            break
        # If '1' or anything else, loop back to full menu


if __name__ == '__main__':
    # 启动时检查环境
    print(f'Python: {sys.executable}')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {"Yes" if torch.cuda.is_available() else "No"}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)')
    main()
