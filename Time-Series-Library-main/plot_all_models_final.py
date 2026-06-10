"""
Generate 5 comparison plots for selected models.
Usage: python plot_all_models_final.py
       交互式选择要对比的模型，生成5张对比图到 ../plots/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

results_dir = './results'
output_dir = '../plots'

feature_names_cn = [
    'ERAB流量', 'PDCCH利用率', 'PDSCH利用率', 'PUSCH利用率',
    '上行流量', '下行流量', '有效连接数', '总流量'
]

# ── Build StandardScaler matching data_loader test mode ──
train_fp = './data_provider/4g_traffic/df_4g_train_100.parquet'
df_train = pd.read_parquet(train_fp)
drop_cols = ['ID编号', '厂商', '频段', '场景']
for col in drop_cols:
    if col in df_train.columns:
        df_train = df_train.drop(columns=[col])
target = '总流量'
cols = list(df_train.columns)
cols.remove(target)
cols.remove('date')
df_train = df_train[['date'] + cols + [target]]
train_data = df_train[df_train.columns[1:]].values
scaler = StandardScaler().fit(train_data)

# ── Model discovery ──
model_dirs = {}
for folder in sorted(os.listdir(results_dir)):
    fp = os.path.join(results_dir, folder)
    # Check both naming conventions (pred.npy and preds.npy)
    for pred_name, true_name in [('pred.npy', 'true.npy'), ('preds.npy', 'trues.npy')]:
        pred_p = os.path.join(fp, pred_name)
        true_p = os.path.join(fp, true_name)
        if os.path.isdir(fp) and os.path.exists(pred_p) and os.path.exists(true_p):
            break
    else:
        continue
    name = folder
    # Clean up model name: remove _4G suffix and variant tags
    for sep in ['_4G_', '_ZeroShot_', '_4G']:
        if sep in name:
            name = name.split(sep)[0]
            break
    # Map known names
    if name == 'External_BaseModel': name = 'BaseModel'
    elif name == 'HistoricalAverage': name = 'HA'
    elif name == 'Persistent24h': name = 'Persistent_24h'
    model_dirs[name] = folder
    # Also store reverse mapping for folder lookup
    model_dirs[name] = folder

# ── Interactive model selection ──
available = sorted(model_dirs.keys())
print('\n' + '=' * 60)
print('  4G Traffic Prediction — 模型对比绘图')
print('=' * 60)
print(f'  发现 {len(available)} 个有预测结果的模型:\n')
for i, name in enumerate(available, 1):
    print(f'    {i:>2}. {name}')
print(f'\n    0. 全部选择')
print(f'   -1. 退出')
print('=' * 60)

choice = input('请选择模型编号 (逗号分隔, 如 1,3,5, 或 -1 退出): ').strip()

if choice == '-1':
    print('退出。')
    sys.exit(0)

if choice == '0' or choice == '':
    selected_names = available[:min(8, len(available))]
    print(f'\n  已选择(默认): {", ".join(selected_names)}')
else:
    try:
        indices = [int(x.strip()) - 1 for x in choice.split(',')]
        selected_names = [available[i] for i in indices if 0 <= i < len(available)]
        print(f'\n  已选择: {", ".join(selected_names)}')
    except:
        print('  输入无效，使用全部模型')
        selected_names = available

if len(selected_names) < 2:
    print('  至少需要2个模型才能对比！已选择全部。')
    selected_names = available[:min(8, len(available))]

model_names = selected_names
n_models = len(model_names)
print(f'  共 {n_models} 个模型参与对比绘图...')

# ── Load reference data (use first available model as reference) ──
_first_folder = list(model_dirs.values())[0]
_first_pred = os.path.join(results_dir, _first_folder, 'pred.npy')
_first_true = os.path.join(results_dir, _first_folder, 'true.npy')
if not os.path.exists(_first_pred):
    _first_pred = os.path.join(results_dir, _first_folder, 'preds.npy')
    _first_true = os.path.join(results_dir, _first_folder, 'trues.npy')
true_orig = np.load(_first_true)
# Find a DL (scaled) model or use the same reference
dl_true_scaled = true_orig.copy()
for _name in model_dirs:
    _folder = model_dirs[_name]
    _t = np.load(os.path.join(results_dir, _folder, 'true.npy' if os.path.exists(os.path.join(results_dir, _folder, 'true.npy')) else 'trues.npy'))
    if not np.array_equal(_t, true_orig):
        dl_true_scaled = _t
        break

# ── Determine original vs scaled space ──
models_orig = {}
models_scaled = {}
for name in model_names:
    folder = model_dirs[name]
    true = np.load(os.path.join(results_dir, folder, 'true.npy'))
    pred = np.load(os.path.join(results_dir, folder, 'pred.npy'))
    if np.array_equal(true, true_orig):
        models_orig[name] = {'pred': pred, 'true': true}
    else:
        models_scaled[name] = {'pred': pred, 'true': true}

# ── Convert all to unified scaled space ──
all_preds_scaled = {}
all_preds_orig = {}

for name, data in {**models_orig, **models_scaled}.items():
    pred = data['pred']
    if name in models_scaled:
        all_preds_scaled[name] = pred
        pred_inv = scaler.inverse_transform(pred.reshape(-1, 8)).reshape(pred.shape)
        pred_inv = np.clip(pred_inv, 0, None)
        pred_inv_display = pred_inv.copy()
        pred_inv_display[:,:,6] = pred_inv[:,:,7]
        pred_inv_display[:,:,7] = pred_inv[:,:,6]
        all_preds_orig[name] = pred_inv_display
    else:
        all_preds_orig[name] = pred
        pred_swapped = pred.copy()
        pred_swapped[:,:,6] = pred[:,:,7]
        pred_swapped[:,:,7] = pred[:,:,6]
        pred_sc = scaler.transform(pred_swapped.reshape(-1, 8)).reshape(pred.shape)
        all_preds_scaled[name] = pred_sc

# ── Compute metrics in scaled space ──
print(f"\n===== Model Metrics ({n_models} models) =====")
metrics = {}
for name in model_names:
    ps = all_preds_scaled[name]
    ts = dl_true_scaled
    flat_p = ps.reshape(-1)
    flat_t = ts.reshape(-1)
    mse = np.mean((flat_p - flat_t) ** 2)
    mae = np.mean(np.abs(flat_p - flat_t))
    rmse = np.sqrt(mse)
    per_ch_rmse = [np.sqrt(np.mean((ps[:,:,ch] - ts[:,:,ch])**2)) for ch in range(8)]
    bias = np.mean(flat_p - flat_t)
    metrics[name] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'Bias': bias, 'PerChRMSE': per_ch_rmse}
    print(f"  {name:<20}  MSE={mse:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  Bias={bias:+.4f}")

# ── Colors ──
tab10 = list(plt.cm.tab10.colors)
tab20 = list(plt.cm.tab20.colors)
all_colors_list = tab10 + tab20 + list(plt.cm.Set3.colors)
colors = [all_colors_list[i % len(all_colors_list)] for i in range(n_models)]
linestyles = ['-', '--', '-.', ':'] * (n_models // 4 + 1)
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*', 'X', 'p', 'h', '+', 'x', 'd', 'H']

os.makedirs(output_dir, exist_ok=True)

# ════════════════════════════════════════════
# PLOT 1: 8-channel (Window 0)
# ════════════════════════════════════════════
print("\n===== Plot 1/5: 8-channel comparison =====")
fig, axes = plt.subplots(2, 4, figsize=(28, 14))
axes = axes.flatten()
for ch_idx in range(8):
    ax = axes[ch_idx]
    hours = np.arange(24)
    ax.plot(hours, true_orig[0,:,ch_idx], 'k-', linewidth=3, label='True', alpha=0.95)
    for i, name in enumerate(model_names):
        ax.plot(hours, all_preds_orig[name][0,:,ch_idx], color=colors[i],
                linestyle=linestyles[i], linewidth=1.0, label=name, alpha=0.7)
    ax.set_xlabel('Hours Ahead', fontsize=9)
    ax.set_ylabel(feature_names_cn[ch_idx], fontsize=9)
    ax.set_title(feature_names_cn[ch_idx], fontsize=11, fontweight='bold')
    ax.legend(fontsize=6.5, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
fig.suptitle(f'4G Traffic: {n_models} Models vs Ground Truth (Window #0, 8 Features)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(output_dir, '01_all_models_per_channel.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved: 01_all_models_per_channel.png")

# ════════════════════════════════════════════
# PLOT 2: Multi-window Total Traffic
# ════════════════════════════════════════════
print("\n===== Plot 2/5: Multi-window Total Traffic =====")
ch_idx = 7
sample_windows = [0, 100, 200, 500, 1000, 2000]
fig, axes = plt.subplots(2, 3, figsize=(24, 12))
axes = axes.flatten()
for si, win_idx in enumerate(sample_windows):
    ax = axes[si]
    hours = np.arange(24)
    ax.plot(hours, true_orig[win_idx,:,ch_idx], 'k-', linewidth=3, label='True', alpha=0.95, marker='o', markersize=3)
    for i, name in enumerate(model_names):
        ax.plot(hours, all_preds_orig[name][win_idx,:,ch_idx], color=colors[i],
                linestyle=linestyles[i], linewidth=1.0, label=name, alpha=0.65, marker=markers[i], markersize=2)
    ax.set_xlabel('Hours Ahead', fontsize=10)
    ax.set_ylabel('Total Traffic', fontsize=10)
    ax.set_title(f'Window #{win_idx}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
plt.tight_layout(pad=2.0)
fig.suptitle(f'Total Traffic: {n_models} Models vs Ground Truth (6 Windows)',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(output_dir, '02_all_models_multi_window.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved: 02_all_models_multi_window.png")

# ════════════════════════════════════════════
# PLOT 3: Hourly MAE
# ════════════════════════════════════════════
print("\n===== Plot 3/5: Hourly MAE =====")
fig, axes = plt.subplots(2, 4, figsize=(24, 13))
axes = axes.flatten()
for ch_idx in range(8):
    ax = axes[ch_idx]
    hours = np.arange(1, 25)
    for i, name in enumerate(model_names):
        err = np.mean(np.abs(all_preds_scaled[name] - dl_true_scaled), axis=0)
        ax.plot(hours, err[:,ch_idx], color=colors[i], linestyle=linestyles[i],
                linewidth=1.5, label=name, marker=markers[i], markersize=3)
    ax.set_xlabel('Prediction Horizon (hours)', fontsize=9)
    ax.set_ylabel('MAE (std units)', fontsize=9)
    ax.set_title(feature_names_cn[ch_idx], fontsize=10, fontweight='bold')
    ax.legend(fontsize=6.5, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
plt.tight_layout(pad=2.0)
fig.suptitle(f'Hourly MAE: Error Growth with Prediction Horizon ({n_models} Models)',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(output_dir, '03_all_models_hourly_mae.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved: 03_all_models_hourly_mae.png")

# ════════════════════════════════════════════
# PLOT 4: Error Distribution (Total Traffic)
# ════════════════════════════════════════════
print("\n===== Plot 4/5: Error Distribution =====")
n_cols = 4
n_rows = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols+2, 3.5*n_rows+1))
axes = np.atleast_1d(axes).flatten()
ch_total = 7
for i, name in enumerate(model_names):
    ax = axes[i]
    errors = (all_preds_scaled[name][:,:,ch_total] - dl_true_scaled[:,:,ch_total]).reshape(-1)
    clip = np.percentile(np.abs(errors), 99)
    errors_c = errors[np.abs(errors) <= clip]
    ax.hist(errors_c, bins=100, density=True, alpha=0.7, color=colors[i], edgecolor='white')
    ax.axvline(0, color='k', linestyle='--', linewidth=1.2)
    m = metrics[name]
    ax.axvline(m['Bias'], color='red', linestyle='-', linewidth=2, label=f"Bias={m['Bias']:.3f}")
    ax.set_xlabel('Error (std)', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.set_title(f"{name}\nRMSE={m['RMSE']:.3f} | Bias={m['Bias']:.3f}", fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
fig.suptitle(f'Total Traffic Error Distributions ({n_models} Models)',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(os.path.join(output_dir, '04_all_models_error_dist.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved: 04_all_models_error_dist.png")

# ════════════════════════════════════════════
# PLOT 5: Performance Bar Chart
# ════════════════════════════════════════════
print("\n===== Plot 5/5: Performance Bar Chart =====")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 5a: MSE
ax = axes[0, 0]
mse_vals = [metrics[n]['MSE'] for n in model_names]
si = np.argsort(mse_vals)
sn = [model_names[i] for i in si]; sv = [mse_vals[i] for i in si]; sc = [colors[i] for i in si]
bars = ax.bar(range(len(sn)), sv, color=sc, edgecolor='black', alpha=0.85)
ax.set_xticks(range(len(sn))); ax.set_xticklabels(sn, fontsize=8, rotation=45, ha='right')
ax.set_title('MSE (Scaled, Lower=Better)', fontsize=12, fontweight='bold')
ax.set_ylabel('MSE'); ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, sv):
    ax.text(bar.get_x()+bar.get_width()/2, v+max(sv)*0.02, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

# 5b: RMSE
ax = axes[0, 1]
rv = [metrics[n]['RMSE'] for n in model_names]
si2 = np.argsort(rv)
sn2 = [model_names[i] for i in si2]; rv2 = [rv[i] for i in si2]; sc2 = [colors[i] for i in si2]
bars = ax.bar(range(len(sn2)), rv2, color=sc2, edgecolor='black', alpha=0.85)
ax.set_xticks(range(len(sn2))); ax.set_xticklabels(sn2, fontsize=8, rotation=45, ha='right')
ax.set_title('RMSE Ranking', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE (std)'); ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, rv2):
    ax.text(bar.get_x()+bar.get_width()/2, v+max(rv2)*0.02, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

# 5c: Per-channel RMSE
ax = axes[1, 0]
x = np.arange(8)
width = max(0.55 / n_models, 0.04)
for i, name in enumerate(model_names):
    offset = (i - n_models/2 + 0.5) * width
    vals = np.clip(metrics[name]['PerChRMSE'], 0, 5)
    ax.bar(x + offset, vals, width, color=colors[i], edgecolor='black', alpha=0.85, label=name, linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(feature_names_cn, fontsize=8, rotation=30, ha='right')
ax.set_title('Per-Channel RMSE', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE (std)'); ax.legend(fontsize=6, loc='upper left', ncol=2); ax.grid(True, alpha=0.3, axis='y')

# 5d: Bias
ax = axes[1, 1]
bv = [metrics[n]['Bias'] for n in model_names]
si3 = np.argsort(bv)
sn3 = [model_names[i] for i in si3]; bv3 = [bv[i] for i in si3]; sc3 = [colors[i] for i in si3]
ax.barh(range(len(sn3)), bv3, color=sc3, edgecolor='black', alpha=0.85)
ax.set_yticks(range(len(sn3))); ax.set_yticklabels(sn3, fontsize=8)
ax.set_title('Model Bias (Negative=Under-predict)', fontsize=12, fontweight='bold')
ax.set_xlabel('Bias (std)'); ax.axvline(0, color='k'); ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout(pad=2.0)
fig.suptitle(f'4G Traffic Prediction: Model Performance ({n_models} Models)',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(os.path.join(output_dir, '05_all_models_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved: 05_all_models_performance.png")

# ════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════
print("\n" + "=" * 85)
print(f"  ALL 5 PLOTS COMPLETE — {n_models} models -> {output_dir}/")
print("=" * 85)
print(f"\n  Rank  Model                 RMSE     MSE      MAE      Bias")
print("  " + "-" * 65)
sorted_by_rmse = sorted(model_names, key=lambda n: metrics[n]['RMSE'])
for rank, name in enumerate(sorted_by_rmse, 1):
    m = metrics[name]
    print(f"  {rank:2d}.   {name:<20s}  {m['RMSE']:.4f}   {m['MSE']:.4f}   {m['MAE']:.4f}   {m['Bias']:+.4f}")

print(f"\n  Best:  {sorted_by_rmse[0]}")
print(f"  Worst: {sorted_by_rmse[-1]}")
print("=" * 85)
