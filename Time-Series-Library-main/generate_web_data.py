"""
Generate JS data files for the React visualization website.
Uses WINDOW-AVERAGED prediction curves for fair model comparison.
Space detection: compare true.npy to reference (as in plot_all_models_final.py).

Usage:
  export PYTHONPATH=".;./models"
  E:/Software/Anaconda3/envs/ETP-exp1/python.exe generate_web_data.py
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

results_dir = './results'
output_dir = '../react-app/src/data'

# ── StandardScaler ──
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
model_info = {}  # folder -> cleaned name
for folder in sorted(os.listdir(results_dir)):
    fp = os.path.join(results_dir, folder)
    if not os.path.isdir(fp):
        continue
    for pred_name, true_name in [('pred.npy', 'true.npy'), ('preds.npy', 'trues.npy')]:
        if os.path.exists(os.path.join(fp, pred_name)) and os.path.exists(os.path.join(fp, true_name)):
            break
    else:
        continue
    name = folder
    for sep in ['_4G_', '_ZeroShot_', '_4G']:
        if sep in name:
            name = name.split(sep)[0]
            break
    if name == 'External_BaseModel': name = '★ BaseModel'
    elif name == 'HistoricalAverage': name = 'Historical Avg'
    elif name == 'Persistent24h': name = 'Persistent 24h'
    elif name == 'IBM_TTM': name = 'IBM TTM'
    elif name.startswith('Mamba_'): name = 'Mamba'
    elif name.startswith('TimeLLM_'): name = 'TimeLLM'
    elif name.startswith('XGBoost_'): name = 'XGBoost'
    elif name.startswith('Informer_'): name = 'Informer'
    elif name.startswith('LightTS_'): name = 'LightTS'
    elif name.startswith('TSMixer_'): name = 'TSMixer'
    elif name.startswith('SCINet_'): name = 'SCINet'
    elif name == 'AutoAR_4G': name = 'AutoAR'
    elif name == 'LinearRegression_4G': name = 'LinearRegression'
    model_info[folder] = name

print(f"Found {len(model_info)} models with prediction data")

# ── Load all pred/true ──
raw = {}  # name -> {pred, true, folder}
for folder, name in model_info.items():
    for pred_name, true_name in [('pred.npy', 'true.npy'), ('preds.npy', 'trues.npy')]:
        pp = os.path.join(results_dir, folder, pred_name)
        tp = os.path.join(results_dir, folder, true_name)
        if os.path.exists(pp) and os.path.exists(tp):
            break
    raw[name] = {
        'pred': np.load(pp),
        'true': np.load(tp),
        'folder': folder,
    }

# ── Space detection via true.npy comparison ──
# First group models by true.npy equality, then pick the largest group as DL reference
def _true_equal(a, b):
    return np.allclose(a, b, rtol=1e-5, atol=1e-4)

# Cluster models by true.npy
true_groups = {}  # group_key -> [model_names]
group_true = {}   # group_key -> representative true array
for name, data in raw.items():
    t = data['true']
    matched = False
    for gkey in true_groups:
        if _true_equal(t, group_true[gkey]):
            true_groups[gkey].append(name)
            matched = True
            break
    if not matched:
        gkey = f'group_{len(true_groups)}'
        true_groups[gkey] = [name]
        group_true[gkey] = t

# Find the largest group → this is the DL reference (most models share this pipeline)
largest_group = max(true_groups, key=lambda g: len(true_groups[g]))
dl_ref_true = group_true[largest_group]
dl_ref_models = true_groups[largest_group]

print(f"\nDL reference group ({len(dl_ref_models)} models): {dl_ref_models}")
for gkey, models in true_groups.items():
    if gkey != largest_group:
        print(f"  Other group ({len(models)} models): {models}")

# Classify each model: does it share the DL reference true?
models_dl_ref = {n: raw[n] for n in dl_ref_models}
models_other_true = {n: raw[n] for n in raw if n not in dl_ref_models}

# ── Convert all preds to scaled space (using the website's StandardScaler) ──
pred_scaled = {}  # name -> array in scaled space

for name, data in raw.items():
    p = data['pred']
    t = data['true']
    pmin, pmax = p.min(), p.max()

    if name in models_dl_ref:
        # DL reference models are already in scaled space → use as-is
        pred_scaled[name] = p.copy()
    else:
        # Non-reference models — detect space
        # Original space models have pred values in raw traffic units (typically 0–100000+)
        # Scaled space models have pred values roughly in [-50, 500]
        if pmin >= -500 and pmax <= 500:
            # Scaled space but different scaler → trust as-is
            pred_scaled[name] = p.copy()
            print(f"  {name}: scaled space (range [{pmin:.1f}, {pmax:.1f}])")
        else:
            # Original space → swap channels then scale
            ps = p.copy()
            ps[:,:,6] = p[:,:,7]
            ps[:,:,7] = p[:,:,6]
            pred_scaled[name] = scaler.transform(ps.reshape(-1, 8)).reshape(p.shape)
            print(f"  {name}: original space → scaled (range [{pmin:.1f}, {pmax:.1f}])")

# Use DL reference true as the common ground truth
true_scaled = dl_ref_true  # shape (n_windows, 24, 8)

# ── Compute overall metrics in scaled space ──
print(f"\n===== Scaled-space metrics (all windows, all channels) =====")
print(f"{'Model':<22s} {'MSE':>8s} {'MAE':>8s} {'RMSE':>8s} {'Bias':>8s}")
print("-" * 58)
metrics = {}
for name in sorted(pred_scaled.keys()):
    p = pred_scaled[name]
    t = true_scaled
    # Clip to avoid extreme outlier effects
    err = (p - t).reshape(-1)
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(err))
    metrics[name] = {'mse': mse, 'mae': mae, 'rmse': rmse, 'bias': bias}
    print(f"{name:<22s} {mse:8.4f} {mae:8.4f} {rmse:8.4f} {bias:+8.4f}")

# ── Select models for the website ──
# All models in this list share the same StandardScaler + Dataset_Custom data pipeline
# (The space inconsistency issues for Mamba/TimeLLM/XGBoost were fixed in shared_utils.py)
# IBM TTM, AutoAR, LinearRegression use a different scaler (run.py functions) —
#   their pred.npy is in scaled space but with different column ordering/scaler.
#   They are included in metrics but excluded from prediction curves.
CORE_MODELS = [
    '★ BaseModel',
    'iTransformer',
    'PatchTST',
    'SegRNN',
    'DLinear',
    'TimesNet',
    'Autoformer',
    'Transformer',
    'Informer',
    'LightTS',
    'TSMixer',
    'SCINet',
    'XGBoost',
    'Mamba',
    'TimeLLM',
]

# Verify all core models have valid scaled predictions
VALID = []
for name in CORE_MODELS:
    if name in pred_scaled:
        # Check that scaled range is reasonable (wider range to handle all valid scaled models)
        p = pred_scaled[name]
        if p.min() > -500 and p.max() < 500:
            VALID.append(name)
        else:
            print(f"  EXCLUDING {name}: scaled range [{p.min():.1f}, {p.max():.1f}] unreasonable")
    else:
        print(f"  EXCLUDING {name}: no scaled prediction available")

print(f"\nModels for web: {len(VALID)}")
for n in VALID:
    m = metrics.get(n, {})
    print(f"  {n:<22s}  MAE={m.get('mae', 0):.4f}σ  RMSE={m.get('rmse', 0):.4f}σ")

# ── Pre-compute own true in scaled space for each model ──
# DL reference models share the same true; others have different true
own_true_scaled = {}
for name in VALID:
    if name in models_dl_ref:
        own_true_scaled[name] = true_scaled  # DL reference true
    else:
        # Non-DL-ref model — convert its own true to scaled space
        orig_data = raw.get(name)
        if orig_data is not None:
            t = orig_data['true'].copy()
            tmin, tmax = t.min(), t.max()
            if tmin >= -500 and tmax <= 500:
                # Already in scaled space (different scaler)
                own_true_scaled[name] = t
            else:
                # Original space — swap channels then scale
                t_swapped = t.copy()
                t_swapped[:,:,6] = t[:,:,7]
                t_swapped[:,:,7] = t[:,:,6]
                own_true_scaled[name] = scaler.transform(t_swapped.reshape(-1,8)).reshape(t.shape)
        else:
            own_true_scaled[name] = true_scaled

# ════════════════════════════════════════════════════════════════
# Find the most representative window
# ════════════════════════════════════════════════════════════════
CH_TOTAL = 7  # Index 7 = 总流量 in scaled space
n_windows = true_scaled.shape[0]

# Compute per-window MAE for each model
overall_mae_ch = {}
window_mae = {}
for name in VALID:
    err = np.abs(pred_scaled[name][:, :, CH_TOTAL] - true_scaled[:, :, CH_TOTAL])
    overall_mae_ch[name] = float(err.mean())
    for w in range(n_windows):
        if w not in window_mae:
            window_mae[w] = {}
        window_mae[w][name] = float(err[w].mean())

# Find best window: balance rank correlation + variance in DL true + variance in BM true
dl_true_var = np.var(true_scaled[:, :, CH_TOTAL], axis=1)
bm_own_true = own_true_scaled.get('★ BaseModel', true_scaled)
bm_true_var = np.var(bm_own_true[:, :, CH_TOTAL], axis=1)

geom_var = np.sqrt(dl_true_var * bm_true_var + 1e-10)
geom_var_norm = (geom_var - geom_var.min()) / (geom_var.max() - geom_var.min() + 1e-10)

overall_rank = {n: i for i, (n, _) in enumerate(sorted(overall_mae_ch.items(), key=lambda x: x[1]))}
best_window, best_score = 0, -1
for w in range(n_windows):
    w_rank = {n: i for i, (n, _) in enumerate(sorted(window_mae[w].items(), key=lambda x: x[1]))}
    n = len(overall_rank)
    d2 = sum((overall_rank[m] - w_rank[m])**2 for m in overall_rank)
    corr = 1 - 6*d2/(n*(n*n-1))
    score = 0.2 * geom_var_norm[w] + 0.8 * (corr + 1) / 2
    if score > best_score:
        best_score = score
        best_window = w
        best_corr = corr

REPRESENTATIVE_WINDOW = int(best_window)
print(f"\nRepresentative window: {REPRESENTATIVE_WINDOW} (corr={best_corr:.4f}, dl_var={dl_true_var[best_window]:.2f}, bm_var={bm_true_var[best_window]:.2f}, score={best_score:.4f})")
print(f"Overall MAE ranking (ch=总流量):")
for name in sorted(overall_mae_ch, key=overall_mae_ch.get):
    print(f"  {name:<22s}: {overall_mae_ch[name]:.4f}")
print(f"Window {REPRESENTATIVE_WINDOW} MAE:")
for name in sorted(window_mae[REPRESENTATIVE_WINDOW], key=window_mae[REPRESENTATIVE_WINDOW].get):
    print(f"  {name:<22s}: {window_mae[REPRESENTATIVE_WINDOW][name]:.4f}")

# ════════════════════════════════════════════════════════════════
# Generate JS data files
# ════════════════════════════════════════════════════════════════
os.makedirs(output_dir, exist_ok=True)

print(f"\nTotal windows: {n_windows} | Representative window: {REPRESENTATIVE_WINDOW} | Channel 总流量 index: {CH_TOTAL}")

# ── 1. prediction_curves.js: representative window ──
# Each model uses its OWN true (converted to scaled space) for fair comparison
# DL models share the same true.npy, BaseModel has its own
print(f"\n===== Generating prediction_curves.js (Window #{REPRESENTATIVE_WINDOW}) =====")

curves_data = {
    "window": REPRESENTATIVE_WINDOW,
    "n_windows": int(n_windows),
    "description": f"Representative window #{REPRESENTATIVE_WINDOW} — corr={best_corr:.4f}",
    "models": {}
}

for name in VALID:
    ps = pred_scaled[name]  # (n_w, 24, 8)
    is_dl_ref = name in models_dl_ref
    if is_dl_ref:
        ts_for_chart = true_scaled  # DL reference truth
    else:
        ts_for_chart = own_true_scaled.get(name, true_scaled)  # model's own truth
    ts_own = own_true_scaled.get(name, true_scaled)

    model_data = {}
    for ch in range(8):
        # Swap channels 6,7 to match website CHANNELS order (总流量=6, 有效连接数=7)
        web_ch = ch if ch not in (6, 7) else (7 if ch == 6 else 6)
        model_data[str(web_ch)] = {
            "pred": ps[REPRESENTATIVE_WINDOW, :, ch].tolist(),
            "true": ts_for_chart[REPRESENTATIVE_WINDOW, :, ch].tolist(),
        }
    curves_data["models"][name] = model_data

    ch_mae_chart = float(np.abs(ps[REPRESENTATIVE_WINDOW] - ts_for_chart[REPRESENTATIVE_WINDOW]).mean())
    ch_mae_own = float(np.abs(ps[REPRESENTATIVE_WINDOW] - ts_own[REPRESENTATIVE_WINDOW]).mean())
    truth_label = "DL ref" if is_dl_ref else "own truth"
    print(f"  {name:<22s}  MAE ({truth_label})={ch_mae_chart:.4f}σ  MAE (own true)={ch_mae_own:.4f}σ")

js = f"export default {json.dumps(curves_data, ensure_ascii=False)};\n"
with open(os.path.join(output_dir, 'prediction_curves.js'), 'w', encoding='utf-8') as f:
    f.write(js)
print(f"  -> Saved prediction_curves.js ({len(js)} bytes)")

# ── 2. multi_window.js ──
# Use website channel order: total traffic = index 6
CH_TOTAL_WEB = 6
print("\n===== Generating multi_window.js =====")
sample_windows = [0, 100, 200, 500, 1000, 2000]
mw_data = {"windows": sample_windows, "channel": "总流量", "models": {}}
for name in VALID:
    ps = pred_scaled[name]
    ts = true_scaled
    md = {}
    for w in sample_windows:
        if w < n_windows:
            md[str(w)] = {
                "pred": ps[w, :, CH_TOTAL].tolist(),  # scaler channel 7 = 总流量
                "true": ts[w, :, CH_TOTAL].tolist(),
            }
    mw_data["models"][name] = md

js = f"export default {json.dumps(mw_data, ensure_ascii=False)};\n"
with open(os.path.join(output_dir, 'multi_window.js'), 'w', encoding='utf-8') as f:
    f.write(js)
print(f"  -> Saved multi_window.js ({len(js)} bytes)")

# ── 3. hourly_mae.js ──
print("\n===== Generating hourly_mae.js =====")
hm_data = {"channel": "总流量", "models": {}}
for name in VALID:
    ps = pred_scaled[name]
    ts = true_scaled
    hourly_mae = np.abs(ps[:, :, CH_TOTAL] - ts[:, :, CH_TOTAL]).mean(axis=0)
    hm_data["models"][name] = hourly_mae.tolist()

js = f"export default {json.dumps(hm_data, ensure_ascii=False)};\n"
with open(os.path.join(output_dir, 'hourly_mae.js'), 'w', encoding='utf-8') as f:
    f.write(js)
print(f"  -> Saved hourly_mae.js ({len(js)} bytes)")

# ── 4. error_dist.js ──
print("\n===== Generating error_dist.js =====")
ed_data = {"channel": "总流量", "models": {}}
for name in VALID:
    ps = pred_scaled[name]
    ts = true_scaled
    errors = (ps[:, :, CH_TOTAL] - ts[:, :, CH_TOTAL]).reshape(-1)
    clip = np.percentile(np.abs(errors), 99.5)
    ec = errors[np.abs(errors) <= clip]
    hist, bins = np.histogram(ec, bins=80, density=True)
    x = ((bins[:-1] + bins[1:]) / 2).tolist()
    # Simple smoothing
    from scipy.ndimage import gaussian_filter1d
    density = gaussian_filter1d(hist, sigma=1.5).tolist()
    ed_data["models"][name] = {
        "density": density,
        "x": x,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
    }

js = f"export default {json.dumps(ed_data, ensure_ascii=False)};\n"
with open(os.path.join(output_dir, 'error_dist.js'), 'w', encoding='utf-8') as f:
    f.write(js)
print(f"  -> Saved error_dist.js ({len(js)} bytes)")

print("\n" + "=" * 60)
print("  ALL DATA FILES GENERATED")
print(f"  Models: {len(VALID)} valid")
print(f"  Key: window-AVERAGED across {n_windows} windows")
print("=" * 60)
