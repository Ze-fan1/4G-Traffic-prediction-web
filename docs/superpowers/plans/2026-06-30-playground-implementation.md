# 模型中心 Playground 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增"模型中心"页面和 FastAPI 后端，支持 26 模型实时验证 + 用户自定义 CSV 上传推理

**Architecture:** FastAPI 后端（按需加载模型 → 推理引擎分发 → 返回 JSON） + React 前端（双栏布局：左侧模型选择 + 右侧实验面板）。前后端独立进程，Vite proxy 转发 `/api/*`。

**Tech Stack:** FastAPI, PyTorch, NumPy, Chart.js, React 18, Vite

---

## 文件结构总览

```
可视化网站/
├── backend/                          # 新建
│   ├── server.py                     # FastAPI 入口 + 路由
│   ├── model_registry.py             # 26 模型元数据
│   ├── model_loader.py               # 按需加载 + 缓存
│   ├── inference_engine.py           # 推理分发
│   ├── data_pipeline.py              # CSV/数据预处理
│   └── requirements.txt
├── react-app/
│   ├── vite.config.js                # 修改: proxy /api/* → :8000
│   └── src/
│       ├── App.jsx                   # 修改: 加 playground/compare 路由
│       ├── components/
│       │   ├── Header.jsx            # 修改: 导航标签调整
│       │   ├── ModelSelector.jsx     # 新建
│       │   ├── DemoPanel.jsx         # 新建
│       │   └── UploadPanel.jsx       # 新建
│       └── pages/
│           ├── PlaygroundPage.jsx    # 新建: 模型中心
│           └── ComparePage.jsx       # 新建: 预测对比(合并)
└── Time-Series-Library-main/         # 不变，被 backend import
```

---

# Phase 1: 后端核心

---

### Task 1: 搭建 FastAPI 骨架

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/server.py`
- Create: `backend/__init__.py` (空文件)

- [ ] **Step 1: 创建 requirements.txt**

```txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.19
pandas==2.2.3
numpy==1.26.4
torch>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
transformers>=4.40.0
statsmodels>=0.14.0
pyarrow>=15.0.0
```

- [ ] **Step 2: 创建 server.py 骨架**

```python
"""
FastAPI 后端 — 4G Traffic Model Playground
启动: cd backend && python server.py
"""
import sys, os
from pathlib import Path

# 把 Time-Series-Library 加入 Python path
TSLIB = Path(__file__).resolve().parent.parent / "Time-Series-Library-main"
if str(TSLIB) not in sys.path:
    sys.path.insert(0, str(TSLIB))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="4G Traffic Playground API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    import torch
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_model": None,  # 由 model_loader 填充
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
```

- [ ] **Step 3: 安装依赖并启动验证**

```bash
cd backend && pip install -r requirements.txt && python server.py
```

Expected: 服务在 `http://localhost:8000` 启动，`/api/health` 返回 JSON。

- [ ] **Step 4: Commit**

```bash
git add backend/ && git commit -m "feat: FastAPI skeleton with health check endpoint"
```

---

### Task 2: 实现 model_registry.py（26 模型元数据）

**Files:**
- Create: `backend/model_registry.py`

- [ ] **Step 1: 编写完整模型注册表**

```python
"""
26 模型完整注册表 — 每个模型的类型、checkpoint 路径、类别、Tier 等级
"""
import os
from pathlib import Path

TSLIB_ROOT = Path(__file__).resolve().parent.parent / "Time-Series-Library-main"
CHECKPOINTS_DIR = TSLIB_ROOT / "checkpoints"
DATA_DIR = TSLIB_ROOT / "data_provider" / "4g_traffic"

MODEL_REGISTRY = {
    # ═══ Statistical — 纯计算，无需模型文件 ═══
    "Naive": {
        "type": "statistical", "method": "naive",
        "tier": 1, "category": "Statistical",
    },
    "Persistent 24h": {
        "type": "statistical", "method": "persistent",
        "tier": 1, "category": "Statistical",
    },
    "Historical Avg": {
        "type": "statistical", "method": "historical_avg",
        "tier": 1, "category": "Statistical",
    },
    "AutoARIMA": {
        "type": "statistical", "method": "autoarima",
        "tier": 1, "category": "Statistical",
    },
    "AutoAR": {
        "type": "statistical", "method": "autoar",
        "tier": 1, "category": "Statistical",
    },
    "LinearRegression": {
        "type": "statistical", "method": "linear_regression",
        "tier": 1, "category": "Statistical",
    },

    # ═══ Tree ═══
    "XGBoost": {
        "type": "xgboost",
        "model_file": str(TSLIB_ROOT / "results" / "XGBoost_n100_d5_lr0.1" / "xgb_model.json"),
        "tier": 1, "category": "Tree",
    },

    # ═══ Baseline ═══
    "★ BaseModel": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "External_BaseModel_4G_Base_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 512, "d_ff": 2048, "n_heads": 8},
        "tier": 2, "category": "Baseline",
        "tier2_reason": "自研模型 checkpoint 待确认",
    },

    # ═══ Transformer — 7 个 checkpoint ═══
    "PatchTST": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "PatchTST_4G_PatchTST_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 128, "d_ff": 256, "n_heads": 4},
        "tier": 1, "category": "Transformer",
    },
    "iTransformer": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "iTransformer_4G_iTransformer_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 256, "d_ff": 512, "n_heads": 8},
        "tier": 1, "category": "Transformer",
    },
    "Autoformer": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "Autoformer_4G_Autoformer_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 256, "d_ff": 512, "n_heads": 8},
        "tier": 1, "category": "Transformer",
    },
    "Transformer": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "Transformer_4G_Transformer_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 256, "d_ff": 512, "n_heads": 8},
        "tier": 1, "category": "Transformer",
    },
    "Informer": {
        "type": "pytorch",
        "checkpoint": None,
        "args_override": {"d_model": 128, "d_ff": 256, "n_heads": 4},
        "tier": 2, "category": "Transformer",
        "tier2_reason": "无 checkpoint，需训练 ~1h",
    },

    # ═══ MLP ═══
    "DLinear": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "DLinear_4G_DLinear_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 128, "individual": False},
        "tier": 1, "category": "MLP",
    },
    "LightTS": {
        "type": "pytorch",
        "checkpoint": None,
        "args_override": {"d_model": 128},
        "tier": 2, "category": "MLP",
        "tier2_reason": "无 checkpoint，需训练 ~30min",
    },
    "TSMixer": {
        "type": "pytorch",
        "checkpoint": None,
        "args_override": {"d_model": 128},
        "tier": 2, "category": "MLP",
        "tier2_reason": "无 checkpoint，需训练 ~1h",
    },
    "IBM TTM": {
        "type": "huggingface",
        "model_id": "ibm-granite/granite-timeseries-ttm-r1",
        "context_len": 512,
        "tier": 3, "category": "MLP",
        "tier3_reason": "需 512 步上下文，待修复",
    },

    # ═══ CNN ═══
    "TimesNet": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "TimesNet_4G_TimesNet_v2_Verify" / "checkpoint.pth"),
        "args_override": {},
        "tier": 1, "category": "CNN",
    },
    "SCINet": {
        "type": "pytorch",
        "checkpoint": None,
        "args_override": {"d_model": 128},
        "tier": 2, "category": "CNN",
        "tier2_reason": "无 checkpoint，需训练 ~1h",
    },

    # ═══ RNN ═══
    "SegRNN": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "SegRNN_4G_SegRNN_v2_Verify" / "checkpoint.pth"),
        "args_override": {"seg_len": 6, "d_model": 256},
        "tier": 1, "category": "RNN",
    },

    # ═══ SSM ═══
    "Mamba": {
        "type": "pytorch",
        "checkpoint": None,
        "args_override": {"d_model": 128, "d_ff": 32, "expand": 2},
        "tier": 2, "category": "SSM",
        "tier2_reason": "独立脚本运行，无标准 .pth checkpoint",
    },

    # ═══ LLM ═══
    "TimeLLM": {
        "type": "pytorch",
        "checkpoint": None,
        "args_override": {},
        "tier": 2, "category": "LLM",
        "tier2_reason": "依赖 GPT-2 权重，需特殊加载",
    },
    "Chronos-tiny": {
        "type": "huggingface",
        "model_id": "amazon/chronos-t5-tiny",
        "tier": 1, "category": "LLM",
    },
    "Chronos-small": {
        "type": "huggingface",
        "model_id": "amazon/chronos-t5-small",
        "tier": 1, "category": "LLM",
    },
    "Chronos-base": {
        "type": "huggingface",
        "model_id": "amazon/chronos-t5-base",
        "tier": 1, "category": "LLM",
    },
    "Chronos2": {
        "type": "huggingface",
        "model_id": "amazon/chronos-2",
        "tier": 1, "category": "LLM",
    },
}


def get_model_info(name: str) -> dict:
    """获取单个模型元数据，不存在返回 None"""
    return MODEL_REGISTRY.get(name)


def list_models(tier_filter: int = None) -> list:
    """列出所有模型，可选按 tier 过滤"""
    result = []
    for name, info in MODEL_REGISTRY.items():
        entry = {
            "name": name,
            "category": info["category"],
            "tier": info["tier"],
            "type": info["type"],
            "available": info["tier"] == 1,
        }
        if tier_filter is None or info["tier"] == tier_filter:
            result.append(entry)
    return result
```

- [ ] **Step 2: 验证模块可导入**

```bash
cd backend && python -c "from model_registry import MODEL_REGISTRY; print(len(MODEL_REGISTRY), 'models registered')"
```

Expected: `26 models registered`

- [ ] **Step 3: Commit**

```bash
git add backend/model_registry.py && git commit -m "feat: 26-model registry with tier classification"
```

---

### Task 3: 实现 data_pipeline.py

**Files:**
- Create: `backend/data_pipeline.py`

- [ ] **Step 1: 编写数据管道**

```python
"""
数据预处理管道: CSV 解析 / 标准化 / 滑动窗口 / 测试数据加载
"""
import numpy as np
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from pathlib import Path

TSLIB_ROOT = Path(__file__).resolve().parent.parent / "Time-Series-Library-main"
DATA_DIR = TSLIB_ROOT / "data_provider" / "4g_traffic"

SEQ_LEN = 24
PRED_LEN = 24
STEP = 3
NUM_CHANNELS = 8

# 通道顺序（与 memory 中记录的 plot_all_models_final.py 一致）
FEATURE_COLS = [
    "ERAB流量", "PDCCH利用率", "PDSCH利用率", "PUSCH利用率",
    "上行流量", "下行流量", "有效连接数", "总流量"
]


def load_test_data():
    """加载测试集 + 在训练集上拟合的 Scaler"""
    train_fp = DATA_DIR / "df_4g_train_100.parquet"
    test_fp = DATA_DIR / "df_4g_test_100.parquet"

    if not train_fp.exists() or not test_fp.exists():
        raise FileNotFoundError(f"数据文件不存在: {train_fp} / {test_fp}")

    df_train = pd.read_parquet(train_fp)
    df_test = pd.read_parquet(test_fp)

    # 提取特征列（去掉 ID/厂商/频段/场景/date）
    train_cols = [c for c in FEATURE_COLS if c in df_train.columns]
    test_cols = [c for c in FEATURE_COLS if c in df_test.columns]

    if len(train_cols) < NUM_CHANNELS:
        # 回退：使用所有数值列
        skip = ["ID编号", "厂商", "频段", "场景", "date"]
        train_cols = [c for c in df_train.columns if c not in skip]
        test_cols = [c for c in df_test.columns if c not in skip]

    scaler = StandardScaler()
    scaler.fit(df_train[train_cols].values)

    return df_train, df_test, scaler, train_cols


def get_test_window(window_idx: int, channel_idx: int):
    """
    获取指定测试窗口的输入/输出数据
    返回: X (24, 8), Y (24, 8), scaler
    """
    _, df_test, scaler, cols = load_test_data()
    test_data = scaler.transform(df_test[cols].values)

    start = window_idx * STEP
    X = test_data[start : start + SEQ_LEN]      # (24, 8)
    Y = test_data[start + SEQ_LEN : start + SEQ_LEN + PRED_LEN]  # (24, 8)

    return X, Y, scaler, cols


def get_all_test_windows():
    """获取全部 5378 个测试窗口"""
    _, df_test, scaler, cols = load_test_data()
    test_data = scaler.transform(df_test[cols].values)

    n_windows = (len(test_data) - SEQ_LEN - PRED_LEN) // STEP + 1
    windows = []
    for i in range(n_windows):
        start = i * STEP
        X = test_data[start : start + SEQ_LEN]
        Y = test_data[start + SEQ_LEN : start + SEQ_LEN + PRED_LEN]
        windows.append((X, Y))

    return windows, scaler, cols


def parse_csv(file_bytes: bytes) -> dict:
    """
    解析上传的 CSV 文件
    返回: {headers, rows_preview, numeric_cols, df}
    """
    # 尝试不同编码和分隔符
    content = file_bytes.decode("utf-8-sig")

    # 自动检测分隔符
    first_line = content.split("\n")[0]
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line:
        sep = ";"
    else:
        sep = ","

    df = pd.read_csv(BytesIO(file_bytes), sep=sep)

    # 识别数值列
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_cols) == 0:
        raise ValueError("CSV 中没有有效的数值列")

    return {
        "headers": list(df.columns),
        "numeric_cols": numeric_cols,
        "rows_preview": df.head(5).to_dict(orient="records"),
        "total_rows": len(df),
        "df": df,
    }


def build_windows_from_csv(df: pd.DataFrame, target_col: str, pred_len: int):
    """
    从上传的 CSV DataFrame 构建推理窗口
    返回: X_windows (list of np.array), scaler
    """
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在 CSV 中")

    # 只保留数值列
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    data = df[numeric_cols].values.astype(np.float32)

    if len(data) < SEQ_LEN + pred_len:
        raise ValueError(
            f"数据行数不足: 需要至少 {SEQ_LEN + pred_len} 行，实际 {len(data)} 行"
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # 构建滑动窗口
    step = max(1, pred_len // 4)  # 自适应步长
    windows = []
    for i in range(0, len(scaled) - SEQ_LEN - pred_len + 1, step):
        X = scaled[i : i + SEQ_LEN]
        windows.append(X)

    return windows, scaler, numeric_cols


def inverse_transform_and_clip(predictions: np.ndarray, scaler: StandardScaler, cols: list):
    """逆标准化 + clip 到 >= 0"""
    # predictions shape: (n_windows, pred_len, n_channels)
    orig_shape = predictions.shape
    flat = predictions.reshape(-1, len(cols))
    inv = scaler.inverse_transform(flat)
    inv = np.clip(inv, 0, None)
    return inv.reshape(orig_shape)
```

- [ ] **Step 2: 验证数据加载**

```bash
cd backend && python -c "
from data_pipeline import load_test_data, get_test_window
_, _, scaler, cols = load_test_data()
print(f'Columns: {cols}')
X, Y, _, _ = get_test_window(3825, 1)
print(f'X shape: {X.shape}, Y shape: {Y.shape}')
"
```

Expected: Columns 列出 8 个特征，X shape (24, 8), Y shape (24, 8)

- [ ] **Step 3: Commit**

```bash
git add backend/data_pipeline.py && git commit -m "feat: data pipeline with CSV parsing, scaling, and windowing"
```

---

### Task 4: 实现 model_loader.py

**Files:**
- Create: `backend/model_loader.py`

- [ ] **Step 1: 编写模型加载器**

```python
"""
模型加载器 — 按需加载，全局单例缓存，自动识别模型类型
"""
import torch
import numpy as np
from pathlib import Path
from model_registry import get_model_info

TSLIB_ROOT = Path(__file__).resolve().parent.parent / "Time-Series-Library-main"

_current_model_name = None
_current_model = None
_current_model_info = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_loaded_model_name():
    return _current_model_name


def unload():
    """卸载当前模型，释放显存"""
    global _current_model, _current_model_name, _current_model_info
    if _current_model is not None:
        del _current_model
        _current_model = None
        _current_model_name = None
        _current_model_info = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load(model_name: str):
    """
    按需加载模型。如果已加载同一模型则直接返回。
    返回: (model_instance, model_info)
    """
    global _current_model, _current_model_name, _current_model_info

    info = get_model_info(model_name)
    if info is None:
        raise ValueError(f"未知模型: {model_name}")

    if model_name == _current_model_name and _current_model is not None:
        return _current_model, info

    # 卸载旧模型
    unload()

    device = get_device()
    mtype = info["type"]

    if mtype == "statistical":
        # 统计模型不需要加载，直接返回 method 名
        model = info["method"]

    elif mtype == "pytorch":
        ckpt_path = info.get("checkpoint")
        if ckpt_path is None or not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

        # 动态构建 Args + Exp
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

        class Args:
            task_name = "long_term_forecast"
            is_training = 0
            model_id = "4G"
            model = model_name
            data = "custom"
            root_path = str(TSLIB_ROOT / "data_provider" / "4g_traffic")
            data_path = "df_4g_base_100.parquet"
            features = "M"
            target = "总流量"
            freq = "h"
            checkpoints = str(TSLIB_ROOT / "checkpoints") + "/"
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
            embed = "timeF"
            activation = "gelu"
            num_workers = 0
            itr = 1
            train_epochs = 10
            batch_size = 32
            patience = 3
            learning_rate = 0.0001
            des = f"{model_name}_v2_Verify"
            loss = "MSE"
            lradj = "type1"
            use_amp = False
            use_gpu = (device.type == "cuda")
            gpu = 0
            gpu_type = "cuda"
            use_multi_gpu = False
            devices = "0"
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
            extra_tag = ""
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
            decomp_method = "moving_avg"
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
            seasonal_patterns = "Monthly"

        args = Args()
        args.device = device

        # 模型特定参数覆盖
        overrides = info.get("args_override", {})
        for k, v in overrides.items():
            setattr(args, k, v)

        exp = Exp_Long_Term_Forecast(args)
        setting = f"{model_name}_4G_{args.des}"

        # 加载 checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        exp.model.load_state_dict(checkpoint)
        exp.model.to(device)
        exp.model.eval()
        model = exp

    elif mtype == "huggingface":
        model_id = info["model_id"]
        if "chronos" in model_id.lower():
            from chronos import ChronosPipeline
            model = ChronosPipeline.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            )
        elif "ttm" in model_id.lower():
            from tsfm_public import TinyTimeMixerForPrediction
            model = TinyTimeMixerForPrediction.from_pretrained(model_id, revision="main")
            model.to(device)
            model.eval()
        else:
            raise ValueError(f"不支持的 HuggingFace 模型: {model_id}")

    elif mtype == "xgboost":
        import xgboost as xgb
        model_path = info.get("model_file")
        if model_path and Path(model_path).exists():
            model = xgb.Booster()
            model.load_model(model_path)
        else:
            # 如果没有保存的模型文件，则现场训练
            from data_pipeline import load_test_data, FEATURE_COLS
            df_train, _, _, cols = load_test_data()
            X_train = df_train[cols].values
            model = _train_xgboost_quick(X_train, cols)

    else:
        raise ValueError(f"未知模型类型: {mtype}")

    _current_model = model
    _current_model_name = model_name
    _current_model_info = info
    return model, info


def _train_xgboost_quick(X: np.ndarray, cols: list):
    """快速训练 XGBoost（用于无模型文件时）"""
    import xgboost as xgb
    # 多步预测：每个通道训练一个模型
    models = {}
    for i, col in enumerate(cols):
        # 简化：用 t-1 预测 t
        dtrain = xgb.DMatrix(X[:-1, i:i+1], label=X[1:, i])
        params = {"max_depth": 5, "eta": 0.1, "objective": "reg:squarederror", "verbosity": 0}
        models[col] = xgb.train(params, dtrain, num_boost_round=100)
    return models
```

- [ ] **Step 2: 验证加载逻辑**

```bash
cd backend && python -c "
from model_loader import load, get_device
print(f'Device: {get_device()}')
# 测试统计模型
m, info = load('Naive')
print(f'Naive loaded: {m}')
"
```

Expected: `Device: cuda/cpu` + `Naive loaded: naive`

- [ ] **Step 3: Commit**

```bash
git add backend/model_loader.py && git commit -m "feat: lazy model loader with PyTorch/HF/statistical support"
```

---

### Task 5: 实现 inference_engine.py

**Files:**
- Create: `backend/inference_engine.py`

- [ ] **Step 1: 编写推理引擎**

```python
"""
统一推理引擎 — 按模型类型分发推理逻辑
"""
import torch
import numpy as np
from model_loader import load, get_device
from model_registry import get_model_info


def infer(model_name: str, X: np.ndarray, pred_len: int = 24) -> np.ndarray:
    """
    对输入 X 执行推理

    Args:
        model_name: 模型名（如 "PatchTST"）
        X: 输入数据 (seq_len, n_channels) 或 (batch, seq_len, n_channels)
        pred_len: 预测长度（仅统计/HF 模型可动态；PyTorch 固定 24 后切片）

    Returns:
        predictions: (pred_len, n_channels)
    """
    info = get_model_info(model_name)
    if info is None:
        raise ValueError(f"未知模型: {model_name}")

    mtype = info["type"]

    # 确保 X 是 (batch, seq_len, n_channels)
    if X.ndim == 2:
        X_batch = X[np.newaxis, :, :]
    else:
        X_batch = X

    if mtype == "statistical":
        result = _infer_statistical(info["method"], X_batch, pred_len)

    elif mtype == "pytorch":
        # PyTorch 固定 24h 输出 → 切片
        result = _infer_pytorch(model_name, X_batch)
        result = result[:, :pred_len, :]

    elif mtype == "huggingface":
        result = _infer_huggingface(model_name, X_batch, pred_len)

    elif mtype == "xgboost":
        result = _infer_xgboost(model_name, X_batch, pred_len)

    else:
        raise ValueError(f"未知模型类型: {mtype}")

    # 去掉 batch 维度（如果输入是单样本）
    if result.shape[0] == 1 and X.ndim == 2:
        result = result[0]

    return result


def _infer_statistical(method: str, X: np.ndarray, pred_len: int) -> np.ndarray:
    """
    统计模型推理
    X: (batch, seq_len, n_channels)
    Returns: (batch, pred_len, n_channels)
    """
    batch, seq_len, n_channels = X.shape
    preds = np.zeros((batch, pred_len, n_channels))

    for b in range(batch):
        for c in range(n_channels):
            series = X[b, :, c]

            if method == "naive":
                # 重复最后一个值
                preds[b, :, c] = series[-1]

            elif method == "persistent":
                # 24h 前值（如果 seq_len >= pred_len）
                offset = max(0, seq_len - pred_len)
                preds[b, :, c] = series[offset]

            elif method == "historical_avg":
                # 历史均值
                avg = np.mean(series)
                preds[b, :, c] = avg

            elif method == "autoarima":
                # 简易自回归：用最后 pred_len 个值作为起始
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    model = ARIMA(series, order=(2, 0, 1))
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=pred_len)
                    preds[b, :, c] = forecast
                except Exception:
                    # 回退：重复均值
                    preds[b, :, c] = np.mean(series)

            elif method == "autoar":
                # 自回归：线性外推
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                t = np.arange(seq_len).reshape(-1, 1)
                lr.fit(t, series)
                t_future = np.arange(seq_len, seq_len + pred_len).reshape(-1, 1)
                preds[b, :, c] = lr.predict(t_future)

            elif method == "linear_regression":
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                t = np.arange(seq_len).reshape(-1, 1)
                lr.fit(t, series)
                t_future = np.arange(seq_len, seq_len + pred_len).reshape(-1, 1)
                preds[b, :, c] = lr.predict(t_future)

    return preds


def _infer_pytorch(model_name: str, X: np.ndarray) -> np.ndarray:
    """
    PyTorch 模型推理
    X: (batch, seq_len, n_channels)
    Returns: (batch, 24, n_channels)
    """
    model_inst, info = load(model_name)
    device = get_device()
    exp = model_inst  # Exp_Long_Term_Forecast 实例

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # 构建 decoder 输入（label_len + pred_len）
    label_len = 12
    dec_inp = torch.zeros((X_tensor.shape[0], label_len + 24, X_tensor.shape[2]),
                          dtype=torch.float32).to(device)
    dec_inp[:, :label_len, :] = X_tensor[:, -label_len:, :]

    with torch.no_grad():
        outputs = exp.model(X_tensor, None, dec_inp, None)

    if isinstance(outputs, tuple):
        outputs = outputs[0]

    return outputs.cpu().numpy()


def _infer_huggingface(model_name: str, X: np.ndarray, pred_len: int) -> np.ndarray:
    """
    HuggingFace 预训练模型推理
    X: (batch, seq_len, n_channels)
    Returns: (batch, pred_len, n_channels)
    """
    model_inst, info = load(model_name)
    model_id = info["model_id"]
    device = get_device()
    batch, seq_len, n_channels = X.shape
    preds = np.zeros((batch, pred_len, n_channels))

    for b in range(batch):
        if "chronos" in model_id.lower():
            # Chronos: 输入 (1, seq_len) 单变量
            for c in range(n_channels):
                context = torch.tensor(X[b, :, c], dtype=torch.float32).to(device)
                forecast = model_inst.predict(
                    context,
                    prediction_length=pred_len,
                    limit_prediction_length=False,
                )
                preds[b, :, c] = forecast[0].cpu().numpy()

        elif "ttm" in model_id.lower():
            # IBM TTM: 需要 512 上下文
            context_len = info.get("context_len", 512)
            pad_len = context_len - seq_len
            past = np.pad(X[b], ((pad_len, 0), (0, 0)), mode="constant", constant_values=0)
            past_tensor = torch.tensor(past, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model_inst(past_values=past_tensor)
                if hasattr(outputs, "prediction_outputs"):
                    p = outputs.prediction_outputs.squeeze(0).cpu().numpy()
                elif hasattr(outputs, "logits"):
                    p = outputs.logits.squeeze(0).cpu().numpy()
                else:
                    p = outputs[0].squeeze(0).cpu().numpy()
            preds[b, :, :] = p[:pred_len, :n_channels]

    return preds


def _infer_xgboost(model_name: str, X: np.ndarray, pred_len: int) -> np.ndarray:
    """
    XGBoost 推理
    X: (batch, seq_len, n_channels)
    Returns: (batch, pred_len, n_channels)
    """
    import xgboost as xgb
    model_inst, info = load(model_name)
    batch, seq_len, n_channels = X.shape
    preds = np.zeros((batch, pred_len, n_channels))

    for b in range(batch):
        for c in range(n_channels):
            last_val = X[b, -1, c]
            if isinstance(model_inst, dict):
                # 多模型字典
                col_model = list(model_inst.values())[min(c, len(model_inst) - 1)]
                for t in range(pred_len):
                    dtest = xgb.DMatrix(np.array([[last_val]]))
                    last_val = col_model.predict(dtest)[0]
                    preds[b, t, c] = last_val
            else:
                # 单模型
                for t in range(pred_len):
                    dtest = xgb.DMatrix(np.array([[last_val]]))
                    last_val = model_inst.predict(dtest)[0]
                    preds[b, t, c] = last_val

    return preds
```

- [ ] **Step 2: Commit**

```bash
git add backend/inference_engine.py && git commit -m "feat: unified inference engine dispatching by model type"
```

---

### Task 6: 实现 server.py API 路由

**Files:**
- Modify: `backend/server.py`

- [ ] **Step 1: 在 server.py 中添加完整路由**

```python
"""
FastAPI 后端 — 4G Traffic Model Playground
启动: cd backend && python server.py
"""
import sys, os, time
from pathlib import Path

TSLIB = Path(__file__).resolve().parent.parent / "Time-Series-Library-main"
if str(TSLIB) not in sys.path:
    sys.path.insert(0, str(TSLIB))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from model_registry import MODEL_REGISTRY, get_model_info, list_models
from model_loader import load, unload, get_loaded_model_name
from inference_engine import infer
from data_pipeline import (
    get_test_window, get_all_test_windows,
    parse_csv, build_windows_from_csv,
    inverse_transform_and_clip, FEATURE_COLS,
)

app = FastAPI(title="4G Traffic Playground API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══ Health ═══

@app.get("/api/health")
async def health():
    import torch
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_model": get_loaded_model_name(),
    }


# ═══ Model List ═══

@app.get("/api/models")
async def api_list_models():
    """返回全部 26 模型 + 前端指标"""
    # 导入前端模型指标（MSE/MAE/ACC 等）
    frontend_metrics = {}
    try:
        import json
        metrics_path = Path(__file__).resolve().parent.parent / "react-app" / "src" / "data" / "models.js"
        if metrics_path.exists():
            content = metrics_path.read_text(encoding="utf-8")
            # 简单解析 JS export
            for line in content.split("\n"):
                if "{ cat:" in line and "model:" in line:
                    pass  # 后续在 Task 7 中完善
    except Exception:
        pass

    result = []
    for name, info in MODEL_REGISTRY.items():
        result.append({
            "name": name,
            "category": info["category"],
            "tier": info["tier"],
            "type": info["type"],
            "available": info["tier"] == 1,
            "tier_reason": info.get("tier2_reason") or info.get("tier3_reason", ""),
        })
    return result


# ═══ Demo: Quick Verification ═══

@app.post("/api/demo/{model_name}/quick")
async def demo_quick(model_name: str, body: dict):
    """
    单窗口快速验证
    Request: {channel_idx: int, window_idx?: int}
    Response: {pred: [24], truth: [24], window: int, mae: float}
    """
    info = get_model_info(model_name)
    if info is None:
        raise HTTPException(404, detail={"error": "unknown_model", "available": list(MODEL_REGISTRY.keys())[:10]})
    if info["tier"] >= 2:
        raise HTTPException(503, detail={"error": "model_not_available", "tier": info["tier"], "reason": info.get("tier2_reason", info.get("tier3_reason", ""))})

    channel_idx = body.get("channel_idx", 1)
    window_idx = body.get("window_idx", 3825)  # 默认代表性窗口

    # 加载数据
    X, Y, scaler, cols = get_test_window(window_idx, channel_idx)

    try:
        t0 = time.time()
        pred = infer(model_name, X, pred_len=24)
        elapsed = time.time() - t0
    except Exception as e:
        raise HTTPException(500, detail={"error": "inference_failed", "detail": str(e)})

    # 逆标准化
    X_inv = inverse_transform_and_clip(X[np.newaxis, :, :], scaler, cols)[0]
    Y_inv = inverse_transform_and_clip(Y[np.newaxis, :, :], scaler, cols)[0]
    pred_inv = inverse_transform_and_clip(pred[np.newaxis, :, :] if pred.ndim == 2 else pred, scaler, cols)[0]

    # 通道重排（修复 6/7 交换 bug）
    if pred_inv.shape[1] >= 8:
        pred_inv[:, [6, 7]] = pred_inv[:, [7, 6]]
        Y_inv[:, [6, 7]] = Y_inv[:, [7, 6]]

    channel_pred = pred_inv[:, channel_idx].tolist()
    channel_truth = Y_inv[:, channel_idx].tolist()
    mae = float(np.mean(np.abs(np.array(channel_pred) - np.array(channel_truth))))

    return {
        "model": model_name,
        "window": window_idx,
        "channel": FEATURE_COLS[channel_idx] if channel_idx < len(FEATURE_COLS) else f"ch{channel_idx}",
        "channel_idx": channel_idx,
        "pred": channel_pred,
        "truth": channel_truth,
        "mae": round(mae, 6),
        "elapsed_s": round(elapsed, 2),
        "device": str(get_device()),
    }


# ═══ Demo: Full Evaluation ═══

def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@app.post("/api/demo/{model_name}/full")
async def demo_full(model_name: str, body: dict):
    """
    全量评估
    Request: {channel_idx: int}
    Response: {metrics: {mse, mae, rmse, mape, acc}, n_windows: int, elapsed_s: float}
    """
    info = get_model_info(model_name)
    if info is None:
        raise HTTPException(404, detail={"error": "unknown_model"})
    if info["tier"] >= 2:
        raise HTTPException(503, detail={"error": "model_not_available", "tier": info["tier"]})

    channel_idx = body.get("channel_idx", 1)

    windows, scaler, cols = get_all_test_windows()

    all_preds = []
    all_trues = []
    t0 = time.time()

    for i, (X, Y) in enumerate(windows):
        try:
            pred = infer(model_name, X, pred_len=24)
            all_preds.append(pred[0] if pred.ndim == 3 else pred)
            all_trues.append(Y)
        except Exception as e:
            raise HTTPException(500, detail={"error": "inference_failed", "window": i, "detail": str(e)})

    elapsed = time.time() - t0

    preds_arr = np.array(all_preds).reshape(-1, len(cols))
    trues_arr = np.array(all_trues).reshape(-1, len(cols))

    # 逆标准化
    preds_inv = scaler.inverse_transform(preds_arr)
    trues_inv = scaler.inverse_transform(trues_arr)

    # 只对目标通道计算指标
    # 全通道
    mse = float(np.mean((trues_inv - preds_inv) ** 2))
    mae = float(np.mean(np.abs(trues_inv - preds_inv)))
    rmse = float(np.sqrt(mse))

    mask = trues_inv > 1e-5
    mape = float(np.mean(np.abs((trues_inv[mask] - preds_inv[mask]) / trues_inv[mask]))) if np.sum(mask) > 0 else 0.0

    return {
        "model": model_name,
        "metrics": {"mse": round(mse, 4), "mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)},
        "n_windows": len(windows),
        "elapsed_s": round(elapsed, 1),
        "device": str(get_device()),
    }


# ═══ Custom Prediction ═══

@app.post("/api/predict/{model_name}")
async def predict(
    model_name: str,
    csv_file: UploadFile = File(...),
    target_col: str = Form(...),
    pred_len: int = Form(24),
):
    """
    用户上传 CSV 自定义预测
    Response: {predictions: [[...]], meta: {...}}
    """
    info = get_model_info(model_name)
    if info is None:
        raise HTTPException(404, detail={"error": "unknown_model"})
    if info["tier"] >= 2:
        raise HTTPException(503, detail={"error": "model_not_available", "tier": info["tier"]})

    if pred_len not in [6, 12, 18, 24]:
        pred_len = 24

    try:
        file_bytes = await csv_file.read()
        parsed = parse_csv(file_bytes)
    except ValueError as e:
        raise HTTPException(400, detail={"error": "csv_parse_error", "detail": str(e)})

    df = parsed["df"]

    if target_col not in parsed["numeric_cols"]:
        raise HTTPException(400, detail={"error": "invalid_target", "numeric_cols": parsed["numeric_cols"]})

    try:
        windows, scaler, numeric_cols = build_windows_from_csv(df, target_col, pred_len)
    except ValueError as e:
        raise HTTPException(400, detail={"error": str(e)})

    if len(windows) == 0:
        raise HTTPException(400, detail={"error": "insufficient_data", "min_rows": 30, "actual": len(df)})

    try:
        X_last = windows[-1]  # 取最后一窗口
        pred = infer(model_name, X_last, pred_len=pred_len)
    except Exception as e:
        raise HTTPException(500, detail={"error": "inference_failed", "detail": str(e)})

    # 逆标准化（只用目标列）
    pred_flat = pred.reshape(-1, len(numeric_cols)) if pred.ndim == 3 else pred
    if pred_flat.ndim == 1:
        pred_flat = pred_flat.reshape(-1, 1)
    pred_inv = scaler.inverse_transform(pred_flat)
    pred_inv = np.clip(pred_inv, 0, None)

    target_idx = numeric_cols.index(target_col)
    predictions = pred_inv[:, target_idx].tolist()

    return {
        "model": model_name,
        "predictions": predictions,
        "meta": {
            "target_col": target_col,
            "pred_len": pred_len,
            "actual_pred_len": pred.shape[1] if pred.ndim >= 2 else len(pred),
            "input_rows": len(df),
            "n_windows": len(windows),
            "device": str(get_device()),
        },
    }
```

- [ ] **Step 2: 启动服务并验证 API**

```bash
cd backend && python server.py
```

另开终端:
```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/models | head -c 500
```

- [ ] **Step 3: Commit**

```bash
git add backend/server.py && git commit -m "feat: complete API routes for demo quick/full and custom prediction"
```

---

### Task 7: 端到端验证 Tier 1 模型

**Files:** 无新建，测试验证

- [ ] **Step 1: 验证统计模型**

```bash
curl -X POST http://localhost:8000/api/demo/Naive/quick \
  -H "Content-Type: application/json" \
  -d '{"channel_idx": 1}'
```

Expected: 返回 `{pred: [...24 floats], truth: [...24 floats], mae: ..., window: 3825}`

- [ ] **Step 2: 验证 PyTorch 模型**

```bash
curl -X POST http://localhost:8000/api/demo/PatchTST/quick \
  -H "Content-Type: application/json" \
  -d '{"channel_idx": 1}'
```

Expected: 同上格式，elapsed_s 应 < 5s

- [ ] **Step 3: 验证 HuggingFace 模型（需联网）**

```bash
curl -X POST http://localhost:8000/api/demo/Chronos-tiny/quick \
  -H "Content-Type: application/json" \
  -d '{"channel_idx": 1}'
```

Expected: 首次较慢（下载模型），后续快

- [ ] **Step 4: 验证 CSV 上传推理**

```bash
# 用测试数据生成一个 CSV
cd backend && python -c "
import pandas as pd
from data_pipeline import load_test_data, FEATURE_COLS
_, df_test, _, cols = load_test_data()
sample = df_test[cols].head(100)
sample.to_csv('test_upload.csv', index=False)
print('test_upload.csv created')
"
curl -X POST http://localhost:8000/api/predict/Naive \
  -F "csv_file=@test_upload.csv" \
  -F "target_col=总流量" \
  -F "pred_len=24"
```

Expected: 返回 `{predictions: [...], meta: {...}}`

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "verify: all Tier 1 model inference paths confirmed"
```

---

# Phase 2: 前端

---

### Task 8: 修改导航和路由

**Files:**
- Modify: `react-app/src/components/Header.jsx`
- Modify: `react-app/src/App.jsx`

- [ ] **Step 1: 修改 Header.jsx 导航标签**

在 Header.jsx 中，替换 TABS 数组：

```jsx
const TABS = [
  { id: 'overview', label: '首页' },
  { id: 'playground', label: '模型中心' },
  { id: 'compare', label: '预测对比' },
  { id: 'errors', label: '误差分析' },
  { id: 'details', label: '详细报告' },
];
```

原文件 [Header.jsx](react-app/src/components/Header.jsx) 的 TABS 定义在第 4-10 行。

- [ ] **Step 2: 修改 App.jsx 添加新页面导入和路由**

在 App.jsx 中：

```jsx
import PlaygroundPage from './pages/PlaygroundPage';
import ComparePage from './pages/ComparePage';

const PAGES = {
  overview: OverviewPage,
  playground: PlaygroundPage,
  compare: ComparePage,
  errors: ErrorsPage,
  details: DetailsPage,
};
```

- [ ] **Step 3: 创建占位页面验证导航**

创建最小 PlaygroundPage.jsx:
```jsx
export default function PlaygroundPage() {
  return <div className="page-enter pt-20"><h2>模型中心</h2></div>;
}
```

创建最小 ComparePage.jsx（复制 CurvesPage 内容临时）:
```jsx
import CurvesPage from './CurvesPage';
export default function ComparePage() {
  return <CurvesPage />;
}
```

- [ ] **Step 4: 启动前端验证导航**

```bash
cd react-app && npm run dev
```

Expected: 导航栏显示 5 个标签，点击"模型中心"和"预测对比"可正常切换。

- [ ] **Step 5: Commit**

```bash
git add react-app/src/components/Header.jsx react-app/src/App.jsx react-app/src/pages/PlaygroundPage.jsx react-app/src/pages/ComparePage.jsx
git commit -m "feat: add 模型中心 and 预测对比 navigation tabs"
```

---

### Task 9: 实现 ModelSelector 组件

**Files:**
- Create: `react-app/src/components/ModelSelector.jsx`

- [ ] **Step 1: 编写组件**

```jsx
import { useState } from 'react';
import { MODELS, CATS } from '../data/models';
import { CAT_PALETTE } from '../data/palette';

function getCat(model) {
  return CAT_PALETTE[model.cat] || CAT_PALETTE['Statistical'];
}

export default function ModelSelector({ selectedModel, onSelect, modelTiers = {} }) {
  const [search, setSearch] = useState('');
  const [collapsed, setCollapsed] = useState(
    Object.fromEntries(CATS.map(c => [c, c !== 'Baseline' && c !== 'Transformer']))
  );

  const toggleCat = (cat) => setCollapsed(prev => ({ ...prev, [cat]: !prev[cat] }));

  const filtered = MODELS.filter(m =>
    m.model.toLowerCase().includes(search.toLowerCase())
  );

  // 按类别分组
  const grouped = {};
  CATS.forEach(cat => {
    const items = filtered.filter(m => m.cat === cat);
    if (items.length > 0) grouped[cat] = items;
  });

  return (
    <div className="card p-4 h-full overflow-y-auto" style={{ maxHeight: 'calc(100vh - 120px)' }}>
      {/* 搜索框 */}
      <div className="relative mb-3">
        <input
          type="text"
          placeholder="搜索模型..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full text-xs px-3 py-2 rounded-xl border border-[rgba(0,0,0,0.08)] bg-[#FAFAFA] focus:outline-none focus:border-[#3B82F6] focus:bg-white transition-colors"
          style={{ fontSize: '0.75rem' }}
        />
      </div>

      {/* 按类别折叠 */}
      {Object.entries(grouped).map(([cat, items]) => {
        const catInfo = CAT_PALETTE[cat];
        const isCollapsed = collapsed[cat];
        return (
          <div key={cat} className="mb-1">
            {/* 类别标题 */}
            <button
              onClick={() => toggleCat(cat)}
              className="w-full flex items-center gap-2 py-1.5 text-left hover:bg-[#FAFAFA] rounded-lg px-1 transition-colors cursor-pointer"
            >
              <span className="text-[0.55rem] transition-transform" style={{ display: 'inline-block', transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)' }}>
                ▼
              </span>
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: catInfo?.border || '#A1A1AA' }}
              />
              <span className="text-[0.7rem] font-medium text-[#52525B]">
                {catInfo?.label || cat}
              </span>
              <span className="text-[0.6rem] text-[#A1A1AA] ml-auto font-mono">
                {items.length}
              </span>
            </button>

            {/* 模型列表 */}
            {!isCollapsed && (
              <div className="ml-4 space-y-0.5">
                {items.map(m => {
                  const isSelected = selectedModel === m.model;
                  const tier = modelTiers[m.model] || 1;
                  const isUnavailable = tier >= 2;

                  return (
                    <button
                      key={m.model}
                      onClick={() => onSelect(m.model)}
                      className={`w-full flex items-center gap-2 py-1.5 px-2 rounded-lg text-left transition-all duration-150 cursor-pointer ${
                        isSelected
                          ? 'bg-[#EFF6FF] border-l-[3px] border-[#3B82F6]'
                          : 'hover:bg-[#FAFAFA] border-l-[3px] border-transparent'
                      } ${isUnavailable ? 'opacity-50' : ''}`}
                      title={isUnavailable ? '该模型暂不支持实时推理' : `${m.model} · ACC=${m.acc?.toFixed(4)}`}
                    >
                      <span
                        className="w-2 h-2 rounded-full flex-shrink-0"
                        style={{ background: getCat(m).border }}
                      />
                      <span
                        className="text-[0.72rem] truncate flex-1"
                        style={{ color: isSelected ? '#3B82F6' : '#52525B', fontWeight: isSelected ? 600 : 400 }}
                      >
                        {m.model}
                      </span>
                      {isUnavailable && (
                        <span className="text-[0.55rem] flex-shrink-0" title="暂不支持实时推理">🔒</span>
                      )}
                      <span className="text-[0.6rem] text-[#A1A1AA] font-mono flex-shrink-0 w-14 text-right">
                        {m.acc?.toFixed(3) || '—'}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}

      {filtered.length === 0 && (
        <p className="text-xs text-[#A1A1AA] text-center py-4">无匹配模型</p>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add react-app/src/components/ModelSelector.jsx
git commit -m "feat: ModelSelector with category accordion, search, and tier indicators"
```

---

### Task 10: 实现 DemoPanel 组件

**Files:**
- Create: `react-app/src/components/DemoPanel.jsx`

- [ ] **Step 1: 编写 DemoPanel（初始展示 + 验证区 + 预置曲线图）**

```jsx
import { useState, useMemo, useCallback } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { CHANNELS } from '../data/channels';
import predictionCurves from '../data/prediction_curves.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const HOURS = Array.from({ length: 24 }, (_, i) => `${i + 1}h`);
const API_BASE = '/api';  // Vite proxy → localhost:8000

export default function DemoPanel({ model, channelIdx, onChangeChannel, isAvailable }) {
  const [quickResult, setQuickResult] = useState(null);
  const [fullResult, setFullResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const chKey = String(channelIdx);

  // 预置曲线数据
  const presetData = useMemo(() => {
    const modelData = predictionCurves.models?.[model]?.[chKey];
    if (!modelData) return null;
    return {
      pred: modelData.pred,
      truth: modelData.true,
    };
  }, [model, chKey]);

  // 图表数据：预置 + 实时验证叠加
  const chartData = useMemo(() => {
    const datasets = [];

    // 真实值
    if (presetData?.truth) {
      datasets.push({
        label: '真实值 (预置)',
        data: presetData.truth,
        borderColor: '#18181B',
        backgroundColor: 'transparent',
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.35,
        order: 0,
      });
    }

    // 预置预测
    if (presetData?.pred) {
      datasets.push({
        label: `${model} (预置)`,
        data: presetData.pred,
        borderColor: '#3B82F6',
        backgroundColor: '#3B82F620',
        borderWidth: 1.8,
        borderDash: [5, 3],
        pointRadius: 0,
        tension: 0.35,
        order: 1,
      });
    }

    // 实时验证预测（叠加在预置上）
    if (quickResult?.pred) {
      datasets.push({
        label: `${model} (实时)`,
        data: quickResult.pred,
        borderColor: '#22C55E',
        backgroundColor: '#22C55E20',
        borderWidth: 2.2,
        borderDash: [],
        pointRadius: 0,
        tension: 0.35,
        order: 0,
      });
    }

    return {
      labels: HOURS,
      datasets,
    };
  }, [presetData, quickResult, model]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'bottom',
        labels: { boxWidth: 10, padding: 14, font: { size: 10 }, color: '#52525B' },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(0,0,0,0.03)' },
        ticks: { color: '#A1A1AA' },
        title: { display: true, text: '预测时刻 (小时)', color: '#A1A1AA' },
      },
      y: {
        grid: { color: 'rgba(0,0,0,0.03)' },
        ticks: { color: '#A1A1AA', callback: (v) => v?.toFixed(2) },
        title: { display: true, text: '标准化值 (σ)', color: '#A1A1AA' },
      },
    },
  }), []);

  const handleQuickVerify = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/demo/${encodeURIComponent(model)}/quick`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel_idx: channelIdx }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail?.reason || err.detail?.error || '请求失败');
      }
      const data = await res.json();
      setQuickResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, [model, channelIdx]);

  const handleFullEval = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setFullResult(null);
    try {
      const res = await fetch(`${API_BASE}/demo/${encodeURIComponent(model)}/full`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel_idx: channelIdx }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail?.reason || err.detail?.error || '请求失败');
      }
      const data = await res.json();
      setFullResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, [model, channelIdx]);

  return (
    <div className="space-y-4">
      {/* 信息Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">{model}</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mt-0.5">
            数据: df_4g_test_100 · Window #{predictionCurves.window || '—'} · 输入24h → 预测24h
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[0.65rem] text-[#A1A1AA]">通道:</span>
          <select
            value={channelIdx}
            onChange={(e) => onChangeChannel(Number(e.target.value))}
            className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer focus:outline-none focus:border-[#3B82F6] transition-colors"
          >
            {CHANNELS.map((ch, i) => (
              <option key={ch.id} value={i}>{ch.name} — {ch.desc}</option>
            ))}
          </select>
        </div>
      </div>

      {/* 曲线图 */}
      <div className="chart-box-lg">
        <Line data={chartData} options={chartOptions} />
      </div>

      {/* MAE 信息 */}
      {quickResult && (
        <div className="p-2 rounded-lg bg-[#F0FDF4] border border-[#BBF7D0] text-xs text-[#166534]">
          ✅ 实时验证完成 (耗时 {quickResult.elapsed_s}s) — MAE = {quickResult.mae}
          {presetData && (
            <span className="ml-2 text-[#A1A1AA]">
              | 预置 MAE = {(
                presetData.pred.reduce((s, p, i) => s + Math.abs(p - presetData.truth[i]), 0) / 24
              ).toFixed(4)}
            </span>
          )}
        </div>
      )}

      {error && (
        <div className="p-2 rounded-lg bg-[#FEF2F2] border border-[#FECACA] text-xs text-[#991B1B]">
          ❌ {error}
        </div>
      )}

      {/* 操作按钮 */}
      <div className="flex flex-wrap items-center gap-3">
        {isAvailable ? (
          <>
            <button
              onClick={handleQuickVerify}
              disabled={isLoading}
              className="text-xs px-4 py-2 rounded-xl bg-[#3B82F6] text-white hover:bg-[#2563EB] disabled:opacity-50 transition-colors cursor-pointer font-medium"
            >
              {isLoading ? '⏳ 进行中...' : '🔄 快速验证 (1个窗口)'}
            </button>
            <button
              onClick={handleFullEval}
              disabled={isLoading}
              className="text-xs px-4 py-2 rounded-xl border border-[rgba(0,0,0,0.08)] bg-white hover:bg-stone-50 disabled:opacity-50 transition-colors cursor-pointer"
            >
              📊 完整评估 (5378窗口)
            </button>
          </>
        ) : (
          <p className="text-xs text-[#A1A1AA]">🔒 该模型暂不支持实时推理，仅展示预置曲线</p>
        )}
      </div>

      {/* 完整评估结果 */}
      {fullResult && (
        <div className="p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
          <p className="text-xs font-semibold mb-2">📊 完整评估 ({fullResult.n_windows} 窗口，耗时 {fullResult.elapsed_s}s)</p>
          <div className="grid grid-cols-4 gap-2 text-xs">
            {Object.entries(fullResult.metrics).map(([k, v]) => (
              <div key={k} className="bg-white rounded-lg p-2 text-center">
                <p className="text-[0.6rem] text-[#A1A1AA] uppercase">{k}</p>
                <p className="font-mono font-bold text-[#52525B]">{v}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add react-app/src/components/DemoPanel.jsx
git commit -m "feat: DemoPanel with preset curves, quick/full verification, result comparison"
```

---

### Task 11: 实现 UploadPanel 组件

**Files:**
- Create: `react-app/src/components/UploadPanel.jsx`

- [ ] **Step 1: 编写 UploadPanel**

```jsx
import { useState, useCallback, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const API_BASE = '/api';
const PRED_LEN_OPTIONS = [6, 12, 18, 24];

export default function UploadPanel({ model, isAvailable }) {
  const [csvFile, setCsvFile] = useState(null);
  const [csvPreview, setCsvPreview] = useState(null);
  const [targetCol, setTargetCol] = useState('');
  const [predLen, setPredLen] = useState(24);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = useCallback(async (file) => {
    if (!file || !file.name.endsWith('.csv')) {
      setError('请选择 .csv 文件');
      return;
    }
    setCsvFile(file);
    setError(null);
    setResult(null);

    // 前端预览：读前 6 行
    const text = await file.text();
    const lines = text.trim().split('\n');
    const headers = lines[0].split(/[,\t;]/);
    const previewRows = lines.slice(1, 6).map(line => line.split(/[,\t;]/));

    // 识别数值列
    const numericCols = [];
    headers.forEach((h, i) => {
      const vals = previewRows.map(r => parseFloat(r[i])).filter(v => !isNaN(v));
      if (vals.length === previewRows.length) numericCols.push(h.trim());
    });

    setCsvPreview({ headers: headers.map(h => h.trim()), rows: previewRows, numericCols });
    if (numericCols.length > 0) setTargetCol(numericCols[0]);
  }, []);

  const handlePredict = useCallback(async () => {
    if (!csvFile || !targetCol) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('csv_file', csvFile);
      formData.append('target_col', targetCol);
      formData.append('pred_len', String(predLen));

      const res = await fetch(`${API_BASE}/predict/${encodeURIComponent(model)}`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail?.detail || err.detail?.error || '预测失败');
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, [csvFile, targetCol, predLen, model]);

  const downloadCSV = useCallback(() => {
    if (!result?.predictions) return;
    const headers = `hour,${model}_prediction`;
    const rows = result.predictions.map((v, i) => `${i + 1},${v.toFixed(6)}`);
    const csv = [headers, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prediction_${model}_${targetCol}_${predLen}h.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result, model, targetCol, predLen]);

  if (!isAvailable) {
    return (
      <div className="p-4 text-center text-xs text-[#A1A1AA]">
        🔒 该模型暂不支持自定义预测
      </div>
    );
  }

  const chartData = result ? {
    labels: Array.from({ length: result.predictions.length }, (_, i) => `${i + 1}h`),
    datasets: [{
      label: `${model} 预测 (${targetCol})`,
      data: result.predictions,
      borderColor: '#3B82F6',
      backgroundColor: '#3B82F620',
      borderWidth: 2,
      pointRadius: 2,
      tension: 0.35,
    }],
  } : null;

  return (
    <div className="space-y-4">
      {/* 上传区 */}
      <div
        className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-colors cursor-pointer ${
          dragOver ? 'border-[#3B82F6] bg-[#EFF6FF]' : 'border-[rgba(0,0,0,0.08)] hover:border-[#3B82F6]'
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        {csvFile ? (
          <p className="text-xs text-[#52525B]">📁 {csvFile.name} ({(csvFile.size / 1024).toFixed(1)} KB)</p>
        ) : (
          <p className="text-xs text-[#A1A1AA]">拖拽 CSV 文件到此处，或点击上传</p>
        )}
      </div>

      {/* CSV 预览 */}
      {csvPreview && (
        <div className="text-xs">
          <p className="font-medium text-[#52525B] mb-1">数据预览 (前 5 行)</p>
          <div className="overflow-x-auto rounded-lg border border-[rgba(0,0,0,0.05)]">
            <table className="w-full" style={{ fontSize: '0.65rem' }}>
              <thead>
                <tr className="bg-[#FAFAFA]">
                  {csvPreview.headers.map((h, i) => (
                    <th key={i} className="px-2 py-1 text-left text-[#A1A1AA] font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {csvPreview.rows.map((row, i) => (
                  <tr key={i} className="border-t border-[rgba(0,0,0,0.03)]">
                    {row.map((cell, j) => (
                      <td key={j} className="px-2 py-0.5 text-[#52525B]">{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 选项 */}
      {csvPreview && (
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#A1A1AA]">目标列:</span>
            <select
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer"
            >
              {csvPreview.numericCols.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#A1A1AA]">预测时长:</span>
            <select
              value={predLen}
              onChange={(e) => setPredLen(Number(e.target.value))}
              className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer"
            >
              {PRED_LEN_OPTIONS.map(n => (
                <option key={n} value={n}>{n}h</option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* 预测按钮 */}
      {csvFile && (
        <button
          onClick={handlePredict}
          disabled={isLoading || !targetCol}
          className="text-xs px-4 py-2 rounded-xl bg-[#3B82F6] text-white hover:bg-[#2563EB] disabled:opacity-50 transition-colors cursor-pointer font-medium"
        >
          {isLoading ? '⏳ 推理中...' : '🚀 开始预测'}
        </button>
      )}

      {/* 结果 */}
      {error && (
        <div className="p-2 rounded-lg bg-[#FEF2F2] border border-[#FECACA] text-xs text-[#991B1B]">❌ {error}</div>
      )}

      {result && (
        <div className="space-y-3">
          <div className="chart-box-lg">
            <Line data={chartData} options={{
              responsive: true, maintainAspectRatio: false,
              plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } },
              scales: {
                x: { title: { display: true, text: '预测时刻 (小时)', color: '#A1A1AA' } },
                y: { title: { display: true, text: '原始值', color: '#A1A1AA' } },
              },
            }} />
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={downloadCSV}
              className="text-xs px-4 py-2 rounded-xl border border-[rgba(0,0,0,0.08)] bg-white hover:bg-stone-50 transition-colors cursor-pointer"
            >
              📥 下载预测 CSV
            </button>
            <span className="text-[0.6rem] text-[#A1A1AA]">
              模型: {result.model} · 输入 {result.meta.input_rows} 行 · 设备: {result.meta.device}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add react-app/src/components/UploadPanel.jsx
git commit -m "feat: UploadPanel with CSV drag-drop, preview, column/duration select, prediction chart"
```

---

### Task 12: 组装 PlaygroundPage

**Files:**
- Modify: `react-app/src/pages/PlaygroundPage.jsx`

- [ ] **Step 1: 替换占位页面为完整版**

```jsx
import { useState } from 'react';
import ModelSelector from '../components/ModelSelector';
import DemoPanel from '../components/DemoPanel';
import UploadPanel from '../components/UploadPanel';

const MODEL_TIERS = {
  '★ BaseModel': 2,
  'Informer': 2, 'LightTS': 2, 'TSMixer': 2, 'SCINet': 2, 'Mamba': 2, 'TimeLLM': 2,
  'IBM TTM': 3,
};

export default function PlaygroundPage() {
  const [selectedModel, setSelectedModel] = useState('★ BaseModel');
  const [channelIdx, setChannelIdx] = useState(1);
  const [activeSection, setActiveSection] = useState('demo'); // 'demo' | 'upload'

  const isAvailable = (MODEL_TIERS[selectedModel] || 1) === 1;

  return (
    <div className="page-enter mt-4">
      <div className="flex gap-4" style={{ minHeight: 'calc(100vh - 140px)' }}>
        {/* 左侧: 模型列表 30% */}
        <div style={{ width: '280px', flexShrink: 0 }}>
          <ModelSelector
            selectedModel={selectedModel}
            onSelect={setSelectedModel}
            modelTiers={MODEL_TIERS}
          />
        </div>

        {/* 右侧: 实验面板 70% */}
        <div className="flex-1 space-y-4">
          <div className="card p-5">
            {/* 子Tab: Demo验证 / 自定义上传 */}
            <div className="flex items-center gap-1 mb-4">
              <button
                onClick={() => setActiveSection('demo')}
                className={`text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${
                  activeSection === 'demo'
                    ? 'bg-[#3B82F6] text-white'
                    : 'bg-[#F5F5F5] text-[#52525B] hover:bg-[#E5E5E5]'
                }`}
              >
                📈 Demo 验证
              </button>
              <button
                onClick={() => setActiveSection('upload')}
                className={`text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${
                  activeSection === 'upload'
                    ? 'bg-[#3B82F6] text-white'
                    : 'bg-[#F5F5F5] text-[#52525B] hover:bg-[#E5E5E5]'
                }`}
              >
                📁 自定义预测
              </button>
            </div>

            {activeSection === 'demo' ? (
              <DemoPanel
                model={selectedModel}
                channelIdx={channelIdx}
                onChangeChannel={setChannelIdx}
                isAvailable={isAvailable}
              />
            ) : (
              <UploadPanel
                model={selectedModel}
                isAvailable={isAvailable}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add react-app/src/pages/PlaygroundPage.jsx
git commit -m "feat: PlaygroundPage with dual-panel layout, model selection, demo/upload tabs"
```

---

### Task 13: 合并曲线+对比 → ComparePage

**Files:**
- Modify: `react-app/src/pages/ComparePage.jsx`

- [ ] **Step 1: 替换占位页面为 Tab 版**

```jsx
import { useState } from 'react';
import CurvesPageContent from './CurvesPage';
import PerformancePageContent from './PerformancePage';

export default function ComparePage() {
  const [tab, setTab] = useState('curves');

  return (
    <div className="page-enter">
      {/* Tab 切换 */}
      <div className="flex items-center gap-1 mt-4 mb-2">
        <button
          onClick={() => setTab('curves')}
          className={`text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${
            tab === 'curves'
              ? 'bg-[#3B82F6] text-white'
              : 'bg-[#F5F5F5] text-[#52525B] hover:bg-[#E5E5E5]'
          }`}
        >
          📈 曲线叠图
        </button>
        <button
          onClick={() => setTab('ranking')}
          className={`text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${
            tab === 'ranking'
              ? 'bg-[#3B82F6] text-white'
              : 'bg-[#F5F5F5] text-[#52525B] hover:bg-[#E5E5E5]'
          }`}
        >
          📊 ACC 排名
        </button>
      </div>

      {tab === 'curves' ? <CurvesPageContent /> : <PerformancePageContent />}
    </div>
  );
}
```

**注意**: CurvesPage 和 PerformancePage 当前是 `export default function CurvesPage()` 形式。需要将内容提取为命名导出。修改 CurvesPage.jsx: `export function CurvesPageContent()` 保留 `export default`；PerformancePage 同理。

- [ ] **Step 2: 修改 CurvesPage.jsx 添加命名导出**

在原 CurvesPage.jsx 中，将 `export default function CurvesPage()` 改为两步：
```jsx
export function CurvesPageContent() { /* 原全部内容 */ }
export default CurvesPageContent;
```

对 PerformancePage.jsx 同样处理: `export function PerformancePageContent()` + `export default PerformancePageContent`。

- [ ] **Step 3: Commit**

```bash
git add react-app/src/pages/ComparePage.jsx react-app/src/pages/CurvesPage.jsx react-app/src/pages/PerformancePage.jsx
git commit -m "feat: ComparePage with tab switching between curves overlay and ACC ranking"
```

---

### Task 14: 配置 Vite Proxy

**Files:**
- Modify: `react-app/vite.config.js`

- [ ] **Step 1: 添加 proxy 配置**

在 `vite.config.js` 的 `defineConfig` 中添加 server.proxy:

```js
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
```

- [ ] **Step 2: 验证 proxy 工作**

启动两个终端:
```
终端1: cd backend && python server.py
终端2: cd react-app && npm run dev
```

浏览器访问 `http://localhost:5173`，打开开发者工具 Network 面板，切换到"模型中心"，观察 `/api/*` 请求是否正确代理到后端。

- [ ] **Step 3: Commit**

```bash
git add react-app/vite.config.js && git commit -m "feat: Vite proxy /api/* to FastAPI backend on port 8000"
```

---

# Phase 3: 补齐 & 修复

---

### Task 15: 训练缺失模型（Informer, LightTS, TSMixer, SCINet）

**Files:**
- Modify: `Time-Series-Library-main/run.py` (间接使用)

- [ ] **Step 1: 逐个训练**

对每个缺失模型，使用 run.py 训练并保存 checkpoint：

```bash
cd Time-Series-Library-main

# Informer
python -c "
from run import run_dl_model; run_dl_model('Informer')
"

# LightTS
python -c "
from run import run_dl_model; run_dl_model('LightTS')
"

# TSMixer
python -c "
from run import run_dl_model; run_dl_model('TSMixer')
"

# SCINet
python -c "
from run import run_dl_model; run_dl_model('SCINet')
"
```

每个训练完成后，将生成的 checkpoint 路径更新到 `backend/model_registry.py`，将 tier 改为 1。

- [ ] **Step 2: 验证新 checkpoint**

```bash
ls checkpoints/ | grep -E "Informer|LightTS|TSMixer|SCINet"
```

Expected: 每个模型一个 checkpoint 文件夹。

- [ ] **Step 3: Commit**

```bash
git add backend/model_registry.py && git commit -m "feat: enable Informer/LightTS/TSMixer/SCINet as Tier 1 after training"
```

---

### Task 16: 修复 IBM TTM

**Files:**
- 可能修改: `backend/inference_engine.py` 的 `_infer_huggingface` 中 TTM 分支

- [ ] **Step 1: 调查并实施修复**

TTM 核心问题：`seq_len=24` vs `context_len=512`，零填充 488 步导致预测平线。

修复方案：使用反射填充（reflect padding）代替零填充，增加历史信息的有效利用。

修改 `inference_engine.py` 中 TTM 分支的填充逻辑：

```python
elif "ttm" in model_id.lower():
    context_len = info.get("context_len", 512)
    pad_len = context_len - seq_len
    # 反射填充代替零填充，保留波动特征
    past = np.pad(X[b], ((pad_len, 0), (0, 0)), mode='reflect')
    # ... 其余不变
```

- [ ] **Step 2: 更新 model_registry，TTM tier 1**

修改 `backend/model_registry.py`，将 IBM TTM 的 tier 改为 1。

- [ ] **Step 3: Commit**

```bash
git add backend/inference_engine.py backend/model_registry.py
git commit -m "fix: IBM TTM reflect padding to address flat prediction issue"
```

---

### Task 17: XGBoost 模型文件持久化

**Files:**
- 新建/修改: `Time-Series-Library-main/models/model_xgboost.py`

- [ ] **Step 1: 训练并保存 XGBoost 模型文件**

```bash
cd Time-Series-Library-main
python models/model_xgboost.py --data_path data_provider/4g_traffic --output_dir ./results --save_model
```

确保 `xgb_model.json` 保存到 `results/XGBoost_n100_d5_lr0.1/` 目录。

- [ ] **Step 2: 验证加载**

```bash
cd backend && python -c "
from model_loader import load
model, info = load('XGBoost')
print('XGBoost loaded:', type(model))
"
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: XGBoost model persistence and loading verification"
```

---

### Task 18: 最终集成测试

- [ ] **Step 1: 全链路测试**

启动前后端，验证：
1. 导航到"模型中心"→ 默认显示 BaseModel 预置曲线
2. 切换模型 → 图表更新
3. 快速验证 → 调用后端，曲线叠加
4. 完整评估 → 返回指标
5. 上传 CSV → 返回预测曲线
6. 下载预测 CSV
7. 切换到"预测对比"→ 曲线叠图 / ACC 排名正常

- [ ] **Step 2: 错误场景测试**

1. 选择 Tier 2 模型 → 显示"暂不支持"
2. 上传非 CSV 文件 → 错误提示
3. 上传空 CSV → 错误提示
4. 后端未启动 → 前端显示友好错误

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "verify: full integration test passed for all flows"
```

---

## 验证清单

完成全部任务后逐项确认：
- [ ] `GET /api/health` 返回 GPU 状态
- [ ] `GET /api/models` 返回 26 模型
- [ ] `POST /api/demo/Naive/quick` 统计模型推理正常
- [ ] `POST /api/demo/PatchTST/quick` PyTorch 推理正常
- [ ] `POST /api/demo/Chronos-tiny/quick` HuggingFace 推理正常
- [ ] `POST /api/predict/{model}` CSV 上传推理正常
- [ ] 前端模型列表可搜索/筛选/选择
- [ ] 预置曲线即时展示
- [ ] 快速验证叠加曲线
- [ ] 完整评估返回指标
- [ ] 自定义上传全流程
- [ ] 预测对比双 Tab 正常
- [ ] 导航 5 标签切换正常
