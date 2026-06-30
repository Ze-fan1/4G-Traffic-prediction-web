# Playground（模型中心）设计规范

> 日期: 2026-06-30 | 状态: 待审批

## 1. 概述

在现有 React 前端基础上新增 **"模型中心"** 页面，并在项目内新建 **FastAPI 后端**，使用户能够：
1. 浏览全部 26 个模型并查看预置预测曲线
2. 对选中模型发起实时推理验证（快速/完整两档）
3. 上传自定义 CSV 时间序列，运行模型预测

导航调整：`首页 → 模型中心 → 预测对比 → 误差分析 → 详细报告`
- "模型中心"：单模型深度实验（本规范主目标）
- "预测对比"：合并原「预测曲线」叠图 + 「模型对比」ACC 排名（本规范次要目标）

---

## 2. 模型就绪度分析

### 2.1 分层统计

| 层级 | 数量 | 说明 |
|------|------|------|
| Tier 1 — 可立即提供实时推理 | ~17 | 有 checkpoint 或简单统计算法 |
| Tier 2 — 仅展示预置数据 | ~6 | 缺少 checkpoint，需额外训练 |
| Tier 3 — 不适用/不可用 | ~3 | IBM TTM(待修复)、特殊模型 |

### 2.2 逐模型详情

**Tier 1（可实时推理）：**

| 模型 | 推理方式 | 备注 |
|------|----------|------|
| Naive | NumPy | 重复最后一个值 |
| Persistent 24h | NumPy | 24h 前值 |
| Historical Avg | NumPy | 历史均值 |
| AutoARIMA | NumPy/statsmodels | 逐窗口拟合 |
| AutoAR | NumPy/statsmodels | 逐窗口拟合 |
| LinearRegression | NumPy/sklearn | 逐窗口拟合 |
| XGBoost | xgboost.predict | 需保存/加载模型文件 |
| BaseModel | PyTorch | 自研模型，检查点待确认 |
| PatchTST | PyTorch | checkpoint.pth ✓ |
| iTransformer | PyTorch | checkpoint.pth ✓ |
| Autoformer | PyTorch | checkpoint.pth ✓ |
| Transformer | PyTorch | checkpoint.pth ✓ |
| DLinear | PyTorch | checkpoint.pth ✓ |
| TimesNet | PyTorch | checkpoint.pth ✓ |
| SegRNN | PyTorch | checkpoint.pth ✓ |
| Chronos-tiny | HuggingFace | amazon/chronos-t5-tiny |
| Chronos-small | HuggingFace | amazon/chronos-t5-small |
| Chronos-base | HuggingFace | amazon/chronos-t5-base |
| Chronos2 | HuggingFace | amazon/chronos-2 |

**Tier 2（缺少 checkpoint，需训练）：**

| 模型 | 原因 | 预计修复 |
|------|------|----------|
| Informer | 无 checkpoint | 需训练 ~1h |
| LightTS | 无 checkpoint | 需训练 ~30min |
| TSMixer | 无 checkpoint | 需训练 ~1h |
| SCINet | 无 checkpoint | 需训练 ~1h |
| Mamba | 独立脚本运行，无 .pth | 需适配加载逻辑 |
| TimeLLM | 无 checkpoint，依赖 GPT-2 | 需训练 + 特殊加载 |

**Tier 3（需特殊处理）：**

| 模型 | 问题 | 处理 |
|------|------|------|
| IBM TTM | 512 上下文 vs 24 步输入 | 待修复后归入 Tier 1 |
| TiDE | 有 checkpoint 但不在 26 模型列表 | 不展示 |

### 2.3 模型列表数据结构

```python
# backend/model_registry.py
MODEL_REGISTRY = {
    "PatchTST": {
        "type": "pytorch",
        "checkpoint": "checkpoints/PatchTST_4G_PatchTST_v2_Verify/checkpoint.pth",
        "args_override": {"d_model": 128, "d_ff": 256, "n_heads": 4},
        "tier": 1,
        "category": "Transformer",
    },
    "Chronos2": {
        "type": "huggingface",
        "model_id": "amazon/chronos-2",
        "tier": 1,
        "category": "LLM",
    },
    "Naive": {
        "type": "statistical",
        "method": "naive",
        "tier": 1,
        "category": "Statistical",
    },
    # ... 全部 26 个
}
```

---

## 3. 架构

```
react-app/ (Vite :5173)
  │
  ├─ vite.config.js proxy: /api/* → localhost:8000
  │
  └─ fetch ──→ backend/ (FastAPI :8000)
                 ├── server.py              # 路由定义
                 ├── model_registry.py      # 26 模型元数据 + 路由表
                 ├── model_loader.py        # 按需加载，自动识别类型
                 ├── inference_engine.py    # 统一推理入口，按类型分发
                 ├── data_pipeline.py       # CSV解析 + 标准化 + 窗口构建
                 └── requirements.txt
```

**本地启动（两个终端）：**
```
终端1: cd react-app && npm run dev
终端2: cd backend && python server.py
```

---

## 4. 后端 API

### 4.1 路由表

| 方法 | 路径 | 请求体 | 响应 | 说明 |
|------|------|--------|------|------|
| GET | `/api/models` | — | `[{name, cat, metrics, tier, available}]` | 26 模型清单 |
| GET | `/api/health` | — | `{status, gpu, loaded_model}` | 健康检查 |
| POST | `/api/demo/{model}/quick` | `{channel_idx: int}` | `{pred: [24], truth: [24], window: int}` | 单窗口验证 |
| POST | `/api/demo/{model}/full` | `{channel_idx: int}` | `{metrics: {...}, elapsed_s: float}` | 全量评估 |
| POST | `/api/predict/{model}` | FormData: csv_file, target_col, pred_len | `{predictions: [[...]], meta: {...}}` | 自定义预测 |

### 4.2 模型加载策略

- 全局单例 `ModelCache`，同一时刻内存中只有一个模型
- 请求模型 A：命中缓存直接用；未命中则 `unload()` → `load(A)`
- `unload()`：`del model; torch.cuda.empty_cache()`
- Tier 2 模型请求时返回 `{"error": "model_not_available", "tier": 2}`
- HuggingFace 模型首次下载后本地缓存

### 4.3 推理分发

```
inference_engine.infer(model_name, data, params)
  │
  ├─ type=="pytorch"      → 加载 .pth → model.eval() → forward
  ├─ type=="huggingface"  → AutoModel.from_pretrained → generate
  ├─ type=="statistical"  → 纯 NumPy/sklearn 计算
  └─ type=="xgboost"      → xgb.Booster.load → predict
```

### 4.4 预测时长（pred_len）处理

- 统计模型：原生支持任意 pred_len
- HuggingFace：`model.generate(prediction_length=pred_len)` 原生支持
- PyTorch：始终推理 24h 输出，按用户选择 `predictions[:pred_len]` 切片
- API 响应 `meta.actual_pred_len` 始终为 24，`meta.requested_pred_len` 为用户选择值

---

## 5. 前端设计

### 5.1 新增/修改文件

```
react-app/src/
├── pages/
│   ├── PlaygroundPage.jsx    # 新增：「模型中心」
│   └── ComparePage.jsx        # 新增：「预测对比」(合并曲线+对比)
├── components/
│   ├── ModelSelector.jsx      # 新增：左侧模型列表
│   ├── DemoPanel.jsx          # 新增：初始展示 + 验证区
│   └── UploadPanel.jsx        # 新增：上传预测区
├── App.jsx                    # 修改：加路由
└── Header.jsx                 # 修改：加/改导航标签
```

### 5.2 页面布局

双栏布局 (30% + 70%)：

```
┌──────────────────────────────────────────────────────┐
│  导航栏: [首页] [★模型中心] [预测对比] [误差分析] [详细报告] │
├──────────────┬───────────────────────────────────────┤
│              │                                       │
│  模型列表     │  右侧面板                              │
│  ┌──────────┐│  ┌── 初始展示（预置数据）────────┐     │
│  │🔍搜索    ││  │ 模型: PatchTST | 数据: test_100│     │
│  │          ││  │ 通道: [PDCCH ▼]              │     │
│  │▸Baseline ││  │ 输入:24h → 预测:24h          │     │
│  │ ★BaseM.. ││  │                              │     │
│  │▸Transf.. ││  │    ┌── Chart.js 曲线 ──┐      │     │
│  │ PatchTST◀││  │    │ 黑线=真值 彩线=预测│      │     │
│  │ iTrans.. ││  │    └──────────────────┘      │     │
│  │▸MLP      ││  └──────────────────────────────┘     │
│  │▸CNN      ││  ┌── 验证区域 ─────────────────┐     │
│  │▸RNN      ││  │ [🔄快速验证] [📊完整评估]    │     │
│  │▸SSM      ││  │ 结果: ✅ MAE=0.042 (与预置一致)│    │
│  │▸LLM      ││  └──────────────────────────────┘     │
│  │▸Stats    ││  ┌── 上传区域 ─────────────────┐     │
│  │▸Tree     ││  │ [📁上传CSV] 目标列:[▼] 时长:[24h▼]│  │
│  └──────────┘│  │ [🚀开始预测] [📥下载结果]    │     │
│              │  └──────────────────────────────┘     │
└──────────────┴───────────────────────────────────────┘
```

### 5.3 组件行为

**ModelSelector：**
- 按类别折叠/展开，默认展开 Baseline + Transformer
- 搜索框模糊过滤模型名
- 每行：色点 + 模型名 + 迷你 ACC
- 当前选中：蓝色左边框高亮
- Tier 2 模型：灰色文字 + 🔒图标，hover 提示"暂不支持实时推理"
- 点击 Tier 2 模型仍可看预置曲线，但验证/上传区显示"该模型暂不支持实时推理"

**DemoPanel（初始展示）：**
- 页面加载默认选中 BaseModel
- 图表数据从前端 `prediction_curves.js` 读取（瞬时渲染）
- 通道下拉切换 8 个 KPI 指标
- 图表下方显示当前窗口 MAE、数据来源等元信息

**DemoPanel（验证区）：**
- 快速验证：POST → 后端跑 1 窗口 → 图表上叠加「实时预测」vs「预置预测」双线对比
- 完整评估：POST → 后端跑全量 → 表格展示 MSE/MAE/RMSE/MAPE/ACC + 与预置值对比
- Loading 态显示进度条 + 预计耗时

**UploadPanel：**
- 拖拽或点击上传 CSV
- 上传后显示前 5 行预览 + 自动识别的列名
- 目标列下拉：自动列出数值列
- 预测时长：6h / 12h / 18h / 24h
- 提交后后端推理 → 新图表展示预测结果
- 下载按钮导出预测 CSV

### 5.4 「预测对比」页面（合并）

```
┌──────────────────────────────────────┐
│  Tab1: 曲线叠图  |  Tab2: ACC 排名   │
├──────────────────────────────────────┤
│  Tab1 = 原 CurvesPage 全部内容       │
│  Tab2 = 原 PerformancePage 全部内容  │
└──────────────────────────────────────┘
```

- 两个 Tab 切换，无信息丢失
- 共享同一套导航位置

### 5.5 状态管理

PlaygroundPage 顶层状态：

```js
state = {
  selectedModel: "★ BaseModel",   // 当前选中
  channelIdx: 1,                  // 通道
  predLen: 24,                    // 预测时长
  mode: "demo",                   // "demo" | "upload"
  // 验证结果
  quickResult: null,              // {pred, truth, window, mae}
  fullResult: null,               // {metrics, elapsed_s}
  isVerifying: false,
  // 上传
  csvFile: null,
  csvPreview: null,               // {headers, rows, numericCols}
  targetCol: null,
  uploadResult: null,             // {predictions, meta}
  isPredicting: false,
}
```

---

## 6. 数据管道

### 6.1 Demo 验证数据流

```
前端请求 {model: "PatchTST", channel_idx: 1, mode: "quick"}
  → server.py 路由到 demo_quick()
  → model_loader.load("PatchTST")  # 加载 .pth
  → data_pipeline.load_test_window(window_idx, channel_idx)
     ├── 读取 df_4g_test_100.parquet
     ├── StandardScaler (在训练集上拟合)
     ├── 提取 seq_len=24 输入 → X
     └── 提取 pred_len=24 真值 → Y
  → inference_engine.infer(model, X)
     → model(X).detach().numpy() → pred (24, 8)
  → scaler.inverse_transform(pred) → 原始空间
  → 通道重排（修复 6/7 交换）
  → 返回 {pred[channel], truth[channel], window_idx}
```

### 6.2 自定义上传数据流

```
前端上传 CSV + target_col + pred_len
  → server.py 路由到 predict()
  → data_pipeline.parse_csv(file)
     ├── 自动检测分隔符
     ├── 识别数值列
     ├── 要求行数 ≥ seq_len + pred_len (最少 30 行)
     └── 返回 DataFrame + 列元数据
  → data_pipeline.build_windows(df, target_col, seq_len, pred_len, step)
     ├── 对全表做 StandardScaler（如果多列）
     ├── 滑动窗口构建 (seq_len + pred_len) 切片
     └── 最后 1 个窗口作为预测输入
  → model_loader.load(model_name)
  → inference_engine.infer(model, X_windows)
  → scaler.inverse_transform(predictions)
  → 返回 {predictions: [[t1, t2, ... t24]], meta: {n_windows, target_col, pred_len}}
```

---

## 7. 错误处理

| 场景 | HTTP 码 | 响应 |
|------|---------|------|
| 模型不在注册表 | 404 | `{"error": "unknown_model", "available": [...]}` |
| Tier 2 模型请求实时推理 | 503 | `{"error": "model_not_available", "tier": 2, "reason": "..."}` |
| CSV 行数不足 | 400 | `{"error": "insufficient_data", "min_rows": 30, "actual": N}` |
| CSV 无有效数值列 | 400 | `{"error": "no_numeric_columns"}` |
| 模型加载失败 | 500 | `{"error": "model_load_failed", "detail": "..."}` |
| GPU 不可用 | 200 | 自动回退 CPU，`meta.device: "cpu"` |
| HuggingFace 下载失败 | 503 | `{"error": "model_download_failed", "retry_hint": "..."}` |

---

## 8. 实施阶段

### Phase 1: 后端核心（优先级最高）
1. 搭建 FastAPI 骨架（server.py + requirements.txt）
2. 实现 `model_registry.py`（26 模型元数据）
3. 实现 `data_pipeline.py`（CSV 解析 + 标准化 + 窗口）
4. 实现 `model_loader.py` + `inference_engine.py`（PyTorch + 统计 + HF 三条路径）
5. 实现 `/api/demo/{model}/quick` 和 `/api/demo/{model}/full`
6. 实现 `/api/predict/{model}`（上传推理）
7. 手动测试所有 Tier 1 模型

### Phase 2: 前端
1. 新增导航标签：`模型中心` + `预测对比`
2. 实现 `ModelSelector.jsx`
3. 实现 `DemoPanel.jsx`（初始展示 + 验证区）
4. 实现 `UploadPanel.jsx`
5. 实现 `PlaygroundPage.jsx`（组装）
6. 合并 CurvesPage + PerformancePage → `ComparePage.jsx`
7. Vite proxy 配置

### Phase 3: 补齐 & 修复
1. 训练缺失模型（Informer/LightTS/TSMixer/SCINet）
2. 修复 IBM TTM（512 上下文适配）
3. Mamba/TimeLLM 推理适配
4. XGBoost 模型文件保存/加载

---

## 9. 风险 & 待确认

| 风险 | 影响 | 缓解 |
|------|------|------|
| HuggingFace 模型需联网下载 | 首次冷启动慢 | 文档说明 + 预下载脚本 |
| GPU 显存不足（RTX 5050） | 大模型加载失败 | 按需加载策略 |
| IBM TTM 修复失败 | 1 个模型不可用 | 降级为 Tier 3 不展示 |
| 统计模型推理慢（逐窗口 ARIMA） | 完整评估耗时长 | 显示进度 + 可中断 |
| CSV 格式千变万化 | 解析失败率高 | 严格校验 + 友好错误提示 |
