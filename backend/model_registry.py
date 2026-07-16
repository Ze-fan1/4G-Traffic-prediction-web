"""
26 模型完整注册表 — 每个模型的类型、checkpoint 路径、类别、Tier 等级、运行方式
"""
from pathlib import Path

from project_paths import CHECKPOINTS_DIR, DATA_DIR, RESULTS_DIR, TSLIB_ROOT

MODEL_REGISTRY = {
    # ═══ Statistical — 纯计算，无需模型文件 ═══
    "Naive": {
        "type": "statistical", "method": "naive",
        "tier": 1, "category": "Statistical",
        "run_type": "inference_stat",
        "result_dir": str(RESULTS_DIR / "Naive_4G"),
    },
    "Persistent 24h": {
        "type": "statistical", "method": "persistent",
        "tier": 1, "category": "Statistical",
        "run_type": "inference_stat",
        "result_dir": str(RESULTS_DIR / "Persistent24h_4G"),
    },
    "Historical Avg": {
        "type": "statistical", "method": "historical_avg",
        "tier": 1, "category": "Statistical",
        "run_type": "inference_stat",
        "result_dir": str(RESULTS_DIR / "HistoricalAverage_4G"),
    },
    "AutoARIMA": {
        "type": "statistical", "method": "autoarima",
        "tier": 1, "category": "Statistical",
        "run_type": "inference_stat",
        "result_dir": str(RESULTS_DIR / "AutoARIMA_4G"),
    },
    "AutoAR": {
        "type": "statistical", "method": "autoar",
        "tier": 1, "category": "Statistical",
        "run_type": "inference_stat",
        "result_dir": str(RESULTS_DIR / "AutoAR_4G"),
    },
    "LinearRegression": {
        "type": "statistical", "method": "linear_regression",
        "tier": 1, "category": "Statistical",
        "run_type": "inference_stat",
        "result_dir": str(RESULTS_DIR / "LinearRegression_4G"),
    },

    # ═══ Tree ═══
    "XGBoost": {
        "type": "xgboost",
        "model_file": str(TSLIB_ROOT / "results" / "XGBoost_n100_d5_lr0.1" / "xgb_model.pkl"),
        "weights_file": str(TSLIB_ROOT / "results" / "XGBoost_n100_d5_lr0.1" / "xgb_model.pkl"),
        "tier": 1, "category": "Tree",
        "run_type": "train_xgboost",
        "result_dir": str(RESULTS_DIR / "XGBoost_n100_d5_lr0.1"),
    },

    # ═══ Baseline ═══
    "★ BaseModel": {
        "type": "external_forecast",
        "tier": 1, "category": "Baseline",
        "run_type": "external_base",
        "result_dir": str(RESULTS_DIR / "External_BaseModel_4G_Base_v2_Verify"),
    },

    # ═══ Transformer — 5 个 ═══
    "PatchTST": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "PatchTST_4G_PatchTST_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 128, "d_ff": 256, "n_heads": 4},
        "tier": 1, "category": "Transformer",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "PatchTST_4G_PatchTST_v2_Verify"),
    },
    "iTransformer": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "iTransformer_4G_iTransformer_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 256, "d_ff": 512, "n_heads": 8},
        "tier": 1, "category": "Transformer",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "iTransformer_4G_iTransformer_v2_Verify"),
    },
    "Autoformer": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "Autoformer_4G_Autoformer_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 64, "d_ff": 128, "n_heads": 8, "e_layers": 1},
        "tier": 1, "category": "Transformer",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "Autoformer_4G_Autoformer_v2_Verify"),
    },
    "Transformer": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "Transformer_4G_Transformer_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 512, "d_ff": 2048, "n_heads": 8},
        "tier": 1, "category": "Transformer",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "Transformer_4G_Transformer_v2_Verify"),
    },
    "Informer": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "Informer_4G_Informer_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 128, "d_ff": 256, "n_heads": 4},
        "tier": 1, "category": "Transformer",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "Informer_4G_Informer_v2_Verify"),
    },

    # ═══ MLP ═══
    "DLinear": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "DLinear_4G_DLinear_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 128, "individual": False},
        "tier": 1, "category": "MLP",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "DLinear_4G_DLinear_v2_Verify"),
    },
    "LightTS": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "LightTS_4G_LightTS_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 128},
        "tier": 1, "category": "MLP",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "LightTS_4G_LightTS_v2_Verify"),
    },
    "TSMixer": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "TSMixer_4G_TSMixer_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 128},
        "tier": 1, "category": "MLP",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "TSMixer_4G_TSMixer_v2_Verify"),
    },
    "IBM TTM": {
        "type": "huggingface",
        "model_id": "ibm-granite/granite-timeseries-ttm-r1",
        "context_len": 512,
        "tier": 1, "category": "MLP",
        "run_type": "inference_pretrained",
        "result_dir": str(RESULTS_DIR / "IBM_TTM_ZeroShot_sl24_pl24_step3"),
    },

    # ═══ CNN ═══
    "TimesNet": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "TimesNet_4G_TimesNet_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 64, "d_ff": 128, "num_kernels": 3, "e_layers": 1},
        "tier": 1, "category": "CNN",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "TimesNet_4G_TimesNet_v2_Verify"),
    },
    "SCINet": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "SCINet_4G_SCINet_v2_Verify" / "checkpoint.pth"),
        "args_override": {"d_model": 128},
        "tier": 1, "category": "CNN",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "SCINet_4G_SCINet_v2_Verify"),
    },

    # ═══ RNN ═══
    "SegRNN": {
        "type": "pytorch",
        "checkpoint": str(CHECKPOINTS_DIR / "SegRNN_4G_SegRNN_v2_Verify" / "checkpoint.pth"),
        "args_override": {"seg_len": 24, "d_model": 256},
        "tier": 1, "category": "RNN",
        "run_type": "train_dl",
        "result_dir": str(RESULTS_DIR / "SegRNN_4G_SegRNN_v2_Verify"),
    },

    # ═══ SSM ═══
    "Mamba": {
        "type": "mamba",
        "checkpoint": str(TSLIB_ROOT / "results" / "Mamba_d128_ex2_ds32_dc4_el2" / "checkpoint.pth"),
        "config_file": str(TSLIB_ROOT / "results" / "Mamba_d128_ex2_ds32_dc4_el2" / "model_config.json"),
        "args_override": {"d_model": 128, "d_ff": 32, "expand": 2},
        "tier": 1, "category": "SSM",  # 改为 Tier 1，通过独立脚本训练
        "run_type": "train_mamba",
        "result_dir": str(RESULTS_DIR / "Mamba_d128_ex2_ds32_dc4_el2"),
    },

    # ═══ LLM ═══
    "TimeLLM": {
        "type": "timellm",
        "checkpoint": str(TSLIB_ROOT / "results" / "TimeLLM_gpt2_pl6_s3" / "checkpoint.pth"),
        "config_file": str(TSLIB_ROOT / "results" / "TimeLLM_gpt2_pl6_s3" / "model_config.json"),
        "args_override": {},
        "tier": 1, "category": "LLM",  # 改为 Tier 1，通过独立脚本训练
        "run_type": "train_timellm",
        "result_dir": str(RESULTS_DIR / "TimeLLM_gpt2_pl6_s3"),
    },
    "Chronos2": {
        "type": "huggingface",
        "model_id": "amazon/chronos-2",
        "tier": 1, "category": "LLM",
        "run_type": "inference_pretrained",
        "result_dir": str(RESULTS_DIR / "Chronos2_amazon_chronos-2"),
    },
}


def get_model_info(name: str) -> dict:
    """获取单个模型元数据，不存在返回 None"""
    return MODEL_REGISTRY.get(name)


def resolve_model_checkpoint(name: str, info: dict | None = None) -> str | None:
    """Find the configured or most recent checkpoint from an earlier local run."""
    info = info or MODEL_REGISTRY.get(name) or {}
    configured = info.get("checkpoint")
    if configured and Path(configured).exists():
        return str(Path(configured))
    candidates = list(CHECKPOINTS_DIR.glob(f"{name}_4G_*/checkpoint.pth"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def list_models(tier_filter: int = None) -> list:
    """列出所有模型，可选按 tier 过滤"""
    result = []
    for name, info in MODEL_REGISTRY.items():
        entry = {
            "name": name,
            "category": info["category"],
            "tier": info["tier"],
            "type": info["type"],
            "run_type": info.get("run_type", "unknown"),
            "available": info["tier"] == 1,
        }
        if tier_filter is None or info["tier"] == tier_filter:
            result.append(entry)
    return result
