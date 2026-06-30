"""
26 模型完整注册表 — 每个模型的类型、checkpoint 路径、类别、Tier 等级
"""
import os
from pathlib import Path

TSLIB_ROOT = Path(__file__).resolve().parent.parent / ".." / "网络流量预测项目新修改2" / "Time-Series-Library-main"
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
        "tier": 1, "category": "MLP",
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
