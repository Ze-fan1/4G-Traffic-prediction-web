"""
模型加载器 — 按需加载，全局单例缓存，自动识别模型类型
"""
import sys
import torch
import numpy as np
from pathlib import Path
from model_registry import get_model_info

TSLIB_ROOT = Path(__file__).resolve().parent.parent / ".." / "网络流量预测项目新修改2" / "Time-Series-Library-main"
TSLIB_ROOT = TSLIB_ROOT.resolve()
MODELS_DIR = TSLIB_ROOT / "models"

sys.path.insert(0, str(TSLIB_ROOT))
sys.path.insert(0, str(MODELS_DIR))  # utils/ 在 models/utils/ 中

import os as _os
_os.chdir(str(TSLIB_ROOT))  # exp_basic 使用相对路径 models/ 扫描模型文件

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
        import pickle
        model_path = info.get("model_file")
        # Try pickle first, then xgb native, then fallback
        if model_path:
            pkl_path = model_path.replace('.json', '.pkl')
            if Path(pkl_path).exists():
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                model = data['models']  # dict of per-channel models
            elif Path(model_path).exists():
                model = xgb.Booster()
                model.load_model(model_path)
            else:
                from data_pipeline import load_test_data
                df_train, _, _, cols = load_test_data()
                X_train = df_train[cols].values
                model = _train_xgboost_quick(X_train, cols)
        else:
            from data_pipeline import load_test_data
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
    models = {}
    for i, col in enumerate(cols):
        dtrain = xgb.DMatrix(X[:-1, i:i+1], label=X[1:, i])
        params = {"max_depth": 5, "eta": 0.1, "objective": "reg:squarederror", "verbosity": 0}
        models[col] = xgb.train(params, dtrain, num_boost_round=100)
    return models
