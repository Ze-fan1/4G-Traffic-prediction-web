"""Run the six special playground models locally with real progress output."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmark_artifacts import write_benchmark_artifact
from four_g_protocol import FEATURE_COLS, build_windows, fit_training_scaler, load_observations
from inference_engine import _infer_statistical, infer
from model_registry import MODEL_REGISTRY
from project_paths import TSLIB_ROOT

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODELS_DIR = TSLIB_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))


def _data(max_windows: int | None = None):
    print("DATA_LOAD_START", flush=True)
    train = load_observations("train")
    test = load_observations("test")
    scaler = fit_training_scaler(train)
    train_x, train_y, _ = build_windows(train, scaler, step=3)
    test_x, test_y, refs = build_windows(test, scaler, step=3)
    if max_windows is not None:
        test_x, test_y, refs = test_x[:max_windows], test_y[:max_windows], refs[:max_windows]
    print(f"DATA_LOAD_DONE:{len(train_x)}:{len(test_x)}", flush=True)
    return scaler, train_x, train_y, test_x, test_y, refs


def _result_dir(model_name: str, output_root: str | None) -> Path:
    return Path(output_root) / model_name if output_root else Path(MODEL_REGISTRY[model_name]["result_dir"])


def _save(model_name: str, prediction, truth, refs, scaler, run_kind: str, output_root: str | None):
    manifest = write_benchmark_artifact(
        _result_dir(model_name, output_root), model_name,
        np.asarray(prediction, dtype=np.float32), np.asarray(truth, dtype=np.float32), refs,
        run_kind=run_kind, scaler=scaler,
    )
    metrics = manifest["metrics"]
    print("METRICS:" + json.dumps(metrics), flush=True)


def run_statistical(model_name: str, max_windows: int | None, output_root: str | None):
    method = MODEL_REGISTRY[model_name]["method"]
    scaler, _, _, test_x, test_y, refs = _data(max_windows)
    total = len(test_x)
    print(f"INFERENCE_START:{total}", flush=True)
    prediction = []
    for index, window in enumerate(test_x, start=1):
        forecast = _infer_statistical(method, window[None, :, :], 24)[0]
        if not np.isfinite(forecast).all():
            raise RuntimeError(f"{model_name} produced a non-finite forecast at window {index}")
        prediction.append(forecast)
        print(f"INFERENCE_PROGRESS:{index}:{total}", flush=True)
    _save(model_name, prediction, test_y, refs, scaler, "local_statistical_refit", output_root)


def run_xgboost(max_windows: int | None, output_root: str | None, train_windows: int | None, estimators: int):
    import xgboost as xgb

    model_name = "XGBoost"
    scaler, train_x, train_y, test_x, test_y, refs = _data(max_windows)
    x_train = train_x.reshape(len(train_x), -1)
    y_train = train_y.reshape(len(train_y), -1)
    if train_windows is not None:
        x_train, y_train = x_train[:train_windows], y_train[:train_windows]
    total_models = y_train.shape[1]
    print(f"TRAIN_START:{total_models}:{total_models}", flush=True)
    models = []
    for output in range(total_models):
        model = xgb.XGBRegressor(
            n_estimators=estimators, max_depth=5, learning_rate=0.1,
            objective="reg:squarederror", verbosity=0, n_jobs=-1,
        )
        model.fit(x_train, y_train[:, output])
        models.append(model)
        print(f"TRAIN_PROGRESS:{output + 1}:{total_models}:{output + 1}:{total_models}:0", flush=True)

    result_dir = _result_dir(model_name, output_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "xgb_model.pkl").open("wb") as handle:
        pickle.dump({
            "models": models, "seq_len": 24, "pred_len": 24,
            "num_channels": len(FEATURE_COLS), "feature_order": list(FEATURE_COLS),
        }, handle)

    total = len(test_x)
    print(f"INFERENCE_START:{total}", flush=True)
    flat_test = test_x.reshape(total, -1)
    prediction_flat = np.empty((total, total_models), dtype=np.float32)
    for output, model in enumerate(models):
        prediction_flat[:, output] = model.predict(flat_test)
        if (output + 1) % 8 == 0 or output + 1 == total_models:
            done = max(1, min(total, round(total * (output + 1) / total_models)))
            print(f"INFERENCE_PROGRESS:{done}:{total}", flush=True)
    prediction = prediction_flat.reshape(total, 24, len(FEATURE_COLS))
    _save(model_name, prediction, test_y, refs, scaler, "local_xgboost_training", output_root)


def _train_torch(model, train_x, train_y, epochs: int, batch_size: int, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    criterion = torch.nn.MSELoss()
    total_steps = epochs * len(loader)
    print(f"TRAIN_START:{epochs}:{total_steps}", flush=True)
    step = 0
    model.to(device).train()
    for epoch in range(1, epochs + 1):
        for batch_x, batch_y in loader:
            step += 1
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 20 == 0 or step == total_steps:
                print(f"TRAIN_PROGRESS:{epoch}:{epochs}:{step}:{total_steps}:{loss.item():.8f}", flush=True)


def _infer_torch(model, test_x, device):
    loader = DataLoader(torch.from_numpy(test_x), batch_size=64, shuffle=False)
    total = len(test_x)
    print(f"INFERENCE_START:{total}", flush=True)
    prediction = []
    done = 0
    model.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch.float().to(device)).detach().cpu().numpy()
            prediction.append(output)
            done += len(output)
            print(f"INFERENCE_PROGRESS:{done}:{total}", flush=True)
    return np.concatenate(prediction, axis=0)


def run_mamba(max_windows: int | None, output_root: str | None, train_windows: int | None, epochs: int):
    from model_mamba import MambaModel

    model_name = "Mamba"
    scaler, train_x, train_y, test_x, test_y, refs = _data(max_windows)
    config = {
        "enc_in": len(FEATURE_COLS), "d_model": 128, "d_inner": 256,
        "d_state": 32, "d_conv": 4, "e_layers": 2, "pred_len": 24,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaModel(**config)
    if train_windows is not None:
        train_x, train_y = train_x[:train_windows], train_y[:train_windows]
    _train_torch(model, train_x, train_y, epochs=epochs, batch_size=16, device=device)
    result_dir = _result_dir(model_name, output_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), result_dir / "checkpoint.pth")
    (result_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("CHECKPOINT_SAVED", flush=True)
    prediction = _infer_torch(model, test_x, device)
    _save(model_name, prediction, test_y, refs, scaler, "local_mamba_training", output_root)


def run_timellm(max_windows: int | None, output_root: str | None, train_windows: int | None, epochs: int):
    from model_timellm import TimeLLM_Model

    model_name = "TimeLLM"
    scaler, train_x, train_y, test_x, test_y, refs = _data(max_windows)
    config = {
        "seq_len": 24, "pred_len": 24, "num_channels": len(FEATURE_COLS),
        "llm_name": "gpt2", "patch_len": 6, "stride": 3,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MODEL_LOAD_START", flush=True)
    model = TimeLLM_Model(**config)
    # A reproducible subset per epoch keeps the local run practical while
    # still fitting the trainable TimeLLM layers from scratch.
    rng = np.random.default_rng(2026)
    sample_count = min(train_windows or 2000, len(train_x))
    selection = rng.choice(len(train_x), size=sample_count, replace=False)
    _train_torch(model, train_x[selection], train_y[selection], epochs=epochs, batch_size=4, device=device)
    result_dir = _result_dir(model_name, output_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), result_dir / "checkpoint.pth")
    (result_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("CHECKPOINT_SAVED", flush=True)
    prediction = np.clip(_infer_torch(model, test_x, device), -10, 10)
    _save(model_name, prediction, test_y, refs, scaler, "local_timellm_training", output_root)


def run_chronos(max_windows: int | None, output_root: str | None):
    model_name = "Chronos2"
    scaler, _, _, test_x, test_y, refs = _data(max_windows)
    total = len(test_x)
    print("MODEL_LOAD_START", flush=True)
    print(f"INFERENCE_START:{total}", flush=True)
    prediction = []
    for index, window in enumerate(test_x, start=1):
        prediction.append(infer(model_name, window, pred_len=24))
        print(f"INFERENCE_PROGRESS:{index}:{total}", flush=True)
    _save(model_name, prediction, test_y, refs, scaler, "local_chronos2_inference", output_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("XGBoost", "AutoARIMA", "AutoAR", "Mamba", "Chronos2", "TimeLLM"))
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--train-windows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--estimators", type=int, default=100)
    args = parser.parse_args()
    if args.model in ("AutoARIMA", "AutoAR"):
        run_statistical(args.model, args.max_windows, args.output_root)
    elif args.model == "XGBoost":
        run_xgboost(args.max_windows, args.output_root, args.train_windows, args.estimators)
    elif args.model == "Mamba":
        run_mamba(args.max_windows, args.output_root, args.train_windows, args.epochs or 10)
    elif args.model == "TimeLLM":
        run_timellm(args.max_windows, args.output_root, args.train_windows, args.epochs or 5)
    else:
        run_chronos(args.max_windows, args.output_root)


if __name__ == "__main__":
    main()
