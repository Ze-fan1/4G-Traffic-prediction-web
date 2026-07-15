"""
FastAPI 后端 — 4G Traffic Model Playground
功能: 统一运行(训练/推理)全部 23 模型 + 预置曲线 + 数据上传预测
"""
import sys, os, time, threading, uuid, subprocess, re, json
from pathlib import Path
from project_paths import TSLIB_ROOT as TSLIB
if str(TSLIB) not in sys.path:
    sys.path.insert(0, str(TSLIB))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from model_registry import get_model_info, MODEL_REGISTRY
from data_pipeline import parse_file, build_4g_window_from_upload, build_windows_from_csv, FEATURE_COLS, NUM_CHANNELS
from four_g_protocol import build_windows, fit_training_scaler, load_observations
from model_loader import get_loaded_model_name, get_device, unload
from inference_engine import infer
from preset_curves import load_preset_curves, get_available_preset_models
from benchmark_artifacts import write_benchmark_artifact
from generic_forecast import SUPPORTED_METHODS, forecast as generic_forecast

app = FastAPI(title="4G Traffic Playground API", version="0.4.0")

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://ze-fan1.github.io"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 全局异常处理 — 防止未捕获异常导致 HTTP 500 无响应
from fastapi.responses import JSONResponse
from fastapi import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = traceback.format_exc()
    # 只打印前1000字符避免日志爆炸
    print(f"[ERROR] 未捕获异常: {exc}\n{tb[:1000]}", flush=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)[:200]},
    )

# ═══ Job system ═══
_jobs = {}
_jobs_lock = threading.Lock()
_train_procs = {}
_JOB_MAX_AGE_SEC = 3600  # 1小时后自动清理已完成的 job
_JOB_CLEANUP_INTERVAL = 300  # 每5分钟清理一次

SEQ_LEN = 24
PRED_LEN = 24
STEP = 3


def _parse_progress_line(line):
    """解析训练输出,返回 dict 或 None"""
    m = re.search(r'iters:\s*(\d+),\s*epoch:\s*(\d+)\s*\|\s*loss:\s*([\d.]+)', line)
    if m:
        return {"batch": int(m.group(1)), "epoch": int(m.group(2)), "loss": float(m.group(3))}
    m = re.search(r'Epoch:\s*(\d+).*cost time:\s*([\d.]+)', line)
    if m:
        return {"epoch_done": int(m.group(1)), "epoch_time": round(float(m.group(2)), 1)}
    m = re.search(r'Train Loss:\s*([\d.]+)\s+Vali Loss:\s*([\d.]+)\s+Test Loss:\s*([\d.]+)', line)
    if m:
        return {"metrics": {"train_loss": round(float(m.group(1)), 6),
                             "val_loss": round(float(m.group(2)), 6),
                             "test_loss": round(float(m.group(3)), 6)}}
    # 新格式：METRICS:{"mse":...,"mae":..., ...}
    if 'METRICS:' in line:
        try:
            d = json.loads(line.split('METRICS:', 1)[1])
            return {"metrics": d}
        except: pass
    return None


def _cleanup_old_jobs():
    """定期清理已完成的旧 job，防止内存泄漏"""
    while True:
        time.sleep(_JOB_CLEANUP_INTERVAL)
        now = time.time()
        with _jobs_lock:
            stale = [jid for jid, j in _jobs.items()
                     if j.get("status") in ("done", "error", "cancelled")
                     and now - j.get("_created_at", now) > _JOB_MAX_AGE_SEC]
            for jid in stale:
                del _jobs[jid]

# 启动后台清理线程
threading.Thread(target=_cleanup_old_jobs, daemon=True).start()


# ═══ Health ═══
@app.get("/api/health")
async def health():
    import torch
    gpu_mem_mb = 0
    if torch.cuda.is_available():
        gpu_mem_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)
    return {
        "status": "ok", "gpu": str(get_device()),
        "loaded_model": get_loaded_model_name(),
        "gpu_memory_mb": gpu_mem_mb,
        "active_jobs": len(_jobs),
    }


@app.post("/api/unload")
async def api_unload():
    """手动卸载当前模型释放显存"""
    try:
        unload()
        return {"status": "ok", "message": "模型已卸载，显存已释放"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══ Model List ═══
@app.get("/api/models")
async def api_list_models():
    result = []
    for name, info in MODEL_REGISTRY.items():
        verified = load_preset_curves(name) is not None
        result.append({
            "name": name, "category": info["category"], "tier": info["tier"],
            "type": info["type"], "run_type": info.get("run_type", "unknown"),
            "verified": verified,
            "available": info["tier"] == 1,
            "tier_reason": info.get("tier2_reason", ""),
        })
    return result


# ═══ Preset Curves ═══
@app.get("/api/preset-curves/{model_name}")
async def api_preset_curves(model_name: str, window: int = 0):
    """Return a validated panel-benchmark curve by its reproducible index."""
    info = get_model_info(model_name)
    if info is None:
        raise HTTPException(404, detail={"error": "unknown_model"})
    idx = max(0, window)
    data = load_preset_curves(model_name, window_idx=idx)
    if data is None:
        return {"model": model_name, "curves": None, "available": False}
    return {"model": model_name, **data, "available": True}


@app.get("/api/preset-models")
async def api_preset_models():
    """列出有预置曲线的模型"""
    return {"models": get_available_preset_models()}


@app.get("/api/benchmark-models")
async def api_benchmark_models():
    """Return only models with an auditable panel-benchmark artifact."""
    models = []
    for name in get_available_preset_models():
        curve = load_preset_curves(name)
        info = get_model_info(name)
        if curve is None or info is None:
            continue
        models.append({
            "model": name,
            "cat": info["category"],
            "metrics": curve["metrics_summary"],
            "protocol": curve["protocol"],
            "experiment_id": curve.get("experiment_id"),
        })
    return {"models": models, "protocol": "4g-panel-v1"}


# ═══ Unified Run (训练/推理) ═══
@app.post("/api/run/{model_name}")
async def run_start(model_name: str):
    """统一运行端点 — 根据 run_type 分发给不同 worker"""
    info = get_model_info(model_name)
    if info is None:
        raise HTTPException(404, detail={"error": "unknown_model"})
    if info["tier"] >= 2:
        raise HTTPException(400, detail={"error": "model_not_available", "tier": info["tier"]})

    run_type = info.get("run_type", "unknown")
    job_id = str(uuid.uuid4())[:8]

    if run_type == "train_dl":
        # PyTorch DL: 从零训练 + 生成曲线
        return _start_dl_training(model_name, job_id, info)

    elif run_type in ("inference_stat", "inference_pretrained"):
        # 统计模型 / 预训练模型: 直接推理验证集
        return _start_inference_run(model_name, job_id, info, run_type)

    elif run_type in ("train_xgboost", "train_mamba", "train_timellm", "external_base"):
        # 独立脚本训练
        return _start_script_training(model_name, job_id, info, run_type)

    else:
        raise HTTPException(400, detail={"error": "unsupported_run_type", "run_type": run_type})


def _start_dl_training(model_name, job_id, info):
    """PyTorch DL 训练: spawn train_runner.py"""
    runner_path = Path(__file__).parent / "train_runner.py"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(runner_path), model_name, job_id],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    with _jobs_lock:
        _jobs[job_id] = {"status": "training", "progress": 0, "total": 0, "epoch": 0,
                          "total_epochs": 0, "loss": 0, "metrics": None, "curves": None,
                          "model": model_name, "run_type": "train_dl",
                          "_created_at": time.time()}
        _train_procs[job_id] = proc

    def _read_progress():
        try:
            while True:
                line = proc.stdout.readline()
                if not line: break
                line = line.strip()
                if not line: continue

                with _jobs_lock:
                    parsed = _parse_progress_line(line)
                    if parsed:
                        if "batch" in parsed:
                            _jobs[job_id]["progress"] = parsed["batch"]
                            _jobs[job_id]["epoch"] = parsed["epoch"]
                            _jobs[job_id]["loss"] = parsed["loss"]
                        elif "epoch_done" in parsed:
                            _jobs[job_id]["epoch"] = parsed["epoch_done"]
                            _jobs[job_id]["epoch_time"] = parsed["epoch_time"]
                        elif "metrics" in parsed:
                            _jobs[job_id]["metrics"] = parsed["metrics"]
                        continue

                    if 'TRAIN_START:' in line:
                        parts = line.split(':')
                        for i, p in enumerate(parts):
                            if p == 'EPOCHS' and i + 1 < len(parts):
                                _jobs[job_id]["total_epochs"] = int(parts[i + 1])
                                _jobs[job_id]["total"] = 3007 * int(parts[i + 1])
                        continue

                    if 'CURVES:' in line:
                        try:
                            curves_data = json.loads(line.split('CURVES:', 1)[1])
                            # 提取元数据（窗口号等）
                            meta = curves_data.pop("_meta", None)
                            _jobs[job_id]["curves"] = curves_data
                            if meta:
                                _jobs[job_id]["curves_meta"] = meta
                        except: pass
                        continue

                    if 'CURVE_START:' in line:
                        _jobs[job_id]["phase"] = "curves"
                        parts = line.split(':')
                        for i, p in enumerate(parts):
                            if p == 'WINDOWS' and i + 1 < len(parts):
                                _jobs[job_id]["total_curves"] = int(parts[i + 1])
                        continue

                    if 'CURVE_PROGRESS:' in line:
                        parts = line.split(':')
                        if len(parts) >= 3:
                            _jobs[job_id]["curve_done"] = int(parts[1])
                            _jobs[job_id]["curve_total"] = int(parts[2])
                        continue

                    if 'CURVE_DONE:' in line:
                        _jobs[job_id]["curve_done"] = _jobs[job_id].get("curve_total", 1)
                        continue

                    if 'TRAIN_DONE:' in line:
                        _jobs[job_id]["status"] = "done"
                        continue

                    if 'TRAIN_ERROR:' in line:
                        _jobs[job_id]["status"] = "error"
                        _jobs[job_id]["error"] = line
                        continue
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(e)
        proc.wait()
        with _jobs_lock:
            if _jobs[job_id]["status"] == "training":
                _jobs[job_id]["status"] = "done" if proc.returncode == 0 else "error"
        _train_procs.pop(job_id, None)

    threading.Thread(target=_read_progress, daemon=True).start()
    return {"job_id": job_id, "status": "training", "model": model_name, "run_type": "train_dl"}


def _start_inference_run(model_name, job_id, info, run_type):
    """统计/预训练模型: 后台线程推理验证集生成曲线"""
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "progress": 0, "total": 0, "epoch": 0,
                          "loss": 0, "metrics": None, "curves": None,
                          "model": model_name, "run_type": run_type,
                          "phase": "inference", "_created_at": time.time()}

    def _run_inference():
        try:
            if run_type == "inference_pretrained":
                with _jobs_lock:
                    _jobs[job_id]["phase"] = "loading_model"
                # 触发模型加载
                from inference_engine import infer as _infer_single
                _ = _infer_single  # ensure imports

            # Benchmark windows never cross a cell ID or a missing hour.
            df_train = load_observations("train")
            df_test = load_observations("test")
            test_x, test_y, refs = build_windows(df_test, fit_training_scaler(df_train))
            cols = FEATURE_COLS
            n_channels = len(cols)
            n_windows = len(test_x)

            with _jobs_lock:
                _jobs[job_id]["total"] = n_windows
                _jobs[job_id]["phase"] = "inference"
                _jobs[job_id]["total_curves"] = n_windows

            all_pred = []
            all_true = []

            for i in range(n_windows):
                X = test_x[i]
                Y = test_y[i]
                all_true.append(Y)

                try:
                    pred = infer(model_name, X, pred_len=PRED_LEN)
                    if pred.ndim == 3:
                        pred = pred[0]
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    all_pred.append(pred)
                except Exception as e:
                    raise RuntimeError(f"Inference failed at benchmark window {i}: {e}") from e

                if (i + 1) % 200 == 0 or i == n_windows - 1:
                    with _jobs_lock:
                        _jobs[job_id]["progress"] = i + 1
                        _jobs[job_id]["curve_done"] = i + 1
                        _jobs[job_id]["curve_total"] = n_windows

            # σ空间（保持标准化，与 prediction_curves.js 一致）
            pred_arr = np.array(all_pred)  # (n_windows, pred_len, n_channels)
            true_arr = np.array(all_true)

            # 整体指标（σ空间）
            p_flat = pred_arr.reshape(-1, n_channels)
            t_flat = true_arr.reshape(-1, n_channels)
            mse = float(np.mean((t_flat - p_flat) ** 2))
            mae = float(np.mean(np.abs(t_flat - p_flat)))

            # Persist first, so a completed job is immediately eligible for public views.
            write_benchmark_artifact(
                info["result_dir"], model_name, pred_arr, true_arr, refs,
                run_kind=run_type,
                scaler=fit_training_scaler(df_train),
            )

            # Use the first reproducible panel window; the metadata identifies it.
            display_idx = 0

            curves = {}
            for ch_idx in range(n_channels):
                ch_name = cols[ch_idx] if ch_idx < len(cols) else f"通道{ch_idx+1}"
                curves[ch_name] = {
                    "pred":  [round(float(v), 4) for v in pred_arr[display_idx, :, ch_idx]],
                    "truth": [round(float(v), 4) for v in true_arr[display_idx, :, ch_idx]],
                }

            with _jobs_lock:
                _jobs[job_id]["curves"] = curves
                _jobs[job_id]["curves_meta"] = {
                    "window_idx": display_idx,
                    "total_windows": n_windows,
                    "cell_id": refs[display_idx].cell_id,
                    "start": refs[display_idx].start.isoformat(),
                    "protocol": "4g-panel-v1",
                }
                _jobs[job_id]["metrics"] = {"mse": round(mse, 4), "mae": round(mae, 4),
                                             "rmse": round(np.sqrt(mse), 4)}
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["phase"] = "done"

        except Exception as e:
            import traceback
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = f"{e}\n{traceback.format_exc()}"

    threading.Thread(target=_run_inference, daemon=True).start()
    return {"job_id": job_id, "status": "running", "model": model_name, "run_type": run_type}


def _start_script_training(model_name, job_id, info, run_type):
    """独立脚本训练 (XGBoost/Mamba/TimeLLM)"""
    tslib_str = str(TSLIB)
    models_dir = str(TSLIB / "models")
    data_dir = str(TSLIB / "data_provider" / "4g_traffic")

    if run_type == "train_xgboost":
        script = str(TSLIB / "models" / "model_xgboost.py")
        cmd = [sys.executable, "-u", script, "--data_path", data_dir,
               "--output_dir", str(TSLIB / "results")]
    elif run_type == "train_mamba":
        script = str(TSLIB / "models" / "model_mamba.py")
        cmd = [sys.executable, "-u", script, "--data_path", data_dir,
               "--output_dir", str(TSLIB / "results"),
               "--d_model", "128", "--d_ff", "32", "--epochs", "10"]
    elif run_type == "train_timellm":
        script = str(TSLIB / "models" / "model_timellm.py")
        cmd = [sys.executable, "-u", script, "--data_path", data_dir,
               "--output_dir", str(TSLIB / "results"),
               "--llm_name", "gpt2", "--epochs", "5", "--batch_size", "4"]
    elif run_type == "external_base":
        script = str(Path(__file__).parent / "evaluate_external_base.py")
        cmd = [sys.executable, "-u", script]
    else:
        raise HTTPException(400, detail={"error": "unknown_run_type"})

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(Path(__file__).parent),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    with _jobs_lock:
        _jobs[job_id] = {"status": "training", "progress": 0, "total": 0, "epoch": 0,
                          "loss": 0, "metrics": None, "curves": None,
                          "model": model_name, "run_type": run_type, "phase": "training",
                          "_created_at": time.time()}
        _train_procs[job_id] = proc

    def _read_script():
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line: continue

                with _jobs_lock:
                    # 尝试解析常见格式
                    m = re.search(r'(\d+)%', line)
                    if m:
                        _jobs[job_id]["progress"] = int(m.group(1))
                        continue
                    m = re.search(r'Epoch\s*(\d+)', line)
                    if m:
                        _jobs[job_id]["epoch"] = int(m.group(1))
                        continue
                    m = re.search(r'loss[:\s]*([\d.]+)', line, re.IGNORECASE)
                    if m:
                        _jobs[job_id]["loss"] = float(m.group(1))
                        continue

                if 'Error' in line or 'Traceback' in line:
                    with _jobs_lock:
                        _jobs[job_id]["error"] = (_jobs[job_id].get("error") or "") + line + "\n"
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(e)
        proc.wait()

        # 训练完成后，从 results 目录加载曲线
        with _jobs_lock:
            if _jobs[job_id]["status"] != "error":
                _jobs[job_id]["status"] = "done" if proc.returncode == 0 else "error"
                _jobs[job_id]["phase"] = "done"

        # 尝试加载预置曲线作为结果
        if proc.returncode == 0:
            preset = load_preset_curves(model_name)
            if preset and preset.get("curves"):
                with _jobs_lock:
                    _jobs[job_id]["curves"] = preset["curves"]
                    if preset.get("metrics_summary"):
                        _jobs[job_id]["metrics"] = preset["metrics_summary"]

        _train_procs.pop(job_id, None)

    threading.Thread(target=_read_script, daemon=True).start()
    return {"job_id": job_id, "status": "training", "model": model_name, "run_type": run_type}


# ═══ Legacy: keep /api/train/{model_name} for backward compat ═══
@app.post("/api/train/{model_name}")
async def train_start(model_name: str):
    """向后兼容 — 重定向到 /api/run/{model_name}"""
    return await run_start(model_name)


@app.post("/api/train/{model_name}/cancel")
async def train_cancel(model_name: str):
    with _jobs_lock:
        for job_id, proc in list(_train_procs.items()):
            _jobs[job_id]["cancel"] = True
            proc.kill()
            _jobs[job_id]["status"] = "cancelled"
            return {"job_id": job_id, "status": "cancelled"}
    raise HTTPException(404, detail={"error": "no_running_training"})


# ═══ Job Status ═══
@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail={"error": "unknown_job"})
    return {
        "job_id": job_id, "status": job["status"],
        "progress": job.get("progress", 0), "total": job.get("total", 0),
        "epoch": job.get("epoch", 0), "total_epochs": job.get("total_epochs", 0),
        "loss": job.get("loss", 0),
        "metrics": job.get("metrics"), "curves": job.get("curves"),
        "curves_meta": job.get("curves_meta"),
        "phase": job.get("phase", "training"),
        "curve_done": job.get("curve_done", 0),
        "curve_total": job.get("curve_total", 0),
        "error": job.get("error"),
    }


# ═══ File Parse ═══
@app.post("/api/parse-file")
async def api_parse_file(data_file: UploadFile = File(...)):
    allowed = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}
    fn = data_file.filename or "upload.csv"
    ext = Path(fn).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, detail={"error": "unsupported_format", "ext": ext, "allowed": sorted(allowed)})
    try:
        file_bytes = await data_file.read()
        parsed = parse_file(file_bytes, fn)
    except ValueError as e:
        raise HTTPException(400, detail={"error": "parse_error", "detail": str(e)})
    return {
        "filename": fn, "format": parsed["format"],
        "headers": parsed["headers"], "numeric_cols": parsed["numeric_cols"],
        "rows_preview": parsed["rows_preview"], "total_rows": parsed["total_rows"],
    }


# ═══ Custom Prediction ═══
@app.post("/api/generic-predict")
async def generic_predict(data_file: UploadFile = File(...), target_col: str = Form(...),
                          pred_len: int = Form(24), method: str = Form("autoar")):
    """Forecast a user series on its own scale with a holdout backtest."""
    if method not in SUPPORTED_METHODS:
        raise HTTPException(400, detail={"error": "unsupported_method", "supported": SUPPORTED_METHODS})
    try:
        parsed = parse_file(await data_file.read(), data_file.filename or "data.csv")
        result = generic_forecast(parsed["df"], target_col, int(pred_len), method)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": "invalid_series", "detail": str(exc)})
    return {
        "method": method,
        "predictions": [round(float(value), 6) for value in result.prediction],
        "meta": {
            "target_col": target_col, "pred_len": int(pred_len),
            "input_rows": result.train_rows + result.validation_rows,
            "validation_rows": result.validation_rows,
            "validation_mae": round(result.validation_mae, 6),
            "validation_rmse": round(result.validation_rmse, 6),
            "space": "original", "protocol": "generic-series-v1",
        },
    }


@app.post("/api/predict/{model_name}")
async def predict(model_name: str, data_file: UploadFile = File(...),
                   target_col: str = Form(...), pred_len: int = Form(24),
                   num_channels: int = Form(0)):
    info = get_model_info(model_name)
    if info is None: raise HTTPException(404, detail={"error": "unknown_model"})
    if info["tier"] >= 2:
        raise HTTPException(503, detail={"error": "model_not_available", "tier": info["tier"]})
    if pred_len not in (6, 12, 18, 24): pred_len = 24
    fn = data_file.filename or "data.csv"
    try:
        fb = await data_file.read()
        parsed = parse_file(fb, fn)
    except ValueError as e:
        raise HTTPException(400, detail={"error": "file_parse_error", "detail": str(e)})
    df = parsed["df"]
    nc_all = parsed["numeric_cols"]
    if 0 < num_channels < len(nc_all):
        df = df[nc_all[:num_channels] + [c for c in df.columns if c not in set(nc_all)]]
    if target_col not in nc_all:
        raise HTTPException(400, detail={"error": f"目标列 '{target_col}' 不在数值列中: {numeric_cols}"})

    mtype = info["type"]

    # ─── 不支持自定义数据预测的模型 ───
    if mtype in ("mamba", "timellm"):
        raise HTTPException(400, detail={
            "error": "model_not_for_custom_prediction",
            "detail": (f"{model_name} 使用独立模型架构，仅支持在基准4G数据上「从零训练」生成benchmark曲线。"
                       f"其训练脚本不保存可加载的模型权重，因此无法用于自定义数据预测。"
                       f"请选择其他模型（如 DLinear/PatchTST/TimesNet 等）进行自定义预测。")
        })
    # SCINet 的二叉树架构对输入通道结构极其敏感，非8通道真实4G数据会退化为恒值预测
    if model_name == "SCINet":
        raise HTTPException(400, detail={
            "error": "model_channel_sensitive",
            "detail": (f"SCINet 的二叉树下采样架构依赖训练时的8通道4G数据协方差结构。"
                       f"自定义数据与训练数据通道结构不同时模型退化为均值预测。"
                       f"建议使用 DLinear/PatchTST/TimesNet/Autoformer 等模型进行自定义预测。")
        })

    # ─── HuggingFace 预训练模型：直接用原始空间数据 ───
    if mtype == "huggingface":
        raw_arr = df[nc_all].values.astype(np.float32)
        if len(raw_arr) < SEQ_LEN + pred_len:
            raise HTTPException(400, detail={"error": f"数据行数不足，需要至少 {SEQ_LEN + pred_len} 行"})
        X_raw = raw_arr[-SEQ_LEN:]  # (24, n_user_cols)
        try:
            pred = infer(model_name, X_raw, pred_len=pred_len)
        except Exception as e:
            raise HTTPException(500, detail={"error": "inference_failed", "detail": str(e)})
        if pred.ndim == 3:
            pred = pred[0]
        idx = nc_all.index(target_col)
        result = pred[:, idx] if pred.shape[1] > 1 else pred[:, 0]
        result = np.clip(result, 0, None)
        return {
            "model": model_name,
            "predictions": [round(float(v), 4) for v in result],
            "meta": {"target_col": target_col, "pred_len": pred_len,
                      "input_rows": len(df), "device": str(get_device()),
                      "space": "original", "note": "预训练模型直接使用原始值"},
        }

    # ─── 4G-only models: require the real feature contract ───
    try:
        window, scaler, target_model_ch = build_4g_window_from_upload(df, target_col)
    except ValueError as e:
        raise HTTPException(400, detail={"error": str(e)})
    try:
        pred = infer(model_name, window, pred_len=pred_len)
    except Exception as e:
        raise HTTPException(500, detail={"error": "inference_failed", "detail": str(e)})

    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    ch = min(target_model_ch, pred.shape[1] - 1)
    target_pred = pred[:, ch]
    raw_val = target_pred * scaler.scale_[ch] + scaler.mean_[ch]
    raw_val = np.clip(raw_val, 0, None)

    return {
        "model": model_name,
        "predictions": [round(float(v), 4) for v in raw_val],
        "meta": {
            "target_col": target_col, "pred_len": pred_len,
            "input_rows": len(df), "device": str(get_device()),
            "channels_matched": list(FEATURE_COLS),
            "total_model_channels": NUM_CHANNELS,
            "protocol": "4g-feature-contract-v1",
        },
    }


@app.post("/api/generic-predict")
async def generic_predict(data_file: UploadFile = File(...), target_col: str = Form(...),
                          pred_len: int = Form(24), method: str = Form("autoar")):
    """Forecast a user series on its own scale with a visible holdout backtest."""
    if method not in SUPPORTED_METHODS:
        raise HTTPException(400, detail={"error": "unsupported_method", "supported": SUPPORTED_METHODS})
    fn = data_file.filename or "data.csv"
    try:
        parsed = parse_file(await data_file.read(), fn)
        result = generic_forecast(parsed["df"], target_col, int(pred_len), method)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": "invalid_series", "detail": str(exc)})
    return {
        "method": method,
        "predictions": [round(float(value), 6) for value in result.prediction],
        "meta": {
            "target_col": target_col,
            "pred_len": int(pred_len),
            "input_rows": result.train_rows + result.validation_rows,
            "validation_rows": result.validation_rows,
            "validation_mae": round(result.validation_mae, 6),
            "validation_rmse": round(result.validation_rmse, 6),
            "space": "original",
            "protocol": "generic-series-v1",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
