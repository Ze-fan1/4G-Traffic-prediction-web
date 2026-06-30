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
        "loaded_model": None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
