# 后端部署指南

## 现状

- ✅ **前端**: GitHub Pages `https://ze-fan1.github.io/4G-Traffic-prediction-web/` (静态文件)
- ❌ **后端**: 仅本地运行 `localhost:8000`，其他用户无法访问

**GitHub Pages 不能运行 Python 后端！** 所以需要单独部署后端服务。

---

## 推荐方案对比

| 方案 | GPU | 免费额度 | 难度 | 适合场景 |
|------|-----|---------|------|---------|
| **Hugging Face Spaces** | ✅ T4 (付费) / CPU (免费) | 免费CPU, GPU $0.6/h | ⭐ 简单 | ML Demo |
| **Railway** | ❌ 仅CPU | $5 试用金 | ⭐ 简单 | 轻量API |
| **Render** | ❌ 仅CPU | 免费 750h/月 | ⭐ 简单 | 轻量API |
| **Fly.io** | ✅ 可加GPU | 免费 3 VM | ⭐⭐ 中等 | 全球部署 |
| **RunPod** | ✅ 多GPU | 按量付费, $0.3/h起 | ⭐⭐ 中等 | GPU推理 |
| **AutoDL** (国内) | ✅ 多GPU | ¥1.5/h起 | ⭐⭐ 中等 | 国内GPU |

---

## 方案一：Hugging Face Spaces (推荐 ⭐)

最省心的方案，Docker 一键部署，有免费 CPU 版可测试。

### 1. 创建 Dockerfile

```dockerfile
# backend/Dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个 benchmark 目录（因为后端依赖 Time-Series-Library）
COPY . /app/benchmark/

WORKDIR /app/benchmark/backend
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 2. 创建 requirements.txt

```txt
fastapi==0.111.0
uvicorn==0.30.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.0
openpyxl==3.1.5
xgboost==2.1.0
statsmodels==0.14.2
python-multipart==0.0.9
```

**注意**: PyTorch 已包含在基础镜像中，不需要重复安装。Chronos/TTM 需要额外安装：

```txt
# 仅在需要 Chronos2/TTM 时取消注释
# git+https://github.com/amazon-science/chronos-forecasting.git
# tsfm-public
```

### 3. 部署步骤

```bash
# 1. 在 Hugging Face 创建 Space: https://huggingface.co/new-space
#    - SDK: Docker
#    - Space name: 4g-traffic-api (或其他)

# 2. 克隆 Space 仓库
git clone https://huggingface.co/spaces/<你的用户名>/4g-traffic-api
cd 4g-traffic-api

# 3. 复制文件
cp -r c:\Users\Admin\Desktop\benchmark\* .

# 4. 推送
git add . && git commit -m "Initial deploy" && git push

# HF Spaces 会自动构建 Docker 镜像并启动
```

### 4. 前端指向新后端

修改 [react-app/vite.config.js](../react-app/vite.config.js)：

```js
// 将 proxy 替换为 Hugging Face Space URL
server: {
  proxy: {
    '/api': {
      target: 'https://<用户名>-4g-traffic-api.hf.space',
      changeOrigin: true,
    }
  }
}
```

或者：在前端直接写死 API 地址（生产环境）：

```js
// react-app/src/components/DemoPanel.jsx
const API_BASE = 'https://<用户名>-4g-traffic-api.hf.space/api';
```

---

## 方案二：Railway (简单，但 CPU only)

适合非 GPU 模型（统计模型、XGBoost）的轻量部署。

```bash
# 1. 安装 Railway CLI
npm i -g @railway/cli

# 2. 登录并创建项目
railway login
railway init

# 3. 部署
railway up

# 4. 设置启动命令
# Web UI → Settings → Start Command:
uvicorn server:app --host 0.0.0.0 --port $PORT
```

Railway 自动分配 `https://xxx.up.railway.app` 域名。

---

## 方案三：AutoDL / 恒源云 (国内推荐)

国内最实用的 GPU 方案，按小时计费，¥1.5-3/h 起。

### 部署步骤

```bash
# 1. 在 AutoDL 官网 (autodl.com) 租用实例
#    - 镜像选: PyTorch 2.x + Python 3.10 + CUDA 12.x
#    - GPU: RTX 3080 / 3090 (¥1.5-2/h)

# 2. SSH 登录后，上传代码
scp -rP <端口> benchmark/ root@<IP>:/root/

# 3. SSH 进入实例
ssh -p <端口> root@<IP>

# 4. 安装依赖
pip install fastapi uvicorn pandas scikit-learn openpyxl xgboost statsmodels python-multipart

# 5. 启动后端
cd /root/benchmark/backend
nohup uvicorn server:app --host 0.0.0.0 --port 8000 &

# 6. 前端指向 AutoDL 实例的公网 IP:8000
```

**注意**: AutoDL 实例停止后数据保留但 IP 会变，需要更新前端配置。

---

## 方案四：自建 Linux 服务器

如果有闲置 Linux 机器（带 NVIDIA GPU >6GB 显存）：

```bash
# 1. SSH 登录
ssh user@your-server-ip

# 2. 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 3. 创建环境 + 安装 PyTorch (CUDA)
conda create -n benchmark python=3.10 -y
conda activate benchmark
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. 上传代码
rsync -avz --exclude 'node_modules' --exclude '.git' benchmark/ user@server:/home/user/benchmark/

# 5. 安装依赖 + 启动
cd /home/user/benchmark/backend
pip install -r requirements.txt

# 6. 使用 systemd 保持后台运行
sudo tee /etc/systemd/system/benchmark-api.service << 'EOF'
[Unit]
Description=4G Traffic Benchmark API
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/benchmark/backend
Environment="PATH=/home/user/miniconda3/envs/benchmark/bin"
ExecStart=/home/user/miniconda3/envs/benchmark/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now benchmark-api
```

### Nginx 反向代理（可选：添加 HTTPS）

```bash
sudo apt install nginx certbot python3-certbot-nginx

# 配置 Nginx
sudo tee /etc/nginx/sites-available/benchmark << 'EOF'
server {
    listen 80;
    server_name api.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 100M;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/benchmark /etc/nginx/sites-enabled/
sudo certbot --nginx -d api.your-domain.com
```

---

## 前端如何适配多后端地址

推荐做法：让前端自动判断环境：

```js
// react-app/src/api.js (新建)
const API_BASE = import.meta.env.VITE_API_URL || '/api';

export async function fetchPresetCurves(model, window) {
  const winParam = window === 'avg' ? 'window=-1' : `window=${window}`;
  const res = await fetch(`${API_BASE}/preset-curves/${encodeURIComponent(model)}?${winParam}`);
  return res.json();
}
```

然后在不同环境下设不同的 `VITE_API_URL`：
- 本地开发: 不设（走 Vite proxy → localhost:8000）
- 生产 GitHub Pages: 设为 `https://your-backend-url.com/api`

---

## 总结：路径选择

```
你的需求："用户可以浏览并且在网页上用自己的数据测试模型"
                    │
                    ▼
    ┌───────────────────────────────────┐
    │ 需要 GPU 吗？                       │
    │ • 仅统计模型(XGBoost/ARIMA等) → CPU │
    │ • DL模型(PatchTST等)推理 → CPU可用但慢│
    │ • DL模型训练 → GPU必需               │
    └───────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   需要 GPU                  不需要 GPU
        │                       │
        ▼                       ▼
  Hugging Face Spaces       Railway / Render
  (Docker + GPU,           (免费即可)
   按需付费)                 
        │                   
  或 AutoDL (国内GPU)      
        │                   
  或 自建 Linux Server     
```

**当前最佳路径**：
1. **测试阶段**: 先用 Hugging Face Spaces CPU 版 (免费) 部署，确认功能正常
2. **正式上线**: 升级到 Spaces GPU 版 ($0.6/h)，或租用 AutoDL 实例
3. **前端**: 在 GitHub Pages 设置中切换 API 地址即可
