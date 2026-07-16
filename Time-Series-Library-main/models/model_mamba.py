"""
Mamba 状态空间模型时间序列预测
===============================
论文: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
纯 Python 实现 (无需 mamba-ssm CUDA 库)，基于 mamba-minimal 参考实现

架构:
  DataEmbedding → ResidualBlock×2 (RMSNorm + MambaBlock) → Output Linear
  MambaBlock = in_proj → Conv1d → SSM(selective_scan) → out_proj

参数控制:
  --d_model   模型维度 (默认256)
  --expand    扩展因子 (默认2, d_inner = d_model * expand)
  --d_ff      SSM状态维度 (默认64, 不宜过大否则OOM)
  --d_conv    卷积核大小 (默认4)
  --e_layers  残差块层数 (默认2)
"""
import sys, os, argparse, math, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange, repeat, einsum
sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import load_data, fit_scaler, generate_windows, compute_metrics, save_results


# ============================================================
# Mamba 组件
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state, d_conv):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, bias=True, kernel_size=d_conv,
                                padding=d_conv-1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_inner//16 + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner//16, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        A = repeat(torch.arange(1, d_state+1), 'n -> d n', d=d_inner).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        b, l, d = x.shape
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)

        y = self._ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def _ssm(self, x):
        d_in, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        dt_rank = d_in // 16
        delta, B, C = x_dbl.split([dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        # Selective scan (纯Python实现)
        b, l, d_inner = x.shape
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, x, 'b l d, b l n, b l d -> b l d n')

        x_ssm = torch.zeros((b, d_inner, n), device=x.device)
        ys = []
        for i in range(l):
            x_ssm = deltaA[:, i] * x_ssm + deltaB_u[:, i]
            y = einsum(x_ssm, C[:, i, :], 'b d n, b n -> b d')
            ys.append(y)
        y = torch.stack(ys, dim=1) + x * D
        return y


class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state, d_conv):
        super().__init__()
        self.mixer = MambaBlock(d_model, d_inner, d_state, d_conv)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class MambaModel(nn.Module):
    def __init__(self, enc_in, d_model, d_inner, d_state, d_conv, e_layers, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.embedding = nn.Linear(enc_in, d_model)
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, d_inner, d_state, d_conv) for _ in range(e_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.out_layer = nn.Linear(d_model, enc_in, bias=False)

    def forward(self, x_enc):
        mean = x_enc.mean(1, keepdim=True).detach()
        std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x_enc - mean) / std
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.out_layer(x) * std + mean
        return x[:, -self.pred_len:, :]


# ============================================================
# 训练和评估
# ============================================================
def train_model(model, train_loader, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    model.train()
    model.to(device)

    for epoch in range(args.epochs):
        epoch_loss, n_batches = 0.0, 0
        for batch_x, batch_y in train_loader:
            bx = batch_x.float().to(device)
            by = batch_y.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch+1) % max(1, args.epochs//3) == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}/{args.epochs}, Loss={epoch_loss/max(1,n_batches):.6f}')


def make_dataloader(data, seq_len, pred_len, batch_size):
    """从时序数据构建 DataLoader"""
    from torch.utils.data import DataLoader, TensorDataset
    X_list, Y_list = [], []
    for i in range(0, len(data) - seq_len - pred_len + 1):
        X_list.append(data[i:i+seq_len])
        Y_list.append(data[i+seq_len:i+seq_len+pred_len])
    dataset = TensorDataset(
        torch.FloatTensor(np.array(X_list)),
        torch.FloatTensor(np.array(Y_list)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--expand', type=int, default=2, help='d_inner = d_model * expand')
    parser.add_argument('--d_ff', type=int, default=64, help='SSM state dimension')
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    print('=' * 60)
    print('  Mamba (Pure Python SSM)')
    print('=' * 60)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'  Device: {device}')

    # 数据
    df_train, df_test, feature_cols = load_data(args.data_path)
    num_channels = len(feature_cols)
    scaler = fit_scaler(df_train, feature_cols)
    train_scaled = scaler.transform(df_train[feature_cols].values)
    test_scaled = scaler.transform(df_test[feature_cols].values)
    print(f'  Features: {num_channels} | Train: {len(train_scaled)} | Test: {len(test_scaled)}')

    # 模型
    d_inner = args.d_model * args.expand
    model = MambaModel(
        enc_in=num_channels, d_model=args.d_model, d_inner=d_inner,
        d_state=args.d_ff, d_conv=args.d_conv, e_layers=args.e_layers,
        pred_len=24,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,} (d_model={args.d_model}, d_inner={d_inner}, d_state={args.d_ff})')

    # 训练
    train_loader = make_dataloader(train_scaled, 24, 24, args.batch_size)
    print(f'  Training ({args.epochs} epochs, {len(train_loader)} batches/epoch)...')
    train_model(model, train_loader, args, device)

    # 预测
    X_windows, trues_list, _ = generate_windows(test_scaled)
    n_windows = len(X_windows)
    print(f'  Predicting {n_windows} windows...')

    model.eval()
    preds_list = []
    with torch.no_grad():
        for idx, X in enumerate(X_windows):
            if (idx+1) % 1000 == 0:
                print(f'    {idx+1}/{n_windows}')
            bx = torch.FloatTensor(X).unsqueeze(0).to(device)
            pred = model(bx).squeeze(0).cpu().numpy()
            preds_list.append(pred)

    # 指标
    metrics = compute_metrics(preds_list, trues_list, scaler, 24, num_channels)
    model_name = f'Mamba_d{args.d_model}_ex{args.expand}_ds{args.d_ff}_dc{args.d_conv}_el{args.e_layers}'
    print(f'\n  {model_name}')
    print(f'  MSE={metrics["MSE"]:.4f}  MAE={metrics["MAE"]:.4f}  RMSE={metrics["RMSE"]:.4f}')
    print(f'  MAPE={metrics["MAPE"]:.4f}  Custom_ACC={metrics["Custom_ACC"]:.4f}')

    # Save architecture and weights together; pred.npy alone cannot be used
    # for inference on a user's uploaded history.
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, 'checkpoint.pth'))
    with open(os.path.join(model_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'enc_in': num_channels, 'd_model': args.d_model,
            'd_inner': d_inner, 'd_state': args.d_ff, 'd_conv': args.d_conv,
            'e_layers': args.e_layers, 'pred_len': 24,
        }, f, indent=2)

    save_results(model_name, metrics, args.output_dir)


if __name__ == '__main__':
    main()
