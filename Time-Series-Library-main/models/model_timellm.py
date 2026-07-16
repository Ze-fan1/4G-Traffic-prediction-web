"""
TimeLLM: 用大语言模型重编程进行时间序列预测
=============================================
论文: Time-LLM (https://arxiv.org/abs/2312.01728)
实现: 使用 GPT-2 (124M) 作为冻结骨干 + 可训练重编程层

核心流程:
  Input(24h,8ch) → Patch → Reprogramming → GPT-2(frozen) → Output Projection → (24h,8ch)

模型大小选择:
  --llm_name gpt2          (124M参数, 约500MB显存)
  --llm_name gpt2-medium   (355M参数, 约1.5GB显存, 需要更多资源)
"""
import sys, os, argparse, json, numpy as np, torch, torch.nn as nn, warnings
sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import load_data, fit_scaler, generate_windows, compute_metrics, save_results

warnings.filterwarnings('ignore')


class TimeLLM_Model(nn.Module):
    """Time-LLM 重编程机制的轻量实现"""

    def __init__(self, seq_len=24, pred_len=24, num_channels=8,
                 llm_name='gpt2', patch_len=6, stride=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_channels = num_channels
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_dim = patch_len * num_channels

        from transformers import GPT2Model
        self.llm = GPT2Model.from_pretrained(llm_name, local_files_only=True)
        d_llm = self.llm.config.n_embd

        # 冻结 LLM
        for p in self.llm.parameters():
            p.requires_grad = False

        # 重编程层: patch_dim → d_llm
        self.reprogram = nn.Sequential(
            nn.Linear(self.patch_dim, d_llm * 2),
            nn.GELU(),
            nn.Linear(d_llm * 2, d_llm),
        )

        # 可学习 soft prompt
        self.num_prompts = 5
        self.prompt_embeds = nn.Parameter(torch.randn(1, self.num_prompts, d_llm))

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_llm, d_llm // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_llm // 2, pred_len * num_channels),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'  TimeLLM: trainable={trainable:,}/{total:,} ({100*trainable/total:.1f}%)')

    def forward(self, x):
        B, L, C = x.shape
        # Patch
        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patches.append(x[:, i:i+self.patch_len, :].reshape(B, -1))
        patches = torch.stack(patches, dim=1)  # (B, N, P*C)

        # Reprogramming → LLM space
        reprogrammed = self.reprogram(patches)  # (B, N, d_llm)

        # Concat with prompt prefix
        prompt = self.prompt_embeds.expand(B, -1, -1)  # (B, K, d_llm)
        llm_input = torch.cat([prompt, reprogrammed], dim=1)  # (B, K+N, d_llm)

        # Through frozen GPT-2
        outputs = self.llm(inputs_embeds=llm_input)
        hidden = outputs.last_hidden_state
        patch_out = hidden[:, self.num_prompts:, :]  # (B, N, d_llm)
        pooled = patch_out.mean(dim=1)  # (B, d_llm)

        # Project to predictions
        out = self.output_proj(pooled).reshape(B, self.pred_len, C)
        return out


def train_model(model, train_data, args, device):
    """在训练集上训练重编程层和输出投影层"""
    seq_len, pred_len = args.seq_len, args.pred_len
    total_len = len(train_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    model.train()
    model.to(device)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        indices = np.random.choice(
            total_len - seq_len - pred_len,
            size=min(args.samples_per_epoch, total_len - seq_len - pred_len),
            replace=False
        )

        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            bx_list, by_list = [], []
            for i in batch_idx:
                bx_list.append(train_data[i:i+seq_len])
                by_list.append(train_data[i+seq_len:i+seq_len+pred_len])

            bx = torch.FloatTensor(np.array(bx_list)).to(device)
            by = torch.FloatTensor(np.array(by_list)).to(device)

            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}/{args.epochs}, Loss={epoch_loss/max(1,n_batches):.6f}')

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--llm_name', type=str, default='gpt2',
                        help='gpt2 (124M) or gpt2-medium (355M)')
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--patch_len', type=int, default=6)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--samples_per_epoch', type=int, default=2000)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    print('=' * 60)
    print(f'  TimeLLM (backbone: {args.llm_name})')
    print('=' * 60)

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'  Device: {device}')

    # 加载数据
    df_train, df_test, feature_cols = load_data(args.data_path)
    num_channels = len(feature_cols)
    scaler = fit_scaler(df_train, feature_cols)
    train_scaled = scaler.transform(df_train[feature_cols].values)
    test_scaled = scaler.transform(df_test[feature_cols].values)
    print(f'  Features: {num_channels}')

    # 构建模型
    model = TimeLLM_Model(
        seq_len=args.seq_len, pred_len=args.pred_len,
        num_channels=num_channels, llm_name=args.llm_name,
        patch_len=args.patch_len, stride=args.stride,
    )

    # 训练
    print(f'  Training ({args.epochs} epochs)...')
    model = train_model(model, train_scaled, args, device)

    # 生成窗口
    X_windows, trues_list, _ = generate_windows(test_scaled)
    n_windows = len(X_windows)
    print(f'  Predicting {n_windows} windows...')

    # 预测
    model.eval()
    preds_list = []
    with torch.no_grad():
        for idx, X in enumerate(X_windows):
            if (idx + 1) % 1000 == 0:
                print(f'    {idx+1}/{n_windows}')
            bx = torch.FloatTensor(X).unsqueeze(0).to(device)
            pred = model(bx).squeeze(0).cpu().numpy()
            pred = np.clip(pred, -10, 10)
            preds_list.append(pred)

    # 指标
    metrics = compute_metrics(preds_list, trues_list, scaler, args.pred_len, num_channels)
    model_name = f'TimeLLM_{args.llm_name}_pl{args.patch_len}_s{args.stride}'
    print(f'\n  {model_name}')
    print(f'  MSE={metrics["MSE"]:.4f}  MAE={metrics["MAE"]:.4f}  RMSE={metrics["RMSE"]:.4f}')
    print(f'  MAPE={metrics["MAPE"]:.4f}  Custom_ACC={metrics["Custom_ACC"]:.4f}')

    # The frozen GPT-2 cache is not enough: retain the trained projection and
    # prompt parameters so uploaded data uses this exact local model.
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, 'checkpoint.pth'))
    with open(os.path.join(model_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'seq_len': args.seq_len, 'pred_len': args.pred_len,
            'num_channels': num_channels, 'llm_name': args.llm_name,
            'patch_len': args.patch_len, 'stride': args.stride,
        }, f, indent=2)

    save_results(model_name, metrics, args.output_dir)


if __name__ == '__main__':
    main()
