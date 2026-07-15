import { MODELS } from '../data/models';
import RevealCard from '../components/RevealCard';

export default function DetailsPage() {
  return (
    <div className="page-enter mt-5 space-y-4">
      <RevealCard>
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-3">可信评测协议</h3>
          <div className="space-y-2 text-xs text-[#52525B] leading-relaxed">
            <p>数据包含 100 个小区。每个小区被视为独立小时级时间序列，不再将整张表按行拼接。</p>
            <p>有效样本仅来自同一小区内连续 48 小时片段：前 24 小时输入，后 24 小时作为预测真值，步长为 3 小时。</p>
            <p>训练数据按小区内时间顺序划分训练与验证；测试集只用于最终评估。StandardScaler 仅在训练部分拟合。</p>
            <p>测试集中目前有 <b>3,514</b> 个有效窗口。此前扁平化得到的 5,378 窗口可能跨小区或跨时间缺口，已停止用于公开指标。</p>
          </div>
        </div>
      </RevealCard>

      <RevealCard delay={80}>
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-3">已完成重评测</h3>
          <div className="overflow-x-auto">
            <table className="dt w-full">
              <thead><tr><th>模型</th><th>类别</th><th>MSE (标准化)</th><th>MAE (标准化)</th><th>RMSE (标准化)</th></tr></thead>
              <tbody>{MODELS.map(model => (
                <tr key={model.model}>
                  <td className="font-medium">{model.model}</td><td>{model.cat}</td>
                  <td className="font-mono">{model.mse.toFixed(4)}</td><td className="font-mono">{model.mae.toFixed(4)}</td><td className="font-mono">{model.rmse.toFixed(4)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-[#52525B] leading-relaxed">
            外部 BaseModel 仅使用 `df_4g_base_100.parquet` 中的 `forecast_*` 作为外部预测结果，与测试集同窗口比较；它不参与任何训练。
            其余模型将在相同协议下重训或重评测后加入表格。
          </p>
        </div>
      </RevealCard>
    </div>
  );
}
