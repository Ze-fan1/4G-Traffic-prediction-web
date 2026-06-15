import { useMemo } from 'react';
import { MODELS, CATS } from '../data/models';
import { getPalette } from '../data/palette';

export default function DetailsPage() {
  const sorted = useMemo(() => [...MODELS].sort((a, b) => b.acc - a.acc).slice(0, 20), []);
  const catSummary = useMemo(() => CATS.map(cat => { const models = MODELS.filter(m => m.cat === cat); const avgMSE = models.reduce((s, m) => s + m.mse, 0) / models.length; const avgACC = models.reduce((s, m) => s + m.acc, 0) / models.length; const best = models.reduce((a, b) => a.acc > b.acc ? a : b); return { cat, count: models.length, avgMSE, avgACC, best: best.model }; }), []);
  const medals = { 0: '🥇', 1: '🥈', 2: '🥉' };

  return (
    <div className="page-enter">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-3">数据集说明</h3>
          <div className="space-y-2.5 text-xs text-[#52525B] leading-relaxed">
            {[['数据来源', '4G 基站实测流量数据', true], ['特征通道', '8 维：ERAB流量、PDCCH利用率、PDSCH利用率、PUSCH利用率、上行流量、下行流量、总流量、有效连接数'], ['时间粒度', '小时级数据，数千小时连续记录'], ['预测目标', '未来 24 小时逐小时流量预测'], ['滚动策略', '每 3 小时滑动窗口重新预测，共 5378 个窗口'], ['评估指标', 'MSE · MAE · RMSE · MAPE · MSPE · Custom ACC'], ['模型总数', '26 个，涵盖统计 / ML / DL / SSM / LLM 五大范式']].map(([label, value, bold]) => (<div key={label} className="flex gap-2"><span className="text-[#A1A1AA] w-24 flex-shrink-0">{label}</span><span className={bold ? 'font-medium' : ''}>{value}</span></div>))}
          </div>
          <div className="mt-4 pt-3 border-t" style={{ borderColor: 'rgba(0,0,0,0.04)' }}>
            <a href="https://github.com/Ze-fan1/4G-Traffic-prediction-web" target="_blank" rel="noopener" className="inline-flex items-center gap-1.5 text-xs font-medium text-[#6152F2] hover:underline transition-all">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" /></svg>View on GitHub ↗
            </a>
          </div>
        </div>
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-3">实验设置</h3>
          <div className="space-y-2.5 text-xs text-[#52525B] leading-relaxed">
            {[['预测任务', '未来 24 小时 4G 流量预测（8 通道多变量输入）'], ['8 通道说明', 'RAN 侧多维度 KPI（ERAB流量 + 3项信道利用率 + 3项流量指标 + 连接数），比单变量流量预测更全面刻画小区负载状态'], ['滚动策略', '每 3 小时滑动窗口重新预测，步长 = 3h'], ['评估指标', 'MSE / MAE / RMSE / MAPE / MSPE / Custom ACC'], ['数据划分', '时间序列交叉验证，避免未来信息泄露'], ['模型总数', '26 个，覆盖统计 / ML / DL / SSM / LLM 五大范式'], ['BaseModel', '自研多尺度特征融合架构（External-Base），综合表现最优'], ['运行环境', 'Python 3.10 · PyTorch 2.x · CUDA 12'], ['代码仓库', 'Time-Series-Library']].map(([label, value]) => (<div key={label} className="flex gap-2"><span className="text-[#A1A1AA] w-24 flex-shrink-0">{label}</span><span className={label === '代码仓库' ? 'font-mono text-[0.7rem]' : ''}>{value}</span></div>))}
          </div>
        </div>
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-3">按类别指标汇总</h3>
          <div className="overflow-x-auto"><table className="dt w-full"><thead><tr><th>Category</th><th>N</th><th>Avg MSE</th><th>Avg ACC</th><th>Best</th></tr></thead><tbody>{catSummary.map(row => { const c = getPalette(MODELS.find(m => m.cat === row.cat)); return (<tr key={row.cat}><td><span className="inline-flex items-center gap-1.5 text-[0.65rem] font-medium" style={{ color: c.text }}><span className="accent-dot" style={{ background: c.border }} />{row.cat}</span></td><td className="font-mono text-[0.72rem]">{row.count}</td><td className="font-mono text-[0.72rem]">{row.avgMSE.toFixed(3)}</td><td className="font-mono text-[0.72rem] font-medium">{row.avgACC.toFixed(4)}</td><td className="text-[0.72rem]">{row.best}</td></tr>); })}</tbody></table></div>
        </div>
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-3">综合排名 · TOP 20</h3>
          <div className="space-y-0.5">{sorted.map((m, i) => { const c = getPalette(m); return (<div key={m.model} className="flex items-center gap-2.5 text-xs py-1.5 px-2 rounded-lg hover:bg-stone-50 transition-colors" title={`MSE: ${m.mse.toFixed(3)} · MAE: ${m.mae.toFixed(3)} · RMSE: ${m.rmse.toFixed(3)}`}><span className="w-5 text-center font-mono font-medium" style={{ fontSize: '0.72rem' }}>{medals[i] || i + 1}</span><span className={`flex-1 font-medium text-[0.72rem] ${m.model.includes('BaseModel') ? 'text-[#6152F2]' : ''}`}>{m.model}</span><span className="text-[#A1A1AA]" style={{ fontSize: '0.62rem' }}>{m.cat}</span><span className="w-12 text-right font-mono font-semibold" style={{ fontSize: '0.72rem', color: c.text }}>{m.acc.toFixed(4)}</span></div>); })}</div>
        </div>
        <div className="card p-5 lg:col-span-2">
          <h3 className="text-sm font-semibold tracking-tight mb-3">详细分析</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-[#52525B] leading-relaxed">
            {[
              ['1. BaseModel 全面领先', '6 项指标全部第一 (MSE=1.0628, ACC=0.5901)，相比第二名 iTransformer ACC 提升 12.0%，验证了基线模型在 4G 流量预测上的有效性。'],
              ['2. Transformer 分化显著', 'iTransformer (ACC=0.5267) 和 PatchTST (ACC=0.5239) 表现接近，但相比原生 Transformer (ACC=0.4140) 提升超过 27%，注意力机制设计是核心差异。'],
              ['3. 轻量基线有竞争力', 'Persistent_24h (ACC=0.4477) 和 XGBoost (ACC=0.4729) 的 ACC 超过 Autoformer、Informer 等复杂模型，表明强基线设定是公平评估的前提。'],
              ['4. LLM Zero-shot 泛化不足', 'Chronos 全系和 TimeLLM ACC 均低于 0.45，通用预训练知识无法直接迁移至网络流量预测，需要领域微调或适配层。'],
              ['5. MAPE 指标不稳定', '标准化空间中大量模型 MAPE 超过 100%（LinearRegression=294.99%），在 scaled 空间使用百分比误差需谨慎，建议以 ACC 和 MSE 为主。'],
              ['6. 滚动窗口需注意', '3h 滑动步长使同一时间点被多次预测，整体方差降低但分析单点精度时需注意自相关偏差。窗口数 5378，覆盖多种时间模式。'],
            ].map(([title, body]) => (<div key={title} className="bg-[#FAFAFA] rounded-xl p-3"><strong className="text-[#18181B]">{title}</strong><br />{body}</div>))}
          </div>
        </div>
      </div>
    </div>
  );
}
