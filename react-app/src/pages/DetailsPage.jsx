import { useMemo } from 'react';
import { MODELS, CATS } from '../data/models';
import { getPalette } from '../data/palette';
import RevealCard from '../components/RevealCard';

export default function DetailsPage() {
  const sorted = useMemo(() => [...MODELS].sort((a, b) => b.acc - a.acc).slice(0, 20), []);
  const catSummary = useMemo(() => CATS.map(cat => { const models = MODELS.filter(m => m.cat === cat); const avgMSE = models.reduce((s, m) => s + m.mse, 0) / models.length; const avgACC = models.reduce((s, m) => s + m.acc, 0) / models.length; const best = models.reduce((a, b) => a.acc > b.acc ? a : b); return { cat, count: models.length, avgMSE, avgACC, best: best.model }; }), []);
  const medals = { 0: '🥇', 1: '🥈', 2: '🥉' };

  return (
    <div className="page-enter">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
        {/* Dataset Description */}
        <RevealCard className="h-full">
          <div className="card p-5 h-full">
          <h3 className="text-sm font-semibold tracking-tight mb-3">数据集说明</h3>
          <div className="space-y-2.5 text-xs text-[#52525B] leading-relaxed">
            {[
              ['数据来源', '4G 基站 RAN 侧实测 KPI 数据，涵盖数千小时连续采集'],
              ['特征通道', '8 维多变量输入：ERAB 流量、PDCCH / PDSCH / PUSCH 信道利用率、上行流量、下行流量、总流量、有效连接数'],
              ['时间粒度', '小时级聚合数据，保留日周期和趋势性模式'],
              ['预测目标', '未来 24 小时逐小时流量预测（多变量输入，多变量输出）'],
              ['滚动策略', '每 3 小时滑动窗口重新预测，共生成 5378 个测试窗口，覆盖多种时间模式'],
              ['评估指标', 'MSE · MAE · RMSE · MAPE · MSPE · Custom ACC（针对高峰流量的分段准确率）'],
              ['模型总数', `${MODELS.length} 个，覆盖传统统计 / 树模型 / Transformer / MLP / CNN / RNN / SSM / LLM 八大范式`],
            ].map(([label, value]) => (
              <div key={label} className="flex gap-2">
                <span className="text-[#A1A1AA] w-24 flex-shrink-0">{label}</span>
                <span>{value}</span>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-3 border-t" style={{ borderColor: 'rgba(0,0,0,0.04)' }}>
            <a href="https://github.com/Ze-fan1/4G-Traffic-prediction-web" target="_blank" rel="noopener" className="inline-flex items-center gap-1.5 text-xs font-medium text-[#3B82F6] hover:underline transition-all">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" /></svg>View on GitHub ↗
            </a>
          </div>
        </div>
        </RevealCard>

        {/* Experiment Setup */}
        <RevealCard className="h-full" delay={80}>
          <div className="card p-5 h-full">
            <h3 className="text-sm font-semibold tracking-tight mb-3">实验设置</h3>
          <div className="space-y-2.5 text-xs text-[#52525B] leading-relaxed">
            {[
              ['预测任务', '多变量时间序列预测：输入过去 24 小时 × 8 通道 KPI 数据，输出未来 24 小时 × 8 通道预测'],
              ['8 通道说明', 'RAN 侧多维度 KPI——涵盖 ERAB 承载流量、三项信道利用率（PDCCH/PDSCH/PUSCH）、上下行流量、总流量和有效连接数。比单变量流量预测更全面刻画基站负载状态'],
              ['标准化方案', 'StandardScaler 拟合于训练集（均值和标准差来自训练数据），统一应用于所有模型。图表中的"σ"即标准差单位：0 = 历史均值水平，±1 = 偏离 1 个标准差'],
              ['滑动协议', '每 3 小时滑动一次窗口，24h 输入 → 24h 预测。共生成 5378 个测试窗口，时间序列交叉验证，严格避免未来信息泄露'],
              ['评估指标', 'MSE / MAE / RMSE：标准回归指标；MAPE / MSPE：相对误差；Custom ACC：分段准确率（高于均值样本单独评估后取平均）'],
              ['基准模型', '★ BaseModel = External-Base 架构，在当前 benchmark 上六项指标全面领先'],
              ['运行环境', 'Python 3.10 · PyTorch 2.10+cu130 · CUDA 12 · NVIDIA RTX 5050'],
              ['代码仓库', 'Time-Series-Library（已在 GitHub 开源，含完整训练和评估脚本）'],
            ].map(([label, value]) => (
              <div key={label} className="flex gap-2">
                <span className="text-[#A1A1AA] w-24 flex-shrink-0">{label}</span>
                <span className={label === '代码仓库' ? 'font-mono text-[0.7rem]' : ''}>{value}</span>
              </div>
            ))}
          </div>
        </div>
        </RevealCard>

        {/* Category Summary */}
        <RevealCard className="h-full" delay={160}>
          <div className="card p-5 h-full">
            <h3 className="text-sm font-semibold tracking-tight mb-3">按类别指标汇总</h3>
          <div className="overflow-x-auto">
            <table className="dt w-full">
              <thead>
                <tr>
                  <th>模型类别</th><th>数量</th><th>平均 MSE</th><th>平均 ACC</th><th>最佳模型</th>
                </tr>
              </thead>
              <tbody>
                {catSummary.map(row => {
                  const c = getPalette(MODELS.find(m => m.cat === row.cat));
                  return (
                    <tr key={row.cat}>
                      <td><span className="inline-flex items-center gap-1.5 text-[0.65rem] font-medium" style={{ color: c.text }}><span className="accent-dot" style={{ background: c.border }} />{row.cat}</span></td>
                      <td className="font-mono text-[0.72rem]">{row.count}</td>
                      <td className="font-mono text-[0.72rem]">{row.avgMSE.toFixed(3)}</td>
                      <td className="font-mono text-[0.72rem] font-medium">{row.avgACC.toFixed(4)}</td>
                      <td className="text-[0.72rem]">{row.best}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
            <p className="text-xs text-[#52525B] leading-relaxed">
              不同架构范式在 4G 流量预测任务上表现各异。Baseline（★ BaseModel）以最低 MSE（1.0615）和最高 ACC（0.5901）领跑，
              MLP 类别的平均 ACC 最高（受益于 DLinear 和 LightTS 的简单高效设计），而 LLM 零样本模型的平均性能较弱，提示领域迁移仍是开放挑战。
            </p>
          </div>
        </div>
        </RevealCard>

        {/* TOP 20 Ranking */}
        <RevealCard className="h-full" delay={240}>
          <div className="card p-5 h-full">
            <h3 className="text-sm font-semibold tracking-tight mb-3">综合排名 · TOP 20</h3>
          <div className="space-y-0.5">
            {sorted.map((m, i) => {
              const c = getPalette(m);
              return (
                <div key={m.model} className="flex items-center gap-2.5 text-xs py-1.5 px-2 rounded-lg hover:bg-stone-50 transition-colors" title={`MSE: ${m.mse.toFixed(3)} · MAE: ${m.mae.toFixed(3)} · RMSE: ${m.rmse.toFixed(3)}`}>
                  <span className="w-5 text-center font-mono font-medium" style={{ fontSize: '0.72rem' }}>{medals[i] || i + 1}</span>
                  <span className={`flex-1 font-medium text-[0.72rem] ${m.model.includes('BaseModel') ? 'text-[#3B82F6]' : ''}`}>{m.model}</span>
                  <span className="text-[#A1A1AA]" style={{ fontSize: '0.62rem' }}>{m.cat}</span>
                  <span className="w-12 text-right font-mono font-semibold" style={{ fontSize: '0.72rem', color: c.text }}>{m.acc.toFixed(4)}</span>
                </div>
              );
            })}
          </div>
        </div>
        </RevealCard>

        {/* Detailed Analysis */}
        <RevealCard delay={320}>
          <div className="card p-5 lg:col-span-2">
          <h3 className="text-sm font-semibold tracking-tight mb-3">深度分析</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-[#52525B] leading-relaxed">
            {[
              ['1. ★ BaseModel 六项指标全面领先', 'MSE=1.0615, ACC=0.5901，相对第二名 iTransformer 的 ACC 提升 12.0%。核心优势来源于多尺度特征融合——同时捕捉小时级波动和日级趋势，而非单一尺度的模式学习。'],
              ['2. Transformer 内部代际演进清晰', '第一代（Transformer/Autoformer/Informer，ACC 0.41-0.46）→ 第二代（PatchTST/iTransformer，ACC 0.52-0.53），提升超 25%。关键改进：通道独立建模 + 图块化注意力替代逐点注意力，显著降低了建模难度。'],
              ['3. 轻量模型竞争力不容忽视', 'DLinear（ACC=0.5202）和 LightTS（ACC=0.5236）仅用简单线性/MLP 结构，性能却超越多数 Transformer 变体。XGBoost（ACC=0.4729）作为唯一树模型，展示了经典 ML 在结构化时序数据上的稳健性。这印证了"强基线"对公平 benchmark 的重要性。'],
              ['4. 零样本 LLM 的代际跃迁', '旧版 Chronos-t5 全系（tiny/small/base，ACC 0.418-0.445）和 TimeLLM（ACC=0.4260）受限于通用时序分布与 4G 流量的领域差异。但 Amazon Chronos2（120M 参数，ACC=0.4903）通过多变量联合预测和更强的预训练，大幅缩小了零样本与训练模型之间的差距，超越 XGBoost（ACC=0.4729），接近 TSMixer（ACC=0.4665）等训练模型，证明了零样本方法的持续进步。'],
              ['5. MAPE 在标准化空间中的不稳定性', '标准化空间中真值接近零时，MAPE 因分母过小而产生极端值（LinearRegression=294.99%）。建议优先使用 Custom ACC 和 MSE 作为主要评估依据，MAPE/MSPE 作为辅助参考。'],
              ['6. 滑动窗口协议对结论的影响', '3 小时步长意味着相邻窗口高度重叠（窗口 i 与 i+1 共享 21 小时数据），降低了指标的方差但仍保持了 ranking 一致性。窗口数 5378 覆盖了多种时间模式（日间高峰/夜间低谷/过渡期），确保了统计显著性。'],
            ].map(([title, body]) => (
              <div key={title} className="bg-[#FAFAFA] rounded-xl p-3">
                <strong className="text-[#18181B]">{title}</strong>
                <p className="mt-1">{body}</p>
              </div>
            ))}
          </div>
        </div>
        </RevealCard>
      </div>
    </div>
  );
}
