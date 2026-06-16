import { MODELS, CATS } from '../data/models';
import { CAT_PALETTE } from '../data/palette';

function getCat(model) {
  return CAT_PALETTE[model.cat] || CAT_PALETTE['Statistical'];
}

export default function OverviewPage() {
  const sorted = [...MODELS].sort((a, b) => b.acc - a.acc);
  const maxACC = sorted[0].acc;
  const legendItems = CATS.map(cat => {
    const c = CAT_PALETTE[cat];
    const n = MODELS.filter(m => m.cat === cat).length;
    return { dot: c.border, label: c.label || cat, count: n };
  });
  const half = Math.ceil(legendItems.length / 2);
  const left = legendItems.slice(0, half);
  const right = legendItems.slice(half);
  const rowCount = Math.max(left.length, right.length);

  return (
    <div className="page-enter">
      {/* Welcome Banner */}
      <div className="relative mt-5 mb-6 text-center overflow-hidden py-8">
        <h1 className="welcome-blur" aria-hidden="true">4G Traffic Prediction</h1>
        <p className="text-sm text-[#A1A1AA] mt-2 tracking-wide" style={{ animation: 'fadeIn 0.8s 0.3s cubic-bezier(0.16,1,0.3,1) both' }}>
          26 Models · 8 Channels · 24h Forecast
        </p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
        <div className="card card-glow shape-blur p-5 relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-0.5" style={{ background: 'var(--accent)' }} />
          <p className="text-[0.65rem] text-[#A1A1AA] uppercase tracking-wider font-medium mb-1.5">Best Model</p>
          <p className="text-xl font-semibold tracking-tight">BaseModel</p>
        </div>
        <div className="card card-glow shape-blur p-5 relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-0.5" style={{ background: 'var(--accent)' }} />
          <p className="text-[0.65rem] text-[#A1A1AA] uppercase tracking-wider font-medium mb-1.5">Custom ACC</p>
          <p className="text-2xl font-bold tracking-tight" style={{ color: 'var(--accent)' }}>0.5901</p>
        </div>
        <div className="card card-glow shape-blur p-5">
          <p className="text-[0.65rem] text-[#A1A1AA] uppercase tracking-wider font-medium mb-1.5">Lowest MSE</p>
          <p className="text-2xl font-bold tracking-tight">1.0615</p>
        </div>
        <div className="card card-glow shape-blur p-5">
          <p className="text-[0.65rem] text-[#A1A1AA] uppercase tracking-wider font-medium mb-1.5">Experiment Setup</p>
          <p className="text-2xl font-bold tracking-tight">8 channels</p>
          <p className="text-[0.65rem] text-[#A1A1AA] mt-0.5">24h forecast · 3h rolling</p>
        </div>
      </div>

      <div className="card p-5 mb-5">
        <h3 className="text-sm font-semibold tracking-tight mb-3">Custom ACC 排名 · 全部模型</h3>
        <div className="space-y-1.5">
          {sorted.map((m, i) => {
            const c = getCat(m);
            const w = (m.acc / maxACC) * 100;
            return (
              <div key={m.model} className="flex items-center gap-2.5 text-xs group cursor-default" title={`${m.model} · ${m.cat}\nMSE: ${m.mse.toFixed(3)}  ACC: ${m.acc.toFixed(4)}`}>
                <span className="w-4 text-right font-mono text-[#A1A1AA] flex-shrink-0" style={{ fontSize: '0.65rem' }}>{i + 1}</span>
                <span className="w-24 flex-shrink-0 truncate" style={{ fontSize: '0.72rem', color: m.model.includes('BaseModel') ? 'var(--accent)' : '#52525B', fontWeight: m.model.includes('BaseModel') ? 600 : 400 }}>{m.model}</span>
                <div className="flex-1 h-5 bg-stone-100 rounded-full relative overflow-hidden">
                  <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-700" style={{ width: `${w}%`, background: c.border, opacity: 0.65 }} />
                </div>
                <span className="w-12 text-right font-mono font-medium flex-shrink-0" style={{ fontSize: '0.72rem', color: c.text }}>{m.acc.toFixed(4)}</span>
              </div>
            );
          })}
        </div>
        <div className="mt-4 pt-3 border-t" style={{ borderColor: 'rgba(0,0,0,0.04)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
            <tbody>
              {Array.from({ length: rowCount }).map((_, i) => (
                <tr key={i}>
                  {[left, right].map((col, ci) => {
                    const item = col[i];
                    if (!item) return <td key={ci} style={{ padding: '3px 14px 3px 0', width: '50%' }} />;
                    return (
                      <td key={ci} style={{ padding: '3px 14px 3px 0', width: '50%' }}>
                        <span className="accent-dot" style={{ background: item.dot, width: 8, height: 8, display: 'inline-block', verticalAlign: 'middle', marginRight: 6 }} />
                        <span style={{ fontSize: '0.72rem', color: '#52525B' }}>{item.label}</span>
                        <span style={{ fontSize: '0.65rem', color: '#A1A1AA', marginLeft: 3, fontFamily: '"JetBrains Mono", monospace' }}>{item.count}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card p-5">
        <h3 className="text-sm font-semibold tracking-tight mb-2">核心结论</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-[#52525B] leading-relaxed">
          <div className="flex gap-2"><span className="text-[#52525B] font-bold flex-shrink-0">1.</span> BaseModel 在全部 6 项指标上排名第一 (MSE=1.0628, ACC=0.5901)，验证了基线模型在 4G 流量预测任务上的有效性。</div>
          <div className="flex gap-2"><span className="text-[#52525B] font-bold flex-shrink-0">2.</span> Transformer 架构内部分化显著：iTransformer (ACC=0.5267) 与 PatchTST (ACC=0.5239) 表现接近，但相比传统 Transformer (ACC=0.4140) 提升超 25%，注意力机制设计是核心差异。</div>
          <div className="flex gap-2"><span className="text-[#52525B] font-bold flex-shrink-0">3.</span> 简单基线仍有竞争力：Persistent_24h (ACC=0.4477) 和 XGBoost (ACC=0.4729) 优于多个复杂深度学习模型，强基线是公平评估的前提。</div>
          <div className="flex gap-2"><span className="text-[#52525B] font-bold flex-shrink-0">4.</span> MAPE 指标需谨慎使用：标准化空间中大量模型 MAPE 超过 100%，建议以 Custom ACC 和 MSE 为主要评估依据。</div>
        </div>
      </div>
    </div>
  );
}
