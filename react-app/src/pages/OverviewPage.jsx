import { CAT_PALETTE } from '../data/palette';
import ModelMarquee from '../components/ModelMarquee';
import RevealCard from '../components/RevealCard';
import useBenchmarkModels from '../hooks/useBenchmarkModels';

function getCat(model) {
  return CAT_PALETTE[model.cat] || CAT_PALETTE['Statistical'];
}

export default function OverviewPage() {
  const { models: benchmarkModels } = useBenchmarkModels();
  const MODELS = benchmarkModels;
  const CATS = [...new Set(MODELS.map(model => model.cat))];
  const sorted = [...MODELS].sort((a, b) => a.mse - b.mse);
  const bestMSE = sorted[0]?.mse ?? null;
  const bestMSEModel = [...MODELS].sort((a, b) => a.mse - b.mse)[0];
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
      {/* ═══ Hero: 蓝粉渐变 ═══ */}
      <div
        className="relative -mx-3 sm:-mx-5 mt-12 pt-12 pb-0 mb-0 overflow-hidden flex items-center justify-center"
        style={{
          height: 'min(68vh, 600px)',
          minHeight: 420,
          background: 'linear-gradient(170deg, #F6F5FA 0%, #F6F5FA 16%, #EFF6FF 25%, #F5F3FF 38%, #FDF2F8 50%, #EFF6FF 64%, #F5F3FF 80%, #F6F5FA 100%)',
        }}
      >
        {/* 顶部柔和过渡 — 从 body 背景渐进浮现 */}
        <div className="hero-fade-top" />

        {/* 柔光椭圆 — 蓝粉淡色 */}
        <div className="absolute inset-0 z-0 pointer-events-none" style={{
          background: 'radial-gradient(ellipse 750px 520px at 60% 18%, rgba(59,130,246,0.08) 0%, transparent 60%), radial-gradient(ellipse 500px 400px at 30% 65%, rgba(236,72,153,0.06) 0%, transparent 55%), radial-gradient(ellipse 450px 380px at 78% 55%, rgba(139,92,246,0.06) 0%, transparent 55%)',
        }} />

        {/* 欢迎文字 — 极慢动效 */}
        <div className="hero-content relative z-[1] flex flex-col items-center justify-center px-4 text-center">
          <p
            className="text-xs sm:text-sm uppercase tracking-[0.22em] text-[#A1A1AA] mb-5"
            style={{ animation: 'fadeIn 1.6s 0.3s cubic-bezier(0.16,1,0.3,1) both' }}
          >
            4G Traffic Benchmark Platform
          </p>
          <h1
            className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-[#1A1A2E] leading-[1.08]"
            style={{
              animation: 'fadeIn 1.8s 0.6s cubic-bezier(0.16,1,0.3,1) both',
            }}
          >
            4G Traffic<br />
            <span style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6, #EC4899)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Prediction
            </span>
            {' '}Benchmark
          </h1>
          <p
            className="mt-5 text-sm text-[#A1A1AA] tracking-wide text-center"
            style={{ animation: 'fadeIn 1.4s 1.0s cubic-bezier(0.16,1,0.3,1) both' }}
          >
            8 特征通道 · 多变量输入 · 24h 预测 · 3h 步长 · 已严格重评测 {MODELS.length} 个模型
          </p>
        </div>

        {/* 底部模糊过渡 */}
        <div className="hero-fade-bottom" />
      </div>

      {/* ═══ 28 模型滚动展示墙 — 参考 Devraj Chatribin ═══ */}
      <div className="relative z-10 mt-2 mb-6">
        <ModelMarquee models={MODELS} />
      </div>

      {/* ═══ Hero 卡片 — 等高统一排版 ═══ */}
      <div className="relative z-10 mt-6 mb-6">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {[
            { label: '最低 MSE 模型', value: bestMSEModel?.model || '待评测', color: 'var(--accent)' },
            { label: '最低 MSE', value: bestMSEModel ? bestMSEModel.mse.toFixed(4) : '—', color: 'var(--accent)' },
            { label: '已验证模型', value: String(MODELS.length) },
            { label: '实验协议', value: '同小区 · 连续48h · 24→24' },
          ].map((item, i) => (
            <RevealCard key={i}>
              <div className="card shape-blur p-5 flex flex-col justify-center" style={{ minHeight: 88 }}>
                <p className="text-[0.65rem] text-[#A1A1AA] uppercase tracking-wider font-medium mb-1">
                  {item.label}
                </p>
                <p
                  className="text-lg font-bold tracking-tight leading-tight"
                  style={item.color ? { color: item.color } : { color: '#1A1A2E' }}
                >
                  {item.value}
                </p>
              </div>
            </RevealCard>
          ))}
        </div>
      </div>

      {/* ═══ MSE ranking ═══ */}
      <RevealCard className="mb-5">
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-3">
            MSE 排名 · 全部 {MODELS.length} 已验证模型
          </h3>
          <div className="space-y-1.5">
            {sorted.length ? sorted.map((m, i) => {
              const c = getCat(m);
              const w = bestMSE ? (bestMSE / m.mse) * 100 : 0;
              return (
                <div
                  key={m.model}
                  className="flex items-center gap-2.5 text-xs group cursor-default"
                  title={`${m.model} · ${m.cat}\nMSE: ${m.mse.toFixed(4)} · MAE: ${m.mae.toFixed(4)}`}
                >
                  <span className="w-4 text-right font-mono text-[#A1A1AA] flex-shrink-0" style={{ fontSize: '0.65rem' }}>
                    {i + 1}
                  </span>
                  <span
                    className="w-24 flex-shrink-0 truncate"
                    style={{
                      fontSize: '0.72rem',
                      color: m.model.includes('BaseModel') ? 'var(--accent)' : '#52525B',
                      fontWeight: m.model.includes('BaseModel') ? 600 : 400,
                    }}
                  >
                    {m.model}
                  </span>
                  <div className="flex-1 h-5 bg-stone-100 rounded-full relative overflow-hidden">
                    <div
                      className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
                      style={{ width: `${w}%`, background: c.border, opacity: 0.65 }}
                    />
                  </div>
                  <span
                    className="w-12 text-right font-mono font-medium flex-shrink-0"
                    style={{ fontSize: '0.72rem', color: c.text }}
                  >
                    {m.mse.toFixed(4)}
                  </span>
                </div>
              );
            }) : <p className="text-xs text-[#A1A1AA]">暂时没有完成严格评测的模型。</p>}
          </div>

          {/* 图例 */}
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
                          <span
                            className="accent-dot"
                            style={{
                              background: item.dot, width: 8, height: 8,
                              display: 'inline-block', verticalAlign: 'middle', marginRight: 6,
                            }}
                          />
                          <span style={{ fontSize: '0.72rem', color: '#52525B' }}>{item.label}</span>
                          <span style={{ fontSize: '0.65rem', color: '#A1A1AA', marginLeft: 3, fontFamily: '"JetBrains Mono", monospace' }}>
                            {item.count}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </RevealCard>

      {/* ═══ 核心发现 ═══ */}
      <RevealCard>
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-2">核心发现</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-[#52525B] leading-relaxed">
            {[
              ['1.', '已完成重评测的外部 BaseModel 当前 Custom ACC 为 0.5853；Naive 与 Persistent 24h 的主要比较指标为 MSE、MAE 与 RMSE。'],
              ['2.', '全部模型将以“同一小区、连续48小时”的窗口协议重新评测，避免跨小区或时间缺口造成的无效样本。'],
              ['3.', '所有公开曲线均附带模型、窗口所属小区和起始时间，便于追溯实验结果。'],
              ['4.', '任意上传数据使用独立的通用预测模式，并先在文件末段执行留出回测；不会被伪造成 4G 特征。'],
            ].map(([num, text]) => (
              <div key={num} className="flex gap-2">
                <span className="text-[#52525B] font-bold flex-shrink-0">{num}</span>
                <p>{text}</p>
              </div>
            ))}
          </div>
        </div>
      </RevealCard>
    </div>
  );
}
