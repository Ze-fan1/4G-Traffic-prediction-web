import { useState, useMemo, useCallback } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { CHANNELS } from '../data/channels';
import { MODEL_COLORS_8 } from '../data/palette';
import predictionCurves from '../data/prediction_curves.js';
import multiWindow from '../data/multi_window.js';
import RevealCard from '../components/RevealCard';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const HOURS = Array.from({ length: 24 }, (_, i) => i + 1);
const ALL_MODELS = Object.keys(predictionCurves.models);

function PredictionCurvesChart({ channelIdx, hiddenModels, onToggleModel, onToggleAll, models, groundTruthModel }) {
  const hours = useMemo(() => HOURS.map(h => `${h}h`), []);
  const chKey = String(channelIdx);
  const allVisible = hiddenModels.size === 0;

  const data = useMemo(() => {
    const refTruth = predictionCurves.models[groundTruthModel][chKey].true;
    const datasets = [
      { label: '真实值', data: refTruth, borderColor: '#18181B', backgroundColor: 'transparent', borderWidth: 2.8, borderDash: [], pointRadius: 0, tension: 0.35, order: 0 },
    ];
    models.forEach((name, i) => {
      const m = predictionCurves.models[name][chKey];
      const globalIdx = ALL_MODELS.indexOf(name);
      datasets.push({
        label: name,
        data: m.pred,
        borderColor: MODEL_COLORS_8[globalIdx % MODEL_COLORS_8.length],
        backgroundColor: MODEL_COLORS_8[globalIdx % MODEL_COLORS_8.length] + '08',
        borderWidth: name.includes('BaseModel') ? 2.2 : 1.3,
        borderDash: name.includes('BaseModel') ? [] : [5, 3],
        pointRadius: 0, tension: 0.35, order: i + 1,
        hidden: hiddenModels.has(name),
      });
    });
    return { labels: hours, datasets };
  }, [hours, chKey, hiddenModels, models, groundTruthModel]);

  const options = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: '预测时刻 (小时)', color: '#A1A1AA' } },
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', callback: (v) => v.toFixed(2) }, title: { display: true, text: '标准化值 (σ)', color: '#A1A1AA' } },
    },
  }), []);

  return (
    <>
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <button
          onClick={onToggleAll}
          className="text-[0.6rem] px-2.5 py-1 rounded-full border border-[rgba(0,0,0,0.08)] bg-white hover:bg-stone-50 transition-colors cursor-pointer text-[#52525B] font-medium"
        >
          {allVisible ? '取消全选' : '全选'}
        </button>
        <div className="flex flex-wrap gap-1.5">
          {models.map((name) => {
            const globalIdx = ALL_MODELS.indexOf(name);
            return (
              <span key={name} className={`chip ${hiddenModels.has(name) ? '' : 'active'}`} style={{ fontSize: '0.68rem' }} onClick={() => onToggleModel(name)}>
                <span className="accent-dot" style={{ background: MODEL_COLORS_8[globalIdx % MODEL_COLORS_8.length], width: 6, height: 6 }} />{name}
              </span>
            );
          })}
        </div>
      </div>
      <div className="chart-box-lg"><Line data={data} options={options} /></div>
    </>
  );
}

function MultiWindowChart() {
  const mwModels = Object.keys(multiWindow.models);
  const windows = multiWindow.windows;

  const data = useMemo(() => {
    const firstModel = multiWindow.models[mwModels[0]];
    const truthData = windows.map(w => firstModel[String(w)].true[12]);
    const datasets = [
      { label: '真实值', data: truthData, borderColor: '#18181B', backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: '#18181B', tension: 0.3, order: 0 },
    ];
    mwModels.forEach((name, i) => {
      const m = multiWindow.models[name];
      const hourIdx = 12;
      datasets.push({
        label: name,
        data: windows.map(w => m[String(w)].pred[hourIdx]),
        borderColor: MODEL_COLORS_8[i],
        backgroundColor: MODEL_COLORS_8[i] + '10',
        borderWidth: name.includes('BaseModel') ? 2.2 : 1.3,
        borderDash: name.includes('BaseModel') ? [] : [5, 3],
        pointRadius: 4, pointBackgroundColor: MODEL_COLORS_8[i], tension: 0.3, order: i + 1,
      });
    });
    return { labels: windows.map(w => `W${w}`), datasets };
  }, []);

  const options = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 12, font: { size: 9 }, color: '#52525B' } } },
    scales: {
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: '窗口编号', color: '#A1A1AA' } },
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', callback: (v) => { if (Math.abs(v) >= 1e6) return (v/1e6).toFixed(1)+'M'; if (Math.abs(v) >= 1000) return (v/1000).toFixed(1)+'k'; if (Math.abs(v) < 0.1) return v.toFixed(4); return v.toFixed(2); } }, title: { display: true, text: '标准化值 (σ)', color: '#A1A1AA' } },
    },
  }), []);

  return <div className="chart-box"><Line data={data} options={options} /></div>;
}

export default function CurvesPage() {
  const [channelIdx, setChannelIdx] = useState(1); // PDCCH 控制信道 — 拟合最佳、最具代表性
  const [hiddenModels, setHiddenModels] = useState(new Set());
  const toggleModel = useCallback((model) => { setHiddenModels(prev => { const next = new Set(prev); if (next.has(model)) next.delete(model); else next.add(model); return next; }); }, []);
  const allVisible = hiddenModels.size === 0;
  const toggleAll = useCallback(() => { if (allVisible) { setHiddenModels(new Set(ALL_MODELS)); } else { setHiddenModels(new Set()); } }, [allVisible]);

  // Compute per-model MAE in this window for ranking analysis
  const chKey = String(channelIdx);
  const modelRanking = useMemo(() => {
    const refTruth = predictionCurves.models[ALL_MODELS[0]]?.[chKey]?.true;
    if (!refTruth) return [];
    return ALL_MODELS
      .map(name => {
        const m = predictionCurves.models[name]?.[chKey];
        if (!m) return null;
        const errs = m.pred.map((p, i) => Math.abs(p - refTruth[i]));
        const mae = errs.reduce((s, v) => s + v, 0) / errs.length;
        return { name, mae };
      })
      .filter(Boolean)
      .sort((a, b) => a.mae - b.mae);
  }, [chKey]);

  const topModel = modelRanking[0]?.name || '';
  const topMAE = modelRanking[0]?.mae?.toFixed(3) || '';
  const lastModel = modelRanking[modelRanking.length - 1]?.name || '';
  const lastMAE = modelRanking[modelRanking.length - 1]?.mae?.toFixed(3) || '';

  return (
    <div className="page-enter">
      {/* Main chart */}
      <RevealCard className="mt-5">
        <div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h3 className="text-sm font-semibold tracking-tight">
              24小时预测曲线对比 <span style={{color:'#16A34A',fontSize:'0.7rem',fontWeight:700}}>【标准化空间 σ】</span>
            </h3>
            <p className="text-[0.65rem] text-[#A1A1AA] mt-0.5">
              Window #{predictionCurves.window} · 代表性窗口（秩相关最优）· {ALL_MODELS.length} 个模型公平对比 · 点击标签显隐曲线
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[0.65rem] text-[#A1A1AA]">通道:</span>
            <select value={channelIdx} onChange={(e) => setChannelIdx(Number(e.target.value))} className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer focus:outline-none focus:border-[#3B82F6] transition-colors">
              {CHANNELS.map((ch, i) => (<option key={ch.id} value={i}>{ch.name} — {ch.desc}</option>))}
            </select>
          </div>
        </div>
        <PredictionCurvesChart channelIdx={channelIdx} hiddenModels={hiddenModels} onToggleModel={toggleModel} onToggleAll={toggleAll} models={ALL_MODELS} groundTruthModel={ALL_MODELS[0]} />

        {/* Dynamic analysis below chart */}
        {modelRanking.length > 1 && (
          <div className="mt-4 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
            <p className="text-xs text-[#52525B] leading-relaxed">
              <strong>📊 当前窗口分析：</strong>
              在 Window #{predictionCurves.window} 的{CHANNELS[channelIdx]?.name || '当前'}通道上，
              <strong style={{color:'#16A34A'}}>{topModel}</strong> 的 24 小时平均绝对误差最低（MAE={topMAE}σ），
              预测曲线与真实值（黑色实线）贴合最紧密；
              其次为{modelRanking[1]?.name || ''}（MAE={modelRanking[1]?.mae?.toFixed(3) || ''}σ）。
              表现最弱的 <strong style={{color:'#DC2626'}}>{lastModel}</strong>（MAE={lastMAE}σ）与前两名差距约 {modelRanking.length >= 2 ? (modelRanking[modelRanking.length-1].mae / modelRanking[0].mae).toFixed(1) : '—'} 倍。
              {topModel.includes('BaseModel') ? '★ BaseModel 凭借自研多尺度特征融合架构，在该窗口继续保持领先。' : ''}
            </p>
          </div>
        )}
      </div>
      </RevealCard>

      {/* Explanation card */}
      <RevealCard className="mt-4" delay={80}>
        <div className="card p-5">
        <h3 className="text-sm font-semibold tracking-tight mb-2">图表说明</h3>
        <div className="text-xs text-[#52525B] leading-relaxed space-y-1.5">
          <p><strong>黑色实线</strong> = 测试集真实值（标准化空间，单位：σ），来源于 4G 基站 RAN 侧实测数据，Window #{predictionCurves.window}</p>
          <p><strong>标准化空间（σ）解读：</strong>0 表示等于历史均值水平，+2 表示高于均值 2 个标准差（流量高峰），−1 表示低于均值 1 个标准差（流量低谷）。所有模型在相同的 StandardScaler（拟合于训练集）下进行标准化，确保预测值在同一尺度上可直接对比。</p>
          <p><strong>代表性窗口选择：</strong>在全部 {predictionCurves.n_windows || 5378} 个滑动窗口中，通过秩相关分析（0.8×Spearman秩相关 + 0.2×方差多样性）选取最优代表窗口（corr≈0.95）。该窗口的模型优劣排序与全局平均高度一致，能够公平地代表模型的典型表现，而非选取某个对特定模型有利或不利的极端窗口。</p>
          <p className="mt-2"><strong>公平基准测试平台：</strong>所有模型共享相同的标准化数据管道——相同的 StandardScaler、相同的训练/测试集划分、相同的 24→24 滑动窗口协议。这是一个面向 4G 流量预测任务的标准化模型 benchmark，而非各模型单独调优后的非公平对比。</p>
          <p className="text-[#A1A1AA] mt-1">💡 使用上方通道下拉菜单切换不同 KPI 指标的预测表现。BaseModel 以实线标注，其余模型以虚线区分。曲线密集时可使用"全选/取消全选"按钮快速切换。</p>
        </div>
      </div>
      </RevealCard>

      {/* Multi-window chart */}
      <RevealCard className="mt-4" delay={160}>
        <div className="card p-5">
        <h3 className="text-sm font-semibold tracking-tight mb-0.5">多时间窗口 · 模型稳定性对比</h3>
        <p className="text-[0.65rem] text-[#A1A1AA] mb-1">6 个窗口采样（标准化空间）· 取每个窗口第 12 小时预测值 · 真实值 vs 各模型预测</p>
        <MultiWindowChart />
        <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
          <p className="text-xs text-[#52525B] leading-relaxed">
            <strong>📊 多窗口解读：</strong>
            横轴展示从 5378 个窗口中采样的 6 个代表性窗口，纵轴为标准化后的总流量（第 12 小时预测值）。
            如果某模型的折线与黑色真实值折线在各窗口间始终保持相近的趋势和距离，说明该模型在不同时间模式（高峰/低谷/过渡期）下均能稳定预测；
            若某些窗口偏离较大，则提示模型对该类时间模式泛化不足。
          </p>
        </div>
      </div>
      </RevealCard>
    </div>
  );
}
