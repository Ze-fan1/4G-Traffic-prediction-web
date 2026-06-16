import { useState, useMemo, useCallback } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { CHANNELS } from '../data/channels';
import { MODEL_COLORS_8 } from '../data/palette';
import predictionCurves from '../data/prediction_curves.js';
import multiWindow from '../data/multi_window.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const HOURS = Array.from({ length: 24 }, (_, i) => i + 1);
const ALL_MODELS = Object.keys(predictionCurves.models);
// All non-BaseModel models share DL reference truth
const DL_MODELS = ALL_MODELS.filter(n => !n.includes('BaseModel'));
const BASEMODEL = ALL_MODELS.find(n => n.includes('BaseModel'));

function PredictionCurvesChart({ channelIdx, hiddenModels, onToggleModel, models, groundTruthModel }) {
  const hours = useMemo(() => HOURS.map(h => `${h}h`), []);
  const chKey = String(channelIdx);

  const data = useMemo(() => {
    // Ground truth from the reference model
    const refTruth = predictionCurves.models[groundTruthModel][chKey].true;
    const datasets = [
      { label: 'Ground Truth', data: refTruth, borderColor: '#18181B', backgroundColor: 'transparent', borderWidth: 2.8, borderDash: [], pointRadius: 0, tension: 0.35, order: 0 },
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
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: 'Forecast Horizon (hours)', color: '#A1A1AA' } },
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', callback: (v) => v.toFixed(2) }, title: { display: true, text: '标准化值 (σ)', color: '#A1A1AA' } },
    },
  }), [channelIdx]);

  return (
    <>
      <div className="flex flex-wrap gap-1.5 mb-3">
        {models.map((name) => {
          const globalIdx = ALL_MODELS.indexOf(name);
          return (
            <span key={name} className={`chip ${hiddenModels.has(name) ? '' : 'active'}`} style={{ fontSize: '0.68rem' }} onClick={() => onToggleModel(name)}>
              <span className="accent-dot" style={{ background: MODEL_COLORS_8[globalIdx % MODEL_COLORS_8.length], width: 6, height: 6 }} />{name}
            </span>
          );
        })}
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
      { label: 'Ground Truth', data: truthData, borderColor: '#18181B', backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: '#18181B', tension: 0.3, order: 0 },
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
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: 'Window Index', color: '#A1A1AA' } },
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', callback: (v) => { if (Math.abs(v) >= 1e6) return (v/1e6).toFixed(1)+'M'; if (Math.abs(v) >= 1000) return (v/1000).toFixed(1)+'k'; if (Math.abs(v) < 0.1) return v.toFixed(4); return v.toFixed(2); } }, title: { display: true, text: '标准化值 (σ)', color: '#A1A1AA' } },
    },
  }), []);

  return <div className="chart-box"><Line data={data} options={options} /></div>;
}

export default function CurvesPage() {
  const [channelIdx, setChannelIdx] = useState(6); // 总流量
  const [hiddenModels, setHiddenModels] = useState(new Set());
  const toggleModel = useCallback((model) => { setHiddenModels(prev => { const next = new Set(prev); if (next.has(model)) next.delete(model); else next.add(model); return next; }); }, []);

  return (
    <div className="page-enter">
      {/* Main chart: 8 DL models (shared test data) */}
      <div className="card p-5 mt-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div><h3 className="text-sm font-semibold tracking-tight">24h 预测曲线对比 <span style={{color:'#16A34A',fontSize:'0.7rem',fontWeight:700}}>【标准化空间 σ】</span></h3><p className="text-[0.65rem] text-[#A1A1AA] mt-0.5">Window #{predictionCurves.window} · 代表性窗口（秩相关最优）· 8 个 DL 模型共享相同测试数据 · 点击模型名称显隐曲线</p></div>
          <div className="flex items-center gap-2">
            <span className="text-[0.65rem] text-[#A1A1AA]">通道:</span>
            <select value={channelIdx} onChange={(e) => setChannelIdx(Number(e.target.value))} className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer focus:outline-none focus:border-[#6152F2] transition-colors">
              {CHANNELS.map((ch, i) => (<option key={ch.id} value={i}>{ch.name} — {ch.desc}</option>))}
            </select>
          </div>
        </div>
        <PredictionCurvesChart channelIdx={channelIdx} hiddenModels={hiddenModels} onToggleModel={toggleModel} models={DL_MODELS} groundTruthModel={DL_MODELS[0]} />
      </div>

      {/* BaseModel standalone card */}
      {BASEMODEL && (
        <div className="card p-5 mt-4" style={{ borderLeft: '3px solid #16A34A' }}>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h3 className="text-sm font-semibold tracking-tight">
                <span style={{ color: '#16A34A' }}>★ BaseModel</span> 预测 vs 真值
                <span style={{ color: '#16A34A', fontSize: '0.7rem', fontWeight: 700, marginLeft: 8 }}>【标准化空间 σ】</span>
              </h3>
              <p className="text-[0.65rem] text-[#A1A1AA] mt-0.5">
                Window #{predictionCurves.window} · BaseModel 与自身测试数据对比（单独窗口对齐）
              </p>
            </div>
          </div>
          <PredictionCurvesChart channelIdx={channelIdx} hiddenModels={hiddenModels} onToggleModel={toggleModel} models={[BASEMODEL]} groundTruthModel={BASEMODEL} />
        </div>
      )}

      <div className="card p-5 mt-4">
        <h3 className="text-sm font-semibold tracking-tight mb-2">图表说明</h3>
        <div className="text-xs text-[#52525B] leading-relaxed space-y-1.5">
          <p><strong>黑色实线</strong> = 测试集真实值（标准化后，单位：σ）— Window #{predictionCurves.window}</p>
          <p><strong>代表性窗口选择：</strong>在全部 {predictionCurves.n_windows || 5378} 个窗口中，选择与整体 MAE 排名秩相关系数最高的窗口（corr≈0.98），确保该窗口的模型相对排序与全局平均一致。</p>
          <p className="mt-2"><strong>主图（8 DL 模型）：</strong>iTransformer、PatchTST、SegRNN、DLinear、TimesNet、Autoformer、Transformer、IBM TTM 共享相同的标准化测试数据，可在此空间直接对比。</p>
          <p><strong>★ BaseModel 单独卡片：</strong>BaseModel 使用原始空间数据（经 swap+scaler 转换为标准化空间），其测试数据窗口与 DL 模型不对齐（同一窗口索引对应不同时间段），因此单独展示与自身真值的对比。</p>
          <p className="text-[#A1A1AA] mt-1">💡 Y 轴单位为标准差（σ）。正值表示高于历史均值，负值表示低于历史均值。切换通道可查看不同指标的预测表现。</p>
        </div>
      </div>

      <div className="card p-5 mt-4"><h3 className="text-sm font-semibold tracking-tight mb-0.5">多时间窗口 · 模型总流量预测对比</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">6 个窗口采样 · 标准化空间 · 真实值 vs 预测值 (midpoint hour=12)</p><MultiWindowChart /></div>
    </div>
  );
}
