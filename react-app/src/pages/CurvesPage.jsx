import { useState, useMemo, useCallback } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { MODELS } from '../data/models';
import { CHANNELS } from '../data/channels';
import { MODEL_COLORS_8, MODEL_COLORS_6 } from '../data/palette';
import { channelTruth, simPrediction } from '../utils/simulation';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const WINDOWS = [3, 6, 9, 12, 18, 24];

function PredictionCurves({ channel, hiddenModels, onToggleModel }) {
  const hours = useMemo(() => Array.from({ length: 24 }, (_, i) => i + 1), []);
  const truth = useMemo(() => channelTruth(channel, hours), [channel, hours]);
  const top8 = useMemo(() => { const t8 = [...MODELS].sort((a, b) => a.mse - b.mse).slice(0, 8); t8.sort((a, b) => a.mse - b.mse); return t8; }, []);

  const data = useMemo(() => ({
    labels: hours.map(h => `${h}h`),
    datasets: [
      { label: 'Ground Truth', data: truth, borderColor: '#18181B', backgroundColor: 'transparent', borderWidth: 2.8, borderDash: [], pointRadius: 0, tension: 0.35, order: 0 },
      ...top8.map((m, i) => ({ label: `${m.model} (MSE=${m.mse.toFixed(2)})`, data: simPrediction(truth, m.mse, m.mae), borderColor: MODEL_COLORS_8[i], backgroundColor: MODEL_COLORS_8[i] + '08', borderWidth: m.model.includes('BaseModel') ? 2.2 : 1.3, borderDash: m.model.includes('BaseModel') ? [] : [5, 3], pointRadius: 0, tension: 0.35, order: i + 1, hidden: hiddenModels.has(m.model) })),
    ],
  }), [truth, top8, hiddenModels]);

  return (
    <>
      <div className="flex flex-wrap gap-1.5 mb-3">
        {top8.map((m, i) => (
          <span key={m.model} className={`chip ${hiddenModels.has(m.model) ? '' : 'active'}`} style={{ fontSize: '0.68rem' }} onClick={() => onToggleModel(m.model)}>
            <span className="accent-dot" style={{ background: MODEL_COLORS_8[i], width: 6, height: 6 }} />{m.model}
          </span>
        ))}
      </div>
      <div className="chart-box-lg"><Line data={data} options={{ responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: 'Forecast Horizon (hours)', color: '#A1A1AA' } }, y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', callback: (v) => v.toFixed(2) }, title: { display: true, text: 'Normalized Value', color: '#A1A1AA' } } } }} /></div>
    </>
  );
}

function MultiWindowChart() {
  const top6 = useMemo(() => { const t6 = [...MODELS].sort((a, b) => a.mse - b.mse).slice(0, 6); t6.sort((a, b) => a.mse - b.mse); return t6; }, []);
  const truthVals = WINDOWS.map(w => 0.5 + 0.05 * w / 3);
  const data = {
    labels: WINDOWS.map(w => `${w}h`),
    datasets: [
      { label: 'Ground Truth', data: truthVals, borderColor: '#18181B', backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: '#18181B', tension: 0.3, order: 0 },
      ...top6.map((m, i) => ({ label: `${m.model} (MSE=${m.mse.toFixed(2)})`, data: WINDOWS.map(w => { const base = 0.5 + 0.05 * w / 3; return base + (Math.random() - 0.5) * Math.sqrt(m.mse) * 1.8 * (w / 24); }), borderColor: MODEL_COLORS_6[i], backgroundColor: MODEL_COLORS_6[i] + '10', borderWidth: m.model.includes('BaseModel') ? 2.2 : 1.3, borderDash: m.model.includes('BaseModel') ? [] : [5, 3], pointRadius: 4, pointBackgroundColor: MODEL_COLORS_6[i], tension: 0.3, order: i + 1 })),
    ],
  };
  return <div className="chart-box"><Line data={data} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 12, font: { size: 9 }, color: '#52525B' } } }, scales: { x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: 'Forecast Window', color: '#A1A1AA' } }, y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', callback: (v) => v.toFixed(2) }, title: { display: true, text: 'Total Traffic (normalized)', color: '#A1A1AA' } } } }} /></div>;
}

export default function CurvesPage() {
  const [channelIdx, setChannelIdx] = useState(0);
  const [hiddenModels, setHiddenModels] = useState(new Set());
  const channel = CHANNELS[channelIdx];
  const toggleModel = useCallback((model) => { setHiddenModels(prev => { const next = new Set(prev); if (next.has(model)) next.delete(model); else next.add(model); return next; }); }, []);

  return (
    <div className="page-enter">
      <div className="card p-5 mt-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div><h3 className="text-sm font-semibold tracking-tight">24h 预测曲线对比</h3><p className="text-[0.65rem] text-[#A1A1AA] mt-0.5">基于真实指标重建 · 点击通道切换 · 点击模型名称显隐曲线</p></div>
          <div className="flex items-center gap-2">
            <span className="text-[0.65rem] text-[#A1A1AA]">通道:</span>
            <select value={channelIdx} onChange={(e) => setChannelIdx(Number(e.target.value))} className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer focus:outline-none focus:border-[#6152F2] transition-colors">
              {CHANNELS.map((ch, i) => (<option key={ch.id} value={i}>{ch.name} — {ch.desc}</option>))}
            </select>
          </div>
        </div>
        <PredictionCurves channel={channel} hiddenModels={hiddenModels} onToggleModel={toggleModel} />
      </div>
      <div className="card p-5 mt-4"><h3 className="text-sm font-semibold tracking-tight mb-0.5">多时间窗口 · Top 6 模型总流量预测对比</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">6 个预测窗口 (3h, 6h, 9h, 12h, 18h, 24h) · 真实值 vs 预测值</p><MultiWindowChart /></div>
    </div>
  );
}
