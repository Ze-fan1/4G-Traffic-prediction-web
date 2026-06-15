import { useState, useMemo, useCallback } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { CHANNELS } from '../data/channels';
import { MODEL_COLORS_8, MODEL_COLORS_6 } from '../data/palette';
import predictionCurves from '../data/prediction_curves.js';
import multiWindow from '../data/multi_window.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const HOURS = Array.from({ length: 24 }, (_, i) => i + 1);
const MODEL_NAMES = Object.keys(predictionCurves.models);

function PredictionCurvesChart({ channelIdx, hiddenModels, onToggleModel }) {
  const hours = useMemo(() => HOURS.map(h => `${h}h`), []);
  const chKey = String(channelIdx);

  const data = useMemo(() => {
    const datasets = [
      { label: 'Ground Truth', data: predictionCurves.models[MODEL_NAMES[0]][chKey].true, borderColor: '#18181B', backgroundColor: 'transparent', borderWidth: 2.8, borderDash: [], pointRadius: 0, tension: 0.35, order: 0 },
    ];
    MODEL_NAMES.forEach((name, i) => {
      const m = predictionCurves.models[name][chKey];
      datasets.push({
        label: name,
        data: m.pred,
        borderColor: MODEL_COLORS_8[i],
        backgroundColor: MODEL_COLORS_8[i] + '08',
        borderWidth: name.includes('BaseModel') ? 2.2 : 1.3,
        borderDash: name.includes('BaseModel') ? [] : [5, 3],
        pointRadius: 0, tension: 0.35, order: i + 1,
        hidden: hiddenModels.has(name),
      });
    });
    return { labels: hours, datasets };
  }, [hours, chKey, hiddenModels]);

  const options = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: 'Forecast Horizon (hours)', color: '#A1A1AA' } },
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', callback: (v) => { if (Math.abs(v) >= 1e6) return (v/1e6).toFixed(1)+'M'; if (Math.abs(v) >= 1000) return (v/1000).toFixed(1)+'k'; if (Math.abs(v) < 0.1) return v.toFixed(4); return v.toFixed(2); } }, title: { display: true, text: CHANNELS[channelIdx]?.desc || '', color: '#A1A1AA' } },
    },
  }), [channelIdx]);

  return (
    <>
      <div className="flex flex-wrap gap-1.5 mb-3">
        {MODEL_NAMES.map((name, i) => (
          <span key={name} className={`chip ${hiddenModels.has(name) ? '' : 'active'}`} style={{ fontSize: '0.68rem' }} onClick={() => onToggleModel(name)}>
            <span className="accent-dot" style={{ background: MODEL_COLORS_8[i], width: 6, height: 6 }} />{name}
          </span>
        ))}
      </div>
      <div className="chart-box-lg"><Line data={data} options={options} /></div>
    </>
  );
}

function MultiWindowChart() {
  const mwModels = Object.keys(multiWindow.models);
  const windows = multiWindow.windows;

  const data = useMemo(() => {
    // Use first model's true as ground truth
    const firstModel = multiWindow.models[mwModels[0]];
    const truthData = windows.map(w => firstModel[String(w)].true[12]); // midpoint hour
    const datasets = [
      { label: 'Ground Truth', data: truthData, borderColor: '#18181B', backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: '#18181B', tension: 0.3, order: 0 },
    ];
    mwModels.forEach((name, i) => {
      const m = multiWindow.models[name];
      const hourIdx = 12; // midpoint of 24h
      datasets.push({
        label: name,
        data: windows.map(w => m[String(w)].pred[hourIdx]),
        borderColor: MODEL_COLORS_6[i],
        backgroundColor: MODEL_COLORS_6[i] + '10',
        borderWidth: name.includes('BaseModel') ? 2.2 : 1.3,
        borderDash: name.includes('BaseModel') ? [] : [5, 3],
        pointRadius: 4, pointBackgroundColor: MODEL_COLORS_6[i], tension: 0.3, order: i + 1,
      });
    });
    return { labels: windows.map(w => `W${w}`), datasets };
  }, []);

  const options = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 12, font: { size: 9 }, color: '#52525B' } } },
    scales: {
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: 'Window Index', color: '#A1A1AA' } },
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', callback: (v) => { if (Math.abs(v) >= 1e6) return (v/1e6).toFixed(1)+'M'; if (Math.abs(v) >= 1000) return (v/1000).toFixed(1)+'k'; if (Math.abs(v) < 0.1) return v.toFixed(4); return v.toFixed(2); } }, title: { display: true, text: 'Total Traffic', color: '#A1A1AA' } },
    },
  }), []);

  return <div className="chart-box"><Line data={data} options={options} /></div>;
}

export default function CurvesPage() {
  const [channelIdx, setChannelIdx] = useState(1);
  const [hiddenModels, setHiddenModels] = useState(new Set());
  const toggleModel = useCallback((model) => { setHiddenModels(prev => { const next = new Set(prev); if (next.has(model)) next.delete(model); else next.add(model); return next; }); }, []);

  return (
    <div className="page-enter">
      <div className="card p-5 mt-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div><h3 className="text-sm font-semibold tracking-tight">24h 预测曲线对比</h3><p className="text-[0.65rem] text-[#A1A1AA] mt-0.5">Window #{predictionCurves.window} · 真实数据 · 点击模型名称显隐曲线</p></div>
          <div className="flex items-center gap-2">
            <span className="text-[0.65rem] text-[#A1A1AA]">通道:</span>
            <select value={channelIdx} onChange={(e) => setChannelIdx(Number(e.target.value))} className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer focus:outline-none focus:border-[#6152F2] transition-colors">
              {CHANNELS.map((ch, i) => (<option key={ch.id} value={i}>{ch.name} — {ch.desc}</option>))}
            </select>
          </div>
        </div>
        <PredictionCurvesChart channelIdx={channelIdx} hiddenModels={hiddenModels} onToggleModel={toggleModel} />
      </div>
      <div className="card p-5 mt-4"><h3 className="text-sm font-semibold tracking-tight mb-0.5">多时间窗口 · Top 6 模型ERAB流量预测</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">6 个窗口采样 · 真实值 vs 预测值 (midpoint hour=12)</p><MultiWindowChart /></div>
    </div>
  );
}
