import { useEffect, useMemo, useState } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { CHANNELS } from '../data/channels';
import { getModelColor } from '../data/palette';
import RevealCard from '../components/RevealCard';
import useBenchmarkModels from '../hooks/useBenchmarkModels';
import CurvesPage from './CurvesPage';
import PerformancePage from './PerformancePage';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

const API_BASE = import.meta.env.VITE_API_URL || '/api';
const HOURS = Array.from({ length: 24 }, (_, index) => `${index + 1}h`);
const WINDOW_OPTIONS = [0, 11, 100, 1000, 2000];
const curveCache = new Map();
const PREFERENCES_KEY = 'current_compare_preferences_v1';

function loadPreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem(PREFERENCES_KEY) || '{}');
    return {
      windowIndex: WINDOW_OPTIONS.includes(saved.windowIndex) ? saved.windowIndex : 11,
      channelIndex: Number.isInteger(saved.channelIndex) && saved.channelIndex >= 0 && saved.channelIndex < CHANNELS.length ? saved.channelIndex : 0,
      scaleMode: saved.scaleMode === 'robust' ? 'robust' : 'visible',
    };
  } catch { return { windowIndex: 11, channelIndex: 0, scaleMode: 'visible' }; }
}

function loadCurve(modelName, windowIndex) {
  const cacheKey = `${modelName}:${windowIndex}`;
  return fetch(`${API_BASE}/preset-curves/${encodeURIComponent(modelName)}?window=${windowIndex}`)
    .then(response => response.ok ? response.json() : null)
    .then(data => {
      if (data?.available && data.curves) {
        curveCache.set(cacheKey, data);
        try { localStorage.setItem(`benchmark_curve_v2:${cacheKey}`, JSON.stringify(data)); } catch {}
        return data;
      }
      return curveCache.get(cacheKey) || (() => {
        try { return JSON.parse(localStorage.getItem(`benchmark_curve_v2:${cacheKey}`) || 'null'); } catch { return null; }
      })();
    })
    .catch(() => curveCache.get(cacheKey) || (() => {
      try { return JSON.parse(localStorage.getItem(`benchmark_curve_v2:${cacheKey}`) || 'null'); } catch { return null; }
    })());
}

function CurrentCompare() {
  const { models } = useBenchmarkModels();
  const [preferences, setPreferences] = useState(loadPreferences);
  const { channelIndex, windowIndex, scaleMode } = preferences;
  const [curves, setCurves] = useState({});
  const [visibleModels, setVisibleModels] = useState([]);

  const updatePreference = (field, value) => setPreferences(current => ({ ...current, [field]: value }));

  useEffect(() => {
    try { localStorage.setItem(PREFERENCES_KEY, JSON.stringify(preferences)); } catch {}
  }, [preferences]);

  useEffect(() => {
    setVisibleModels(current => {
      const available = new Set(models.map(model => model.model));
      const kept = current.filter(name => available.has(name));
      const additions = models.map(model => model.model).filter(name => !kept.includes(name));
      return [...kept, ...additions];
    });
  }, [models]);

  useEffect(() => {
    let cancelled = false;
    setCurves({});
    Promise.all(models.map(async model => [model.model, await loadCurve(model.model, windowIndex)])).then(entries => {
      if (!cancelled) setCurves(Object.fromEntries(entries.filter(([, data]) => data?.available && data.curves)));
    }).catch(() => {
      if (!cancelled) setCurves({});
    });
    return () => { cancelled = true; };
  }, [models, windowIndex]);

  const hiddenModels = models.map(model => model.model).filter(name => !visibleModels.includes(name));
  const channel = CHANNELS[channelIndex];
  const chartData = useMemo(() => {
    const first = models.find(model => curves[model.model]?.curves?.[channel.name]);
    const datasets = [];
    if (first) {
      datasets.push({
        label: '真实值', data: curves[first.model].curves[channel.name].truth,
        borderColor: '#18181B', borderDash: [6, 3], borderWidth: 2.2, pointRadius: 0,
      });
    }
    visibleModels.forEach(modelName => {
      const curve = curves[modelName]?.curves?.[channel.name];
      if (!curve) return;
      datasets.push({
        label: modelName, data: curve.pred, borderColor: getModelColor(modelName),
        borderWidth: 1.8, pointRadius: 0, tension: 0.3,
      });
    });
    return { labels: HOURS, datasets };
  }, [channel.name, curves, models, visibleModels]);

  const yRange = useMemo(() => {
    if (scaleMode !== 'robust' || !chartData.datasets.length) return {};
    const values = chartData.datasets.flatMap(dataset => dataset.data).filter(Number.isFinite).sort((a, b) => a - b);
    if (values.length < 4) return {};
    const low = values[Math.floor((values.length - 1) * 0.02)];
    const high = values[Math.ceil((values.length - 1) * 0.98)];
    const padding = Math.max((high - low) * 0.12, 0.05);
    return { min: low - padding, max: high + padding };
  }, [chartData, scaleMode]);

  const firstCurve = Object.values(curves)[0];
  const windowText = firstCurve
    ? `窗口 #${firstCurve.window_idx} / ${firstCurve.total_windows}${firstCurve.window_ref ? ` · 小区 ${firstCurve.window_ref.cell_id} · ${firstCurve.window_ref.start}` : ''}`
    : `窗口 #${windowIndex}`;

  return (
    <div className="page-enter mt-5 space-y-4">
      <RevealCard><div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div>
            <h2 className="text-lg font-semibold">当前本地评测曲线</h2>
            <p className="mt-1 text-xs text-[#52525B]">同一窗口、同一通道、同一标准化坐标；ACC 是全部 3,514 个窗口与全部通道的汇总指标，不等同于这一张局部曲线。</p>
            <p className="mt-1 text-[0.65rem] text-[#A1A1AA]">{windowText}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select value={windowIndex} onChange={event => updatePreference('windowIndex', Number(event.target.value))} className="text-xs px-3 py-2 rounded-lg border bg-white">
              {WINDOW_OPTIONS.map(value => <option key={value} value={value}>有效窗口 #{value}</option>)}
            </select>
            <select value={channelIndex} onChange={event => updatePreference('channelIndex', Number(event.target.value))} className="text-xs px-3 py-2 rounded-lg border bg-white">
              {CHANNELS.map((item, index) => <option key={item.id} value={index}>{item.name}</option>)}
            </select>
            <select value={scaleMode} onChange={event => updatePreference('scaleMode', event.target.value)} className="text-xs px-3 py-2 rounded-lg border bg-white">
              <option value="visible">可见曲线自适应</option>
              <option value="robust">弱化极端值</option>
            </select>
          </div>
        </div>

        <div className="flex flex-wrap gap-1.5 mb-2">
          {visibleModels.map(name => (
            <span key={name} className="text-[0.65rem] px-2.5 py-1 rounded-full border inline-flex items-center gap-1"
              style={{ borderColor: getModelColor(name), color: getModelColor(name) }}>
              <span className="accent-dot" style={{ background: getModelColor(name), width: 6, height: 6 }} />
              {name}
              <button type="button" onClick={() => setVisibleModels(current => current.filter(item => item !== name))}
                className="ml-0.5 w-3.5 h-3.5 rounded-full inline-flex items-center justify-center text-[0.55rem] hover:bg-black/10 cursor-pointer leading-none"
                title={`隐藏 ${name}`}>×</button>
            </span>
          ))}
        </div>
        {hiddenModels.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 mb-3">
            <span className="text-[0.62rem] text-[#A1A1AA]">添加模型:</span>
            {hiddenModels.map(name => <button key={name} type="button" onClick={() => setVisibleModels(current => [...current, name])}
              className="text-[0.62rem] px-2 py-0.5 rounded-full border border-[#D4D4D8] text-[#71717A] bg-white hover:border-[#A1A1AA]">+ {name}</button>)}
          </div>
        )}

        {chartData.datasets.length ? <div style={{ height: 420 }}><Line data={chartData} options={{
          responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
          plugins: { legend: { display: false } },
          scales: { y: { ...yRange, title: { display: true, text: '标准化值 (sigma)' } } },
        }} /></div> : <p className="py-16 text-center text-sm text-[#A1A1AA]">等待至少一个模型完成可信评测。</p>}

        <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)] text-xs text-[#52525B] leading-relaxed">
          Chronos2 和 IBM TTM 的预训练输出通常比真实序列更平滑，这是真实模型输出，不是前端画成直线。若某个通道在所选窗口自身波动很小，曲线会进一步显得接近平直。切换窗口和通道可核对不同局部片段；模型优劣仍应结合全量 MSE、MAE、RMSE 与 Custom ACC 判断。
        </div>
      </div></RevealCard>
    </div>
  );
}

export default function ComparePage() {
  const [view, setView] = useState('current');
  return (
    <div className="page-enter">
      <div className="flex items-center gap-1 mt-4 mb-2">
        <button onClick={() => setView('current')} className={`text-xs px-4 py-1.5 rounded-lg font-medium ${view === 'current' ? 'bg-[#3B82F6] text-white' : 'bg-[#F5F5F5] text-[#52525B]'}`}>当前本地评测曲线</button>
        <button onClick={() => setView('history')} className={`text-xs px-4 py-1.5 rounded-lg font-medium ${view === 'history' ? 'bg-[#3B82F6] text-white' : 'bg-[#F5F5F5] text-[#52525B]'}`}>历史曲线与稳定性</button>
        <button onClick={() => setView('history-ranking')} className={`text-xs px-4 py-1.5 rounded-lg font-medium ${view === 'history-ranking' ? 'bg-[#3B82F6] text-white' : 'bg-[#F5F5F5] text-[#52525B]'}`}>历史指标对比</button>
      </div>
      {view === 'current' ? <CurrentCompare /> : view === 'history' ? <CurvesPage /> : <PerformancePage />}
    </div>
  );
}
