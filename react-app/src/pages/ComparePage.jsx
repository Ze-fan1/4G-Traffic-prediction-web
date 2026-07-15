import { useEffect, useMemo, useState } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { CHANNELS } from '../data/channels';
import { getModelColor } from '../data/palette';
import RevealCard from '../components/RevealCard';
import useBenchmarkModels from '../hooks/useBenchmarkModels';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

const API_BASE = import.meta.env.VITE_API_URL || '/api';
const HOURS = Array.from({ length: 24 }, (_, index) => `${index + 1}h`);

export default function ComparePage() {
  const { models } = useBenchmarkModels();
  const [channelIndex, setChannelIndex] = useState(0);
  const [curves, setCurves] = useState({});

  useEffect(() => {
    let cancelled = false;
    Promise.all(models.map(async model => {
      const response = await fetch(`${API_BASE}/preset-curves/${encodeURIComponent(model.model)}?window=0`);
      const data = await response.json();
      return [model.model, data];
    })).then(entries => {
      if (!cancelled) setCurves(Object.fromEntries(entries.filter(([, data]) => data.available && data.curves)));
    }).catch(() => {
      if (!cancelled) setCurves({});
    });
    return () => { cancelled = true; };
  }, [models]);

  const channel = CHANNELS[channelIndex];
  const chartData = useMemo(() => {
    const first = models.find(model => curves[model.model]?.curves?.[channel.name]);
    const datasets = [];
    if (first) {
      datasets.push({
        label: '真实值', data: curves[first.model].curves[channel.name].truth,
        borderColor: '#18181B', borderDash: [6, 3], borderWidth: 2, pointRadius: 0,
      });
    }
    models.forEach(model => {
      const curve = curves[model.model]?.curves?.[channel.name];
      if (!curve) return;
      datasets.push({
        label: model.model, data: curve.pred, borderColor: getModelColor(model.model),
        borderWidth: 1.7, pointRadius: 0, tension: 0.3,
      });
    });
    return { labels: HOURS, datasets };
  }, [channel.name, curves, models]);

  return (
    <div className="page-enter mt-5 space-y-4">
      <RevealCard><div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div><h2 className="text-lg font-semibold">已验证模型预测对比</h2><p className="mt-1 text-xs text-[#52525B]">只展示已按同小区连续 48 小时协议完成评测的模型；新结果会自动加入。</p></div>
          <select value={channelIndex} onChange={event => setChannelIndex(Number(event.target.value))} className="text-xs px-3 py-2 rounded-lg border bg-white">
            {CHANNELS.map((item, index) => <option key={item.id} value={index}>{item.name}</option>)}
          </select>
        </div>
        {chartData.datasets.length ? <div style={{ height: 400 }}><Line data={chartData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } }, scales: { y: { title: { display: true, text: '标准化值 (sigma)' } } } }} /></div> : <p className="py-16 text-center text-sm text-[#A1A1AA]">等待至少一个模型完成可信评测。</p>}
      </div></RevealCard>
    </div>
  );
}
