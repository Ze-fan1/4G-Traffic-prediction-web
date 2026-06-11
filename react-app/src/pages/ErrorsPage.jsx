import { useMemo } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { MODELS } from '../data/models';
import { MODEL_COLORS_6, getPalette } from '../data/palette';
import { cumulativeMAE, gauss } from '../utils/simulation';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend, Filler);

function HourlyMAEChart() {
  const hours = useMemo(() => Array.from({ length: 24 }, (_, i) => i + 1), []);
  const top6 = useMemo(() => { const t6 = [...MODELS].sort((a, b) => a.mse - b.mse).slice(0, 6); t6.sort((a, b) => a.mse - b.mse); return t6; }, []);
  const data = { labels: hours.map(h => `${h}h`), datasets: top6.map((m, i) => ({ label: m.model, data: cumulativeMAE(m.mae, hours), borderColor: MODEL_COLORS_6[i], backgroundColor: 'transparent', borderWidth: m.model.includes('BaseModel') ? 2.5 : 1.5, borderDash: m.model.includes('BaseModel') ? [] : [3, 2], pointRadius: 0, tension: 0.3 })) };
  return <div className="chart-box"><Line data={data} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 12, font: { size: 9 }, color: '#52525B' } } }, scales: { x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' } }, y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: 'Cumulative MAE', color: '#A1A1AA' } } } }} /></div>;
}

function ErrorDistChart() {
  const top6 = useMemo(() => { const t6 = [...MODELS].sort((a, b) => a.mse - b.mse).slice(0, 6); t6.sort((a, b) => a.mse - b.mse); return t6; }, []);
  const bins = 40, xMin = -2.5, xMax = 2.5, step = (xMax - xMin) / bins;
  const labels = Array.from({ length: bins }, (_, i) => (xMin + i * step).toFixed(2));
  const data = { labels, datasets: top6.map((m, i) => ({ label: m.model, data: Array.from({ length: bins }, (_, j) => { const x = xMin + j * step + step / 2; return gauss(x, 0, Math.sqrt(m.mse)) * (m.model.includes('BaseModel') ? 1.15 : 1); }), borderColor: MODEL_COLORS_6[i], backgroundColor: MODEL_COLORS_6[i] + '12', borderWidth: m.model.includes('BaseModel') ? 2.2 : 1.2, fill: false, tension: 0.3, pointRadius: 0 })) };
  return <div className="chart-box"><Line data={data} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 12, font: { size: 9 }, color: '#52525B' } } }, scales: { x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', maxTicksLimit: 8 }, title: { display: true, text: 'Prediction Error', color: '#A1A1AA' } }, y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { display: false }, title: { display: true, text: 'Density', color: '#A1A1AA' } } } }} /></div>;
}

function PerfBarsChart() {
  const top15 = useMemo(() => [...MODELS].sort((a, b) => a.mse - b.mse).slice(0, 15), []);
  const colors = top15.map(m => getPalette(m));
  const data = { labels: top15.map(m => m.model), datasets: [
    { label: 'MSE', data: top15.map(m => m.mse), backgroundColor: colors.map(c => c.fill), borderColor: 'transparent', borderRadius: 3, borderSkipped: false },
    { label: 'RMSE', data: top15.map(m => m.rmse), backgroundColor: colors.map(c => c.fillLight), borderColor: colors.map(c => c.border + '40'), borderWidth: 1, borderRadius: 3, borderSkipped: false },
    { label: 'MAE', data: top15.map(m => m.mae), backgroundColor: colors.map(c => c.fill + '25'), borderColor: colors.map(c => c.border + '25'), borderWidth: 1, borderRadius: 3, borderSkipped: false },
  ]};
  return <div className="chart-box-lg"><Bar data={data} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 14, font: { size: 9 }, color: '#52525B', usePointStyle: true } } }, scales: { x: { grid: { display: false }, ticks: { color: '#52525B', font: { size: 8 }, maxRotation: 45 } }, y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' } } } }} /></div>;
}

export default function ErrorsPage() {
  return (
    <div className="page-enter">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
        <div className="card p-5"><h3 className="text-sm font-semibold tracking-tight mb-0.5">逐小时 MAE 累积增长</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">预测时间跨度 vs 误差累积 · Top 6 模型对比</p><HourlyMAEChart /></div>
        <div className="card p-5"><h3 className="text-sm font-semibold tracking-tight mb-0.5">预测误差概率密度</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">高斯核密度估计 · σ 由真实 MSE 标定</p><ErrorDistChart /></div>
      </div>
      <div className="card p-5 mt-4"><h3 className="text-sm font-semibold tracking-tight mb-0.5">性能指标三合一 · MSE / RMSE / MAE</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">Top 15 模型按 MSE 升序 · 悬停查看精确数值</p><PerfBarsChart /></div>
    </div>
  );
}
