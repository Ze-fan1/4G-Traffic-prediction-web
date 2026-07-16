import { useMemo } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { MODELS } from '../data/models';
import { MODEL_COLORS_8, getPalette } from '../data/palette';
import hourlyMAE from '../data/hourly_mae.js';
import errorDist from '../data/error_dist.js';
import RevealCard from '../components/RevealCard';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend, Filler);

const MAE_MODELS = Object.keys(hourlyMAE.models);
const DIST_MODELS = Object.keys(errorDist.models);

function HourlyMAEChart() {
  const hours = Array.from({ length: 24 }, (_, i) => `${i + 1}h`);
  const data = useMemo(() => ({ labels: hours, datasets: MAE_MODELS.map((name, i) => ({
    label: name, data: hourlyMAE.models[name], borderColor: MODEL_COLORS_8[i % MODEL_COLORS_8.length],
    backgroundColor: 'transparent', borderWidth: name.includes('BaseModel') ? 2.5 : 1.5,
    borderDash: name.includes('BaseModel') ? [] : [3, 2], pointRadius: 0, tension: 0.3,
  })) }), []);
  return <div className="chart-box"><Line data={data} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, font: { size: 9 } } } }, scales: { x: { title: { display: true, text: '预测时刻 (小时)' } }, y: { title: { display: true, text: 'MAE (σ)' } } } }} /></div>;
}

function ErrorDistChart() {
  const data = useMemo(() => ({
    labels: errorDist.models[DIST_MODELS[0]].x.map((_, i) => i),
    datasets: DIST_MODELS.map((name, i) => ({ label: name, data: errorDist.models[name].density,
      borderColor: MODEL_COLORS_8[i % MODEL_COLORS_8.length], borderWidth: name.includes('BaseModel') ? 2.2 : 1.2,
      fill: false, tension: 0.3, pointRadius: 0 })),
  }), []);
  return <div className="chart-box"><Line data={data} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, font: { size: 9 } } } }, scales: { x: { display: false }, y: { display: false } } }} /></div>;
}

function PerfBarsChart() {
  const top15 = [...MODELS].sort((a, b) => a.mse - b.mse).slice(0, 15);
  const colors = top15.map(getPalette);
  const data = { labels: top15.map(m => m.model), datasets: [
    { label: 'MSE', data: top15.map(m => m.mse), backgroundColor: colors.map(c => c.fill), borderRadius: 3 },
    { label: 'RMSE', data: top15.map(m => m.rmse), backgroundColor: colors.map(c => c.fillLight), borderRadius: 3 },
    { label: 'MAE', data: top15.map(m => m.mae), backgroundColor: colors.map(c => c.fill + '40'), borderRadius: 3 },
  ] };
  return <div className="chart-box-lg"><Bar data={data} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } }, scales: { x: { ticks: { maxRotation: 45, font: { size: 8 } } } } }} /></div>;
}

export default function ErrorsPage() {
  return <div className="page-enter">
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
      <RevealCard><div className="card p-5"><h3 className="text-sm font-semibold mb-0.5">逐小时 MAE 对比</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">总流量通道 · 历史基准</p><HourlyMAEChart /><p className="mt-3 text-xs text-[#52525B]">每个点是对应预测时刻的 MAE，并非累加值。曲线越低，表示该预测时刻的平均误差越小；局部升降反映不同预测步的难度差异。</p></div></RevealCard>
      <RevealCard delay={80}><div className="card p-5"><h3 className="text-sm font-semibold mb-0.5">预测误差概率密度分布</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">高斯核密度估计 · 历史基准</p><ErrorDistChart /><p className="mt-3 text-xs text-[#52525B]">分布越靠近 0 且越集中，代表系统性偏差和预测不确定性越小。</p></div></RevealCard>
    </div>
    <RevealCard className="mt-4"><div className="card p-5"><h3 className="text-sm font-semibold mb-0.5">三项指标综合对比 · MSE / RMSE / MAE</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">历史 Top 15 模型</p><PerfBarsChart /><p className="mt-3 text-xs text-[#52525B]">MSE 对大误差更敏感，MAE 反映典型偏差，RMSE 用于观察整体误差尺度。</p></div></RevealCard>
  </div>;
}
