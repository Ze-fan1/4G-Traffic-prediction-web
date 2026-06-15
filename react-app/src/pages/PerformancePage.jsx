import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  PointElement, LineElement, RadialLinearScale, Filler, Tooltip, Legend
} from 'chart.js';
import { Bar, Radar, Scatter } from 'react-chartjs-2';
import { MODELS, CATS } from '../data/models';
import { MODEL_COLORS_8, getPalette } from '../data/palette';
import DataTable from '../components/DataTable';

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, RadialLinearScale, Filler, Tooltip, Legend);

function ACCBarChart() {
  const sorted = [...MODELS].sort((a, b) => a.acc - b.acc);
  const data = {
    labels: sorted.map(m => m.model),
    datasets: [{
      data: sorted.map(m => m.acc),
      backgroundColor: sorted.map(m => getPalette(m).fill),
      borderColor: sorted.map(m => m.model.includes('BaseModel') ? getPalette(m).border : 'transparent'),
      borderWidth: sorted.map(m => m.model.includes('BaseModel') ? 2.5 : 0),
      borderRadius: 3,
      borderSkipped: false,
    }],
  };
  const options = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', font: { size: 9 } }, min: 0.15, max: 0.65 },
      y: { grid: { display: false }, ticks: { color: '#52525B', font: { size: 9 } } },
    },
  };
  return <div className="chart-box"><Bar data={data} options={options} /></div>;
}

function RadarChart_() {
  const top8 = [...MODELS].sort((a, b) => b.acc - a.acc).slice(0, 8);
  const data = {
    labels: ['Custom ACC', '1/MSE', '1/MAE', '1/RMSE', '1-MAPE%'],
    datasets: top8.map((m, i) => ({
      label: m.model,
      data: [m.acc, 1 - m.mse / 5, 1 - m.mae / 1, 1 - m.rmse / 2.5, Math.max(0, 1 - m.mape / 300)],
      borderColor: MODEL_COLORS_8[i],
      backgroundColor: MODEL_COLORS_8[i] + '12',
      borderWidth: 1.5,
      pointRadius: 3,
      pointBackgroundColor: MODEL_COLORS_8[i],
    })),
  };
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 14, font: { size: 9 }, color: '#52525B' } } },
    scales: { r: { grid: { color: 'rgba(0,0,0,0.04)' }, angleLines: { color: 'rgba(0,0,0,0.04)' }, pointLabels: { color: '#52525B', font: { size: 9 } }, ticks: { display: false }, min: 0, max: 0.9 } },
  };
  return <div className="chart-box"><Radar data={data} options={options} /></div>;
}

function ScatterChart_() {
  const datasets = CATS.map(cat => {
    const models = MODELS.filter(m => m.cat === cat);
    const c = getPalette(models[0]);
    return {
      label: cat,
      data: models.map(m => ({ x: m.mse, y: m.acc, model: m.model, mse: m.mse, mae: m.mae, rmse: m.rmse })),
      backgroundColor: c.fill,
      borderColor: c.border,
      pointRadius: cat === 'Baseline' ? 9 : 5,
      pointHoverRadius: cat === 'Baseline' ? 11 : 7,
    };
  });
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 12, font: { size: 9 }, color: '#52525B', usePointStyle: true } } },
    scales: {
      x: { title: { display: true, text: 'MSE', color: '#A1A1AA' }, grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' } },
      y: { title: { display: true, text: 'Custom ACC', color: '#A1A1AA' }, grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, min: 0.15, max: 0.65 },
    },
  };
  return <div className="chart-box"><Scatter data={{ datasets }} options={options} /></div>;
}

function CategoryBarChart() {
  const catStats = CATS.map(cat => {
    const models = MODELS.filter(m => m.cat === cat);
    const vals = models.map(m => m.mse);
    return { cat, avg: vals.reduce((a, b) => a + b, 0) / vals.length, min: Math.min(...vals), max: Math.max(...vals), count: models.length };
  });
  const data = {
    labels: catStats.map(c => c.cat),
    datasets: [{
      data: catStats.map(c => c.avg),
      backgroundColor: catStats.map(c => getPalette(MODELS.find(m => m.cat === c.cat)).fill),
      borderColor: catStats.map(c => getPalette(MODELS.find(m => m.cat === c.cat)).border),
      borderWidth: 1,
      borderRadius: 5,
    }],
  };
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#52525B', font: { size: 9 } } },
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: 'MSE', color: '#A1A1AA' } },
    },
  };
  return <div className="chart-box"><Bar data={data} options={options} /></div>;
}

export default function PerformancePage() {
  return (
    <div className="page-enter">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">Custom ACC 横向对比</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">所有模型按 ACC 升序排列 · 悬停查看详情</p>
          <ACCBarChart />
        </div>
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">Top 8 多维雷达</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">综合评估</p>
          <RadarChart_ />
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">MSE vs ACC 帕累托前沿</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">气泡半径 · 悬停显示完整指标</p>
          <ScatterChart_ />
        </div>
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">模型类别 MSE 汇总</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">柱高 = 类别均值 · 悬停查看最值范围</p>
          <CategoryBarChart />
        </div>
      </div>
      <div className="mt-4">
        <DataTable />
      </div>
    </div>
  );
}
