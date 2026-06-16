import { useMemo } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { MODELS } from '../data/models';
import { MODEL_COLORS_8, getPalette } from '../data/palette';
import hourlyMAE from '../data/hourly_mae.js';
import errorDist from '../data/error_dist.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend, Filler);

const MAE_MODELS = Object.keys(hourlyMAE.models);
const DIST_MODELS = Object.keys(errorDist.models);

function HourlyMAEChart() {
  const hours = useMemo(() => Array.from({ length: 24 }, (_, i) => `${i + 1}h`), []);
  const data = useMemo(() => ({
    labels: hours,
    datasets: MAE_MODELS.map((name, i) => ({
      label: name,
      data: hourlyMAE.models[name],
      borderColor: MODEL_COLORS_8[i],
      backgroundColor: 'transparent',
      borderWidth: name.includes('BaseModel') ? 2.5 : 1.5,
      borderDash: name.includes('BaseModel') ? [] : [3, 2],
      pointRadius: 0,
      tension: 0.3,
    })),
  }), [hours]);
  const options = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 12, font: { size: 9 }, color: '#52525B' } } },
    scales: {
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: '预测时刻 (小时)', color: '#A1A1AA' } },
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: 'MAE (σ)', color: '#A1A1AA' } },
    },
  };
  return <div className="chart-box"><Line data={data} options={options} /></div>;
}

function ErrorDistChart() {
  // Compute skewness analysis
  const skewInfo = useMemo(() => {
    const info = {};
    for (const name of DIST_MODELS) {
      const m = errorDist.models[name];
      const x = m.x;
      const d = m.density;
      // Find peak location
      let maxIdx = 0;
      for (let i = 1; i < d.length; i++) { if (d[i] > d[maxIdx]) maxIdx = i; }
      info[name] = { peak: x[maxIdx], bias: m.bias, rmse: m.rmse };
    }
    return info;
  }, []);

  const data = useMemo(() => ({
    labels: errorDist.models[DIST_MODELS[0]].x.map((_, i) => i),
    datasets: DIST_MODELS.map((name, i) => ({
      label: name,
      data: errorDist.models[name].density,
      borderColor: MODEL_COLORS_8[i],
      backgroundColor: MODEL_COLORS_8[i] + '12',
      borderWidth: name.includes('BaseModel') ? 2.2 : 1.2,
      fill: false, tension: 0.3, pointRadius: 0,
    })),
  }), []);
  const options = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { boxWidth: 8, padding: 12, font: { size: 9 }, color: '#52525B' } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: 密度=${ctx.raw.toFixed(4)}` } },
    },
    scales: {
      x: { display: false },
      y: { display: false },
    },
  };

  // Find most asymmetric model
  let mostSkewed = '', maxAbsBias = 0;
  for (const [name, info] of Object.entries(skewInfo)) {
    if (Math.abs(info.bias) > maxAbsBias) { maxAbsBias = Math.abs(info.bias); mostSkewed = name; }
  }

  return (
    <>
      <div className="chart-box"><Line data={data} options={options} /></div>
    </>
  );
}

function PerfBarsChart() {
  const top15 = useMemo(() => [...MODELS].sort((a, b) => a.mse - b.mse).slice(0, 15), []);
  const colors = top15.map(m => getPalette(m));
  const data = {
    labels: top15.map(m => m.model),
    datasets: [
      { label: 'MSE', data: top15.map(m => m.mse), backgroundColor: colors.map(c => c.fill), borderColor: 'transparent', borderRadius: 3, borderSkipped: false },
      { label: 'RMSE', data: top15.map(m => m.rmse), backgroundColor: colors.map(c => c.fillLight), borderColor: colors.map(c => c.border + '40'), borderWidth: 1, borderRadius: 3, borderSkipped: false },
      { label: 'MAE', data: top15.map(m => m.mae), backgroundColor: colors.map(c => c.fill + '25'), borderColor: colors.map(c => c.border + '25'), borderWidth: 1, borderRadius: 3, borderSkipped: false },
    ],
  };
  const options = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 8, padding: 14, font: { size: 9 }, color: '#52525B', usePointStyle: true } } },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#52525B', font: { size: 8 }, maxRotation: 45 } },
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' } },
    },
  };
  return <div className="chart-box-lg"><Bar data={data} options={options} /></div>;
}

export default function ErrorsPage() {
  return (
    <div className="page-enter">
      {/* Hourly MAE */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">逐小时 MAE 累积增长</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">总流量通道 · 标准化空间 · 全窗口平均</p>
          <HourlyMAEChart />
          <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
            <p className="text-xs text-[#52525B] leading-relaxed">
              <strong>📊 趋势解读：</strong>
              所有模型的逐小时 MAE 均呈单调上升趋势——预测越远的未来，不确定性越大。前 6 小时各模型差距较小（误差累积尚未拉开），
              12 小时后分化加速：表现优异的模型（如 ★ BaseModel、iTransformer）增长斜率明显更缓，
              说明其对长期依赖的建模能力更强。若某模型在 20-24h 段出现陡增，提示其长程预测存在系统性偏差。
            </p>
          </div>
        </div>

        {/* Error Distribution */}
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">预测误差概率密度分布</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">高斯核密度估计 · 标准化空间 · 基于全部窗口误差</p>
          <ErrorDistChart />
          <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
            <p className="text-xs text-[#52525B] leading-relaxed">
              <strong>📊 分布形态解读：</strong>
              横轴为预测误差（pred − true，单位 σ），纵轴为概率密度。理想情况下，误差分布应关于 0 对称（无系统性偏差）且峰值尖锐（大部分预测接近真值）。
              若分布整体右偏（峰值 &gt; 0），说明模型倾向于高估流量；左偏则倾向于低估。
              分布越窄越尖锐的模型，预测精度越高、不确定性越低。
            </p>
          </div>
        </div>
      </div>

      {/* Performance Bars */}
      <div className="card p-5 mt-4">
        <h3 className="text-sm font-semibold tracking-tight mb-0.5">三项指标综合对比 · MSE / RMSE / MAE</h3>
        <p className="text-[0.65rem] text-[#A1A1AA] mb-3">Top 15 模型按 MSE 升序排列 · 悬停查看精确数值</p>
        <PerfBarsChart />
        <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
          <p className="text-xs text-[#52525B] leading-relaxed">
            <strong>📊 指标解读：</strong>
            MSE（均方误差）对大误差惩罚更重，RMSE 是其平方根（量纲与原始数据一致），MAE（平均绝对误差）反映典型偏差大小。
            三者联合观察可判断误差结构：若 MSE ≫ MAE²，说明存在少量大偏差样本（拖尾），模型在极端情况下可能不稳定。
            ★ BaseModel 在三项指标上均最低，说明其在正常和极端样本上都有最优表现。
          </p>
        </div>
      </div>
    </div>
  );
}
