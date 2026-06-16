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
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', font: { size: 9 } }, min: 0.15, max: 0.65, title: { display: true, text: 'Custom ACC', color: '#A1A1AA' } },
      y: { grid: { display: false }, ticks: { color: '#52525B', font: { size: 9 } } },
    },
  };
  return <div className="chart-box"><Bar data={data} options={options} /></div>;
}

function RadarChart_() {
  const top8 = [...MODELS].sort((a, b) => b.acc - a.acc).slice(0, 8);
  const data = {
    labels: ['Custom ACC', '1/MSE', '1/MAE', '1/RMSE', '1−MAPE'],
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
      x: { title: { display: true, text: 'MSE（越低越好）', color: '#A1A1AA' }, grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' } },
      y: { title: { display: true, text: 'Custom ACC（越高越好）', color: '#A1A1AA' }, grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, min: 0.15, max: 0.65 },
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
      y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA' }, title: { display: true, text: '平均 MSE（越低越好）', color: '#A1A1AA' } },
    },
  };
  return <div className="chart-box"><Bar data={data} options={options} /></div>;
}

export default function PerformancePage() {
  return (
    <div className="page-enter">
      {/* Data table */}
      <div className="mt-5">
        <DataTable />
      </div>

      {/* ACC Bar + Radar */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">Custom ACC 横向对比</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">所有模型按 ACC 升序排列 · 绿色边框标注 ★ BaseModel</p>
          <ACCBarChart />
          <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
            <p className="text-xs text-[#52525B] leading-relaxed">
              <strong>📊 图表解读：</strong>
              Custom ACC 是我们设计的复合评估指标，综合考虑了高低流量区间的预测精度（对高于均值的样本单独计算准确率后取平均）。
              该指标比 MSE 更贴近业务需求——运营商更关心高峰时段的预测是否准确。
              ★ BaseModel 以 0.5901 领先，意味着在流量高峰区段的平均预测准确率约 59%。
            </p>
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">Top 8 多维雷达图</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">五项指标归一化至 [0, 0.9] · 面积越大综合表现越好</p>
          <RadarChart_ />
          <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
            <p className="text-xs text-[#52525B] leading-relaxed">
              <strong>📊 图表解读：</strong>
              雷达图将 ACC、1/MSE、1/MAE、1/RMSE、1−MAPE 五项指标归一化后叠放，便于多维度综合比较。
              正五边形的面积越大，代表模型在多个维度上的综合表现越均衡。
              若某模型在 ACC 上表现出色但在 MAPE 上塌陷，说明其在相对误差指标上存在短板——这在使用百分比误差评估标准化空间预测时并不罕见。
            </p>
          </div>
        </div>
      </div>

      {/* Scatter + Category */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">MSE vs ACC 帕累托前沿</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">按模型类别着色 · ★ BaseModel 突出显示 · 悬停查看完整指标</p>
          <ScatterChart_ />
          <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
            <p className="text-xs text-[#52525B] leading-relaxed">
              <strong>📊 图表解读：</strong>
              散点图将 MSE 与 ACC 交叉对比，寻找"帕累托最优"模型——即在两个指标上都没有被其他模型同时超越的模型。
              理想位置在左上角（低 MSE + 高 ACC）。若两模型 MSE 相近但 ACC 差距明显，说明它们在高峰流量区间的表现有显著差异。
              同类别模型（同色）通常聚集在相近区域，反映了架构范式的整体水平。
            </p>
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-sm font-semibold tracking-tight mb-0.5">模型类别 MSE 汇总</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mb-3">柱高 = 类别平均 MSE · 误差线示最值范围</p>
          <CategoryBarChart />
          <div className="mt-3 p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
            <p className="text-xs text-[#52525B] leading-relaxed">
              <strong>📊 图表解读：</strong>
              按模型架构范式（Transformer、MLP、CNN、RNN、SSM、LLM 等）汇总平均 MSE，快速判断哪类架构在 4G 流量预测任务上整体最优。
              柱高越低越好。注意柱内的模型数量不同（如 LLM 类含 3 个 Chronos 变体），柱间对比时应结合样本量判断。
              目前 Baseline（自研融合架构）以最低单模型 MSE 领跑，CNN/Tree 类别整体表现稳健。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
