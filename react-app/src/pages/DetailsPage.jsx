import { useMemo, useState } from 'react';
import { MODELS, CATS } from '../data/models';
import { getPalette } from '../data/palette';
import RevealCard from '../components/RevealCard';
import useBenchmarkModels from '../hooks/useBenchmarkModels';

const CURRENT_COLUMNS = [
  ['model', '模型', null],
  ['cat', '类别', null],
  ['mse', 'MSE', 'lower'],
  ['mae', 'MAE', 'lower'],
  ['rmse', 'RMSE', 'lower'],
  ['mape', 'MAPE', 'lower'],
  ['mspe', 'MSPE', 'lower'],
  ['acc', 'Custom ACC', 'higher'],
];

function valueForSort(model, field, preference) {
  if (field === 'model' || field === 'cat') return String(model[field] || '');
  const value = model[field];
  if (!Number.isFinite(value)) return preference === 'higher' ? -Infinity : Infinity;
  return value;
}

export default function DetailsPage() {
  const { models: currentModels } = useBenchmarkModels();
  const [sortState, setSortState] = useState({ field: 'acc', direction: 'desc' });
  const sorted = useMemo(() => [...MODELS].sort((a, b) => b.acc - a.acc).slice(0, 20), []);
  const catSummary = useMemo(() => CATS.map(cat => {
    const models = MODELS.filter(m => m.cat === cat);
    const best = models.reduce((a, b) => a.acc > b.acc ? a : b);
    return { cat, count: models.length, avgMSE: models.reduce((s, m) => s + m.mse, 0) / models.length, avgACC: models.reduce((s, m) => s + m.acc, 0) / models.length, best: best.model };
  }), []);
  const sortedCurrent = useMemo(() => [...currentModels].sort((a, b) => {
    const left = valueForSort(a, sortState.field, sortState.field === 'acc' ? 'higher' : 'lower');
    const right = valueForSort(b, sortState.field, sortState.field === 'acc' ? 'higher' : 'lower');
    if (typeof left === 'string') return sortState.direction === 'asc' ? left.localeCompare(right) : right.localeCompare(left);
    return sortState.direction === 'asc' ? left - right : right - left;
  }), [currentModels, sortState]);

  const toggleSort = (field, preference) => {
    setSortState(current => {
      if (current.field === field) return { field, direction: current.direction === 'asc' ? 'desc' : 'asc' };
      return { field, direction: preference === 'higher' ? 'desc' : 'asc' };
    });
  };
  const arrow = field => sortState.field === field ? (sortState.direction === 'asc' ? ' ↑' : ' ↓') : '';
  const renderMetric = value => Number.isFinite(value) ? value.toFixed(4) : '—';

  return <div className="page-enter">
    <RevealCard className="mt-5"><div className="card p-5"><h3 className="text-sm font-semibold mb-1">当前严格重评测结果</h3><p className="text-[0.65rem] text-[#A1A1AA] mb-3">仅含已完成本地运行并写入 manifest 的结果。点击列名切换升序/降序；MSE、MAE、RMSE、MAPE、MSPE 越低越好，Custom ACC 越高越好。</p>{currentModels.length ? <div className="overflow-x-auto"><table className="dt w-full"><thead><tr>{CURRENT_COLUMNS.map(([field, label, preference]) => <th key={field} onClick={() => toggleSort(field, preference)}>{label}{arrow(field)}</th>)}</tr></thead><tbody>{sortedCurrent.map(m => <tr key={m.model}><td>{m.model}</td><td>{m.cat}</td><td>{renderMetric(m.mse)}</td><td>{renderMetric(m.mae)}</td><td>{renderMetric(m.rmse)}</td><td>{renderMetric(m.mape)}</td><td>{renderMetric(m.mspe)}</td><td>{renderMetric(m.acc)}</td></tr>)}</tbody></table></div> : <p className="text-xs text-[#A1A1AA]">等待本地模型完成重评测。</p>}</div></RevealCard>

    <div className="mt-4 p-3 rounded-xl bg-[#FFF7ED] border border-[#FED7AA] text-xs text-[#9A3412]">以下区域保留最初 GitHub 版本的历史报告数据，与上方当前严格重评测结果分开显示。</div>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
      <RevealCard><div className="card p-5 h-full"><h3 className="text-sm font-semibold mb-3">数据集说明</h3><div className="space-y-2.5 text-xs text-[#52525B]">
        {[['数据来源','4G 基站 RAN 侧实测 KPI 数据，覆盖 100 个小区'],['特征通道','ERAB、PDCCH、PDSCH、PUSCH、上下行流量、总流量、有效连接数'],['时间粒度','小时级时间序列'],['预测目标','过去 24 小时输入，预测未来 24 小时'],['当前协议','同一小区内连续 48 小时，3 小时步长，共 3,514 个有效测试窗口'],['历史视图','最初 GitHub 页面使用 5,378 窗口，保留用于展示旧版图表']].map(([k,v]) => <div key={k} className="flex gap-2"><span className="text-[#A1A1AA] w-24 flex-shrink-0">{k}</span><span>{v}</span></div>)}
      </div></div></RevealCard>
      <RevealCard delay={80}><div className="card p-5 h-full"><h3 className="text-sm font-semibold mb-3">实验设置</h3><div className="space-y-2.5 text-xs text-[#52525B]">
        {[['任务','多变量 24→24 时间序列预测'],['标准化','StandardScaler 仅拟合训练观测'],['当前评估','MSE、MAE、RMSE、MAPE、MSPE、Custom ACC'],['训练环境','本机 Python / PyTorch / CUDA'],['结果更新','模型完成本地运行后写入 manifest，并自动进入首页和当前预测对比'],['上传预测','完整八通道按训练协议输入；任意单列使用逐通道适配。已有 checkpoint 时无需重复训练']].map(([k,v]) => <div key={k} className="flex gap-2"><span className="text-[#A1A1AA] w-24 flex-shrink-0">{k}</span><span>{v}</span></div>)}
      </div></div></RevealCard>
      <RevealCard delay={160}><div className="card p-5 h-full"><h3 className="text-sm font-semibold mb-3">历史类别指标汇总</h3><table className="dt w-full"><thead><tr><th>类别</th><th>数量</th><th>平均 MSE</th><th>平均 ACC</th><th>最佳模型</th></tr></thead><tbody>{catSummary.map(row => { const c = getPalette(MODELS.find(m => m.cat === row.cat)); return <tr key={row.cat}><td style={{color:c.text}}>{row.cat}</td><td>{row.count}</td><td>{row.avgMSE.toFixed(3)}</td><td>{row.avgACC.toFixed(4)}</td><td>{row.best}</td></tr>; })}</tbody></table></div></RevealCard>
      <RevealCard delay={240}><div className="card p-5 h-full"><h3 className="text-sm font-semibold mb-3">历史综合排名 · TOP 20</h3><div className="space-y-1">{sorted.map((m,i) => <div key={m.model} className="flex gap-2 text-xs py-1"><span className="w-6 text-right">{i+1}</span><span className="flex-1">{m.model}</span><span className="text-[#A1A1AA]">{m.cat}</span><span className="w-14 text-right font-mono">{m.acc.toFixed(4)}</span></div>)}</div></div></RevealCard>
    </div>
  </div>;
}
