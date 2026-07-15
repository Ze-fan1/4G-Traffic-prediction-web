import RevealCard from '../components/RevealCard';
import useBenchmarkModels from '../hooks/useBenchmarkModels';

export default function ErrorsPage() {
  const { models } = useBenchmarkModels();
  const sorted = [...models].sort((left, right) => left.mse - right.mse);

  return (
    <div className="page-enter mt-5"><RevealCard><div className="card p-6">
      <h2 className="text-lg font-semibold">已验证模型误差分析</h2>
      <p className="mt-2 text-sm text-[#52525B] leading-relaxed">数值来自每个模型自己的完整 3,514 个有效测试窗口。训练或评测完成后会自动更新，失败任务不会写入此表。</p>
      {sorted.length ? <div className="overflow-x-auto mt-5"><table className="dt w-full"><thead><tr><th>排名</th><th>模型</th><th>MSE</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>{sorted.map((model, index) => <tr key={model.model}><td>{index + 1}</td><td className="font-medium">{model.model}</td><td className="font-mono">{model.mse.toFixed(4)}</td><td className="font-mono">{model.mae.toFixed(4)}</td><td className="font-mono">{model.rmse.toFixed(4)}</td></tr>)}</tbody></table></div> : <p className="mt-6 text-sm text-[#A1A1AA]">暂时没有完成严格评测的模型。</p>}
    </div></RevealCard></div>
  );
}
