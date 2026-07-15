import RevealCard from '../components/RevealCard';

export default function ErrorsPage() {
  return (
    <div className="page-enter mt-5"><RevealCard><div className="card p-6">
      <h2 className="text-lg font-semibold">误差分析正在重评测</h2>
      <p className="mt-3 text-sm text-[#52525B] leading-relaxed">误差分布将在所有模型完成同一小区、连续时间窗口协议下的重新评估后发布。</p>
    </div></RevealCard></div>
  );
}
