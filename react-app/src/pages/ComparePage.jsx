import RevealCard from '../components/RevealCard';

export default function ComparePage() {
  return (
    <div className="page-enter mt-5"><RevealCard><div className="card p-6">
      <h2 className="text-lg font-semibold">预测比较正在重评测</h2>
      <p className="mt-3 text-sm text-[#52525B] leading-relaxed">旧比较数据采用扁平化窗口，现已下线。请在模型中心查看带小区 ID、起始时间和实验清单的已验证曲线。</p>
    </div></RevealCard></div>
  );
}
