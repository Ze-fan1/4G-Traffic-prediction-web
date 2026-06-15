const TABS = [
  { id: 'overview', label: '首页' },
  { id: 'performance', label: '性能与数据' },
  { id: 'curves', label: '预测曲线' },
  { id: 'errors', label: '误差分析' },
  { id: 'details', label: '详细报告' },
];

export default function Header({ activeTab, onTabChange }) {
  return (
    <header className="sticky top-0 z-30 glass-nav">
      <div className="max-w-7xl mx-auto px-4 sm:px-5 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #6152F2, #0E7490)' }}>
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <span className="font-semibold text-[15px] tracking-tight">4G Traffic Prediction <span style={{ color: 'var(--accent)' }}>Benchmark</span></span>
        </div>
        <nav className="flex items-center gap-0.5 bg-[#F5F5F5] rounded-xl p-0.5 flex-wrap justify-center">
          {TABS.map(tab => (
            <button key={tab.id} className={`tab-btn${activeTab === tab.id ? ' active' : ''}`} onClick={() => onTabChange(tab.id)}>{tab.label}</button>
          ))}
        </nav>
        <div className="hidden lg:flex items-center gap-2 text-xs text-[#A1A1AA]">
          <span className="accent-dot" style={{ background: '#22C55E' }} /> 26 Models
        </div>
      </div>
    </header>
  );
}
