import { useState, useEffect } from 'react';
import { MODELS } from '../data/models';

const TABS = [
  { id: 'overview', label: '首页' },
  { id: 'playground', label: '模型中心' },
  { id: 'compare', label: '预测对比' },
  { id: 'errors', label: '误差分析' },
  { id: 'details', label: '详细报告' },
];

export default function Header({ activeTab, onTabChange }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className={`glass-nav${scrolled ? ' scrolled' : ''}`}>
      <div className="max-w-7xl mx-auto px-5 h-16 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-[10px] flex items-center justify-center flex-shrink-0"
            style={{
              background: 'linear-gradient(135deg, #3B82F6, #EC4899)',
              boxShadow: '0 2px 8px rgba(59,130,246,0.30)',
            }}
          >
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <span className="font-semibold text-[17px] tracking-tight whitespace-nowrap text-[#18181B]">
            4G Traffic Prediction{' '}
            <span style={{ color: 'var(--accent)' }}>Benchmark</span>
          </span>
        </div>

        {/* Nav tabs */}
        <nav className="flex items-center gap-1 bg-[#F5F5F5]/80 rounded-xl p-0.5">
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={`tab-btn${activeTab === tab.id ? ' active' : ''}`}
              onClick={() => onTabChange(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Right */}
        <div className="hidden lg:flex items-center gap-2 text-xs text-[#A1A1AA]">
          <span className="accent-dot" style={{ background: '#22C55E' }} />
          {MODELS.length} Models
        </div>
      </div>
    </header>
  );
}
