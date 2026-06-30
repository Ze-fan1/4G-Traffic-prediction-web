import { useState } from 'react';
import CurvesPage from './CurvesPage';
import PerformancePage from './PerformancePage';

export default function ComparePage() {
  const [tab, setTab] = useState('curves');
  return (
    <div className="page-enter">
      <div className="flex items-center gap-1 mt-4 mb-2">
        <button onClick={() => setTab('curves')} className={`text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${tab === 'curves' ? 'bg-[#3B82F6] text-white' : 'bg-[#F5F5F5] text-[#52525B] hover:bg-[#E5E5E5]'}`}>📈 曲线叠图</button>
        <button onClick={() => setTab('ranking')} className={`text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${tab === 'ranking' ? 'bg-[#3B82F6] text-white' : 'bg-[#F5F5F5] text-[#52525B] hover:bg-[#E5E5E5]'}`}>📊 ACC 排名</button>
      </div>
      {tab === 'curves' ? <CurvesPage /> : <PerformancePage />}
    </div>
  );
}
