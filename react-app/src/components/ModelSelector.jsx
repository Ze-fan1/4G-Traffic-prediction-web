import { useState } from 'react';
import { CAT_PALETTE } from '../data/palette';

function getCat(model) {
  return CAT_PALETTE[model.cat] || CAT_PALETTE.Statistical;
}

export default function ModelSelector({ selectedModel, onSelect, models = [], modelTiers = {} }) {
  const [search, setSearch] = useState('');
  const [collapsed, setCollapsed] = useState({});
  const categories = [...new Set(models.map(model => model.cat))];
  const filtered = models.filter(model => model.model.toLowerCase().includes(search.toLowerCase()));
  const grouped = categories.reduce((groups, category) => {
    const items = filtered.filter(model => model.cat === category);
    if (items.length) groups[category] = items;
    return groups;
  }, {});

  return (
    <div className="card p-4 h-full flex flex-col" style={{ maxHeight: 'calc(100vh - 140px)' }}>
      <div className="relative mb-3">
        <input value={search} onChange={event => setSearch(event.target.value)} placeholder="搜索模型..."
          className="w-full text-xs px-3 py-2 rounded-xl border border-[rgba(0,0,0,0.08)] bg-[#FAFAFA] focus:outline-none focus:border-[#3B82F6]" />
      </div>
      <div className="overflow-y-auto flex-1 pr-1" style={{ scrollbarWidth: 'thin' }}>
        {Object.entries(grouped).map(([category, items]) => {
          const categoryInfo = CAT_PALETTE[category] || CAT_PALETTE.Statistical;
          const isCollapsed = collapsed[category] === true;
          return (
            <div key={category} className="mb-1">
              <button onClick={() => setCollapsed(previous => ({ ...previous, [category]: !previous[category] }))}
                className="w-full flex items-center gap-2 py-1.5 text-left hover:bg-[#FAFAFA] rounded-lg px-1">
                <span className="w-2 h-2 rounded-full" style={{ background: categoryInfo.border }} />
                <span className="text-[0.7rem] font-medium text-[#52525B]">{categoryInfo.label || category}</span>
                <span className="text-[0.6rem] text-[#A1A1AA] ml-auto font-mono">{items.length}</span>
              </button>
              {!isCollapsed && <div className="ml-3 space-y-0.5">
                {items.map(model => {
                  const selected = selectedModel === model.model;
                  const unavailable = (modelTiers[model.model] || 1) >= 2;
                  return (
                    <button key={model.model} onClick={() => onSelect(model.model)} disabled={unavailable}
                      title={unavailable ? '暂不可用' : `${model.model} · ${model.runType}`}
                      className={`w-full flex items-center gap-2 py-1.5 px-2 rounded-lg text-left transition-all ${selected ? 'bg-[#EFF6FF] border-l-[3px] border-[#3B82F6]' : 'hover:bg-[#FAFAFA] border-l-[3px] border-transparent'} ${unavailable ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
                      <span className="w-2 h-2 rounded-full" style={{ background: getCat(model).border }} />
                      <span className="text-[0.72rem] truncate flex-1" style={{ color: selected ? '#3B82F6' : '#52525B', fontWeight: selected ? 600 : 400 }}>{model.model}</span>
                      <span className="text-[0.55rem] text-[#A1A1AA]">{model.verified ? '已验证' : '待重评测'}</span>
                    </button>
                  );
                })}
              </div>}
            </div>
          );
        })}
        {!filtered.length && <p className="text-xs text-[#A1A1AA] text-center py-4">无匹配模型</p>}
      </div>
    </div>
  );
}
