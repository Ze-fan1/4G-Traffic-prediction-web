import { useState } from 'react';
import { MODELS, CATS } from '../data/models';
import { CAT_PALETTE } from '../data/palette';

function getCat(model) {
  return CAT_PALETTE[model.cat] || CAT_PALETTE['Statistical'];
}

export default function ModelSelector({ selectedModel, onSelect, modelTiers = {} }) {
  const [search, setSearch] = useState('');
  const [collapsed, setCollapsed] = useState(() => {
    const init = {};
    CATS.forEach(c => { init[c] = c !== 'Baseline' && c !== 'Transformer'; });
    return init;
  });

  const toggleCat = (cat) => setCollapsed(prev => ({ ...prev, [cat]: !prev[cat] }));

  const filtered = MODELS.filter(m =>
    m.model.toLowerCase().includes(search.toLowerCase())
  );

  // Group by category
  const grouped = {};
  CATS.forEach(cat => {
    const items = filtered.filter(m => m.cat === cat);
    if (items.length > 0) grouped[cat] = items;
  });

  return (
    <div className="card p-4 h-full flex flex-col" style={{ maxHeight: 'calc(100vh - 140px)' }}>
      <div className="overflow-y-auto flex-1 pr-1" style={{ scrollbarWidth: 'thin' }}>
      {/* Search */}
      <div className="relative mb-3">
        <input
          type="text"
          placeholder="搜索模型..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full text-xs px-3 py-2 rounded-xl border border-[rgba(0,0,0,0.08)] bg-[#FAFAFA] focus:outline-none focus:border-[#3B82F6] focus:bg-white transition-colors"
          style={{ fontSize: '0.75rem' }}
        />
      </div>

      {/* Category accordion */}
      {Object.entries(grouped).map(([cat, items]) => {
        const catInfo = CAT_PALETTE[cat];
        const isCollapsed = collapsed[cat];
        return (
          <div key={cat} className="mb-1">
            {/* Category header */}
            <button
              onClick={() => toggleCat(cat)}
              className="w-full flex items-center gap-2 py-1.5 text-left hover:bg-[#FAFAFA] rounded-lg px-1 transition-colors cursor-pointer select-none"
            >
              <svg
                className={`w-2.5 h-2.5 transition-transform duration-200 flex-shrink-0 ${isCollapsed ? '-rotate-90' : 'rotate-0'}`}
                viewBox="0 0 24 24"
                fill="none"
                stroke="#A1A1AA"
                strokeWidth="3"
                strokeLinecap="round"
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: catInfo?.border || '#A1A1AA' }}
              />
              <span className="text-[0.7rem] font-medium text-[#52525B]">
                {catInfo?.label || cat}
              </span>
              <span className="text-[0.6rem] text-[#A1A1AA] ml-auto font-mono">
                {items.length}
              </span>
            </button>

            {/* Model list */}
            {!isCollapsed && (
              <div className="ml-4 space-y-0.5">
                {items.map(m => {
                  const isSelected = selectedModel === m.model;
                  const tier = modelTiers[m.model] || 1;
                  const isUnavailable = tier >= 2;

                  return (
                    <button
                      key={m.model}
                      onClick={() => onSelect(m.model)}
                      className={`w-full flex items-center gap-2 py-1.5 px-2 rounded-lg text-left transition-all duration-150 cursor-pointer ${
                        isSelected
                          ? 'bg-[#EFF6FF] border-l-[3px] border-[#3B82F6]'
                          : 'hover:bg-[#FAFAFA] border-l-[3px] border-transparent'
                      } ${isUnavailable ? 'opacity-50' : ''}`}
                      title={isUnavailable ? '该模型暂不支持实时推理' : `${m.model} · ACC=${m.acc?.toFixed(4)}`}
                    >
                      <span
                        className="w-2 h-2 rounded-full flex-shrink-0"
                        style={{ background: getCat(m).border }}
                      />
                      <span
                        className="text-[0.72rem] truncate flex-1"
                        style={{ color: isSelected ? '#3B82F6' : '#52525B', fontWeight: isSelected ? 600 : 400 }}
                      >
                        {m.model}
                      </span>
                      {isUnavailable && (
                        <span className="text-[0.55rem] flex-shrink-0" title="暂不支持实时推理">🔒</span>
                      )}
                      <span className="text-[0.6rem] text-[#A1A1AA] font-mono flex-shrink-0 w-14 text-right">
                        {m.acc?.toFixed(3) || '—'}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}

      {filtered.length === 0 && (
        <p className="text-xs text-[#A1A1AA] text-center py-4">无匹配模型</p>
      )}
      </div>
    </div>
  );
}
