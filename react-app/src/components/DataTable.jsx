import { useState, useMemo, useCallback } from 'react';
import { MODELS, CATS } from '../data/models';
import { getPalette } from '../data/palette';

export default function DataTable() {
  const [sortCol, setSortCol] = useState('acc');
  const [sortDir, setSortDir] = useState(-1);
  const [activeCat, setActiveCat] = useState('all');
  const [search, setSearch] = useState('');
  const [shimmer, setShimmer] = useState(false);

  const triggerShimmer = useCallback(() => { setShimmer(true); setTimeout(() => setShimmer(false), 800); }, []);
  const handleSort = useCallback((col) => { setSortCol(prev => { if (prev === col) { setSortDir(d => -d); return prev; } setSortDir(col === 'acc' || col === 'model' ? -1 : 1); return col; }); triggerShimmer(); }, [triggerShimmer]);

  const filtered = useMemo(() => {
    let data = [...MODELS];
    if (activeCat !== 'all') data = data.filter(m => m.cat === activeCat);
    if (search) { const q = search.toLowerCase(); data = data.filter(m => m.model.toLowerCase().includes(q) || m.cat.toLowerCase().includes(q)); }
    data.sort((a, b) => { let va = a[sortCol], vb = b[sortCol]; if (sortCol === 'cat') { va = a.cat; vb = b.cat; return sortDir === 1 ? va.localeCompare(vb) : vb.localeCompare(va); } if (sortCol === 'model') { va = a.model; vb = b.model; return sortDir === 1 ? va.localeCompare(vb) : vb.localeCompare(va); } return sortDir === 1 ? (va - vb) : (vb - va); });
    return data;
  }, [activeCat, search, sortCol, sortDir]);

  const bestMSE = Math.min(...MODELS.map(m => m.mse));
  const bestACC = Math.max(...MODELS.filter(m => Number.isFinite(m.acc)).map(m => m.acc));
  const sortHeaders = ['cat', 'model', 'mse', 'mae', 'rmse', 'mape', 'acc'];
  const sortLabels = { cat: 'Category', model: 'Model', mse: 'MSE ↓', mae: 'MAE ↓', rmse: 'RMSE ↓', mape: 'MAPE% ↓', acc: 'Custom ACC ↑' };

  return (
    <div className="card overflow-hidden">
      <div className="p-4 border-b flex flex-wrap items-center gap-3 justify-between" style={{ borderColor: 'rgba(0,0,0,0.04)' }}>
        <h3 className="text-sm font-semibold tracking-tight">模型数据表格</h3>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex flex-wrap gap-1.5">
            <span className={`chip ${activeCat === 'all' ? 'active' : ''}`} onClick={() => { setActiveCat('all'); triggerShimmer(); }}>All</span>
            {CATS.map(cat => { const c = getPalette(MODELS.find(m => m.cat === cat)); return (<span key={cat} className={`chip ${activeCat === cat ? 'active' : ''}`} onClick={() => { setActiveCat(cat); triggerShimmer(); }}><span className="accent-dot" style={{ background: c.border }} />{cat}</span>); })}
          </div>
          <div className="relative">
            <svg className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[#A1A1AA]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
            <input type="text" placeholder="搜索模型…" value={search} onChange={(e) => { setSearch(e.target.value); triggerShimmer(); }} className="pl-8 pr-4 py-2 text-xs rounded-xl border bg-[#FAFAFA] w-60 focus:outline-none focus:border-[#3B82F6] transition-all" style={{ fontSize: '0.72rem', borderColor: 'rgba(0,0,0,0.05)' }} />
          </div>
        </div>
      </div>
      <div className="overflow-x-auto max-h-[500px] overflow-y-auto relative">
        {shimmer && <div className="shimmer-overlay" />}
        <table className="dt w-full">
          <thead className="sticky top-0 bg-white z-10">
            <tr>{sortHeaders.map(col => (<th key={col} onClick={() => handleSort(col)} style={{ cursor: 'pointer' }}>{sortLabels[col]}{sortCol === col ? (sortDir === 1 ? ' ↑' : ' ↓') : ''}</th>))}</tr>
          </thead>
          <tbody>
            {filtered.map(m => { const c = getPalette(m); const isBest = m.model.includes('BaseModel'); return (
              <tr key={m.model} className={isBest ? 'bg-[#DBEAFE]/40' : ''}>
                <td><span className="inline-flex items-center gap-1.5 text-[0.65rem] font-medium px-2 py-0.5 rounded-full" style={{ background: c.bg, color: c.text }}><span className="accent-dot" style={{ background: c.border, width: 5, height: 5 }} />{m.cat}</span></td>
                <td className={`font-medium text-[0.78rem] ${isBest ? 'text-[#3B82F6]' : ''}`}>{m.model}</td>
                <td className={`font-mono text-[0.72rem] ${m.mse === bestMSE ? 'text-emerald-600 font-medium' : ''}`}>{m.mse.toFixed(4)}</td>
                <td className="font-mono text-[0.72rem]">{m.mae.toFixed(4)}</td>
                <td className="font-mono text-[0.72rem]">{m.rmse.toFixed(4)}</td>
                <td className={`font-mono text-[0.72rem] ${m.mape > 150 ? 'text-amber-600' : ''}`}>{Number.isFinite(m.mape) ? m.mape.toFixed(2) : '—'}</td>
                <td className={`font-mono text-[0.72rem] font-semibold ${m.acc === bestACC ? 'text-[#3B82F6]' : m.acc > 0.5 ? 'text-emerald-600' : ''}`}>{Number.isFinite(m.acc) ? m.acc.toFixed(4) : '—'}</td>
              </tr>
            ); })}
            {filtered.length === 0 && <tr><td colSpan="7" className="text-center py-12 text-[#A1A1AA] text-xs">No models match.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
