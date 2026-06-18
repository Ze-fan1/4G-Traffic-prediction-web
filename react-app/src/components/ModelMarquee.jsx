import { MODELS } from '../data/models';
import { getPalette } from '../data/palette';

export default function ModelMarquee() {
  // 按 ACC 排序
  const sorted = [...MODELS].sort((a, b) => b.acc - a.acc);

  return (
    <div className="relative w-full overflow-hidden py-3" style={{ maskImage: 'linear-gradient(to right, transparent 0%, black 8%, black 92%, transparent 100%)', WebkitMaskImage: 'linear-gradient(to right, transparent 0%, black 8%, black 92%, transparent 100%)' }}>
      {/* 第一行 — 向右滚动 */}
      <div className="flex gap-2 mb-2" style={{ animation: 'marqueeRight 60s linear infinite', width: 'max-content' }}>
        {[...sorted, ...sorted].map((m, i) => {
          const p = getPalette(m);
          return (
            <span
              key={`r1-${i}`}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg whitespace-nowrap flex-shrink-0 transition-all duration-300"
              style={{
                background: p.bg,
                border: `1px solid ${p.border}20`,
                color: p.text,
                fontSize: '0.7rem',
                fontWeight: 500,
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: p.border }} />
              {m.model}
            </span>
          );
        })}
      </div>
      {/* 第二行 — 向左滚动 */}
      <div className="flex gap-2" style={{ animation: 'marqueeLeft 55s linear infinite', width: 'max-content' }}>
        {[...sorted.reverse(), ...sorted.reverse()].map((m, i) => {
          const p = getPalette(m);
          return (
            <span
              key={`r2-${i}`}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg whitespace-nowrap flex-shrink-0 transition-all duration-300"
              style={{
                background: p.bg,
                border: `1px solid ${p.border}20`,
                color: p.text,
                fontSize: '0.7rem',
                fontWeight: 500,
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: p.border }} />
              {m.model}
            </span>
          );
        })}
      </div>
    </div>
  );
}
