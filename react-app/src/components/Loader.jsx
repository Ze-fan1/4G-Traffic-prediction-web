import { useState, useEffect } from 'react';

export default function Loader({ onDone }) {
  const [fadeOut, setFadeOut] = useState(false);

  useEffect(() => {
    const t1 = setTimeout(() => setFadeOut(true), 500);
    const t2 = setTimeout(() => onDone(), 900);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [onDone]);

  return (
    <div className="fixed inset-0 z-50 bg-[#FAFAFA] flex items-center justify-center" style={{ opacity: fadeOut ? 0 : 1, transition: 'opacity 0.4s' }}>
      <div className="w-full max-w-4xl px-5 space-y-4">
        <div className="flex gap-2 mb-4">
          {[...Array(4)].map((_, i) => (<div key={i} className="skeleton h-8 w-16 rounded-lg" />))}
        </div>
        <div className="grid grid-cols-4 gap-3">
          {[...Array(4)].map((_, i) => (<div key={i} className="skeleton h-24 rounded-xl" />))}
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="skeleton h-72 rounded-xl" /><div className="skeleton h-72 rounded-xl" />
        </div>
        <p className="text-center text-[#A1A1AA] text-xs pt-2">Loading benchmark data…</p>
      </div>
    </div>
  );
}
