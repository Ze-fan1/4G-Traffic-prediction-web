import { useEffect, useRef, useState } from 'react';

/** Scroll 揭示 Hook — 弹簧弹出 */
export function useReveal(threshold = 0.08) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return [ref, visible];
}

/** 单个揭示卡片包装器 — 弹簧弹出 duangduang */
export default function RevealCard({ children, className = '', style, delay = 0 }) {
  const [ref, visible] = useReveal(0.08);
  return (
    <div
      ref={ref}
      className={`reveal-card ${visible ? 'visible' : ''} ${className}`}
      style={{ ...style, transitionDelay: delay ? `${delay}ms` : undefined }}
    >
      {children}
    </div>
  );
}
