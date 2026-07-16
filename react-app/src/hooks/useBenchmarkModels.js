import { useCallback, useEffect, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '/api';
const CACHE_KEY = 'benchmark_models_cache_v2';
let memoryCache = null;

function loadStoredModels() {
  try {
    return JSON.parse(localStorage.getItem(CACHE_KEY) || sessionStorage.getItem(CACHE_KEY) || '[]');
  } catch { return []; }
}

export function normalizeBenchmarkEntry(entry) {
  const metrics = entry.metrics || {};
  return {
    cat: entry.cat,
    model: entry.model,
    mse: metrics.mse,
    mae: metrics.mae,
    rmse: metrics.rmse,
    mape: metrics.mape ?? null,
    mspe: metrics.mspe ?? null,
    acc: metrics.custom_acc ?? metrics.acc ?? null,
    experimentId: entry.experiment_id,
  };
}

export default function useBenchmarkModels() {
  const [models, setModels] = useState(() => {
    if (memoryCache) return memoryCache;
    return loadStoredModels();
  });
  const [loading, setLoading] = useState(() => models.length === 0);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/benchmark-models`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const next = (data.models || []).map(normalizeBenchmarkEntry);
      // Never erase a useful ranking during a backend restart or while a new
      // training artifact is still being written.
      if (next.length > 0) {
        memoryCache = next;
        try {
          sessionStorage.setItem(CACHE_KEY, JSON.stringify(next));
          localStorage.setItem(CACHE_KEY, JSON.stringify(next));
        } catch {}
        setModels(next);
      }
    } catch (error) {
      console.warn('Unable to load benchmark models:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 30000);
    window.addEventListener('benchmark-updated', refresh);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener('benchmark-updated', refresh);
    };
  }, [refresh]);

  return { models, loading, refresh };
}
