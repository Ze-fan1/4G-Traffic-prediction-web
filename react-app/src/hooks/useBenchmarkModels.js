import { useCallback, useEffect, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

function normalize(entry) {
  const metrics = entry.metrics || {};
  return {
    cat: entry.cat,
    model: entry.model,
    mse: metrics.mse,
    mae: metrics.mae,
    rmse: metrics.rmse,
    mape: metrics.mape ?? null,
    acc: metrics.acc ?? null,
    experimentId: entry.experiment_id,
  };
}

export default function useBenchmarkModels() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/benchmark-models`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setModels((data.models || []).map(normalize));
    } catch (error) {
      console.warn('Unable to load benchmark models:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 5000);
    window.addEventListener('benchmark-updated', refresh);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener('benchmark-updated', refresh);
    };
  }, [refresh]);

  return { models, loading, refresh };
}
