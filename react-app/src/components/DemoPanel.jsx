import { useState, useMemo, useCallback } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { CHANNELS } from '../data/channels';
import predictionCurves from '../data/prediction_curves.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const HOURS = Array.from({ length: 24 }, (_, i) => `${i + 1}h`);
const API_BASE = '/api';

export default function DemoPanel({ model, channelIdx, onChangeChannel, isAvailable }) {
  const [quickResult, setQuickResult] = useState(null);
  const [fullResult, setFullResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const chKey = String(channelIdx);

  // Preset curve data from frontend
  const presetData = useMemo(() => {
    const modelData = predictionCurves.models?.[model]?.[chKey];
    if (!modelData) return null;
    return {
      pred: modelData.pred,
      truth: modelData.true,
    };
  }, [model, chKey]);

  // Chart data: preset + live verification overlay
  const chartData = useMemo(() => {
    const datasets = [];

    // Ground truth
    if (presetData?.truth) {
      datasets.push({
        label: '真实值 (预置)',
        data: presetData.truth,
        borderColor: '#18181B',
        backgroundColor: 'transparent',
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.35,
        order: 0,
      });
    }

    // Preset prediction
    if (presetData?.pred) {
      datasets.push({
        label: `${model} (预置)`,
        data: presetData.pred,
        borderColor: '#3B82F6',
        backgroundColor: '#3B82F620',
        borderWidth: 1.8,
        borderDash: [5, 3],
        pointRadius: 0,
        tension: 0.35,
        order: 1,
      });
    }

    // Live verification prediction (overlaid)
    if (quickResult?.pred) {
      datasets.push({
        label: `${model} (实时)`,
        data: quickResult.pred,
        borderColor: '#22C55E',
        backgroundColor: '#22C55E20',
        borderWidth: 2.2,
        borderDash: [],
        pointRadius: 0,
        tension: 0.35,
        order: 0,
      });
    }

    return {
      labels: HOURS,
      datasets,
    };
  }, [presetData, quickResult, model]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'bottom',
        labels: { boxWidth: 10, padding: 14, font: { size: 10 }, color: '#52525B' },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(0,0,0,0.03)' },
        ticks: { color: '#A1A1AA' },
        title: { display: true, text: '预测时刻 (小时)', color: '#A1A1AA' },
      },
      y: {
        grid: { color: 'rgba(0,0,0,0.03)' },
        ticks: { color: '#A1A1AA', callback: (v) => v?.toFixed(2) },
        title: { display: true, text: '标准化值 (σ)', color: '#A1A1AA' },
      },
    },
  }), []);

  const handleQuickVerify = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/demo/${encodeURIComponent(model)}/quick`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel_idx: channelIdx }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail?.reason || err.detail?.error || '请求失败');
      }
      const data = await res.json();
      setQuickResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, [model, channelIdx]);

  const handleFullEval = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setFullResult(null);
    try {
      const res = await fetch(`${API_BASE}/demo/${encodeURIComponent(model)}/full`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel_idx: channelIdx }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail?.reason || err.detail?.error || '请求失败');
      }
      const data = await res.json();
      setFullResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, [model, channelIdx]);

  return (
    <div className="space-y-4">
      {/* Info header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">{model}</h3>
          <p className="text-[0.65rem] text-[#A1A1AA] mt-0.5">
            数据: df_4g_test_100 · Window #{predictionCurves.window || '—'} · 输入24h → 预测24h
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[0.65rem] text-[#A1A1AA]">通道:</span>
          <select
            value={channelIdx}
            onChange={(e) => onChangeChannel(Number(e.target.value))}
            className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer focus:outline-none focus:border-[#3B82F6] transition-colors"
          >
            {CHANNELS.map((ch, i) => (
              <option key={ch.id} value={i}>{ch.name} — {ch.desc}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Chart */}
      <div className="chart-box-lg">
        <Line data={chartData} options={chartOptions} />
      </div>

      {/* Quick result */}
      {quickResult && (
        <div className="p-2 rounded-lg bg-[#F0FDF4] border border-[#BBF7D0] text-xs text-[#166534]">
          ✅ 实时验证完成 (耗时 {quickResult.elapsed_s}s) — MAE = {quickResult.mae}
          {presetData && (
            <span className="ml-2 text-[#A1A1AA]">
              | 预置 MAE = {(
                presetData.pred.reduce((s, p, i) => s + Math.abs(p - presetData.truth[i]), 0) / 24
              ).toFixed(4)}
            </span>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-2 rounded-lg bg-[#FEF2F2] border border-[#FECACA] text-xs text-[#991B1B]">
          ❌ {error}
        </div>
      )}

      {/* Buttons */}
      <div className="flex flex-wrap items-center gap-3">
        {isAvailable ? (
          <>
            <button
              onClick={handleQuickVerify}
              disabled={isLoading}
              className="text-xs px-4 py-2 rounded-xl bg-[#3B82F6] text-white hover:bg-[#2563EB] disabled:opacity-50 transition-colors cursor-pointer font-medium"
            >
              {isLoading ? '⏳ 进行中...' : '🔄 快速验证 (1个窗口)'}
            </button>
            <button
              onClick={handleFullEval}
              disabled={isLoading}
              className="text-xs px-4 py-2 rounded-xl border border-[rgba(0,0,0,0.08)] bg-white hover:bg-stone-50 disabled:opacity-50 transition-colors cursor-pointer"
            >
              📊 完整评估 (5378窗口)
            </button>
          </>
        ) : (
          <p className="text-xs text-[#A1A1AA]">🔒 该模型暂不支持实时推理，仅展示预置曲线</p>
        )}
      </div>

      {/* Full evaluation results */}
      {fullResult && (
        <div className="p-3 rounded-xl bg-[#FAFAFA] border border-[rgba(0,0,0,0.03)]">
          <p className="text-xs font-semibold mb-2">📊 完整评估 ({fullResult.n_windows} 窗口，耗时 {fullResult.elapsed_s}s)</p>
          <div className="grid grid-cols-4 gap-2 text-xs">
            {Object.entries(fullResult.metrics).map(([k, v]) => (
              <div key={k} className="bg-white rounded-lg p-2 text-center">
                <p className="text-[0.6rem] text-[#A1A1AA] uppercase">{k}</p>
                <p className="font-mono font-bold text-[#52525B]">{v}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
