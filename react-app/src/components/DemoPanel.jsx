import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { CHANNELS } from '../data/channels';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const HOURS = Array.from({ length: 24 }, (_, i) => `${i + 1}h`);
const API_BASE = import.meta.env.VITE_API_URL || '/api';

const RUN_TYPE_LABEL = {
  train_dl: '🔥 从零训练 (10 epochs)',
  inference_stat: '📊 生成预测曲线',
  inference_pretrained: '🔄 加载预训练模型并验证',
  train_xgboost: '🔥 训练 XGBoost',
  train_mamba: '🔥 训练 Mamba',
  train_timellm: '🔥 训练 TimeLLM (需 GPT-2)',
};

const WINDOW_OPTIONS = [
  { value: '4394', label: '窗口 #4394 (代表性)' },
  { value: '100',  label: '窗口 #100' },
  { value: '0',    label: '窗口 #0' },
];

const HISTORY_COLORS = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6'];

// ─── 每个模型的专属颜色（按类别分组，便于区分）───
const MODEL_COLORS = {
  // Statistical — 冷灰/紫
  'Naive':           '#64748B',
  'Persistent 24h':  '#78716C',
  'Historical Avg':  '#A8A29E',
  'AutoARIMA':       '#7C3AED',
  'AutoAR':          '#A78BFA',
  'LinearRegression': '#8B5CF6',
  // Tree
  'XGBoost':         '#65A30D',
  // Baseline
  '★ BaseModel':     '#374151',
  // Transformer 系 — 红/橙
  'PatchTST':        '#DC2626',
  'iTransformer':    '#EA580C',
  'Autoformer':      '#D97706',
  'Transformer':     '#E11D48',
  'Informer':        '#F59E0B',
  // MLP — 蓝/青
  'DLinear':         '#2563EB',
  'LightTS':         '#0891B2',
  'TSMixer':         '#0D9488',
  'IBM TTM':         '#6366F1',
  // CNN — 绿
  'TimesNet':        '#059669',
  'SCINet':          '#10B981',
  // RNN
  'SegRNN':          '#DB2777',
  // SSM / LLM
  'Mamba':           '#9333EA',
  'TimeLLM':         '#4F46E5',
  'Chronos2':        '#8B5CF6',
};

function getModelColor(model) {
  return MODEL_COLORS[model] || '#6B7280';
}

// ─── 全局缓存（跨卡片持久化）───
const modelHistoryCache = new Map();
const modelRunState = new Map();
const globalPollers = new Map();
const modelPredHistory = new Map();
const compareCurvesCache = new Map();  // 缓存其他模型的预置曲线，避免重复请求

function stopGlobalPoll(mKey) {
  const p = globalPollers.get(mKey);
  if (p) { clearInterval(p); globalPollers.delete(mKey); }
}

function startGlobalPoll(mKey, jobId, onUpdate) {
  stopGlobalPoll(mKey);
  const interval = setInterval(async () => {
    try {
      const r = await fetch(`${API_BASE}/job/${jobId}`);
      const d = await r.json();
      onUpdate(d);
      if (d.status === 'done' || d.status === 'error' || d.status === 'cancelled') {
        clearInterval(interval);
        globalPollers.delete(mKey);
      }
    } catch (e) { /* ignore poll errors */ }
  }, 500);
  globalPollers.set(mKey, interval);
}

function ChartBlock({ curves, chKey, label, color, isPreset, windowInfo, compareLines }) {
  const ds = [];
  // Ground truth — always grey dashed
  if (curves?.[chKey]?.truth) {
    ds.push({
      label: '真实值 (Ground Truth)',
      data: curves[chKey].truth,
      borderColor: '#9CA3AF',
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.35,
      order: 0,
      borderDash: [4, 3],
    });
  }
  // Current model's prediction — solid colored line
  if (curves?.[chKey]?.pred) {
    ds.push({
      label: label || '预测',
      data: curves[chKey].pred,
      borderColor: isPreset ? (color || '#6B7280') : (color || '#3B82F6'),
      backgroundColor: (isPreset ? (color || '#6B7280') : (color || '#3B82F6')) + '20',
      borderWidth: isPreset ? 2.5 : 2.5,
      pointRadius: 0,
      tension: 0.35,
      order: 1,
    });
  }
  // Compare model predictions — thinner, dashed/dotted, semi-transparent
  if (compareLines && compareLines.length > 0) {
    const DASH_STYLES = [[6, 3], [2, 2], [10, 4], [4, 2, 1, 2], [1, 1]];
    compareLines.forEach((cl, i) => {
      if (cl.data && cl.data.length > 0) {
        ds.push({
          label: cl.label,
          data: cl.data,
          borderColor: cl.color,
          borderWidth: 1.5,
          borderDash: DASH_STYLES[i % DASH_STYLES.length],
          pointRadius: 0,
          tension: 0.35,
          order: 2,
        });
      }
    });
  }
  return (
    <div className="mb-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[0.65rem] font-medium" style={{ color: color || '#3B82F6' }}>
          {label}{windowInfo ? ` · ${windowInfo}` : ''}
        </span>
      </div>
      <div style={{ height: '200px' }}>
        {ds.length > 0 ? <Line data={{ labels: HOURS, datasets: ds }} options={CHART_OPTIONS} />
          : <div className="h-full flex items-center justify-center text-xs text-[#A1A1AA]">无数据</div>}
      </div>
    </div>
  );
}

const CHART_OPTIONS = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 300 },
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: {
      position: 'bottom',
      labels: {
        boxWidth: 10, padding: 6, font: { size: 7 },
        color: '#52525B',
        usePointStyle: false,
        filter: (item) => item.text !== '', // hide empty labels
      },
    },
    tooltip: {
      titleFont: { size: 9 },
      bodyFont: { size: 8 },
    },
  },
  scales: {
    x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', font: { size: 8 } },
         title: { display: true, text: '预测时刻 (h)', color: '#A1A1AA' } },
    y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { color: '#A1A1AA', font: { size: 8 } },
         title: { display: true, text: '标准化值 (σ)', color: '#A1A1AA' } },
  },
};

export default function DemoPanel({ model, channelIdx, onChangeChannel, isAvailable, runType }) {
  // ─── 从全局缓存恢复状态 ───
  const saved = modelRunState.get(model) || {};
  const [presetData, setPresetData] = useState(null);
  const [presetLoading, setPresetLoading] = useState(false);
  const [presetWindow, setPresetWindow] = useState('4394');

  const [runJobId, setRunJobId] = useState(saved.runJobId || null);
  const [runStatus, setRunStatusRaw] = useState(saved.runStatus || null);
  const [runPhase, setRunPhase] = useState(saved.runPhase || null);
  const [runEpoch, setRunEpoch] = useState(saved.runEpoch || 0);
  const [runTotalEpochs, setRunTotalEpochs] = useState(saved.runTotalEpochs || 0);
  const [runLoss, setRunLoss] = useState(saved.runLoss || 0);
  const [error, setError] = useState(saved.error || null);
  const [curveDone, setCurveDone] = useState(saved.curveDone || 0);
  const [curveTotal, setCurveTotal] = useState(saved.curveTotal || 0);
  const [runHistory, setRunHistory] = useState(modelHistoryCache.get(model) || []);
  const mountedRef = useRef(true);

  // ─── 对比模式 ───
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [compareModels, setCompareModels] = useState(() => {
    // 默认对比同类别的其他模型
    const saved = sessionStorage.getItem('pg_compare');
    return saved ? JSON.parse(saved) : [];
  });
  const [compareCurves, setCompareCurves] = useState([]);

  const setRunStatus = useCallback((v) => {
    setRunStatusRaw(v);
    const s = modelRunState.get(model) || {};
    s.runStatus = v;
    modelRunState.set(model, s);
  }, [model]);

  // ─── 持久化状态到全局缓存 ───
  useEffect(() => {
    modelRunState.set(model, {
      runJobId, runStatus, runPhase, runEpoch, runTotalEpochs,
      runLoss, error, curveDone, curveTotal,
    });
  }, [model, runJobId, runStatus, runPhase, runEpoch, runTotalEpochs, runLoss, error, curveDone, curveTotal]);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // ─── 获取可对比的模型列表 ───
  const [allModels, setAllModels] = useState([]);
  useEffect(() => {
    fetch(`${API_BASE}/preset-models`)
      .then(r => r.json())
      .then(d => setAllModels((d.models || []).filter(m => m !== model)))
      .catch(() => {});
  }, [model]);

  // ─── 加载对比模型的预置曲线 ───
  useEffect(() => {
    if (!compareEnabled || compareModels.length === 0) {
      setCompareCurves([]);
      return;
    }
    const promises = compareModels.map(async (m) => {
      const cacheKey = `${m}_${presetWindow}`;
      if (compareCurvesCache.has(cacheKey)) return compareCurvesCache.get(cacheKey);
      try {
        const r = await fetch(`${API_BASE}/preset-curves/${encodeURIComponent(m)}?window=${presetWindow}`);
        const d = await r.json();
        if (d.available && d.curves) {
          const result = { model: m, curves: d.curves, color: getModelColor(m) };
          compareCurvesCache.set(cacheKey, result);
          return result;
        }
      } catch (e) { /* skip */ }
      return null;
    });
    Promise.all(promises).then(results => {
      setCompareCurves(results.filter(Boolean));
    });
  }, [compareEnabled, compareModels, presetWindow]);

  useEffect(() => { sessionStorage.setItem('pg_compare', JSON.stringify(compareModels)); }, [compareModels]);

  // ─── 模型切换 → 加载预置 + 恢复状态 ───
  useEffect(() => {
    const saved2 = modelRunState.get(model) || {};
    setRunJobId(saved2.runJobId || null);
    setRunStatusRaw(saved2.runStatus || null);
    setRunPhase(saved2.runPhase || null);
    setRunEpoch(saved2.runEpoch || 0);
    setRunTotalEpochs(saved2.runTotalEpochs || 0);
    setRunLoss(saved2.runLoss || 0);
    setError(saved2.error || null);
    setCurveDone(saved2.curveDone || 0);
    setCurveTotal(saved2.curveTotal || 0);
    setPresetLoading(true); setPresetData(null);
    setRunHistory(modelHistoryCache.get(model) || []);

    const url = `${API_BASE}/preset-curves/${encodeURIComponent(model)}?window=${presetWindow}`;
    fetch(url).then(r => r.json()).then(d => {
      if (d.available && d.curves) setPresetData(d);
    }).catch(err => console.warn('[DemoPanel] preset err:', err.message))
      .finally(() => setPresetLoading(false));

    const savedJobId = saved2.runJobId;
    if (savedJobId && saved2.runStatus === 'running') {
      const pollKey = `${model}_${savedJobId}`;
      startGlobalPoll(pollKey, savedJobId, (d) => {
        setRunEpoch(d.epoch || 0);
        setRunLoss(d.loss || 0);
        if (d.total_epochs > 0) setRunTotalEpochs(d.total_epochs);
        setRunPhase(d.phase || 'training');
        if (d.phase === 'curves' || d.phase === 'inference') {
          setCurveDone(d.curve_done || 0);
          setCurveTotal(d.curve_total || 0);
        }
        if (d.status === 'done' && d.curves) {
          const winMeta = d.curves_meta;
          const labelSuffix = winMeta
            ? `#${(modelHistoryCache.get(model) || []).length + 1} · 窗口#${winMeta.window_idx}/${winMeta.total_windows}`
            : `#${(modelHistoryCache.get(model) || []).length + 1}`;
          const entry = { curves: d.curves, metrics: d.metrics, ts: Date.now(), label: labelSuffix, curvesMeta: winMeta };
          const upd = [...(modelHistoryCache.get(model) || []), entry];
          modelHistoryCache.set(model, upd);
          setRunHistory(upd);
          setRunStatus('done');
          setRunJobId(null);
          const s = modelRunState.get(model) || {};
          s.runStatus = 'done'; s.runJobId = null;
          modelRunState.set(model, s);
        } else if (d.status === 'error') {
          setRunStatus('error');
          setError(d.error || '失败');
          setRunJobId(null);
        } else if (d.status === 'cancelled') {
          setRunStatus('cancelled');
          setRunJobId(null);
        }
      });
    }
  }, [model]);

  // ─── 窗口切换 → 重载预置曲线 ───
  useEffect(() => {
    setPresetLoading(true); setPresetData(null);
    const url = `${API_BASE}/preset-curves/${encodeURIComponent(model)}?window=${presetWindow}`;
    fetch(url).then(r => r.json()).then(d => {
      if (d.available && d.curves) setPresetData(d);
    }).catch(err => console.warn('[DemoPanel] preset err:', err.message))
      .finally(() => setPresetLoading(false));
    // 清除对比缓存，新窗口需要重新加载
    compareCurvesCache.clear();
    setCompareCurves([]);
  }, [presetWindow]);

  // ─── 开始运行 ───
  const handleRun = useCallback(async () => {
    setError(null); setRunEpoch(0); setRunTotalEpochs(0); setRunLoss(0);
    setCurveDone(0); setCurveTotal(0); setRunStatus('running');
    setRunPhase(runType === 'inference_stat' || runType === 'inference_pretrained' ? 'inference' : 'training');

    try {
      const res = await fetch(`${API_BASE}/run/${encodeURIComponent(model)}`, { method: 'POST' });
      if (!res.ok) { let m = `HTTP ${res.status}`; try { m = (await res.json()).detail?.error || m; } catch {} throw new Error(m); }
      const { job_id } = await res.json();
      setRunJobId(job_id);

      const pollKey = `${model}_${job_id}`;
      startGlobalPoll(pollKey, job_id, (d) => {
        setRunEpoch(d.epoch || 0);
        setRunLoss(d.loss || 0);
        if (d.total_epochs > 0) setRunTotalEpochs(d.total_epochs);
        setRunPhase(d.phase || 'training');
        if (d.phase === 'curves' || d.phase === 'inference') { setCurveDone(d.curve_done || 0); setCurveTotal(d.curve_total || 0); }
        if (d.status === 'done' && d.curves) {
          const winMeta = d.curves_meta;
          const labelSuffix = winMeta
            ? `#${(modelHistoryCache.get(model) || []).length + 1} · 窗口#${winMeta.window_idx}/${winMeta.total_windows}`
            : `#${(modelHistoryCache.get(model) || []).length + 1}`;
          const entry = { curves: d.curves, metrics: d.metrics, ts: Date.now(), label: labelSuffix, curvesMeta: winMeta };
          const upd = [...(modelHistoryCache.get(model) || []), entry];
          modelHistoryCache.set(model, upd);
          setRunHistory(upd);
          setRunStatus('done');
          setRunJobId(null);
          stopGlobalPoll(pollKey);
        } else if (d.status === 'error') {
          setRunStatus('error');
          setError(d.error || '失败');
          setRunJobId(null);
          stopGlobalPoll(pollKey);
        } else if (d.status === 'cancelled') {
          setRunStatus('cancelled');
          setRunJobId(null);
          stopGlobalPoll(pollKey);
        }
      });
    } catch (e) { console.error('[DemoPanel] run err:', e); setError(e.message); setRunStatus(null); }
  }, [model, runType]);

  const handleCancel = useCallback(async () => {
    if (!runJobId) return;
    try { await fetch(`${API_BASE}/train/${encodeURIComponent(model)}/cancel`, { method: 'POST' }); } catch {}
    stopGlobalPoll(`${model}_${runJobId}`);
    setRunJobId(null); setRunStatus('cancelled');
  }, [runJobId, model]);

  const handleClearHistory = useCallback(() => {
    modelHistoryCache.set(model, []); setRunHistory([]);
  }, [model]);

  const chKey = CHANNELS[channelIdx]?.name || '总流量';
  const isCurvePhase = runPhase === 'curves' || runPhase === 'inference';
  const runPct = isCurvePhase
    ? (curveTotal > 0 ? Math.round((curveDone / curveTotal) * 100) : 0)
    : (runTotalEpochs > 0 ? Math.round((runEpoch / runTotalEpochs) * 100) : 0);

  const presetWindowInfo = presetData
    ? `窗口 #${presetData.window_idx}/${presetData.total_windows}`
    : '';

  const modelColor = getModelColor(model);

  // ─── 构建对比曲线数据集 ───
  const compareLines = useMemo(() => {
    return compareCurves.map(c => ({
      label: c.model,
      data: c.curves?.[chKey]?.pred || [],
      color: c.color,
    }));
  }, [compareCurves, chKey]);

  // ─── 推荐的对比模型（同类别的 top5）───
  const suggestedCompare = useMemo(() => {
    return allModels.slice(0, 8);
  }, [allModels]);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold tracking-tight" style={{ color: modelColor }}>
            ● {model}
          </h3>
          <p className="text-[0.65rem] text-[#A1A1AA]">
            {RUN_TYPE_LABEL[runType] || RUN_TYPE_LABEL.train_dl}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select value={presetWindow} onChange={e => setPresetWindow(e.target.value)}
            className="text-[0.6rem] px-2 py-1 rounded-lg border border-[rgba(0,0,0,0.06)] bg-white cursor-pointer">
            {WINDOW_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          <span className="text-[0.65rem] text-[#A1A1AA]">通道:</span>
          <select value={channelIdx} onChange={e => onChangeChannel(Number(e.target.value))}
            className="text-[0.6rem] px-2 py-1 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer">
            {CHANNELS.map((ch, i) => <option key={ch.id} value={i}>{ch.name}</option>)}
          </select>
        </div>
      </div>

      {/* σ-space explanation */}
      <div className="p-2 rounded-lg bg-[#F0F9FF] border border-[#BAE6FD] text-[0.6rem] text-[#0369A1] leading-relaxed">
        💡 <b>σ空间说明</b>：所有曲线经过 <b>StandardScaler 标准化</b>（减均值 ÷ 标准差）。
        值 &gt; 0 表示<b>高于</b>历史平均水平，值 &lt; 0 表示<b>低于</b>平均水平。
        这是时序预测领域的标准做法——将不同量纲的通道统一到同一尺度，确保各模型在公平条件下比较。
      </div>

      {/* Compare toggle */}
      {allModels.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" checked={compareEnabled}
              onChange={e => setCompareEnabled(e.target.checked)}
              className="w-3 h-3 accent-[#3B82F6]" />
            <span className="text-[0.65rem] text-[#52525B] font-medium">📊 多模型对比</span>
          </label>
          {compareEnabled && (
            <select multiple value={compareModels}
              onChange={e => setCompareModels(Array.from(e.target.selectedOptions, o => o.value))}
              className="text-[0.6rem] px-2 py-1 rounded-lg border border-[rgba(0,0,0,0.06)] bg-white cursor-pointer"
              style={{ minWidth: '140px', height: '24px' }}
              title="按住 Ctrl 多选">
              {suggestedCompare.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          )}
          {compareEnabled && (
            <span className="text-[0.55rem] text-[#A1A1AA]">按住 Ctrl 多选</span>
          )}
        </div>
      )}

      {/* Charts */}
      <div className="space-y-2 max-h-[750px] overflow-y-auto">
        {presetLoading && <div className="h-[180px] flex items-center justify-center text-xs text-[#A1A1AA]">⏳ 加载中...</div>}

        {!presetLoading && presetData?.curves && (
          <div className="p-3 rounded-xl bg-[#F9FAFB] border border-[#E5E7EB]">
            <ChartBlock
              curves={presetData.curves}
              chKey={chKey}
              label={`📋 ${model} 预置`}
              color={modelColor}
              isPreset={true}
              windowInfo={presetWindowInfo}
              compareLines={compareLines}
            />
            {/* Legend for compare lines */}
            {compareLines.length > 0 && (
              <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1">
                {compareLines.map(cl => (
                  <span key={cl.label} className="text-[0.55rem] flex items-center gap-1">
                    <span style={{
                      display: 'inline-block', width: '12px', height: '2px',
                      backgroundColor: cl.color, borderTop: '1px dashed ' + cl.color,
                    }} />
                    {cl.label}
                  </span>
                ))}
              </div>
            )}
            {presetData.metrics_summary && (
              <div className="flex gap-3 text-[0.6rem] text-[#9CA3AF] mt-2 pt-2 border-t border-[#E5E7EB]">
                <span>MSE: <b>{presetData.metrics_summary.mse}</b></span>
                <span>MAE: <b>{presetData.metrics_summary.mae}</b></span>
                <span>RMSE: <b>{presetData.metrics_summary.rmse}</b></span>
              </div>
            )}
          </div>
        )}

        {runHistory.map((entry, i) => (
          <div key={i} className="p-3 rounded-xl bg-white border border-[#DBEAFE]">
            <ChartBlock
              curves={entry.curves}
              chKey={chKey}
              label={`✅ 训练结果 ${entry.label}`}
              color={HISTORY_COLORS[i % HISTORY_COLORS.length]}
            />
            {entry.metrics && (
              <div className="flex gap-2 text-[0.6rem] text-[#6B7280]">
                {Object.entries(entry.metrics).slice(0, 5).map(([k, v]) =>
                  <span key={k}>{k}: <b>{typeof v === 'number' ? v.toFixed(4) : v}</b></span>
                )}
              </div>
            )}
          </div>
        ))}

        {!presetLoading && !presetData?.curves && runHistory.length === 0 && (
          <div className="h-[180px] flex items-center justify-center text-xs text-[#A1A1AA]">
            暂无数据，点击下方按钮生成预测曲线
          </div>
        )}
      </div>

      {/* Progress */}
      {runStatus === 'running' && (
        <div className="p-3 rounded-lg bg-[#FEF3C7] border border-[#FCD34D] text-xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-medium text-[#92400E]">
              {runPhase === 'loading_model' ? '🔄 加载预训练模型...'
                : isCurvePhase ? `📊 生成曲线... ${curveDone}/${curveTotal || '?'}`
                : runType === 'inference_stat' ? `📊 计算中... (${runPct}%)`
                : `🔥 Epoch ${runEpoch}/${runTotalEpochs || '?'} · Loss ${runLoss.toFixed(4)} (${runPct}%)`}
            </span>
            <button onClick={handleCancel} className="text-[0.6rem] px-2 py-0.5 rounded bg-[#EF4444] text-white hover:bg-[#DC2626] cursor-pointer">取消</button>
          </div>
          <div className="w-full h-2 rounded-full bg-[#FDE68A] overflow-hidden">
            <div className="h-full rounded-full bg-[#D97706] transition-all duration-500" style={{ width: `${Math.min(runPct, 100)}%` }} />
          </div>
        </div>
      )}
      {runStatus === 'cancelled' && <div className="p-2 rounded-lg bg-[#FFF7ED] border text-xs text-[#9A3412]">⏹️ 已取消</div>}
      {error && <div className="p-2 rounded-lg bg-[#FEF2F2] border text-xs text-[#991B1B] max-h-32 overflow-auto">❌ {error}</div>}

      {/* Buttons */}
      <div className="flex flex-wrap items-center gap-2">
        {!isAvailable ? <p className="text-xs text-[#A1A1AA]">🔒 暂不可用</p>
          : runStatus === 'running'
          ? <button onClick={handleCancel} className="text-xs px-4 py-2 rounded-xl bg-[#EF4444] text-white hover:bg-[#DC2626] cursor-pointer font-medium">⏹️ 停止</button>
          : <button onClick={handleRun} disabled={presetLoading}
              className="text-xs px-4 py-2 rounded-xl bg-[#D97706] text-white hover:bg-[#B45309] cursor-pointer font-medium disabled:opacity-50">
              {runHistory.length > 0 ? `🔄 再次运行 (#${runHistory.length + 1})` : (RUN_TYPE_LABEL[runType] || RUN_TYPE_LABEL.train_dl)}
            </button>}
        {runHistory.length > 0 && runStatus !== 'running' && (
          <button onClick={handleClearHistory} className="text-xs px-3 py-2 rounded-xl bg-[#F3F4F6] text-[#6B7280] hover:bg-[#E5E7EB] cursor-pointer">🗑️ 清除</button>
        )}
      </div>
    </div>
  );
}
