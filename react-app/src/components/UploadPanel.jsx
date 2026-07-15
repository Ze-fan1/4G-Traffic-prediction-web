import { useState, useCallback, useRef, useEffect } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const API_BASE = import.meta.env.VITE_API_URL || '/api';
const PRED_LEN_OPTIONS = [6, 12, 18, 24];
const ALLOWED_EXTS = ['.csv', '.tsv', '.txt', '.xlsx', '.xls', '.parquet'];

function getFileExt(filename) {
  const dot = filename.lastIndexOf('.');
  return dot >= 0 ? filename.slice(dot).toLowerCase() : '';
}

const PRED_COLORS = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#6366F1', '#14B8A6'];
// 全局缓存：所有模型的预测叠加在同一张图上（切换卡片不丢失，上传新文件才清空）
let globalPredHistory = [];

export default function UploadPanel({ selectedModel, modelInfo = {} }) {
  const [dataFile, setDataFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [targetCol, setTargetCol] = useState('');
  const [predLen, setPredLen] = useState(24);
  const [predHistory, setPredHistory] = useState(globalPredHistory);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // 保持 globalPredHistory 同步
  const updatePredHistory = useCallback((v) => {
    setPredHistory(prev => {
      const val = typeof v === 'function' ? v(prev) : v;
      globalPredHistory = val;
      return val;
    });
  }, []);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    const ext = getFileExt(file.name);
    if (!ALLOWED_EXTS.includes(ext)) {
      setError(`不支持的文件格式: ${ext}。支持: ${ALLOWED_EXTS.join(', ')}`);
      return;
    }
    setDataFile(file);
    setError(null);
    updatePredHistory([]);

    // CSV/TSV: 客户端快速预览
    if (ext === '.csv' || ext === '.tsv' || ext === '.txt') {
      const text = await file.text();
      const lines = text.trim().split('\n');
      const headers = lines[0].split(/[,\t;]/);
      const previewRows = lines.slice(1, 6).map(line => line.split(/[,\t;]/));

      const numericCols = [];
      headers.forEach((h, i) => {
        const vals = previewRows.map(r => parseFloat(r[i])).filter(v => !isNaN(v));
        if (vals.length === previewRows.length) numericCols.push(h.trim());
      });

      setFilePreview({
        format: ext.slice(1),
        headers: headers.map(h => h.trim()),
        rows: previewRows,
        numericCols,
        total_rows: lines.length - 1,
      });
      if (numericCols.length > 0) setTargetCol(numericCols[0]);
      return;
    }

    // Excel/Parquet: 调用后端解析预览
    setFilePreview({ format: ext.slice(1), loading: true });
    try {
      const formData = new FormData();
      formData.append('data_file', file);
      const res = await fetch(`${API_BASE}/parse-file`, { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail?.detail || err.detail?.error || '文件解析失败');
      }
      const data = await res.json();
      setFilePreview({
        format: data.format,
        headers: data.headers,
        numericCols: data.numeric_cols,
        rowsPreview: data.rows_preview,
        total_rows: data.total_rows,
      });
      if (data.numeric_cols?.length > 0) setTargetCol(data.numeric_cols[0]);
    } catch (e) {
      setError(e.message);
      setFilePreview(null);
    }
  }, []);

  const handlePredict = useCallback(async () => {
    if (!dataFile || !targetCol) return;
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('data_file', dataFile);
      formData.append('target_col', targetCol);
      formData.append('pred_len', String(predLen));
      formData.append('num_channels', '0');

      const res = await fetch(`${API_BASE}/predict/${encodeURIComponent(selectedModel)}`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail?.detail || err.detail?.error || '预测失败');
      }
      const data = await res.json();
      updatePredHistory(prev => [...prev.filter(e => e.model !== selectedModel), {
        model: selectedModel,
        predictions: data.predictions, meta: data.meta, ts: Date.now()
      }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, [dataFile, targetCol, predLen, selectedModel, updatePredHistory]);

  const downloadAllCSV = useCallback(() => {
    if (predHistory.length === 0) return;
    const labels = Array.from({ length: predHistory[0].predictions.length }, (_, i) => i + 1);
    const header = ['hour', ...predHistory.map(e => e.model)];
    const rows = labels.map((h, i) => [h, ...predHistory.map(e => e.predictions[i].toFixed(6))]);
    const csv = [header.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `predictions_${targetCol}_${predLen}h.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [predHistory, targetCol, predLen]);

  const formatLabel = (fmt) => {
    const map = { csv: 'CSV', tsv: 'TSV', xlsx: 'Excel', xls: 'Excel', parquet: 'Parquet' };
    return map[fmt] || fmt?.toUpperCase() || '?';
  };

  // 构建叠加图表：所有模型预测在同一张图上
  const allPreds = predHistory.flatMap(e => e.predictions);
  const globalMin = allPreds.length > 0 ? Math.min(...allPreds) : 0;
  const globalMax = allPreds.length > 0 ? Math.max(...allPreds) : 100;
  const yPad = (globalMax - globalMin) * 0.15 || 10;
  const sharedYMin = globalMin - yPad;
  const sharedYMax = globalMax + yPad;

  const chartData = predHistory.length > 0 ? {
    labels: Array.from({ length: predHistory[0].predictions.length }, (_, i) => `${i + 1}h`),
    datasets: predHistory.map((entry, i) => ({
      label: `${entry.model} (${targetCol})`,
      data: entry.predictions,
      borderColor: PRED_COLORS[i % PRED_COLORS.length],
      backgroundColor: PRED_COLORS[i % PRED_COLORS.length] + '18',
      borderWidth: 2.5,
      pointRadius: 1.5,
      tension: 0.35,
    })),
  } : null;

  const chartOptions = chartData ? {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 300 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'bottom', labels: { boxWidth: 10, padding: 12, font: { size: 9 } } },
      tooltip: { callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}` } },
    },
    scales: {
      x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { font: { size: 8 } }, title: { display: true, text: '预测时刻 (h)' } },
      y: {
        min: sharedYMin, max: sharedYMax,
        grid: { color: 'rgba(0,0,0,0.03)' },
        ticks: { font: { size: 8 } },
        title: { display: true, text: targetCol },
      },
    },
  } : null;

  return (
    <div className="space-y-4">
      <div className="p-3 rounded-xl bg-[#EFF6FF] border border-[#BFDBFE] text-xs text-[#1E3A5F] leading-relaxed">
        <b>Local model prediction</b>: using the selected model <b>{selectedModel}</b>.{' '}
        {modelInfo.custom_prediction
          ? (modelInfo.type === 'statistical'
            ? 'Statistical models accept any one numeric target column.'
            : 'This model requires the complete eight-column 4G feature contract. Missing channels are never fabricated.')
          : `This model cannot predict uploaded data: ${modelInfo.custom_prediction_reason || 'no reusable local weights.'}`}
      </div>
      {/* Upload area */}
      <div
        className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-colors cursor-pointer ${
          dragOver ? 'border-[#3B82F6] bg-[#EFF6FF]' : 'border-[rgba(0,0,0,0.08)] hover:border-[#3B82F6]'
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        {dataFile ? (
          <p className="text-xs text-[#52525B]">
            📁 {dataFile.name} ({(dataFile.size / 1024).toFixed(1)} KB)
            {filePreview?.format && (
              <span className="ml-1.5 px-1.5 py-0.5 rounded bg-[#EFF6FF] text-[#3B82F6] text-[0.6rem] font-medium">
                {formatLabel(filePreview.format)}
              </span>
            )}
          </p>
        ) : (
          <p className="text-xs text-[#A1A1AA]">拖拽文件到此处，或点击上传<br /><span className="text-[0.6rem]">支持 CSV / TSV / Excel / Parquet</span></p>
        )}
      </div>

      {/* File preview */}
      {filePreview && !filePreview.loading && (
        <div className="text-xs">
          <p className="font-medium text-[#52525B] mb-1">
            数据预览 ({formatLabel(filePreview.format)}) · {filePreview.total_rows} 行
          </p>
          {(filePreview.rows || filePreview.rowsPreview) && (
            <div className="overflow-x-auto rounded-lg border border-[rgba(0,0,0,0.05)]">
              <table className="w-full" style={{ fontSize: '0.65rem' }}>
                <thead>
                  <tr className="bg-[#FAFAFA]">
                    {filePreview.headers.map((h, i) => (
                      <th key={i} className="px-2 py-1 text-left text-[#A1A1AA] font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(filePreview.rows || filePreview.rowsPreview || []).slice(0, 5).map((row, i) => (
                    <tr key={i} className="border-t border-[rgba(0,0,0,0.03)]">
                      {(Array.isArray(row) ? row : Object.values(row || {})).map((cell, j) => (
                        <td key={j} className="px-2 py-0.5 text-[#52525B]">{typeof cell === 'number' ? cell.toFixed(4) : String(cell ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {filePreview.loading && (
            <p className="text-[#A1A1AA] text-xs">⏳ 正在解析文件...</p>
          )}
        </div>
      )}

      {/* Options */}
      {filePreview && !filePreview.loading && (
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#A1A1AA]">目标列:</span>
            <select
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer"
            >
              {filePreview.numericCols.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#A1A1AA]">预测时长:</span>
            <select
              value={predLen}
              onChange={(e) => setPredLen(Number(e.target.value))}
              className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer"
            >
              {PRED_LEN_OPTIONS.map(n => (
                <option key={n} value={n}>{n}h</option>
              ))}
            </select>
          </div>
          <span className="text-xs px-3 py-1.5 rounded-xl bg-[#F5F5F5] text-[#52525B]">Model: {selectedModel}</span>
        </div>
      )}

      {/* Predict button */}
      {dataFile && (
        <button
          onClick={handlePredict}
          disabled={isLoading || !targetCol || !modelInfo.custom_prediction}
          className="text-xs px-4 py-2 rounded-xl bg-[#3B82F6] text-white hover:bg-[#2563EB] disabled:opacity-50 transition-colors cursor-pointer font-medium"
        >
          {isLoading ? '⏳ 推理中...' : `🚀 使用 ${selectedModel} 预测`}
        </button>
      )}

      {/* Error */}
      {error && (
        <div className="p-2 rounded-lg bg-[#FEF2F2] border border-[#FECACA] text-xs text-[#991B1B]">❌ {error}</div>
      )}

      {/* Results — 所有模型叠加在同一张图上，共享Y轴 */}
      {predHistory.length > 0 && (
        <div className="space-y-3">
          <div style={{ height: '280px' }}>
            <Line data={chartData} options={chartOptions} />
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button onClick={downloadAllCSV}
              className="text-xs px-4 py-2 rounded-xl border border-[rgba(0,0,0,0.08)] bg-white hover:bg-stone-50 transition-colors cursor-pointer">
              📥 下载全部预测 CSV
            </button>
            <button onClick={() => updatePredHistory([])}
              className="text-xs px-3 py-2 rounded-xl bg-[#F3F4F6] text-[#6B7280] hover:bg-[#E5E7EB] cursor-pointer">
              🗑️ 清空全部
            </button>
            <span className="text-[0.6rem] text-[#A1A1AA]">
              已预测 {predHistory.length} 个通用方法 · 目标列: {targetCol} · Y轴: [{sharedYMin.toFixed(1)}, {sharedYMax.toFixed(1)}]
            </span>
            {predHistory[0]?.meta && (
              <span className="text-[0.6rem] text-[#1D4ED8]">
                留出回测 MAE: {predHistory[0].meta.validation_mae} · RMSE: {predHistory[0].meta.validation_rmse}
              </span>
            )}
          </div>
          {/* 模型列表 — 点击 × 删除单个 */}
          <div className="flex flex-wrap gap-1.5">
            {predHistory.map((e, i) => (
              <span key={e.model} className="text-[0.6rem] px-2 py-0.5 rounded-full border inline-flex items-center gap-1 cursor-default"
                style={{ borderColor: PRED_COLORS[i % PRED_COLORS.length], color: PRED_COLORS[i % PRED_COLORS.length] }}>
                {e.model}
                <button onClick={() => updatePredHistory(prev => prev.filter(x => x.model !== e.model))}
                  className="ml-0.5 w-3.5 h-3.5 rounded-full inline-flex items-center justify-center text-[0.55rem] hover:bg-black/10 cursor-pointer leading-none"
                  title={`移除 ${e.model}`}>
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
