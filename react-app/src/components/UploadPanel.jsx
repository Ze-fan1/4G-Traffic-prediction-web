import { useState, useCallback, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const API_BASE = '/api';
const PRED_LEN_OPTIONS = [6, 12, 18, 24];

export default function UploadPanel({ model, isAvailable }) {
  const [csvFile, setCsvFile] = useState(null);
  const [csvPreview, setCsvPreview] = useState(null);
  const [targetCol, setTargetCol] = useState('');
  const [predLen, setPredLen] = useState(24);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = useCallback(async (file) => {
    if (!file || !file.name.endsWith('.csv')) {
      setError('请选择 .csv 文件');
      return;
    }
    setCsvFile(file);
    setError(null);
    setResult(null);

    // Frontend preview: read first 6 lines
    const text = await file.text();
    const lines = text.trim().split('\n');
    const headers = lines[0].split(/[,\t;]/);
    const previewRows = lines.slice(1, 6).map(line => line.split(/[,\t;]/));

    // Identify numeric columns
    const numericCols = [];
    headers.forEach((h, i) => {
      const vals = previewRows.map(r => parseFloat(r[i])).filter(v => !isNaN(v));
      if (vals.length === previewRows.length) numericCols.push(h.trim());
    });

    setCsvPreview({ headers: headers.map(h => h.trim()), rows: previewRows, numericCols });
    if (numericCols.length > 0) setTargetCol(numericCols[0]);
  }, []);

  const handlePredict = useCallback(async () => {
    if (!csvFile || !targetCol) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('csv_file', csvFile);
      formData.append('target_col', targetCol);
      formData.append('pred_len', String(predLen));

      const res = await fetch(`${API_BASE}/predict/${encodeURIComponent(model)}`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail?.detail || err.detail?.error || '预测失败');
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, [csvFile, targetCol, predLen, model]);

  const downloadCSV = useCallback(() => {
    if (!result?.predictions) return;
    const headers = `hour,${model}_prediction`;
    const rows = result.predictions.map((v, i) => `${i + 1},${v.toFixed(6)}`);
    const csv = [headers, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prediction_${model}_${targetCol}_${predLen}h.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result, model, targetCol, predLen]);

  if (!isAvailable) {
    return (
      <div className="p-4 text-center text-xs text-[#A1A1AA]">
        🔒 该模型暂不支持自定义数据上传
      </div>
    );
  }

  const chartData = result ? {
    labels: Array.from({ length: result.predictions.length }, (_, i) => `${i + 1}h`),
    datasets: [{
      label: `${model} 预测 (${targetCol})`,
      data: result.predictions,
      borderColor: '#3B82F6',
      backgroundColor: '#3B82F620',
      borderWidth: 2,
      pointRadius: 2,
      tension: 0.35,
    }],
  } : null;

  return (
    <div className="space-y-4">
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
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        {csvFile ? (
          <p className="text-xs text-[#52525B]">📁 {csvFile.name} ({(csvFile.size / 1024).toFixed(1)} KB)</p>
        ) : (
          <p className="text-xs text-[#A1A1AA]">拖拽 CSV 文件到此处，或点击上传</p>
        )}
      </div>

      {/* CSV preview */}
      {csvPreview && (
        <div className="text-xs">
          <p className="font-medium text-[#52525B] mb-1">数据预览 (前 5 行)</p>
          <div className="overflow-x-auto rounded-lg border border-[rgba(0,0,0,0.05)]">
            <table className="w-full" style={{ fontSize: '0.65rem' }}>
              <thead>
                <tr className="bg-[#FAFAFA]">
                  {csvPreview.headers.map((h, i) => (
                    <th key={i} className="px-2 py-1 text-left text-[#A1A1AA] font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {csvPreview.rows.map((row, i) => (
                  <tr key={i} className="border-t border-[rgba(0,0,0,0.03)]">
                    {row.map((cell, j) => (
                      <td key={j} className="px-2 py-0.5 text-[#52525B]">{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Options */}
      {csvPreview && (
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#A1A1AA]">目标列:</span>
            <select
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="text-xs px-3 py-1.5 rounded-xl border border-[rgba(0,0,0,0.05)] bg-white cursor-pointer"
            >
              {csvPreview.numericCols.map(c => (
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
        </div>
      )}

      {/* Predict button */}
      {csvFile && (
        <button
          onClick={handlePredict}
          disabled={isLoading || !targetCol}
          className="text-xs px-4 py-2 rounded-xl bg-[#3B82F6] text-white hover:bg-[#2563EB] disabled:opacity-50 transition-colors cursor-pointer font-medium"
        >
          {isLoading ? '⏳ 推理中...' : '🚀 开始预测'}
        </button>
      )}

      {/* Error */}
      {error && (
        <div className="p-2 rounded-lg bg-[#FEF2F2] border border-[#FECACA] text-xs text-[#991B1B]">❌ {error}</div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-3">
          <div className="chart-box-lg">
            <Line data={chartData} options={{
              responsive: true, maintainAspectRatio: false,
              plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } },
              scales: {
                x: { title: { display: true, text: '预测时刻 (小时)', color: '#A1A1AA' } },
                y: { title: { display: true, text: '原始值', color: '#A1A1AA' } },
              },
            }} />
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={downloadCSV}
              className="text-xs px-4 py-2 rounded-xl border border-[rgba(0,0,0,0.08)] bg-white hover:bg-stone-50 transition-colors cursor-pointer"
            >
              📥 下载预测 CSV
            </button>
            <span className="text-[0.6rem] text-[#A1A1AA]">
              模型: {result.model} · 输入 {result.meta.input_rows} 行 · 设备: {result.meta.device}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
