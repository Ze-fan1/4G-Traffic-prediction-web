import { useState, useEffect } from 'react';
import ModelSelector from '../components/ModelSelector';
import DemoPanel from '../components/DemoPanel';
import UploadPanel from '../components/UploadPanel';

const API_BASE = import.meta.env.VITE_API_URL || '/api';
const MODELS_CACHE_KEY = 'playground_models_cache_v3';

export default function PlaygroundPage() {
  const [selectedModel, setSelectedModel] = useState(() => {
    return sessionStorage.getItem('pg_model') || 'Naive';
  });
  const [channelIdx, setChannelIdx] = useState(() => {
    const v = sessionStorage.getItem('pg_channel');
    return v != null ? parseInt(v) : 1;
  });
  const [activeSection, setActiveSection] = useState('demo');
  const [modelInfo, setModelInfo] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem(MODELS_CACHE_KEY) || '{}'); } catch { return {}; }
  });

  // 持久化选中模型和通道
  useEffect(() => { sessionStorage.setItem('pg_model', selectedModel); }, [selectedModel]);
  useEffect(() => { sessionStorage.setItem('pg_channel', String(channelIdx)); }, [channelIdx]);

  // 从后端 API 获取真实模型信息
  useEffect(() => {
    const loadModels = () => fetch(`${API_BASE}/models`)
      .then(res => res.json())
      .then(data => {
        const info = {};
        (data.models || data).forEach(m => { info[m.name] = m; });
        sessionStorage.setItem(MODELS_CACHE_KEY, JSON.stringify(info));
        setModelInfo(info);
      })
      .catch(() => {});
    loadModels();
    const timer = window.setInterval(loadModels, 30000);
    window.addEventListener('benchmark-updated', loadModels);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener('benchmark-updated', loadModels);
    };
  }, []);

  const currentInfo = modelInfo[selectedModel] || {};
  const isAvailable = currentInfo.available ?? ((currentInfo.tier || 1) === 1);
  const runType = currentInfo.run_type || 'unknown';
  const selectorModels = Object.values(modelInfo).map(info => ({
    model: info.name,
    cat: info.category,
    runType: info.run_type,
    verified: info.verified,
    available: info.available,
    availabilityReason: info.availability_reason,
  }));

  return (
    <div className="page-enter mt-4">
      <div className="flex gap-4" style={{ minHeight: 'calc(100vh - 140px)' }}>
        {/* Left: Model list 280px */}
        <div style={{ width: '280px', flexShrink: 0 }}>
          <ModelSelector
            selectedModel={selectedModel}
            onSelect={setSelectedModel}
            models={selectorModels}
            modelTiers={Object.fromEntries(
              Object.entries(modelInfo).map(([k, v]) => [k, v.tier])
            )}
          />
        </div>

        {/* Right: Experiment panel */}
        <div className="flex-1 space-y-4">
          <div className="card p-5">
            {/* Sub-tab: Demo / Upload */}
            <div className="flex items-center gap-1 mb-4">
              <button
                onClick={() => setActiveSection('demo')}
                className={`text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${
                  activeSection === 'demo'
                    ? 'bg-[#3B82F6] text-white'
                    : 'bg-[#F5F5F5] text-[#52525B] hover:bg-[#E5E5E5]'
                }`}
              >
                📈 数据验证
              </button>
              <button
                onClick={() => setActiveSection('upload')}
                className={`text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${
                  activeSection === 'upload'
                    ? 'bg-[#3B82F6] text-white'
                    : 'bg-[#F5F5F5] text-[#52525B] hover:bg-[#E5E5E5]'
                }`}
              >
                📁 上传数据
              </button>
            </div>

            {activeSection === 'demo' ? (
              <DemoPanel
                model={selectedModel}
                channelIdx={channelIdx}
                onChangeChannel={setChannelIdx}
                isAvailable={isAvailable}
                runType={runType}
              />
            ) : (
              <UploadPanel selectedModel={selectedModel} modelInfo={currentInfo} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
