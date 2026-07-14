import { useState, useEffect } from 'react';
import ModelSelector from '../components/ModelSelector';
import DemoPanel from '../components/DemoPanel';
import UploadPanel from '../components/UploadPanel';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

export default function PlaygroundPage() {
  const [selectedModel, setSelectedModel] = useState(() => {
    return sessionStorage.getItem('pg_model') || 'Naive';
  });
  const [channelIdx, setChannelIdx] = useState(() => {
    const v = sessionStorage.getItem('pg_channel');
    return v != null ? parseInt(v) : 1;
  });
  const [activeSection, setActiveSection] = useState('demo');
  const [modelInfo, setModelInfo] = useState({});

  // 持久化选中模型和通道
  useEffect(() => { sessionStorage.setItem('pg_model', selectedModel); }, [selectedModel]);
  useEffect(() => { sessionStorage.setItem('pg_channel', String(channelIdx)); }, [channelIdx]);

  // 从后端 API 获取真实模型信息
  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then(res => res.json())
      .then(data => {
        const info = {};
        data.forEach(m => { info[m.name] = m; });
        setModelInfo(info);
      })
      .catch(() => {
        setModelInfo({});
      });
  }, []);

  const currentInfo = modelInfo[selectedModel] || {};
  const isAvailable = (currentInfo.tier || 1) === 1;
  const runType = currentInfo.run_type || 'unknown';

  return (
    <div className="page-enter mt-4">
      <div className="flex gap-4" style={{ minHeight: 'calc(100vh - 140px)' }}>
        {/* Left: Model list 280px */}
        <div style={{ width: '280px', flexShrink: 0 }}>
          <ModelSelector
            selectedModel={selectedModel}
            onSelect={setSelectedModel}
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
                📁 自定义数据
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
              <UploadPanel
                model={selectedModel}
                isAvailable={isAvailable}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
