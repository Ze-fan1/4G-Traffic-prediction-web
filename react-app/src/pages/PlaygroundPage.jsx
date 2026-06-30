import { useState } from 'react';
import ModelSelector from '../components/ModelSelector';
import DemoPanel from '../components/DemoPanel';
import UploadPanel from '../components/UploadPanel';

const MODEL_TIERS = {
  '★ BaseModel': 2,
  'Informer': 2, 'LightTS': 2, 'TSMixer': 2, 'SCINet': 2, 'Mamba': 2, 'TimeLLM': 2,
  'IBM TTM': 3,
};

export default function PlaygroundPage() {
  const [selectedModel, setSelectedModel] = useState('★ BaseModel');
  const [channelIdx, setChannelIdx] = useState(1);
  const [activeSection, setActiveSection] = useState('demo');

  const isAvailable = (MODEL_TIERS[selectedModel] || 1) === 1;

  return (
    <div className="page-enter mt-4">
      <div className="flex gap-4" style={{ minHeight: 'calc(100vh - 140px)' }}>
        {/* Left: Model list 280px */}
        <div style={{ width: '280px', flexShrink: 0 }}>
          <ModelSelector
            selectedModel={selectedModel}
            onSelect={setSelectedModel}
            modelTiers={MODEL_TIERS}
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
                📈 Demo 验证
              </button>
              <button
                onClick={() => setActiveSection('upload')}
                className={`text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer font-medium ${
                  activeSection === 'upload'
                    ? 'bg-[#3B82F6] text-white'
                    : 'bg-[#F5F5F5] text-[#52525B] hover:bg-[#E5E5E5]'
                }`}
              >
                📁 自定义预测
              </button>
            </div>

            {activeSection === 'demo' ? (
              <DemoPanel
                model={selectedModel}
                channelIdx={channelIdx}
                onChangeChannel={setChannelIdx}
                isAvailable={isAvailable}
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
