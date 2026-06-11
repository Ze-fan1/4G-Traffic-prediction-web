"""Rebuild all React source files after git mishap."""
import os

SRC = r'c:\Users\Admin\Desktop\可视化网站\react-app\src'

files = {}

# ── data/palette.js ──
files['data/palette.js'] = r'''export const CAT_PALETTE = {
  'Baseline':    { bg: '#EEECFE', border: '#6152F2', text: '#4F46E5', fill: 'rgba(97,82,242,0.72)', fillLight: 'rgba(97,82,242,0.12)', label: '基准模型' },
  'Statistical': { bg: '#F5F5F4', border: '#78716C', text: '#57534E', fill: 'rgba(120,113,108,0.55)', fillLight: 'rgba(120,113,108,0.08)', label: '传统统计' },
  'Tree':        { bg: '#DCFCE7', border: '#16A34A', text: '#15803D', fill: 'rgba(22,163,74,0.55)', fillLight: 'rgba(22,163,74,0.1)', label: '树模型' },
  'Transformer': { bg: '#FEF3C7', border: '#D97706', text: '#B45309', fill: 'rgba(217,119,6,0.6)', fillLight: 'rgba(217,119,6,0.1)', label: 'Transformer' },
  'MLP':         { bg: '#DBEAFE', border: '#2563EB', text: '#1D4ED8', fill: 'rgba(37,99,235,0.55)', fillLight: 'rgba(37,99,235,0.08)', label: 'MLP' },
  'CNN':         { bg: '#F3E8FF', border: '#7C3AED', text: '#6D28D9', fill: 'rgba(124,58,237,0.55)', fillLight: 'rgba(124,58,237,0.08)', label: 'CNN' },
  'RNN':         { bg: '#FEE2E2', border: '#DC2626', text: '#B91C1C', fill: 'rgba(220,38,38,0.55)', fillLight: 'rgba(220,38,38,0.1)', label: 'RNN' },
  'SSM':         { bg: '#CFFAFE', border: '#0891B2', text: '#0E7490', fill: 'rgba(8,145,178,0.55)', fillLight: 'rgba(8,145,178,0.08)', label: 'SSM' },
  'LLM':         { bg: '#FCE7F3', border: '#BE185D', text: '#9D174D', fill: 'rgba(190,24,93,0.55)', fillLight: 'rgba(190,24,93,0.08)', label: 'LLM' },
};

export const MODEL_COLORS_8 = ['#6152F2', '#DC2626', '#2563EB', '#D97706', '#7C3AED', '#16A34A', '#0891B2', '#BE185D'];
export const MODEL_COLORS_6 = ['#6152F2', '#DC2626', '#2563EB', '#D97706', '#7C3AED', '#16A34A'];

export function getPalette(model) {
  return CAT_PALETTE[model.cat] || CAT_PALETTE['Statistical'];
}
'''

# ── data/channels.js ──
files['data/channels.js'] = r'''export const CHANNELS = [
  { id: 'total', name: '总流量', base: 0.55, amp1: 0.3, amp2: 0.12, noise: 0.08, desc: 'Total Traffic · 上下行合计' },
  { id: 'dl_prb', name: '下行PRB利用率', base: 0.42, amp1: 0.28, amp2: 0.1, noise: 0.1, desc: 'DL PRB Utilization · 资源块占用率' },
  { id: 'ul_prb', name: '上行PRB利用率', base: 0.25, amp1: 0.12, amp2: 0.06, noise: 0.07, desc: 'UL PRB Utilization · 上行资源占用' },
  { id: 'rrc', name: 'RRC连接数', base: 0.6, amp1: 0.2, amp2: 0.1, noise: 0.05, desc: 'RRC Connections · 无线资源控制连接' },
  { id: 'users', name: '活跃用户数', base: 0.5, amp1: 0.22, amp2: 0.08, noise: 0.06, desc: 'Active Users · 在线用户数量' },
  { id: 'dl_tp', name: '下行吞吐量', base: 0.48, amp1: 0.35, amp2: 0.15, noise: 0.12, desc: 'DL Throughput · 下行数据速率' },
  { id: 'ul_tp', name: '上行吞吐量', base: 0.2, amp1: 0.1, amp2: 0.05, noise: 0.08, desc: 'UL Throughput · 上行数据速率' },
  { id: 'latency', name: '平均时延', base: 0.35, amp1: -0.15, amp2: -0.08, noise: 0.06, desc: 'Avg Latency · 端到端时延（越低越好）' },
];
'''

# ── utils/simulation.js ──
files['utils/simulation.js'] = r'''export function channelTruth(ch, hours) {
  return hours.map(h => ch.base + ch.amp1 * Math.sin((h - 6) * Math.PI / 12) + ch.amp2 * Math.cos((h - 12) * Math.PI / 6));
}

export function simPrediction(truth, mse, mae) {
  const std = Math.sqrt(mse);
  return truth.map(t => t + (Math.random() - 0.5) * std * 2.2 + (Math.random() - 0.5) * mae * 0.8);
}

export function cumulativeMAE(baseMAE, hours) {
  return hours.map(h => baseMAE * Math.pow(h / 24, 0.6) * (1 + (Math.random() - 0.5) * 0.25));
}

export function gauss(x, mean, std) {
  return Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI));
}
'''

# ── Write all files ──
for path, content in files.items():
    full = os.path.join(SRC, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        f.write(content.lstrip('\n'))
    print(f'  OK: {path}')

print(f'\nWrote {len(files)} files')
