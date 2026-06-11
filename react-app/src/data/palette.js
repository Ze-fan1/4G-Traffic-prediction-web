export const CAT_PALETTE = {
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
