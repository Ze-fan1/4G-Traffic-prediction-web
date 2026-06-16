export const CAT_PALETTE = {
  'Baseline':    { bg: '#DCFCE7', border: '#16A34A', text: '#15803D', fill: 'rgba(22,163,74,0.72)', fillLight: 'rgba(22,163,74,0.12)', label: '基准模型' },
  'Statistical': { bg: '#F5F5F4', border: '#78716C', text: '#57534E', fill: 'rgba(120,113,108,0.55)', fillLight: 'rgba(120,113,108,0.08)', label: '传统统计' },
  'Tree':        { bg: '#FEF9C3', border: '#B45309', text: '#92400E', fill: 'rgba(180,83,9,0.55)', fillLight: 'rgba(180,83,9,0.1)', label: '树模型' },
  'Transformer': { bg: '#FEF3C7', border: '#D97706', text: '#B45309', fill: 'rgba(217,119,6,0.6)', fillLight: 'rgba(217,119,6,0.1)', label: 'Transformer' },
  'MLP':         { bg: '#EEECFE', border: '#6152F2', text: '#4F46E5', fill: 'rgba(97,82,242,0.55)', fillLight: 'rgba(97,82,242,0.08)', label: 'MLP' },
  'CNN':         { bg: '#F3E8FF', border: '#7C3AED', text: '#6D28D9', fill: 'rgba(124,58,237,0.55)', fillLight: 'rgba(124,58,237,0.08)', label: 'CNN' },
  'RNN':         { bg: '#FEE2E2', border: '#DC2626', text: '#B91C1C', fill: 'rgba(220,38,38,0.55)', fillLight: 'rgba(220,38,38,0.1)', label: 'RNN' },
  'SSM':         { bg: '#CFFAFE', border: '#0891B2', text: '#0E7490', fill: 'rgba(8,145,178,0.55)', fillLight: 'rgba(8,145,178,0.08)', label: 'SSM' },
  'LLM':         { bg: '#FCE7F3', border: '#BE185D', text: '#9D174D', fill: 'rgba(190,24,93,0.55)', fillLight: 'rgba(190,24,93,0.08)', label: 'LLM' },
};

export const MODEL_COLORS_20 = [
  '#16A34A', // Baseline green
  '#DC2626', // Red
  '#2563EB', // Blue
  '#D97706', // Amber
  '#7C3AED', // Purple
  '#6152F2', // Indigo
  '#0891B2', // Cyan
  '#BE185D', // Pink
  '#78716C', // Warm gray
  '#059669', // Emerald
  '#EA580C', // Orange
  '#4F46E5', // Deep indigo
  '#65A30D', // Lime
  '#9333EA', // Violet
  '#0284C7', // Sky
  '#C2410C', // Rust
  '#10B981', // Teal
  '#DB2777', // Magenta
  '#F59E0B', // Yellow
  '#8B5CF6', // Lavender
];
export const MODEL_COLORS_12 = MODEL_COLORS_20;
export const MODEL_COLORS_9 = MODEL_COLORS_20;
export const MODEL_COLORS_8 = MODEL_COLORS_20;
export const MODEL_COLORS_6 = ['#16A34A', '#DC2626', '#2563EB', '#D97706', '#7C3AED', '#6152F2'];

export function getPalette(model) {
  return CAT_PALETTE[model.cat] || CAT_PALETTE['Statistical'];
}
