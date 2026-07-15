export const MODELS = [
  { cat: 'Statistical', model: 'Naive', mse: 3.8318, mae: 0.6734, rmse: 1.9575, mape: null, acc: null },
  { cat: 'Statistical', model: 'Persistent 24h', mse: 1.5664, mae: 0.4079, rmse: 1.2516, mape: null, acc: null },
  { cat: 'Baseline', model: '★ BaseModel', mse: 3.6413, mae: 0.6815, rmse: 1.9082, mape: 87.9599, acc: 0.5853 },
];

export const CATS = [...new Set(MODELS.map(m => m.cat))];
