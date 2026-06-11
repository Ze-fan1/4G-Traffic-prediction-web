export function channelTruth(ch, hours) {
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
