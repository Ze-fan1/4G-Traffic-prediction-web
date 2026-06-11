export const CHANNELS = [
  { id: 'total', name: '总流量', base: 0.55, amp1: 0.3, amp2: 0.12, noise: 0.08, desc: 'Total Traffic · 上下行合计' },
  { id: 'dl_prb', name: '下行PRB利用率', base: 0.42, amp1: 0.28, amp2: 0.1, noise: 0.1, desc: 'DL PRB Utilization · 资源块占用率' },
  { id: 'ul_prb', name: '上行PRB利用率', base: 0.25, amp1: 0.12, amp2: 0.06, noise: 0.07, desc: 'UL PRB Utilization · 上行资源占用' },
  { id: 'rrc', name: 'RRC连接数', base: 0.6, amp1: 0.2, amp2: 0.1, noise: 0.05, desc: 'RRC Connections · 无线资源控制连接' },
  { id: 'users', name: '活跃用户数', base: 0.5, amp1: 0.22, amp2: 0.08, noise: 0.06, desc: 'Active Users · 在线用户数量' },
  { id: 'dl_tp', name: '下行吞吐量', base: 0.48, amp1: 0.35, amp2: 0.15, noise: 0.12, desc: 'DL Throughput · 下行数据速率' },
  { id: 'ul_tp', name: '上行吞吐量', base: 0.2, amp1: 0.1, amp2: 0.05, noise: 0.08, desc: 'UL Throughput · 上行数据速率' },
  { id: 'latency', name: '平均时延', base: 0.35, amp1: -0.15, amp2: -0.08, noise: 0.06, desc: 'Avg Latency · 端到端时延（越低越好）' },
];
