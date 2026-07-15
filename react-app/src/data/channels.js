// Real 4G traffic feature channels (from parquet data)
export const CHANNELS = [
  { id: 'erab',     name: 'erab流量',         desc: 'ERAB Traffic · 主要流量指标' },
  { id: 'pdcch',    name: 'pdcch利用率',      desc: 'PDCCH Utilization · 控制信道占用率' },
  { id: 'pdsch',    name: 'pdsch利用率',      desc: 'PDSCH Utilization · 下行共享信道占用率' },
  { id: 'pusch',    name: 'pusch利用率',      desc: 'PUSCH Utilization · 上行共享信道占用率' },
  { id: 'ul_traffic', name: '上行流量',        desc: 'Uplink Traffic · 上行数据量' },
  { id: 'dl_traffic', name: '下行流量',        desc: 'Downlink Traffic · 下行数据量' },
  { id: 'total',    name: '总流量',           desc: 'Total Traffic · 上下行合计' },
  { id: 'conn',     name: '有效连接数',        desc: 'Active Connections · 在线连接数量' },
];
