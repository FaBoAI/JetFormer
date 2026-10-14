// 確認 r4: 抜け出しを使う標準の物理の開始で、1 フレームの計算の時間(Node)を測る(CPU の混み具合は uptime と一緒に書く)
import os from 'node:os';
import {runOne} from './run_ck4.mjs';
const S = [
  {id: 'LS273', set: 'T', start: {gate: 23, off: 0.041729, yaw: 14.6211}},
  {id: 'fid_escape_rev', set: 'T', start: {gate: 23, off: -0.09944396461695658, yaw: -14.927424}},
  {id: 'DRL154-std', set: 'T', start: {gate: 23, off: 0.041635, yaw: 13.036841}},
  {id: 'plain', set: 'T', start: {gate: 5, off: 0.05, yaw: 3}},
];
for (const s of S) for (let k = 0; k < 2; k++) {
  const load = os.loadavg()[0];
  const r = await runOne({...s, seed: 1});
  console.log(JSON.stringify({id: s.id, k, load1: +load.toFixed(2), esc: r.escapes, rev: r.reverse_frames, ms_p50: +r.ms_p50.toFixed(2), ms_p99: +r.ms_p99.toFixed(2), ms_max: +r.ms_max.toFixed(1), over33: r.ms_over33, ms_mean: +r.ms_mean.toFixed(2)}));
}
