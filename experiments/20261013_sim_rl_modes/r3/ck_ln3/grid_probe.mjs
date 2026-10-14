// 角(左右 −0.2・向き +15°)の近くの細かい格子(2026-10-14、確認の段): ゲート 16・17 × 左右 6 × 向き 6、形は 2 つ(const 0・random 777)、c_v 0.95、シールドあり
//   node grid_probe.mjs <w> <W>
import fs from 'node:fs';
import {runOne, startInfo} from './ck_run.mjs';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/runs';
const [w, W] = [+process.argv[2], +process.argv[3]];
const jobs = [];
let i = 0;
for (const g of [16, 17]) for (const off of [-.2, -.195, -.19, -.18, -.17, -.16]) for (const yaw of [15, 14.5, 14, 13, 12, 11]) for (const line of [{family: 'const', c0: 0}, {family: 'random', seed: 777}]) {
  jobs.push({set: 'GRID', id: `GR-${i}`, start: {gate: g, left_offset_m: off, left_yaw_deg: yaw, seed: 4723000 + i}, params: {line, c_v: .95}, family: line.family, policy: 'line', rt: 'r3', shield: true}); i++;
}
for (const j of jobs.filter((_, k) => k % W === w)) {
  const r = runOne({...j, keepFrames: false});
  delete r.bins; delete r.trk_e4;
  fs.appendFileSync(`${OUT}/grid_${w}.jsonl`, JSON.stringify(r) + '\n');
}
console.log('done grid', w);
