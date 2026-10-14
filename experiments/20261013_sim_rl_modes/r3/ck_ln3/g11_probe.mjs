// ゲート 11(黒シートの線 z = 0 の上の点)からの開始の調べ(2026-10-14、確認の段)。新しい組 U300 の U300-278 が
// required_crossing_missing(黒シートの横断なしで周が閉じた)で無効になったので、ゲート 11 の開始を 左右 5 × 向き 5 × 形 8(c_v 0.95)と最速で走らせる
//   node g11_probe.mjs <w> <W>
import fs from 'node:fs';
import {runOne, startInfo} from './ck_run.mjs';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/runs';
const [w, W] = [+process.argv[2], +process.argv[3]];
const SH = [{family: 'const', c0: .75}, {family: 'const', c0: -.75}, {family: 'const', c0: 0}, {family: 'inside', a: .7}, {family: 'outside', a: .7},
  {family: 'late_apex', a: .7}, {family: 'early_apex', a: .7}, {family: 'random', seed: 12345}];
const jobs = [];
let i = 0;
for (const off of [-.2, -.1, 0, .1, .2]) for (const yaw of [-15, -7.5, 0, 7.5, 15]) {
  const I = startInfo(11, off, yaw);
  if (!(I.clearance >= .12)) continue;
  for (const line of SH) { jobs.push({set: 'G11', id: `G11-${i}`, start: {gate: 11, left_offset_m: off, left_yaw_deg: yaw, seed: 4718000 + i}, params: {line, c_v: .95}, family: line.family, policy: 'line', rt: 'r3', shield: true}); i++; }
  jobs.push({set: 'G11F', id: `G11F-${i}`, start: {gate: 11, left_offset_m: off, left_yaw_deg: yaw, seed: 4718000 + i}, params: {}, family: null, policy: 'fastest', rt: 'web', shield: false}); i++;
}
const file = `${OUT}/g11_${w}.jsonl`;
for (const j of jobs.filter((_, k) => k % W === w)) {
  const r = runOne({...j, keepFrames: false});
  delete r.bins; delete r.trk_e4;
  r.start_z = startInfo(11, j.start.left_offset_m, j.start.left_yaw_deg).z;
  fs.appendFileSync(file, JSON.stringify(r) + '\n');
}
console.log('done', w);
