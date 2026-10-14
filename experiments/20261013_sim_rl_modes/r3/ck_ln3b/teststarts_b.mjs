// 確認 line r3(再実行): 試験の組の開始が開始の分布の中にあり、HEAD60 が全部 |Δψ| ≥ 1 rad かを、自前の幾何・射影で確かめる
import fs from 'node:fs';
import {startOf} from './run_b.mjs';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b';
const {jobs} = JSON.parse(fs.readFileSync(`${OUT}/testjobs.json`, 'utf8'));
const res = {};
for (const j of jobs) {
  const s = startOf(j.gate, j.off, j.yaw);
  const r = res[j.set] ??= {n: 0, out_of_range: 0, clr_lt_12: 0, head: 0, dpsi_vs_training_max: 0, clr_vs_training_max: 0, gates: {}};
  r.n++;
  if (Math.abs(j.off) > .2 + 1e-12 || Math.abs(j.yaw) > 15 + 1e-9 || j.gate < 0 || j.gate > 32) r.out_of_range++;
  if (s.clearance < .12) r.clr_lt_12++;
  if (Math.abs(s.dpsi) >= 1) r.head++;
  if (j.dpsi != null) r.dpsi_vs_training_max = Math.max(r.dpsi_vs_training_max, Math.abs(Math.abs(s.dpsi) - Math.abs(j.dpsi)));
  if (j.c_start != null) r.clr_vs_training_max = Math.max(r.clr_vs_training_max, Math.abs(s.clearance - j.c_start));
  r.gates[j.gate] = (r.gates[j.gate] ?? 0) + 1;
}
fs.writeFileSync(`${OUT}/teststarts_check.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify(res));
