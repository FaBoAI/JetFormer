// 確認 line r4: 追補 r4b の新しい試験の組 LN-G30R120-R4B と、r4 の LN-BOX120-R4 の開始・形・c_v を、決まりの文言だけから自前で作り直し、
// 学習の段の eval_test_r4b.json の値と照らす(開始のすき間は自前の geom4)
import fs from 'node:fs';
import {startInfo} from './run4.mjs';
const mb = a => () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const r4 = x => +x.toFixed(4);
const E = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r4b/eval_test_r4b.json', 'utf8')).tests;
const out = {};
// G30R120: ゲート 29→32 の順に 30 回ずつ、左右 U(0.16, 0.20)・向き U(12, 15)、開始のすき間 ≥ 0.12 まで同じ箱で引き直す(上限 200)
{
  const g = mb(0x4b320001), gs = mb(0x4b320002), mine = [];
  let i = 0;
  for (const gate of [29, 30, 31, 32]) for (let k = 0; k < 30; k++) {
    let st = null, tries = 0;
    for (; tries < 200 && !st; tries++) { const off = .16 + .04 * g(), yaw = 12 + 3 * g(), si = startInfo(gate, off, yaw); if (si.clearance >= .12) st = {gate, off, yaw, c: si.clearance, dpsi: si.dpsi}; }
    const c0 = r4(-.9 + .3 * gs()), cv = r4(.9 + .1 * gs());
    mine.push({...st, c0, cv, seed: 9300000 + i, tries}); i++;
  }
  const ev = E['LN-G30R120-R4B'];
  let maxd = 0, shapeBad = 0, seedBad = 0;
  ev.forEach((r, j) => {
    const m = mine[j];
    maxd = Math.max(maxd, Math.abs(m.off - r.start.left_offset_m), Math.abs(m.yaw - r.start.left_yaw_deg), m.gate === r.start.gate ? 0 : 99);
    if (r.line.family !== 'const' || Math.abs(r.line.c0 - m.c0) > 1e-9 || Math.abs(r.c_v - m.cv) > 1e-9) shapeBad++;
    if (r.start.seed !== m.seed) seedBad++;
  });
  out.G30R120 = {n_mine: mine.length, n_eval: ev.length, max_abs_diff_start: maxd, shape_mismatch: shapeBad, seed_mismatch: seedBad,
    redraws: mine.filter(m => m.tries > 1).length, min_start_c: Math.min(...mine.map(m => m.c)), c0_range: [Math.min(...mine.map(m => m.c0)), Math.max(...mine.map(m => m.c0))],
    cv_range: [Math.min(...mine.map(m => m.cv)), Math.max(...mine.map(m => m.cv))], c0_le_m085: mine.filter(m => m.c0 <= -.85).length,
    per_gate: [29, 30, 31, 32].map(gt => mine.filter(m => m.gate === gt).length)};
}
// 試験の組のほかの組: 開始のすき間 ≥ 0.12・範囲の中・大きく横を向いた開始の数を自前で
for (const [name, runs] of Object.entries(E)) {
  let minC = Infinity, out_of_range = 0, head = 0, cdiff = 0;
  for (const r of runs) {
    const s = r.start, si = startInfo(s.gate, s.left_offset_m, s.left_yaw_deg);
    minC = Math.min(minC, si.clearance);
    if (Math.abs(s.left_offset_m) > .2 + 1e-12 || Math.abs(s.left_yaw_deg) > 15 + 1e-9 || s.gate < 0 || s.gate > 32) out_of_range++;
    if (Math.abs(si.dpsi) >= 1) head++;
    if (Number.isFinite(s.initial_clearance_m) && s.initial_clearance_m < .5) cdiff = Math.max(cdiff, Math.abs(s.initial_clearance_m - si.clearance));
  }
  out[name] = {n: runs.length, min_start_clearance: +minC.toFixed(5), out_of_range, large_heading: head, max_abs_diff_start_clearance_vs_eval: cdiff};
}
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4/teststarts4.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
