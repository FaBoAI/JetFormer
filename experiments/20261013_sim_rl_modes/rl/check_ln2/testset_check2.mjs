// 経路いろいろ(line)の確認 r2: 決めた試験の組 LN-P60-R2 が protocol_line_r2.json sets.test の文言どおりに作られているか(2026-10-14、確認の段)。
// 自前の Halton(基数 2・3・5、番号 26,001〜)と自前の開始のすき間(own_geom.mjs)≥ 0.12 m で飛ばし、学習の段の結果の行の開始・種・形・c_v を比べる。
// あわせて、開始の車の向きが R に対してどれだけずれているか(|Δψ| ≥ 1 rad の数)を数える。報告の組 LN-F120-R2 の開始も同じ下限か確かめる。
import fs from 'node:fs';
import {startClearance, startHeadingVsR} from './run_ln2.mjs';
function halton(i, b) { let f = 1, r = 0, n = i; while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); } return r; }
const R2 = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/line/r2';
const ev = JSON.parse(fs.readFileSync(`${R2}/eval_test_r2.json`, 'utf8'));
const rows = ev.tests['LN-P60'];
const plan = Object.entries({const_left: 2, const_right: 2, const_center: 1, inside: 2, outside: 2, late_apex: 2, early_apex: 1, random: 3}).flatMap(([k, w]) => Array(w).fill(k));
const own = [], skipped = [];
for (let i = 26001; own.length < 60; i++) {
  const st = {gate: Math.min(32, Math.floor(halton(i, 2) * 33)), off: -.2 + .4 * halton(i, 3), yaw: -15 + 30 * halton(i, 5), i};
  const c = startClearance(st.gate, st.off, st.yaw);
  if (!(c >= .12)) { skipped.push({i, c: +c.toFixed(4)}); continue; }
  own.push({...st, c, h: startHeadingVsR(st.gate, st.off, st.yaw)});
}
let maxd = 0, maxClrDiff = 0;
const bad = [];
rows.forEach((r, k) => {
  const o = own[k];
  const d = Math.max(Math.abs(o.off - r.start.left_offset_m), Math.abs(o.yaw - r.start.left_yaw_deg));
  maxd = Math.max(maxd, d);
  if (r.start.initial_clearance_m < .5) maxClrDiff = Math.max(maxClrDiff, Math.abs(Math.min(o.c, .5) - r.start.initial_clearance_m));
  const L = r.params.line, fam = plan[k % 15];
  const famOk = fam === 'const_left' ? L.family === 'const' && L.c0 >= .6 && L.c0 <= .9 : fam === 'const_right' ? L.family === 'const' && L.c0 >= -.9 && L.c0 <= -.6
    : fam === 'const_center' ? L.family === 'const' && Math.abs(L.c0) <= .2 : fam === 'random' ? L.family === 'random' : L.family === fam && L.a >= .5 && L.a <= .9;
  if (o.gate !== r.start.gate || o.i !== r.start.halton_index || r.start.seed !== 904500 + k || !famOk || !(r.params.c_v >= .9 && r.params.c_v <= 1) || r.family !== fam) bad.push(r.id);
});
const big = own.map((o, k) => ({id: `P${k}`, gate: o.gate, dpsi: +o.h.dpsi.toFixed(3), s: +o.h.s.toFixed(2)})).filter(q => Math.abs(q.dpsi) >= 1.0);
// 報告の組 LN-F120-R2(学習の段の結果の行)の開始のすき間と向きのずれ
const fr = JSON.parse(fs.readFileSync(`${R2}/eval_fresh_r2.json`, 'utf8'));
const frRows = Object.values(fr.tests).flat();
let frMinClr = Infinity, frBig = 0;
for (const r of frRows) {
  const c = startClearance(r.start.gate, r.start.left_offset_m, r.start.left_yaw_deg);
  frMinClr = Math.min(frMinClr, c);
  if (Math.abs(startHeadingVsR(r.start.gate, r.start.left_offset_m, r.start.left_yaw_deg).dpsi) >= 1.0) frBig++;
}
const out = {n: rows.length, max_abs_start_diff: maxd, max_abs_start_clearance_diff_below_0_5: maxClrDiff, mismatched: bad, skipped_halton: skipped,
  min_start_clearance_own: Math.min(...own.map(o => o.c)), test_starts_abs_dpsi_ge_1rad: big,
  fresh_report_set: {n: frRows.length, min_start_clearance_own: frMinClr, abs_dpsi_ge_1rad: frBig}};
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_line_r2/testset_check.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
