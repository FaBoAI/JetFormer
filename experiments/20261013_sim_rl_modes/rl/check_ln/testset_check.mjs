// 経路いろいろ(line)の確認 r1: 決めた試験の組 LN-P60 が protocol の文言どおりに作られているか(2026-10-14、確認の段)。
// 自前の Halton(基数 2・3・5、番号 20,001〜、開始で重なる番号は飛ばす)と、学習の段の結果の行の開始・種・形・c_v を比べる。
import fs from 'node:fs';
import {startClearance} from './run_ln.mjs';
function halton(i, b) { let f = 1, r = 0, n = i; while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); } return r; }
const ev = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/line/eval_test_retry.json', 'utf8'));
const rows = ev.tests['LN-P60'];
const plan = Object.entries({const_left: 2, const_right: 2, const_center: 1, inside: 2, outside: 2, late_apex: 2, early_apex: 1, random: 3}).flatMap(([k, w]) => Array(w).fill(k));
const own = [], skipped = [];
for (let i = 20001; own.length < 60; i++) {
  const st = {gate: Math.min(32, Math.floor(halton(i, 2) * 33)), off: -.2 + .4 * halton(i, 3), yaw: -15 + 30 * halton(i, 5), i};
  const c = startClearance(st.gate, st.off, st.yaw);
  if (!(c > 1e-9)) { skipped.push({i, c}); continue; }
  own.push({...st, c});
}
let maxd = 0, bad = [];
rows.forEach((r, k) => {
  const o = own[k];
  const d = Math.max(Math.abs(o.off - r.start.left_offset_m), Math.abs(o.yaw - r.start.left_yaw_deg));
  maxd = Math.max(maxd, d);
  const L = r.params.line, fam = plan[k % 15];
  const famOk = fam === 'const_left' ? L.family === 'const' && L.c0 >= .6 && L.c0 <= .9 : fam === 'const_right' ? L.family === 'const' && L.c0 >= -.9 && L.c0 <= -.6
    : fam === 'const_center' ? L.family === 'const' && Math.abs(L.c0) <= .2 : fam === 'random' ? L.family === 'random' : L.family === fam && L.a >= .5 && L.a <= .9;
  if (o.gate !== r.start.gate || o.i !== r.start.halton_index || r.start.seed !== 904000 + k || !famOk || !(r.params.c_v >= .9 && r.params.c_v <= 1) || r.family !== fam) bad.push(r.id);
});
console.log(JSON.stringify({n: rows.length, max_abs_start_diff: maxd, mismatched: bad, skipped_halton: skipped, low_start_clearance: own.filter(o => o.c < .03).map(o => ({i: o.i, c: +o.c.toFixed(4)}))}));
