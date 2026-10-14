// 経路いろいろ(line)の確認 r2: 目標の線 v2 の幅の表と線そのものの確かめ(2026-10-14、確認の段)。走らせない。
//   1. 自前の幅の表(own_line_v2.mjs、protocol の文言から)と、学習の段の line_v2_table.json(sha256 39e1b1c4…)の差
//   2. 自前の v2 の線(FAM6 + 強さ 0.9 の 6 本 + 範囲の端)で e*·κ_R の最大、線の曲率半径の最小、線の上に車体を置いたときの静的なすき間
//   3. ランタイムの v2(web/rl/line_profiles.mjs)と自前の v2 の e* の差(つなぎなし・つなぎあり)
import fs from 'node:fs';
import {buildOwnTable, OWN_TABLE, createOwnV2, DS, N, L} from './own_line_v2.mjs';
import {rAt} from '../check_ln/own_line.mjs';
import {buildObstacles, ownClearance} from '../check_ln/own_geom.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const TT = JSON.parse(fs.readFileSync(`${EXP}/rl/runtime/line_v2_table.json`, 'utf8'));
const LP = await import(`${EXP}/web/rl/line_profiles.mjs`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
const OB = buildObstacles(await loadCourseWorld());
const out = {};
const cmp = (a, b) => { let m = 0, at = -1; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) { m = d; at = i; } } return {max_abs: +m.toFixed(6), at_s: +(at * DS).toFixed(3)}; };
out.table_vs_trainer = {cap2: {kappa: cmp(OWN_TABLE.kappa, TT.kappa), hi: cmp(OWN_TABLE.hi, TT.hi), lo: cmp(OWN_TABLE.lo, TT.lo)}};
const T1 = buildOwnTable({cap: 1.0}), T10 = buildOwnTable({cap: 10});
out.table_vs_trainer.cap1 = {hi: cmp(T1.hi, TT.hi), lo: cmp(T1.lo, TT.lo)};
out.table_vs_trainer.cap10 = {hi: cmp(T10.hi, TT.hi), lo: cmp(T10.lo, TT.lo)};
out.table_lo_gt_hi = OWN_TABLE.hi.filter((h, i) => OWN_TABLE.lo[i] > h).length;
// 2. 線そのもの
const fams = [['c-0.8', {family: 'const', c0: -.8}], ['c0', {family: 'const', c0: 0}], ['c+0.8', {family: 'const', c0: .8}], ['inside', {family: 'inside', a: .8}],
  ['outside', {family: 'outside', a: .8}], ['late_apex', {family: 'late_apex', a: .8}], ['c-0.9', {family: 'const', c0: -.9}], ['c+0.9', {family: 'const', c0: .9}],
  ['inside0.9', {family: 'inside', a: .9}], ['outside0.9', {family: 'outside', a: .9}], ['late0.9', {family: 'late_apex', a: .9}], ['early0.9', {family: 'early_apex', a: .9}]];
const lines = {};
out.lines = {};
for (const [nm, p] of fams) {
  const own = createOwnV2(p, 0, null), rt = LP.createLineProfile(p, 0, {version: 'v2'});
  let dmax = 0, ekMax = 0, ekAt = 0, rMin = Infinity, rAt_ = 0, clrMin = Infinity, clrAt = 0;
  const pts = [];
  for (let k = 0; k < Math.round(L / .01); k++) {
    const s = k * .01, e = own.eStar(s), d = Math.abs(e - rt.eStar(s));
    if (d > dmax) dmax = d;
    const kap = OWN_TABLE.kappa[Math.round(s / DS) % N];
    if (e * kap > ekMax) { ekMax = e * kap; ekAt = s; }
    const r = rAt(s); pts.push({s, e, x: r.x + e * r.nx, z: r.z + e * r.nz});
  }
  for (let i = 0; i < pts.length; i++) {
    const a = pts[(i - 4 + pts.length) % pts.length], b = pts[i], c = pts[(i + 4) % pts.length];
    const ax = b.x - a.x, az = b.z - a.z, bx = c.x - b.x, bz = c.z - b.z, cx = c.x - a.x, cz = c.z - a.z;
    const den = Math.hypot(ax, az) * Math.hypot(bx, bz) * Math.hypot(cx, cz), k = den > 0 ? 2 * Math.abs(ax * bz - az * bx) / den : 0;
    const R = k > 0 ? 1 / k : Infinity;
    if (R < rMin) { rMin = R; rAt_ = b.s; }
    const yaw = Math.atan2(-(c.z - a.z), c.x - a.x), cl = ownClearance(OB, b.x, b.z, yaw);
    if (cl < clrMin) { clrMin = cl; clrAt = b.s; }
  }
  lines[nm] = pts;
  out.lines[nm] = {own_vs_runtime_max_abs_m: +dmax.toExponential(3), max_e_kappa: +ekMax.toFixed(3), at_s: +ekAt.toFixed(2), min_path_radius_m: +rMin.toFixed(3), r_at_s: +rAt_.toFixed(2),
    static_clearance_min_m: +clrMin.toFixed(4), clr_at_s: +clrAt.toFixed(2)};
}
// R そのものの上(e = 0、R の向き)の静的なすき間が小さい所(P45 の s 11.9〜12.1 m)
const rClr = [];
for (let k = 0; k < Math.round(L / .02); k++) {
  const s = k * .02, a = rAt(s - .05), b = rAt(s + .05), r = rAt(s), yaw = Math.atan2(-(b.z - a.z), b.x - a.x);
  rClr.push({s: +s.toFixed(2), c: ownClearance(OB, r.x, r.z, yaw)});
}
out.R_static_clearance_lt_10cm = rClr.filter(q => q.c < .10).map(q => [q.s, +q.c.toFixed(4)]);
// 目標の線どうしの L4(目標の上を正確に走ったとき)
const ey = nm => { const a = []; for (let i = 0; i < N; i++) a.push(createOwnV2(fams.find(f => f[0] === nm)[1], 0, null).eStar(i * DS)); return a; };
const E = Object.fromEntries(['c-0.8', 'c0', 'c+0.8', 'inside', 'outside', 'late_apex'].map(k => [k, ey(k)]));
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const ids = Object.keys(E), pairs = [];
for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
  const d = E[ids[i]].map((v, k) => Math.abs(v - E[ids[j]][k]));
  let best = 0; for (let k = 0; k < N; k++) { let s = 0; for (let q = 0; q < 20; q++) s += d[(k + q) % N]; best = Math.max(best, s / 20); }
  pairs.push([ids[i], ids[j], +best.toFixed(4)]);
}
out.target_L4 = {left_right: +mean(E['c+0.8'].map((v, k) => Math.abs(v - E['c-0.8'][k]))).toFixed(4), inside_outside: +mean(E.inside.map((v, k) => Math.abs(v - E.outside[k]))).toFixed(4),
  min_pair: pairs.reduce((a, b) => a[2] < b[2] ? a : b)};
// 3. つなぎ: 開始の e0 と s0 をいろいろ変えてランタイムと比べる
let bmax = 0;
for (const [nm, p] of fams.slice(0, 6)) for (const s0 of [0, 3.3, 9.58, 12.0, 20.1]) for (const e0 of [-.4, -.1, 0, .25, .5]) {
  const own = createOwnV2(p, s0, e0), rt = LP.createLineProfile(p, s0, {version: 'v2'}); rt.anchor(e0);
  for (let s = s0 - 1; s < s0 + 4; s += .013) bmax = Math.max(bmax, Math.abs(own.eStar(s) - rt.eStar(s)));
}
out.blend_own_vs_runtime_max_abs_m = bmax;
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_line_r2/table_audit.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1).slice(0, 6000));
