// 離脱復活 r3 の確認: 走らせる回の一覧(2026-10-14)。種・開始はどこでも使っていないもの(種 7,310,000〜7,369,999、乱数の種 0x7c0e3c03・0x7c0e3c04)。
//   node ck_jobs.mjs <out.json>
import fs from 'node:fs';
import {startClearance} from './ck_run.mjs';
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0};
const J = [], add = (set, id, spec) => J.push({set, id, spec});
// 記録の開始の分布(modes.json recover.recording.start_distribution: ゲート U{0..32}・左右 ±0.2 m・向き ±15°)から一様に。開始のすき間 > 0 だけ(試験と同じ決まり)
function freshStarts(rngSeed, n) {
  const u = mulberry32(rngSeed), out = []; let skipped = 0;
  while (out.length < n) {
    const st = {gate: Math.min(32, Math.floor(u() * 33)), left_offset_m: -.2 + .4 * u(), left_yaw_deg: -15 + 30 * u()};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c > 1e-9)) { skipped++; continue; }
    out.push({...st, start_clearance: c});
  }
  return {out, skipped};
}
// 乱しなしの線(protocol definitions.nominal_line): 8 回、種 990000 + i、ゲート 0・±3 cm・±3°
const un = mulberry32(0x7c0e3c04);
for (const pol of ['r3', 'fastest']) for (let i = 0; i < 8; i++) {
  const o = -.03 + .06 * un(), y = -3 + 6 * un();
  add(`NOM-${pol}`, `N${i}`, {policy: pol, start: {gate: 0, left_offset_m: o, left_yaw_deg: y, seed: 990000 + i}, perturb: false});
}
const fp = freshStarts(0x7c0e3c03, 320);
const FP = fp.out.slice(0, 200), FU = fp.out.slice(200, 320);
FP.forEach((st, i) => add('FP', `P${i}`, {policy: 'r3', start: {...st, seed: 7310000 + i}, perturb: true}));
FU.forEach((st, i) => add('FU', `U${i}`, {policy: 'r3', start: {...st, seed: 7330000 + i}, perturb: false}));
// 角(ゲート 33 × 左右 ±0.2 × 向き ±15°)
let ci = 0, cskip = [];
for (let g = 0; g <= 32; g++) for (const o of [-.2, .2]) for (const y of [-15, 15]) {
  const c = startClearance(g, o, y);
  if (!(c > 1e-9)) { cskip.push({gate: g, o, y, c}); continue; }
  add('FC', `C${g}_${o > 0 ? 'L' : 'R'}${y > 0 ? '+' : '-'}`, {policy: 'r3', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 7320000 + ci}, perturb: true});
  add('FUC', `C${g}_${o > 0 ? 'L' : 'R'}${y > 0 ? '+' : '-'}`, {policy: 'r3', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 7335000 + ci}, perturb: false});
  ci++;
}
// キック(STD の開始、時刻 1.1 + 0.5k と 1.35 + 0.5k、k = 0〜35。protocol 1.0・試験 1.25・検証 1.4 と重ならないフレーム)
let ki = 0;
for (const t0 of [1.1, 1.35]) for (let k = 0; k <= 35; k++) for (const sign of [1, -1]) for (const kind of ['yaw', 'lateral']) {
  const t = t0 + .5 * k;
  add('FK', `${kind}${sign > 0 ? '+' : '-'}${t.toFixed(2)}`, {policy: 'r3', start: {...STD, seed: 7340000 + ki++}, perturb: false,
    kick: {frame: Math.round(t * 30), [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind});
}
// 新しい開始・一様な時刻のキック(報告)
const uk = mulberry32(0x7c0e3c05), kr = freshStarts(0x7c0e3c06, 200).out;
kr.forEach((st, i) => {
  const t = 1 + 18 * uk(), kind = uk() < .5 ? 'yaw' : 'lateral', sign = uk() < .5 ? 1 : -1;
  add('FKR', `R${i}`, {policy: 'r3', start: {...st, seed: 7350000 + i}, perturb: false, kick: {frame: Math.round(t * 30), [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind});
});
// 記録の 60 条件(protocol RC-REC60 = F-REC60。乱しなし)と停止から 1 周
const plan = JSON.parse(fs.readFileSync('/home/jetson/sim/extracted/course-simulator/qa/rc-collection-plan-60.json', 'utf8'));
plan.forEach(c => add('REC60', c.name, {policy: 'r3', start: {gate: c['start-gate'] ?? 0, left_offset_m: c['spawn-offset'] ?? 0, left_yaw_deg: c['spawn-yaw'] ?? 0, seed: c.seed}, perturb: false}));
add('STD1', 'single', {policy: 'r3', start: {...STD, seed: 1}, perturb: false, episodeMode: 'single'});
add('STD1F', 'single', {policy: 'fastest', start: {...STD, seed: 1}, perturb: false, episodeMode: 'single'});
// 比べ: 最速の方策 + 同じ乱し v2・r3 の方策 + 元の乱し v1・強さを「強」に固定(記録のプランで固定できる値の端)
FP.slice(0, 100).forEach((st, i) => {
  add('FF', `P${i}`, {policy: 'fastest', start: {...st, seed: 7310000 + i}, perturb: true});
  add('FV1', `P${i}`, {policy: 'r3', start: {...st, seed: 7310000 + i}, perturb: true, params: {perturb_version: 'v1'}});
  add('FSTRONG', `P${i}`, {policy: 'r3', start: {...st, seed: 7370000 + i}, perturb: true, params: {perturb_class: '強'}});
});
// 止まりの格子(乱しなし、全部の角と組み合わせ): ゲート 33 × 左右 5 × 向き 5
let gi = 0;
for (let g = 0; g <= 32; g++) for (const o of [-.2, -.1, 0, .1, .2]) for (const y of [-15, -7.5, 0, 7.5, 15]) {
  const c = startClearance(g, o, y);
  add('GRID', `G${g}_${o}_${y}`, {policy: 'r3', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 7360000 + gi++, start_clearance: c}, perturb: false});
}
fs.writeFileSync(process.argv[2], JSON.stringify(J));
const cnt = {}; for (const j of J) cnt[j.set] = (cnt[j.set] ?? 0) + 1;
console.log(JSON.stringify(cnt), 'FP skipped', fp.skipped, 'corner skipped', JSON.stringify(cskip));
