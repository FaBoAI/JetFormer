// 角の近くの調べ(2026-10-14、確認の段)。CORNER で C129(ゲート 16・左右 −0.2・向き +15°、2.13 cm)と C137(ゲート 17・同じ角、止まり)が
// 崩れたので、その角の近くの開始の割合を測る:
//   NB16・NB17: ゲート 16・17、左右 U(−0.2, −0.1)・向き U(7.5, 15)、開始のすき間 ≥ 0.12 m、60 回ずつ。形は family_plan_15 の順(強さは範囲の一様)、c_v U(0.9, 1.0)
//   NBC: ちょうど角(−0.2, +15°)のゲート 16・17 × 形 8 つ(強さ 0.75)× c_v {0.9, 1.0}
//   BOX16・BOX17(追加): 左右 U(−0.2, −0.16)・向き U(12, 15)(各ゲートの開始の面積の 1%)、80 回ずつ、形 family_plan_15・c_v U(0.9, 1.0)
//   node nb_probe.mjs <variant cand|nosh> <w> <W> [box]
import fs from 'node:fs';
import {runOne, startInfo} from './ck_run.mjs';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/runs';
const [variant, w, W] = [process.argv[2], +process.argv[3], +process.argv[4]];
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const PLAN15 = [['const_left', 2], ['const_right', 2], ['const_center', 1], ['inside', 2], ['outside', 2], ['late_apex', 2], ['early_apex', 1], ['random', 3]].flatMap(([k, n]) => Array(n).fill(k));
function shapeFor(kind, rnd) {
  const u = rnd();
  if (kind === 'const_left') return {family: 'const', c0: .6 + .3 * u};
  if (kind === 'const_right') return {family: 'const', c0: -.9 + .3 * u};
  if (kind === 'const_center') return {family: 'const', c0: -.2 + .4 * u};
  if (kind === 'random') return {family: 'random', seed: Math.floor(u * 4294967296) >>> 0};
  return {family: kind, a: .5 + .4 * u};
}
const jobs = [];
const BOX = process.argv[5] === 'box';
if (BOX) for (const [g, sSeed, base] of [[16, 0x6c6e3312, 4724000], [17, 0x6c6e3313, 4725000]]) {
  const rs = mulberry32(sSeed), rsh = mulberry32(sSeed + 0x100);
  let i = 0;
  while (i < 80) {
    const off = -.2 + .04 * rs(), yaw = 12 + 3 * rs(), I = startInfo(g, off, yaw);
    if (!(I.clearance >= .12)) continue;
    const kind = PLAN15[i % 15], line = shapeFor(kind, rsh), cv = +(0.9 + 0.1 * rsh()).toFixed(4);
    jobs.push({set: `BOX${g}`, id: `BOX${g}-${i}`, start: {gate: g, left_offset_m: off, left_yaw_deg: yaw, seed: base + i}, params: {line, c_v: cv}, family: kind});
    i++;
  }
}
if (!BOX) for (const [g, sSeed, base] of [[16, 0x6c6e3310, 4720000], [17, 0x6c6e3311, 4721000]]) {
  const rs = mulberry32(sSeed), rsh = mulberry32(sSeed + 0x100);
  let i = 0;
  while (i < 60) {
    const off = -.2 + .1 * rs(), yaw = 7.5 + 7.5 * rs(), I = startInfo(g, off, yaw);
    if (!(I.clearance >= .12)) continue;
    const kind = PLAN15[i % 15], line = shapeFor(kind, rsh), cv = +(0.9 + 0.1 * rsh()).toFixed(4);
    jobs.push({set: `NB${g}`, id: `NB${g}-${i}`, start: {gate: g, left_offset_m: off, left_yaw_deg: yaw, seed: base + i}, params: {line, c_v: cv}, family: kind});
    i++;
  }
}
if (!BOX) {
  const SH = [['const_left', {family: 'const', c0: .75}], ['const_right', {family: 'const', c0: -.75}], ['const_center', {family: 'const', c0: 0}], ['inside', {family: 'inside', a: .75}],
    ['outside', {family: 'outside', a: .75}], ['late_apex', {family: 'late_apex', a: .75}], ['early_apex', {family: 'early_apex', a: .75}], ['random', {family: 'random', seed: 777}]];
  let i = 0;
  for (const g of [16, 17]) for (const [kind, line] of SH) for (const cv of [.9, 1.0]) { jobs.push({set: 'NBC', id: `NBC-${i}`, start: {gate: g, left_offset_m: -.2, left_yaw_deg: 15, seed: 4722000 + i}, params: {line, c_v: cv}, family: kind}); i++; }
}
const V = {cand: {policy: 'line', rt: 'r3', shield: true}, nosh: {policy: 'line', rt: 'r3', shield: false}}[variant];
const file = `${OUT}/${BOX ? 'box' : 'nb'}_${variant}_${w}.jsonl`;
for (const j of jobs.filter((_, k) => k % W === w)) {
  const r = runOne({...j, ...V, keepFrames: true});
  r.variant = variant;
  if (!(r.collisions > 0 || r.stall || r.min_clearance < .05 || !r.valid_r3)) delete r.fr;
  fs.appendFileSync(file, JSON.stringify(r) + '\n');
}
console.log('done', variant, w);
