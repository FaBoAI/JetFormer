// 経路いろいろ(line)の確認 r1 の回の一覧(2026-10-14、確認の段)。
//   REPLAY: 決めた試験の組(LN-P60 60 回 + LN-FAM6 6 回)。開始・種・経路の形・c_v は学習の段の結果の行(train/line/eval_test_retry.json)から
//           「データとして」読む(同じ条件でないと比べられないため)。走らせ方・すき間・射影・e*・判定は自前
//   FRESH: 新しい開始(mulberry32 の一様乱数。Halton を使わない)。記録の開始の分布(ゲート U{0..32}・左右 ±0.2 m・向き ±15°・開始で重ならない)、
//          形は family_plan_15 の順に回し、強さ・c0・random の種・c_v U(0.9, 1.0) は自前の乱数。種 2,100,000〜(試験 904xxx・検証 804xxx・記録 704xxx と重ならない)
//   FRESH_FAST: FRESH の最初の 75 回と同じ開始・同じ種で最速の方策(bundle/fastest)
//   FAM6V: LN-FAM6 の 6 つの形を、標準の開始の小さなずれ(±3 cm・±3°)と c_v 0.9 で
//   FB_MIX: 代わりの方法(ベストラップの 7 本を LN-P60 の回 i に i mod 7)を同じ LN-P60 の条件で
import fs from 'node:fs';
import {startClearance} from './run_ln.mjs';

const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const PLAN15 = Object.entries({const_left: 2, const_right: 2, const_center: 1, inside: 2, outside: 2, late_apex: 2, early_apex: 1, random: 3}).flatMap(([k, w]) => Array(w).fill(k));
function lineFor(kind, rnd) {
  const u = rnd();
  if (kind === 'const_left') return {family: 'const', c0: .6 + .3 * u};
  if (kind === 'const_right') return {family: 'const', c0: -.9 + .3 * u};
  if (kind === 'const_center') return {family: 'const', c0: -.2 + .4 * u};
  if (kind === 'random') return {family: 'random', seed: Math.floor(u * 4294967296) >>> 0};
  return {family: kind, a: .5 + .4 * u};
}
export function freshStarts(count, seed) {
  const rnd = mulberry32(seed), out = [];
  while (out.length < count) {
    const st = {gate: Math.floor(rnd() * 33), left_offset_m: -.2 + .4 * rnd(), left_yaw_deg: -15 + 30 * rnd()};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c > 1e-9)) continue;
    out.push({...st, start_clearance: c});
  }
  return out;
}
const FAM6 = [['c-0.8', {family: 'const', c0: -.8}], ['c0', {family: 'const', c0: 0}], ['c+0.8', {family: 'const', c0: .8}],
  ['inside', {family: 'inside', a: .8}], ['outside', {family: 'outside', a: .8}], ['late_apex', {family: 'late_apex', a: .8}]];
const FB = ['main-s1/policies/step_010010624', 'main-s1/policies/step_006012928', 'main-s2/policies/step_010010624', 'r2-dr-s11/policies/step_001015808',
  'r3-s21/policies/step_001015808', 'r3-s22/policies/step_004014080', 'main-s0/policies/step_006012928'].map(p => `/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/${p}.json`);

export function buildSet(name) {
  const J = [];
  if (name === 'REPLAY' || name === 'FB_MIX') {
    const ev = JSON.parse(fs.readFileSync(`${REP}/train/line/eval_test_retry.json`, 'utf8'));
    for (const r of ev.tests['LN-P60']) {
      const st = {gate: r.start.gate, left_offset_m: r.start.left_offset_m, left_yaw_deg: r.start.left_yaw_deg, seed: r.start.seed};
      if (name === 'REPLAY') J.push({id: r.id, test: 'LN-P60', spec: {policy: 'line', start: st, params: {line: r.params.line, c_v: r.params.c_v}, family: r.family}});
      else { const k = +r.id.slice(1) % 7; J.push({id: r.id, test: 'LN-P60', spec: {policy: FB[k], mode: 'fastest', start: st, params: {}, family: r.family, keepFrames: false}}); }
    }
    if (name === 'REPLAY') for (const r of ev.tests['LN-FAM6'])
      J.push({id: r.id, test: 'LN-FAM6', spec: {policy: 'line', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: r.start.seed ?? 1}, params: {line: r.params.line, c_v: r.params.c_v}, family: r.id}});
  } else if (name === 'FRESH' || name === 'FRESH_FAST') {
    const fs_ = freshStarts(240, 0x1e14a7e5), rnd = mulberry32(0x2a11e5);
    fs_.forEach((st, i) => {
      const kind = PLAN15[i % 15], line = lineFor(kind, rnd), cv = +(0.9 + 0.1 * rnd()).toFixed(4);
      const start = {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: 2100000 + i};
      if (name === 'FRESH') J.push({id: `F${i}`, test: 'FRESH', spec: {policy: 'line', start, params: {line, c_v: cv}, family: kind, keepPath: i < 30}});
      else if (i < 75) J.push({id: `F${i}`, test: 'FRESH_FAST', spec: {policy: 'fastest', mode: 'fastest', start, params: {}, family: kind, keepPath: i < 30}});
    });
  } else if (name === 'FAM6V') {
    let k = 0;
    for (const [off, yaw] of [[.03, 0], [-.03, 0], [0, 3], [0, -3], [.03, -3], [-.03, 3]])
      for (const [nm, line] of FAM6) J.push({id: `V${k++}-${nm}`, test: 'FAM6V', spec: {policy: 'line', start: {gate: 0, left_offset_m: off, left_yaw_deg: yaw, seed: 2200000 + k}, params: {line, c_v: 1.0}, family: nm}});
    for (const [nm, line] of FAM6) J.push({id: `C09-${nm}`, test: 'FAM6_CV09', spec: {policy: 'line', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, params: {line, c_v: .9}, family: nm}});
    J.push({id: 'FAST-STD', test: 'FAST_STD', spec: {policy: 'fastest', mode: 'fastest', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, params: {}, family: 'fastest'}});
  } else if (name === 'SMOKE') {
    J.push({id: 'S0', test: 'SMOKE', spec: {policy: 'line', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, params: {line: {family: 'const', c0: .8}, c_v: 1.0}, family: 'c+0.8'}});
  } else throw new Error('知らない組: ' + name);
  return J;
}
