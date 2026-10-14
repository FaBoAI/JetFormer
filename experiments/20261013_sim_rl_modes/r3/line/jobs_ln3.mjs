// r3 の下調べ: 確認 line r2 の jobs_ln2.mjs(5215af26…)の写し(読み込む run を run_ln3.mjs にしただけ)
// 経路いろいろ(line)の確認 r2 の回の一覧(2026-10-14、確認の段)。
//   REPLAY: 決めた試験の組(LN-P60-R2 60 回 + LN-FAM6 6 回)。開始・種・経路の形・c_v は学習の段の結果の行(train/line/r2/eval_test_r2.json)から
//           「データとして」読む(同じ条件でないと比べられないため)。走らせ方・すき間・射影・e*・判定は自前。作り方が protocol どおりかは testset_check2.mjs
//   FRESH: 新しい開始 240 回。protocol_line_r2 start_distribution(ゲート U{0..32}・左右 U(−0.2, 0.2) m・向き U(−15, 15)°・開始のすき間 ≥ 0.12 m)を
//          mulberry32(0x5eed2c42) の一様乱数で(Halton を使わない)。形は family_plan_15 の順、強さ・c0・random の種・c_v U(0.9, 1.0) は mulberry32(0x71e2c0de)。
//          種 2,500,000 + i(試験 904500〜・検証 804500〜・報告 2,300,000〜・r1 の確認 2,100,000〜/2,200,000〜 と重ならない)
//   FRESH_FAST: FRESH の最初の 60 回と同じ開始・同じ種で最速の方策(bundle/fastest)
//   HEAD: 同じ分布から、開始の車の向きが R に対して |Δψ| ≥ 1.0 rad(57°)のものだけを 60 回(P45 の形がどれくらい落ちるか)。
//         mulberry32(0x4ead0001)、種 2,600,000 + i。引いた数と通った数も返す(分布の中の割合)
//   FAM6V: LN-FAM6 の 6 つの形を、標準の開始の小さなずれ(±3 cm・±3°)6 通りと c_v 0.9 で(L4・L6 の頑丈さ)
import fs from 'node:fs';
import {startClearance, startHeadingVsR} from './run_ln3.mjs';

const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
export function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export const PLAN15 = Object.entries({const_left: 2, const_right: 2, const_center: 1, inside: 2, outside: 2, late_apex: 2, early_apex: 1, random: 3}).flatMap(([k, w]) => Array(w).fill(k));
function lineFor(kind, rnd) {
  const u = rnd();
  if (kind === 'const_left') return {family: 'const', c0: .6 + .3 * u};
  if (kind === 'const_right') return {family: 'const', c0: -.9 + .3 * u};
  if (kind === 'const_center') return {family: 'const', c0: -.2 + .4 * u};
  if (kind === 'random') return {family: 'random', seed: Math.floor(u * 4294967296) >>> 0};
  return {family: kind, a: .5 + .4 * u};
}
export function drawStarts(count, seed, {minClr = .12, minAbsDpsi = 0} = {}) {
  const rnd = mulberry32(seed), out = [];
  let drawn = 0, clrOk = 0;
  while (out.length < count) {
    const st = {gate: Math.floor(rnd() * 33), left_offset_m: -.2 + .4 * rnd(), left_yaw_deg: -15 + 30 * rnd()};
    drawn++;
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c >= minClr)) continue;
    clrOk++;
    const h = startHeadingVsR(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (Math.abs(h.dpsi) < minAbsDpsi) continue;
    out.push({...st, start_clearance: c, dpsi: h.dpsi});
  }
  return {starts: out, drawn, clr_ok: clrOk};
}
const FAM6 = [['c-0.8', {family: 'const', c0: -.8}], ['c0', {family: 'const', c0: 0}], ['c+0.8', {family: 'const', c0: .8}],
  ['inside', {family: 'inside', a: .8}], ['outside', {family: 'outside', a: .8}], ['late_apex', {family: 'late_apex', a: .8}]];

export function buildSet(name) {
  const J = [];
  if (name === 'REPLAY') {
    const ev = JSON.parse(fs.readFileSync(`${REP}/train/line/r2/eval_test_r2.json`, 'utf8'));
    for (const r of ev.tests['LN-P60']) {
      const st = {gate: r.start.gate, left_offset_m: r.start.left_offset_m, left_yaw_deg: r.start.left_yaw_deg, seed: r.start.seed};
      J.push({id: r.id, test: 'LN-P60', spec: {policy: 'line', start: st, params: {line: r.params.line, c_v: r.params.c_v}, family: r.family, keepPath: r.id === 'P45' || r.id === 'P43'}});
    }
    for (const r of ev.tests['LN-FAM6'])
      J.push({id: r.id, test: 'LN-FAM6', spec: {policy: 'line', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: r.start.seed ?? 1}, params: {line: r.params.line, c_v: r.params.c_v}, family: r.id, keepPath: true}});
  } else if (name === 'FRESH' || name === 'FRESH_FAST') {
    const {starts} = drawStarts(240, 0x5eed2c42), rnd = mulberry32(0x71e2c0de);
    starts.forEach((st, i) => {
      const kind = PLAN15[i % 15], line = lineFor(kind, rnd), cv = +(0.9 + 0.1 * rnd()).toFixed(4);
      const start = {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: 2500000 + i};
      if (name === 'FRESH') J.push({id: `F${i}`, test: 'FRESH', spec: {policy: 'line', start, params: {line, c_v: cv}, family: kind}});
      else if (i < 60) J.push({id: `F${i}`, test: 'FRESH_FAST', spec: {policy: 'fastest', mode: 'fastest', start, params: {}, family: kind}});
    });
  } else if (name === 'HEAD') {
    const {starts} = drawStarts(60, 0x4ead0001, {minAbsDpsi: 1.0}), rnd = mulberry32(0x4ead0002);
    starts.forEach((st, i) => {
      const kind = PLAN15[i % 15], line = lineFor(kind, rnd), cv = +(0.9 + 0.1 * rnd()).toFixed(4);
      J.push({id: `H${i}`, test: 'HEAD', spec: {policy: 'line', start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: 2600000 + i}, params: {line, c_v: cv}, family: kind}});
    });
  } else if (name === 'FB_MIX') {
    // 代わりの方法(protocol acceptance.fallbacks.line): ベストラップの 7 本を LN-P60-R2 の回 i に i mod 7(読むだけ)
    const FB = ['main-s1/policies/step_010010624', 'main-s1/policies/step_006012928', 'main-s2/policies/step_010010624', 'r2-dr-s11/policies/step_001015808',
      'r3-s21/policies/step_001015808', 'r3-s22/policies/step_004014080', 'main-s0/policies/step_006012928'].map(q => `/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/${q}.json`);
    const ev = JSON.parse(fs.readFileSync(`${REP}/train/line/r2/eval_test_r2.json`, 'utf8'));
    for (const r of ev.tests['LN-P60']) {
      const st = {gate: r.start.gate, left_offset_m: r.start.left_offset_m, left_yaw_deg: r.start.left_yaw_deg, seed: r.start.seed};
      J.push({id: r.id, test: 'LN-P60', spec: {policy: FB[+r.id.slice(1) % 7], mode: 'fastest', start: st, params: {}, family: r.family, keepFrames: false}});
    }
  } else if (name === 'EXT') {
    // 範囲の端: 形の強さを上限(c0 ±0.9・a 0.9)、c_v を 0.9 か 1.0 にして、新しい開始(mulberry32(0xe7e70001)、種 2,800,000 + i)で 60 回
    const {starts} = drawStarts(60, 0xe7e70001), rnd = mulberry32(0xe7e70002);
    const EXT = [{family: 'const', c0: .9}, {family: 'const', c0: -.9}, {family: 'inside', a: .9}, {family: 'outside', a: .9}, {family: 'late_apex', a: .9}, {family: 'early_apex', a: .9}];
    starts.forEach((st, i) => {
      const line = i % 10 < 6 ? EXT[i % 10] : {family: 'random', seed: Math.floor(rnd() * 4294967296) >>> 0}, cv = i % 2 ? 1.0 : 0.9;
      const kind = line.family === 'const' ? (line.c0 > 0 ? 'const_left' : 'const_right') : line.family;
      J.push({id: `E${i}`, test: 'EXT', spec: {policy: 'line', start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: 2800000 + i}, params: {line, c_v: cv}, family: kind}});
    });
  } else if (name === 'FAM6V') {
    let k = 0;
    for (const [off, yaw] of [[.03, 0], [-.03, 0], [0, 3], [0, -3], [.03, -3], [-.03, 3]])
      for (const [nm, line] of FAM6) J.push({id: `V${k++}-${nm}`, test: 'FAM6V', spec: {policy: 'line', start: {gate: 0, left_offset_m: off, left_yaw_deg: yaw, seed: 2700000 + k}, params: {line, c_v: 1.0}, family: nm}});
    for (const [nm, line] of FAM6) J.push({id: `C09-${nm}`, test: 'FAM6_CV09', spec: {policy: 'line', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, params: {line, c_v: .9}, family: nm}});
    J.push({id: 'FAST-STD', test: 'FAST_STD', spec: {policy: 'fastest', mode: 'fastest', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, params: {}, family: 'fastest'}});
  } else if (name === 'SMOKE') {
    J.push({id: 'S0', test: 'SMOKE', spec: {policy: 'line', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, params: {line: {family: 'const', c0: .8}, c_v: 1.0}, family: 'c+0.8'}});
    J.push({id: 'S1', test: 'SMOKE', spec: {policy: 'line', start: {gate: 16, left_offset_m: -.1194, left_yaw_deg: 14.2, seed: 5}, params: {line: {family: 'random', seed: 12345}, c_v: .95}, family: 'random'}});
  } else throw new Error('知らない組: ' + name);
  return J;
}
