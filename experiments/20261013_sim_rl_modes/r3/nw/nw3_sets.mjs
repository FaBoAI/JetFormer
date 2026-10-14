// no_wall r3 の試験の組・検証の組(2026-10-14、r3 の学習の段)。protocol_no_wall_r3.json sets のとおりに、決まった順で回の一覧を作る。
//   試験(sets.test): NW-N300-R3・NW-GRID-R3・NW-STD・NW-G(範囲の中 68 回 + 範囲の外 12 回は報告)・NW-R200-R3・NW-VX-R3・NW-KX-R3・NW-FC30-R3・NW-LONG-R3
//                    + 報告だけ: NW-OFF-R3(N300・R200・VX をシールドなし)・FAST(最速の方策、N300 の最初の 150 の開始、シールドなし)・LOWMU(μ × 0.4、範囲の外)
//   検証(sets.validation = NW-V-R3): 標準 120・DR-T 一様 120・dr_vertex 120・dr_hard_corner 4 通り × 33 ゲート・NW-STD
// 決まりの文で書き切れていない作り方(ここで決めた。train_no_wall_r3.md に書く):
//   - 開始の流れ: Halton i(基数 2・3・5)を no_wall の記録の開始の分布(modes.json no_wall.recording.start_distribution)に写し、
//     開始のすき間(自前の形 own_geom.mjs、確認 r2 と同じ)< 0.12 m の番号を飛ばす。試験は 26,501〜27,999、検証は 16,001〜
//   - DR-T 一様 / 頂点の引き方は確認 r2 の FD3 / VX と同じ順(舵の速さ・ブレーキ・時定数・加速の上限・μ・遅延・舵のゲイン・前進のゲイン・摩擦円)
//   - NW-KX-R3: mulberry32(0x3e3a0003) で先に 12 通りの隅(舵の速さ・時定数・加速の上限・舵のゲイン・前進のゲインの端と遅延 0/1)、
//     続けて 隅 0 のゲート 0〜32、隅 1 の …の順に開始(左右 U(−0.05, 0.05)・向き U(−5°, 5°)、すき間 ≥ 0.12 m まで引き直す)。種 3,114,000 + 33·隅 + ゲート
//   - 検証: mulberry32(0x3e3a0005) で DR-T 一様 120 → 頂点 120 → 隅 4 通り → 隅の開始(4 × 33)の順。開始の流れは 標準 120 → DR-T 120 → 頂点 120
//   - NW-LONG-R3: 流れの最後の 20 の開始のうち前の 10 が標準・後の 10 が DR-T 一様(mulberry32(0x3e3a0004))
import fs from 'node:fs';
import {startClearance} from './run_nw3r.mjs';
import {haltonStart} from './runtime_v4/rng.mjs';

const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
export const PROTOCOL_PATH = `${REP}/r3/protocol_no_wall_r3.json`;
export const PROTOCOL = JSON.parse(fs.readFileSync(PROTOCOL_PATH, 'utf8'));
const MODES = JSON.parse(fs.readFileSync(`${REP}/modes.json`, 'utf8'));
export const REC_DIST = MODES.modes.find(m => m.key === 'no_wall').recording.start_distribution;
export const DRT = MODES.common.dr_specs['DR-T'];
export const STD = Object.freeze({gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1});
const DR_KEYS = ['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier'];

export function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
// DR-T から一様(確認 r2 の FD3 と同じ引き方: 遅延 0/1 半々・摩擦円 30%)
export function drUniform(rnd) {
  const p = k => DRT[k][0] + (DRT[k][1] - DRT[k][0]) * rnd();
  const physics = {}; for (const k of DR_KEYS) physics[k] = p(k);
  return {physics, delay: rnd() < .5 ? 0 : 1, steerGain: p('steerGain'), speedGain: p('speedGain'), frictionCircle: rnd() < .3, kind: 'uniform'};
}
// dr_vertex(確認 r2 の VX と同じ引き方)
export function drVertex(rnd) {
  const e = k => DRT[k][rnd() < .5 ? 0 : 1];
  const physics = {}; for (const k of DR_KEYS) physics[k] = e(k);
  return {physics, delay: rnd() < .5 ? 0 : 1, steerGain: e('steerGain'), speedGain: e('speedGain'), frictionCircle: rnd() < .3, kind: 'vertex'};
}
// dr_hard_corner: 摩擦円・ブレーキ 0.7・μ 0.85 に固定、ほかの 5 つと遅延 0/1 は端を乱数で
export function drHardCorner(rnd) {
  const e = k => DRT[k][rnd() < .5 ? 0 : 1];
  const steerRate = e('steerRate'), driveTau = e('driveTau'), driveAccelerationCap = e('driveAccelerationCap'), steerGain = e('steerGain'), speedGain = e('speedGain');
  const delay = rnd() < .5 ? 0 : 1;
  return {physics: {steerRate, brakeDeceleration: DRT.brakeDeceleration[0], driveTau, driveAccelerationCap, frictionMultiplier: DRT.frictionMultiplier[0]},
    delay, steerGain, speedGain, frictionCircle: true, kind: 'hard_corner'};
}
// 開始の流れ(Halton。開始のすき間 ≥ 0.12 m)
export function haltonStream(from, to, count) {
  const out = [];
  for (let i = from; out.length < count && i <= to; i++) {
    const st = haltonStart(i, REC_DIST);
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c > 1e-9) || c < REC_DIST.min_initial_clearance_m) continue;
    out.push({...st, initial_clearance_m: c});
  }
  if (out.length < count) throw new Error(`nw3_sets: Halton ${from}〜${to} で開始が ${out.length} 個しか作れません(要 ${count})`);
  return out;
}
function cornerStarts(rnd, nCorners, seedBase) {
  const out = [];
  for (let k = 0; k < nCorners; k++) for (let g = 0; g < 33; g++) {
    let st = null;
    for (let t = 0; t < 1000 && !st; t++) {
      const o = -.05 + .1 * rnd(), y = -5 + 10 * rnd();
      if (startClearance(g, o, y) >= .12) st = {gate: g, left_offset_m: o, left_yaw_deg: y};
    }
    if (!st) throw new Error(`nw3_sets: 隅 ${k} のゲート ${g} で開始が作れません`);
    out.push({corner: k, ...st, seed: seedBase + 33 * k + g});
  }
  return out;
}
// 範囲の中の判定(protocol definitions.in_range)
export function inRange(dr) {
  if (!dr) return true;
  const ok = (k, v) => v >= DRT[k][0] - 1e-12 && v <= DRT[k][1] + 1e-12;
  let r = (dr.delay ?? 0) <= 1;
  for (const [k, v] of Object.entries(dr.physics ?? {})) r = r && ok(k, v);
  if (dr.steerGain != null) r = r && ok('steerGain', dr.steerGain);
  if (dr.speedGain != null) r = r && ok('speedGain', dr.speedGain);
  return r;
}
// ベストラップの r2 の条件の表(rl/eval_modes.mjs gridRows の写し。40 行)
export function gridRows() {
  const G = [{name: '標準', cond: {}}, {name: '遅延 1', cond: {delay: 1}}, {name: '遅延 1 + 前進ゲイン 1.01(ONNX の条件)', cond: {delay: 1, speedGain: 1.01}},
    ...[.8, .9, 1.0, 1.1, 1.3, 1.5, 1.6, 1.8].map(v => ({name: `舵の速さ ${v} rad/s`, cond: {physics: {steerRate: v}}})),
    ...[.6, .7, .8, .9, 1.0, 1.1, 1.5, 2.0, 2.5].map(v => ({name: `ブレーキ ${v} m/s²`, cond: {physics: {brakeDeceleration: v}}})),
    ...[.8, .85, .9, .95, 1.05, 1.1].map(v => ({name: `μ × ${v}`, cond: {physics: {frictionMultiplier: v}}})),
    ...[.25, .3, .5, .6].map(v => ({name: `時定数 ${v} 秒`, cond: {physics: {driveTau: v}}})),
    ...[3, 4].map(v => ({name: `加速の上限 ${v} m/s²`, cond: {physics: {driveAccelerationCap: v}}})),
    {name: '舵のゲイン 0.95', cond: {steerGain: .95}}, {name: '舵のゲイン 1.05', cond: {steerGain: 1.05}},
    {name: '前進ゲイン 0.97', cond: {speedGain: .97}}, {name: '前進ゲイン 1.03', cond: {speedGain: 1.03}},
    {name: '組み合わせ: 舵の速さ 1.0 + ブレーキ 1.0', cond: {physics: {steerRate: 1.0, brakeDeceleration: 1.0}}},
    {name: '組み合わせ: 舵の速さ 1.0 + 遅延 1', cond: {physics: {steerRate: 1.0}, delay: 1}},
    {name: '組み合わせ: μ × 0.9 + ブレーキ 1.0 + 遅延 1', cond: {physics: {frictionMultiplier: .9, brakeDeceleration: 1.0}, delay: 1}},
    {name: '組み合わせ: 舵 0.9・ブレーキ 0.8・μ 0.9・時定数 0.5・遅延 1(悪い側をすべて)', cond: {physics: {steerRate: .9, brakeDeceleration: .8, frictionMultiplier: .9, driveTau: .5}, delay: 1}}];
  return G.map(r => {
    const c = r.cond, dr = {physics: c.physics ?? {}, delay: c.delay ?? 0, steerGain: c.steerGain ?? 1, speedGain: c.speedGain ?? 1, frictionCircle: false};
    return {...r, dr, in_range: inRange(dr)};
  });
}

// 回の一覧。job = {set, id, judged(判定に使う組か), physics('nominal'|'dr'), spec(run_nw3r.mjs の runOne に渡す)}
export function buildTestJobs() {
  const S = PROTOCOL.sets.test, J = [];
  const add = (set, id, spec, extra = {}) => J.push({set, id, spec: {mode: 'no_wall', ...spec}, ...extra});
  const stream = haltonStream(26501, 27999, 850);
  const N300 = stream.slice(0, 300), R200 = stream.slice(300, 500), VX = stream.slice(500, 800), FC30 = stream.slice(800, 830), LONG = stream.slice(830, 850);
  N300.forEach((st, i) => add('NW-N300-R3', `N${i}`, {start: {...st, seed: S['NW-N300-R3'].seed_base + i}, keepPath: i < 150}, {physics: 'nominal'}));
  let n = 0;
  for (let g = 0; g < 33; g++) for (const o of [-.15, -.05, .05, .15]) for (const y of [-12, -4, 4, 12]) {
    const c = startClearance(g, o, y);
    if (!(c >= .12)) continue;
    add('NW-GRID-R3', `GR-g${g}o${o}y${y}`, {start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: S['NW-GRID-R3'].seed_base + n++, initial_clearance_m: c}}, {physics: 'nominal'});
  }
  add('NW-STD', 'single', {start: STD, episodeMode: 'single'}, {physics: 'nominal'});
  add('NW-STD', 'continuous', {start: STD}, {physics: 'nominal'});
  gridRows().forEach((r, i) => {
    for (const em of ['single', 'continuous'])
      add(r.in_range ? 'NW-G' : 'NW-G-OUT', `G${String(i).padStart(2, '0')}-${em}`, {start: STD, episodeMode: em, dr: r.dr}, {physics: i === 0 ? 'nominal' : 'dr', name: r.name});
  });
  const r1 = mulberry32(0x3e3a0001), dR = R200.map(() => drUniform(r1));
  R200.forEach((st, i) => add('NW-R200-R3', `R${i}`, {start: {...st, seed: S['NW-R200-R3'].seed_base + i}, dr: dR[i]}, {physics: 'dr'}));
  const r2 = mulberry32(0x3e3a0002), dV = VX.map(() => drVertex(r2));
  VX.forEach((st, i) => add('NW-VX-R3', `VX${i}`, {start: {...st, seed: S['NW-VX-R3'].seed_base + i}, dr: dV[i]}, {physics: 'dr'}));
  const r3 = mulberry32(0x3e3a0003), corners = Array.from({length: 12}, () => drHardCorner(r3)), cs = cornerStarts(r3, 12, S['NW-KX-R3'].seed_base);
  cs.forEach(st => add('NW-KX-R3', `K${st.corner}-g${st.gate}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: st.seed},
    dr: {...corners[st.corner], corner: st.corner}}, {physics: 'dr', corner: st.corner}));
  FC30.forEach((st, i) => add('NW-FC30-R3', `FC${i}`, {start: {...st, seed: S['NW-FC30-R3'].seed_base + i}, dr: {physics: {}, delay: 0, steerGain: 1, speedGain: 1, frictionCircle: true, kind: 'fc'}}, {physics: 'dr'}));
  const r4 = mulberry32(0x3e3a0004);
  LONG.forEach((st, i) => {
    const dr = i < 10 ? null : drUniform(r4);
    add('NW-LONG-R3', `L${i}`, {start: {...st, seed: S['NW-LONG-R3'].seed_base + i}, timeLimit: 120, ...(dr ? {dr} : {})}, {physics: dr ? 'dr' : 'nominal'});
  });
  // 報告だけ
  for (const j of J.filter(x => ['NW-N300-R3', 'NW-R200-R3', 'NW-VX-R3'].includes(x.set)))
    add('NW-OFF-R3', `${j.set}:${j.id}`, {...j.spec, keepPath: false, shield: false}, {physics: j.physics, of: j.set, report: true});
  N300.slice(0, 150).forEach((st, i) => J.push({set: 'FAST', id: `N${i}`, spec: {mode: 'fastest', start: {...st, seed: S['NW-N300-R3'].seed_base + i}, keepPath: true},
    physics: 'nominal', report: true}));
  add('LOWMU', 'STD', {start: STD, dr: {physics: {frictionMultiplier: .4}, delay: 0}}, {physics: 'dr', report: true});
  N300.slice(0, 9).forEach((st, i) => add('LOWMU', `LM${i}`, {start: {...st, seed: S['NW-N300-R3'].seed_base + i}, dr: {physics: {frictionMultiplier: .4}, delay: 0}}, {physics: 'dr', report: true}));
  for (const j of J) j.judged = !j.report && j.set !== 'NW-G-OUT';
  return J;
}

export function buildValidationJobs() {
  const J = [];
  const add = (set, id, spec, extra = {}) => J.push({set, id, spec: {mode: 'no_wall', ...spec}, judged: true, ...extra});
  const stream = haltonStream(16001, 19999, 360);
  const rnd = mulberry32(0x3e3a0005);
  const dU = Array.from({length: 120}, () => drUniform(rnd)), dV = Array.from({length: 120}, () => drVertex(rnd));
  const corners = Array.from({length: 4}, () => drHardCorner(rnd)), cs = cornerStarts(rnd, 4, 3213000);
  stream.slice(0, 120).forEach((st, i) => add('V-N', `VN${i}`, {start: {...st, seed: 3210000 + i}}, {physics: 'nominal'}));
  stream.slice(120, 240).forEach((st, i) => add('V-R', `VR${i}`, {start: {...st, seed: 3211000 + i}, dr: dU[i]}, {physics: 'dr'}));
  stream.slice(240, 360).forEach((st, i) => add('V-VX', `VV${i}`, {start: {...st, seed: 3212000 + i}, dr: dV[i]}, {physics: 'dr'}));
  cs.forEach(st => add('V-KX', `VK${st.corner}-g${st.gate}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: st.seed},
    dr: {...corners[st.corner], corner: st.corner}}, {physics: 'dr', corner: st.corner}));
  add('V-STD', 'single', {start: STD, episodeMode: 'single'}, {physics: 'nominal'});
  add('V-STD', 'continuous', {start: STD}, {physics: 'nominal'});
  return J;
}

// 下調べの組(dev。判定・選びには使わない。シールドの直し v4 → v4.1 を決めるために、試験・検証と重ならない開始と種で)
//   D-K7: 確認 r2 の隅 K0 + 遅延 1(ABX の K7 と同じ物理・開始)× 33 ゲート / D-KX: dr_hard_corner 8 通り(mulberry32(0x3e3a0006))× 33 ゲート(種 3,400,000 +)
//   D-N: 標準の物理 100 回(Halton 18,001〜、種 3,401,000 +)/ D-R: DR-T 一様 60 回(mulberry32(0x3e3a0007)、続きの Halton、種 3,402,000 +)
export function buildDevJobs() {
  const J = [];
  const add = (set, id, spec, extra = {}) => J.push({set, id, spec: {mode: 'no_wall', ...spec}, judged: false, ...extra});
  const K7 = {physics: {steerRate: 1.8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 3.5, frictionMultiplier: .85}, delay: 1, steerGain: 1.06, speedGain: 1.04,
    frictionCircle: true, kind: 'K7'};
  for (let g = 0; g < 33; g++) add('D-K7', `K7-g${g}`, {start: {gate: g, left_offset_m: 0, left_yaw_deg: 0, seed: 2850000 + 7 * 33 + g}, dr: K7}, {physics: 'dr'});
  const rnd = mulberry32(0x3e3a0006), corners = Array.from({length: 8}, () => drHardCorner(rnd)), cs = cornerStarts(rnd, 8, 3400000);
  cs.forEach(st => add('D-KX', `DK${st.corner}-g${st.gate}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: st.seed},
    dr: {...corners[st.corner], corner: st.corner}}, {physics: 'dr', corner: st.corner}));
  const stream = haltonStream(18001, 19999, 160);
  stream.slice(0, 100).forEach((st, i) => add('D-N', `DN${i}`, {start: {...st, seed: 3401000 + i}}, {physics: 'nominal'}));
  const r7 = mulberry32(0x3e3a0007);
  stream.slice(100, 160).forEach((st, i) => add('D-R', `DR${i}`, {start: {...st, seed: 3402000 + i}, dr: drUniform(r7)}, {physics: 'dr'}));
  return J;
}

export function buildJobs(which) {
  if (which === 'dev') return buildDevJobs();
  if (which === 'test') return buildTestJobs();
  if (which === 'validation') return buildValidationJobs();
  throw new Error('nw3_sets: 知らない組 ' + which);
}
