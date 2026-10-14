// no_wall r4(2026-10-14、r4 の段): r3/nw/nw3_sets.mjs の写し(元の sha256 は r3/no_wall_r4/sources_sha256.txt)。作り方の小道具(DR の引き方・Halton の開始・
//   範囲の中の判定・条件の表)はそのまま。回の一覧は protocol_no_wall_r4.json の組(下の buildTest4Jobs・buildValidation4Jobs)と、シールド v5 を決めるための
//   下調べの組(buildDev4Jobs。試験・検証と重ならない乱数)に替えた。以下は r3 の説明。
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
import {startClearance} from './run_nw4r.mjs';
import {haltonStart} from './runtime_v5/rng.mjs';

const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
export const PROTOCOL_PATH = `${REP}/r3/protocol_no_wall_r4.json`;
export const PROTOCOL = fs.existsSync(PROTOCOL_PATH) ? JSON.parse(fs.readFileSync(PROTOCOL_PATH, 'utf8')) : null;
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

// 記録の開始の分布(ゲート U{0..32}・左右 U(−0.2, 0.2)・向き U(−15°, 15°))から一様乱数で引き、開始のすき間が [minC, maxC] のものだけ(引き直す)
export function uniformStart(rnd, minC = .12, maxC = Infinity) {
  for (let t = 0; t < 100000; t++) {
    const g = Math.floor(rnd() * 33), o = -.2 + .4 * rnd(), y = -15 + 30 * rnd();
    const c = startClearance(g, o, y);
    if (c >= minC && c <= maxC) return {gate: g, left_offset_m: o, left_yaw_deg: y, initial_clearance_m: c};
  }
  throw new Error('nw4_sets: 開始が作れません');
}
// dr_hard_corner の全部の組(摩擦円・ブレーキ 0.7・μ 0.85 に、舵の速さ・時定数・加速の上限・舵のゲイン・前進のゲインの端 × 遅延 0/1 = 64 通り)。b の順
export function hardCornerAll(b) {
  const e = (k, bit) => DRT[k][(b >> bit) & 1];
  return {physics: {steerRate: e('steerRate', 0), brakeDeceleration: DRT.brakeDeceleration[0], driveTau: e('driveTau', 1), driveAccelerationCap: e('driveAccelerationCap', 2),
    frictionMultiplier: DRT.frictionMultiplier[0]}, steerGain: e('steerGain', 3), speedGain: e('speedGain', 4), delay: (b >> 5) & 1, frictionCircle: true, kind: 'hard_corner', combo: b};
}
const add0 = J => (set, id, spec, extra = {}) => J.push({set, id, spec: {mode: 'no_wall', ...spec}, ...extra});

// 下調べの組 dev4(シールド v5 の値を決めるためだけ。判定・選びには使わない)。乱数 mulberry32(0x4e4a0d01)、種 6,900,000 +
//   D4-FS 標準 300(記録の分布から一様)・D4-LOWC 標準 300(開始のすき間 12〜15 cm の層)・D4-DRU DR-T 一様 100・D4-VX 頂点 100・D4-HC 止まりにくい隅の全部の組 64 × 開始 1
export function buildDev4Jobs() {
  const J = [], add = add0(J), rnd = mulberry32(0x4e4a0d01);
  let seed = 6900000;
  for (let i = 0; i < 300; i++) add('D4-FS', `F${i}`, {start: {...uniformStart(rnd), seed: seed++}}, {physics: 'nominal'});
  for (let i = 0; i < 300; i++) add('D4-LOWC', `L${i}`, {start: {...uniformStart(rnd, .12, .15), seed: seed++}}, {physics: 'nominal'});
  for (let i = 0; i < 100; i++) add('D4-DRU', `R${i}`, {start: {...uniformStart(rnd), seed: seed++}, dr: drUniform(rnd)}, {physics: 'dr'});
  for (let i = 0; i < 100; i++) add('D4-VX', `V${i}`, {start: {...uniformStart(rnd), seed: seed++}, dr: drVertex(rnd)}, {physics: 'dr'});
  for (let b = 0; b < 64; b++) add('D4-HC', `H${b}`, {start: {...uniformStart(rnd), seed: seed++}, dr: hardCornerAll(b)}, {physics: 'dr', corner: b});
  for (const j of J) j.judged = false;
  return J;
}

// 下調べの組 dev4e(分布の端。シールド v5・学び直しの要否を決めるためだけ): 33 ゲート × 左右 {±0.2, ±0.15} m × 向き {±15, ±12}°(開始のすき間 ≥ 0.12 m)、種 6,950,000 +
export function buildDev4EdgeJobs() {
  const J = [], add = add0(J);
  let n = 0;
  for (let g = 0; g < 33; g++) for (const o of [-.2, -.15, .15, .2]) for (const y of [-15, -12, 12, 15]) {
    const c = startClearance(g, o, y);
    if (!(c >= .12)) continue;
    add('D4-EDGE', `DE-g${g}o${o}y${y}`, {start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 6950000 + n++, initial_clearance_m: c}}, {physics: 'nominal'});
  }
  for (const j of J) j.judged = false;
  return J;
}

export function buildJobs(which) {
  if (which === 'dev4') return buildDev4Jobs();
  if (which === 'dev4e') return buildDev4EdgeJobs();
  if (which === 'test') return buildTest4Jobs();
  if (which === 'validation') return buildValidation4Jobs();
  throw new Error('nw4_sets: 知らない組 ' + which);
}

// ---- protocol_no_wall_r4.json sets(r4 の段。学習・選び・試験の前に固定)----
// 開始の流れ: Halton(protocol seeds.halton と同じ基数 2・3・5 の写し方)を no_wall の記録の開始の分布に写し、開始のすき間 < 0.12 m の番号を飛ばす。
//   試験は 60,001〜64,999、検証は 70,001〜74,999(protocol の範囲 1〜29,999・審査の段の 40,001〜/50,001〜 の外。どの段とも重ならない)
// 一様の開始と DR の物理: mulberry32(定数)(下の各組)。種(シーンの seed)は 6,100,000〜(検証)・6,200,000〜(試験)。no_wall の教師は種を使わない
// 端の格子: 33 ゲート × 左右 {−0.19, −0.16, +0.16, +0.19} m × 向き {−14.5, −11.5, +11.5, +14.5}°(開始のすき間 ≥ 0.12 m のもの。確認 r2 の格子・r3 の格子・
//   確認 r3 の NBR・EDGE2 と同じ点を含まない)
export const EDGE_OFFS = Object.freeze([-.19, -.16, .16, .19]), EDGE_YAWS = Object.freeze([-14.5, -11.5, 11.5, 14.5]);
export const V_EDGE_OFFS = Object.freeze([-.18, .18]), V_EDGE_YAWS = Object.freeze([-13, 13]);
export function buildValidation4Jobs() {
  const J = [], add = add0(J);
  const stream = haltonStream(70001, 74999, 150);
  stream.forEach((st, i) => add('V4-N', `VN${i}`, {start: {...st, seed: 6100000 + i}}, {physics: 'nominal'}));
  const rL = mulberry32(0x4e4a5601);
  for (let i = 0; i < 150; i++) add('V4-LOWC', `VL${i}`, {start: {...uniformStart(rL, .12, .15), seed: 6101000 + i}}, {physics: 'nominal'});
  let n = 0;
  for (let g = 0; g < 33; g++) for (const o of V_EDGE_OFFS) for (const y of V_EDGE_YAWS) {
    const c = startClearance(g, o, y);
    if (!(c >= .12)) continue;
    add('V4-EDGE', `VE-g${g}o${o}y${y}`, {start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 6101500 + n++, initial_clearance_m: c}}, {physics: 'nominal'});
  }
  const rD = mulberry32(0x4e4a5602);
  for (let i = 0; i < 100; i++) add('V4-R', `VR${i}`, {start: {...uniformStart(rD), seed: 6102000 + i}, dr: drUniform(rD)}, {physics: 'dr'});
  for (let i = 0; i < 100; i++) add('V4-VX', `VV${i}`, {start: {...uniformStart(rD), seed: 6103000 + i}, dr: drVertex(rD)}, {physics: 'dr'});
  for (let b = 0; b < 64; b++) add('V4-HC', `VH${b}`, {start: {...uniformStart(rD), seed: 6104000 + b}, dr: hardCornerAll(b)}, {physics: 'dr', corner: b});
  add('V4-STD', 'single', {start: STD, episodeMode: 'single'}, {physics: 'nominal'});
  add('V4-STD', 'continuous', {start: STD}, {physics: 'nominal'});
  for (const j of J) j.judged = true;
  return J;
}
export function buildTest4Jobs() {
  const J = [], add = add0(J);
  const stream = haltonStream(60001, 64999, 300);
  stream.forEach((st, i) => add('NW-N300-R4', `N${i}`, {start: {...st, seed: 6200000 + i}, keepPath: i < 150}, {physics: 'nominal'}));
  const rF = mulberry32(0x4e4a7e01);
  for (let i = 0; i < 1000; i++) add('NW-FS1000-R4', `F${i}`, {start: {...uniformStart(rF), seed: 6201000 + i}}, {physics: 'nominal'});
  const rL = mulberry32(0x4e4a7e02);
  for (let i = 0; i < 400; i++) add('NW-LOWC-R4', `L${i}`, {start: {...uniformStart(rL, .12, .15), seed: 6202000 + i}}, {physics: 'nominal'});
  let n = 0;
  for (let g = 0; g < 33; g++) for (const o of EDGE_OFFS) for (const y of EDGE_YAWS) {
    const c = startClearance(g, o, y);
    if (!(c >= .12)) continue;
    add('NW-EDGE-R4', `E-g${g}o${o}y${y}`, {start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 6202500 + n++, initial_clearance_m: c}}, {physics: 'nominal'});
  }
  add('NW-STD', 'single', {start: STD, episodeMode: 'single'}, {physics: 'nominal'});
  add('NW-STD', 'continuous', {start: STD}, {physics: 'nominal'});
  gridRows().forEach((r, i) => {
    for (const em of ['single', 'continuous'])
      add(r.in_range ? 'NW-G' : 'NW-G-OUT', `G${String(i).padStart(2, '0')}-${em}`, {start: STD, episodeMode: em, dr: r.dr}, {physics: i === 0 ? 'nominal' : 'dr', name: r.name});
  });
  const rD = mulberry32(0x4e4a7e03);
  for (let i = 0; i < 200; i++) add('NW-R200-R4', `R${i}`, {start: {...uniformStart(rD), seed: 6203000 + i}, dr: drUniform(rD)}, {physics: 'dr'});
  for (let i = 0; i < 300; i++) add('NW-VX-R4', `VX${i}`, {start: {...uniformStart(rD), seed: 6204000 + i}, dr: drVertex(rD)}, {physics: 'dr'});
  for (let b = 0; b < 64; b++) for (let k = 0; k < 4; k++)
    add('NW-HC-R4', `H${b}-${k}`, {start: {...uniformStart(rD), seed: 6205000 + 4 * b + k}, dr: hardCornerAll(b)}, {physics: 'dr', corner: b});
  for (let i = 0; i < 30; i++) add('NW-FC30-R4', `FC${i}`, {start: {...uniformStart(rD), seed: 6206000 + i},
    dr: {physics: {}, delay: 0, steerGain: 1, speedGain: 1, frictionCircle: true, kind: 'fc'}}, {physics: 'dr'});
  for (let i = 0; i < 20; i++) {
    const dr = i < 10 ? null : drUniform(rD);
    add('NW-LONG-R4', `LG${i}`, {start: {...uniformStart(rD), seed: 6207000 + i}, timeLimit: 120, ...(dr ? {dr} : {})}, {physics: dr ? 'dr' : 'nominal'});
  }
  // 報告だけ
  for (const j of J.filter(x => ['NW-N300-R4', 'NW-R200-R4', 'NW-VX-R4'].includes(x.set)))
    add('NW-OFF-R4', `${j.set}:${j.id}`, {...j.spec, keepPath: false, shield: false}, {physics: j.physics, of: j.set, report: true});
  stream.slice(0, 150).forEach((st, i) => J.push({set: 'FAST', id: `N${i}`, spec: {mode: 'fastest', start: {...st, seed: 6200000 + i}, keepPath: true,
    policyPath: '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/bundle/fastest/policy.json', rtDir: '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/web/rl'},
    physics: 'nominal', report: true}));
  for (let i = 0; i < 10; i++) add('LOWMU', `LM${i}`, {start: {...stream[i], seed: 6200000 + i}, dr: {physics: {frictionMultiplier: .4}, delay: 0}}, {physics: 'dr', report: true});
  for (const j of J) j.judged = !j.report && j.set !== 'NW-G-OUT';
  return J;
}
