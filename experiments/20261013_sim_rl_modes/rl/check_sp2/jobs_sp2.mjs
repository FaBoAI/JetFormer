// 速さいろいろ(speed)の確認 r2 の回の一覧(2026-10-14、確認の段)。開始・種は protocol_speed_r2.json の文言から自前で作る(eval_modes.mjs の buildJobs を使わない)。
//   REPLAY(_WEB・_R1・_FB): 決めた試験の組 SP-P60-R2 + SP-STD3 の作り直し(Halton 25,001〜、基数 2・3・5、開始のすき間 ≥ 0.12 m、種 905500 + i、c_v は 1.0・0.875・0.75・0.625・0.5 の順)
//   FRESH(_CAP・_FAST・_R1): 新しい開始。mulberry32(0x2c0ffee5) の一様乱数(Halton を使わない)、ゲート U{0..32}・±0.2 m・±15°・開始のすき間 ≥ 0.12 m。種 2,600,000〜
//     (試験 905xxx・検証 805xxx・記録 705xxx・線 990xxx・学習の段の報告 1,900,000〜・確認 r1 の 1,800,000〜 と重ならない。乱数の種も学習の段の報告 6221138・確認 r1 の 0x5bd1e14 と別)
//   FRESH2: 2 つ目の新しい開始(mulberry32(0x51ed2701)、900 回、種 3,100,000〜)
//   GRID: protocol の開始の範囲の端と真ん中(33 ゲート × 左右 {−0.2, 0, +0.2} m × 向き {−15, 0, +15}°)× c_v {1.0, 0.75, 0.5}。開始のすき間 ≥ 0.12 m のものだけ
//   HOT: GRID・FRESH で近かった所のまわりを細かく(ゲートの一覧は HOT_GATES、c_v の一覧は HOT_CV、方策は HOT_POLICY で渡す。左右 0.05 m・向き 2.5° おき)
import {startClearance} from './run_sp2.mjs';

export const LEVELS = [1.0, .875, .75, .625, .5];
const FB_CV = [1.0, 4.0 / 4.5, 3.5 / 4.5, 3.0 / 4.5, 2.5 / 4.5];
const MINC = .12;
function halton(i, b) { let f = 1, r = 0, n = i; while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); } return r; }
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export function testStarts(count = 60, from = 25001) {
  const out = []; let skipped = 0;
  for (let i = from; out.length < count && i < 30000; i++) {
    const st = {gate: Math.min(32, Math.floor(halton(i, 2) * 33)), left_offset_m: -.2 + .4 * halton(i, 3), left_yaw_deg: -15 + 30 * halton(i, 5), halton_index: i};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c >= MINC)) { skipped++; continue; }
    out.push({...st, start_clearance: c});
  }
  out.skipped = skipped;
  return out;
}
export function freshStarts(count, seed = 0x2c0ffee5) {
  const rnd = mulberry32(seed), out = [];
  while (out.length < count) {
    const st = {gate: Math.floor(rnd() * 33), left_offset_m: -.2 + .4 * rnd(), left_yaw_deg: -15 + 30 * rnd()};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c >= MINC)) continue;
    out.push({...st, start_clearance: c});
  }
  return out;
}
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const NFRESH = +(process.env.NFRESH ?? 600);
export function buildSet(name) {
  const J = [];
  if (['REPLAY', 'REPLAY_WEB', 'REPLAY_R1', 'REPLAY_FB'].includes(name)) {
    const fb = name === 'REPLAY_FB', pol = fb ? 'fastest' : name === 'REPLAY_R1' ? 'speed_r1' : 'speed';
    testStarts(60).forEach((st, i) => {
      const cv = fb ? FB_CV[i % 5] : LEVELS[i % 5];
      J.push({id: `P${i}`, spec: {policy: pol, start: {...st, seed: 905500 + i}, params: fb ? {speed_limit: 4.5 * cv} : {c_v: cv}, c_v: cv}});
    });
    for (const cv of [1.0, .75, .5]) {
      const c = fb ? {1: 1, .75: 3.5 / 4.5, .5: 2.5 / 4.5}[cv] : cv;
      J.push({id: `cv${cv}`, spec: {policy: pol, start: STD, episodeMode: 'single', params: fb ? {speed_limit: 4.5 * c} : {c_v: c}, c_v: c}});
    }
  } else if (['FRESH', 'FRESH_CAP', 'FRESH_FAST', 'FRESH_R1'].includes(name)) {
    const fs = freshStarts(NFRESH);
    fs.forEach((st, i) => {
      const cv = LEVELS[i % 5], base = {start: {...st, seed: 2600000 + i}, c_v: cv, keepPath: i < 30};
      if (name === 'FRESH') J.push({id: `F${i}`, spec: {...base, policy: 'speed', params: {c_v: cv}}});
      else if (name === 'FRESH_R1') { if (i < 300) J.push({id: `F${i}`, spec: {...base, policy: 'speed_r1', params: {c_v: cv}}}); }
      else if (name === 'FRESH_CAP') { if (i < 300) J.push({id: `F${i}`, spec: {...base, policy: 'fastest', params: {speed_limit: 4.5 * cv}}}); }
      else if (i < 300 && i % 5 === 0) J.push({id: `F${i}`, spec: {...base, policy: 'fastest', params: {}, c_v: 1.0}});
    });
  } else if (name === 'FRESH2') {
    // 2 つ目の新しい開始(乱数の種 0x51ed2701、種 3,100,000〜)。一様な開始での失敗の割合の上限を狭めるため
    freshStarts(+(process.env.NFRESH2 ?? 900), 0x51ed2701).forEach((st, i) => {
      const cv = LEVELS[i % 5];
      J.push({id: `E${i}`, spec: {start: {...st, seed: 3100000 + i}, c_v: cv, policy: 'speed', params: {c_v: cv}}});
    });
  } else if (name === 'GRID') {
    let k = 0;
    for (let g = 0; g < 33; g++) for (const off of [-.2, 0, .2]) for (const yaw of [-15, 0, 15]) {
      const c = startClearance(g, off, yaw);
      if (!(c >= MINC)) continue;
      for (const cv of [1.0, .75, .5]) J.push({id: `G${k++}`, spec: {policy: 'speed', start: {gate: g, left_offset_m: off, left_yaw_deg: yaw, start_clearance: c, seed: 2700000 + k}, params: {c_v: cv}, c_v: cv}});
    }
  } else if (name === 'HOT') {
    const gates = (process.env.HOT_GATES ?? '6,7').split(',').map(Number), cvs = (process.env.HOT_CV ?? '1,0.875,0.75,0.625,0.5').split(',').map(Number);
    let k = 0;
    for (const g of gates) for (let a = -4; a <= 4; a++) for (let b = -6; b <= 6; b++) {
      const off = .05 * a, yaw = 2.5 * b, c = startClearance(g, off, yaw);
      if (!(c >= MINC)) continue;
      for (const cv of cvs) J.push({id: `H${k++}`, spec: {policy: process.env.HOT_POLICY ?? 'speed', start: {gate: g, left_offset_m: off, left_yaw_deg: yaw, start_clearance: c, seed: 2800000 + k}, params: {c_v: cv}, c_v: cv}});
    }
  } else if (name === 'STDV') {
    let k = 0;
    for (const off of [-.03, 0, .03]) for (const yaw of [-3, 0, 3]) for (const cv of LEVELS)
      J.push({id: `S${k++}`, spec: {policy: 'speed', start: {gate: 0, left_offset_m: off, left_yaw_deg: yaw, seed: 2900000 + k}, episodeMode: 'single', params: {c_v: cv}, c_v: cv}});
  } else if (name === 'VAL67') {
    // 学習の段の検証の組の P67(黒シートを横切らないままゴールの線を越えた回)と P36(初期化元が止まったままだった開始)を元のコアで。まわりも少し
    const P67 = {gate: 11, left_offset_m: -0.14760961235584008, left_yaw_deg: 14.295360000000002};
    const P36 = {gate: 3, left_offset_m: 0.15197886501041508, left_yaw_deg: -11.14464};
    J.push({id: 'P67', spec: {policy: 'speed', start: {...P67, seed: 805067}, params: {c_v: .75}, c_v: .75, keepPath: true, keepClear: true}});
    J.push({id: 'P36', spec: {policy: 'speed', start: {...P36, seed: 805036}, params: {c_v: .875}, c_v: .875, keepPath: true}});
    let k = 0;
    for (const d of [-.02, .02]) for (const y of [-2, 2]) for (const cv of LEVELS)
      J.push({id: `P67n${k++}`, spec: {policy: 'speed', start: {gate: 11, left_offset_m: P67.left_offset_m + d, left_yaw_deg: P67.left_yaw_deg + y, seed: 2950000 + k}, params: {c_v: cv}, c_v: cv, keepPath: true}});
  } else if (name === 'SEEDCHK') {
    for (const seed of [1, 12345, 2600001]) J.push({id: `seed${seed}`, spec: {policy: 'speed', start: {gate: 5, left_offset_m: .05, left_yaw_deg: 4, seed}, params: {c_v: .75}, c_v: .75}});
  } else if (name === 'DR') {
    // DR-T(modes.json common.dr_specs)から一様。遅延 0/1 半々。摩擦円は使わない(元のコアに無いため)。記録は標準の物理なので判定ではない(参考)
    const rnd = mulberry32(0x0d7b2a2), fs = freshStarts(NFRESH);
    const U = (a, b) => a + (b - a) * rnd();
    for (let i = 0; i < 60; i++) {
      const cv = LEVELS[i % 5], st = fs[i];
      const dr = {physics: {steerRate: U(.8, 1.8), brakeDeceleration: U(.7, 2.2), driveTau: U(.28, .55), driveAccelerationCap: U(3.5, 6.0), frictionMultiplier: U(.85, 1.1)},
        delay: rnd() < .5 ? 0 : 1, steerGain: U(.94, 1.06), speedGain: U(.96, 1.04)};
      J.push({id: `D${i}`, spec: {policy: 'speed', start: {...st, seed: 2960000 + i}, params: {c_v: cv}, c_v: cv, dr}});
    }
  } else throw new Error('知らない組: ' + name);
  return J;
}
