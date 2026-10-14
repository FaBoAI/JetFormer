// 速さいろいろ(speed)の確認 r1 の回の一覧(2026-10-14、確認の段)。開始・種は自前で作る(eval_modes.mjs の buildJobs を使わない)。
//   REPLAY: 決めた試験の組 SP-P60 + SP-STD3 を protocol の文言から作り直したもの(Halton 20,001〜、基数 2・3・5、開始のすき間 > 0、種 905000 + i、c_v は 1.0・0.875・0.75・0.625・0.5 の順)
//   新しい開始: mulberry32(0x5bd1e14) の一様乱数(Halton を使わない)。種 1,800,000〜(試験 905xxx・検証 805xxx・記録 705xxx・線 990xxx と重ならない)
import {startClearance} from './run_sp.mjs';

export const LEVELS = [1.0, .875, .75, .625, .5];
const FB_CV = [1.0, 4.0 / 4.5, 3.5 / 4.5, 3.0 / 4.5, 2.5 / 4.5];
function halton(i, b) { let f = 1, r = 0, n = i; while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); } return r; }
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export function testStarts(count = 60) {
  const out = [];
  for (let i = 20001; out.length < count && i < 30000; i++) {
    const st = {gate: Math.min(32, Math.floor(halton(i, 2) * 33)), left_offset_m: -.2 + .4 * halton(i, 3), left_yaw_deg: -15 + 30 * halton(i, 5), halton_index: i};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c > 1e-9)) continue;
    out.push({...st, start_clearance: c});
  }
  return out;
}
export function freshStarts(count, seed = 0x5bd1e14) {
  const rnd = mulberry32(seed), out = [];
  while (out.length < count) {
    const st = {gate: Math.floor(rnd() * 33), left_offset_m: -.2 + .4 * rnd(), left_yaw_deg: -15 + 30 * rnd()};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c > 1e-9)) continue;
    out.push({...st, start_clearance: c});
  }
  return out;
}
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
export function buildSet(name) {
  const J = [];
  if (name === 'REPLAY' || name === 'REPLAY_FB') {
    const fb = name === 'REPLAY_FB';
    testStarts(60).forEach((st, i) => {
      const cv = fb ? FB_CV[i % 5] : LEVELS[i % 5];
      J.push({id: `P${i}`, spec: {policy: fb ? 'fastest' : 'speed', start: {...st, seed: 905000 + i}, params: fb ? {speed_limit: 4.5 * cv} : {c_v: cv}, c_v: cv}});
    });
    for (const cv of [1.0, .75, .5]) {
      const c = fb ? {1: 1, .75: 3.5 / 4.5, .5: 2.5 / 4.5}[cv] : cv;
      J.push({id: `cv${cv}`, spec: {policy: fb ? 'fastest' : 'speed', start: STD, episodeMode: 'single', params: fb ? {speed_limit: 4.5 * c} : {c_v: c}, c_v: c}});
    }
  } else if (name === 'FRESH' || name === 'FRESH_CAP' || name === 'FRESH_FAST') {
    const fs = freshStarts(300);
    fs.forEach((st, i) => {
      const cv = LEVELS[i % 5];
      if (name === 'FRESH') J.push({id: `F${i}`, spec: {policy: 'speed', start: {...st, seed: 1800000 + i}, params: {c_v: cv}, c_v: cv, keepPath: i < 25}});
      else if (name === 'FRESH_CAP') J.push({id: `F${i}`, spec: {policy: 'fastest', start: {...st, seed: 1800000 + i}, params: {speed_limit: 4.5 * cv}, c_v: cv, keepPath: i < 25}});
      else if (i % 5 === 0) J.push({id: `F${i}`, spec: {policy: 'fastest', start: {...st, seed: 1800000 + i}, params: {}, c_v: 1.0, keepPath: i < 25}});
    });
  } else if (name === 'STDV') {
    let k = 0;
    for (const off of [-.03, 0, .03]) for (const yaw of [-3, 0, 3]) for (const cv of LEVELS)
      J.push({id: `S${k++}`, spec: {policy: 'speed', start: {gate: 0, left_offset_m: off, left_yaw_deg: yaw, seed: 1900000 + k}, episodeMode: 'single', params: {c_v: cv}, c_v: cv}});
  } else if (name === 'SEEDCHK') {
    for (const seed of [1, 12345, 1800001]) J.push({id: `seed${seed}`, spec: {policy: 'speed', start: {gate: 5, left_offset_m: .05, left_yaw_deg: 4, seed}, params: {c_v: .75}, c_v: .75}});
  } else if (name === 'DR') {
    // DR-T(modes.json common.dr_specs)から一様。遅延 0/1 半々。摩擦円は使わない(元のコアに無いため)。記録は標準の物理なので判定ではない
    const rnd = mulberry32(0xd27e57), fs = freshStarts(300);
    const U = (a, b) => a + (b - a) * rnd();
    for (let i = 0; i < 60; i++) {
      const cv = LEVELS[i % 5], st = fs[i];
      const dr = {physics: {steerRate: U(.8, 1.8), brakeDeceleration: U(.7, 2.2), driveTau: U(.28, .55), driveAccelerationCap: U(3.5, 6.0), frictionMultiplier: U(.85, 1.1)},
        delay: rnd() < .5 ? 0 : 1, steerGain: U(.94, 1.06), speedGain: U(.96, 1.04)};
      J.push({id: `D${i}`, spec: {policy: 'speed', start: {...st, seed: 1950000 + i}, params: {c_v: cv}, c_v: cv, dr}});
    }
  } else throw new Error('知らない組: ' + name);
  return J;
}
