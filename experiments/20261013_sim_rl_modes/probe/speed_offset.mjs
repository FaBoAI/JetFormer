// いまのベストラップの方策を学習し直さずに「速さ」「線」を変えられるか(条件付けの代わりになるか)を確かめる。
//  S. 速さ: 方策の速さの指令に上限(cap)を掛ける / 倍率を掛ける。観測の「前回の行動」は適用した値。
//  O. 線: 観測の参照ループ R を左右に o [m] ずらして見せる(横ずれ e_y と 16 点の先読みをずらす)。レイ・すき間・ゲートは本物のまま。
//  W. 開始: アプリの入力の範囲いっぱい(ゲート 0〜32・左右 ±0.3 m・向き ±25°)の開始から走れるか。
// どれも連続周回 20 秒。
//   node probe/speed_offset.mjs
import {features, envCore, loadBestPolicy, runPolicyEpisode, writeJson, quantiles} from './common.mjs';

const {loopPointAt, LOOKAHEAD_DS, SCALES, OBS_INDEX} = features;
const {act, sha256} = loadBestPolicy();
const res = {schema: 'sim-rl-modes-probe-speed-offset/v1', policy_sha256: sha256, speed_cap: [], speed_scale: [], offset: [], wide_starts: {}};
const STARTS = [{gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, {gate: 0, left_offset_m: .03, left_yaw_deg: 3, seed: 1},
  {gate: 0, left_offset_m: -.03, left_yaw_deg: -3, seed: 1}, {gate: 20, left_offset_m: 0, left_yaw_deg: 0, seed: 1}];
const sumRun = r => {
  const s = r.summary, tr = r.trace, moving = tr.filter(q => q.v > .3);
  return {reason: s.sim_reason, collisions: s.collisions, collider: s.collision_events?.[0]?.collider, laps: s.completed_laps, valid_laps: s.valid_laps,
    gate_miss: r.env.stats.gate_miss_frame !== null, lap_times: s.laps.map(l => +l.elapsed.toFixed(3)), min_clearance: +s.min_clearance.toFixed(4),
    max_speed: +s.max_speed.toFixed(2), mean_ey: moving.length ? +(moving.reduce((a, q) => a + q.ey, 0) / moving.length).toFixed(3) : null,
    ey_q: quantiles(moving.map(q => +q.ey.toFixed(3)), [.05, .5, .95])};
};
// S. 上限
for (const cap of [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5]) for (const st of STARTS) {
  const hook = (env, frame, a) => { const v = 4.5 * (a[1] + 1) / 2, vc = Math.min(v, cap); return {a0: a[0], a1: 2 * vc / 4.5 - 1}; };
  res.speed_cap.push({cap, start: st, ...sumRun(await runPolicyEpisode({act, hook, start: st}))});
}
// S. 倍率
for (const k of [.9, .8, .7, .6]) for (const st of STARTS) {
  const hook = (env, frame, a) => { const v = 4.5 * (a[1] + 1) / 2 * k; return {a0: a[0], a1: 2 * v / 4.5 - 1}; };
  res.speed_scale.push({scale: k, start: st, ...sumRun(await runPolicyEpisode({act, hook, start: st}))});
}
// O. 線をずらして見せる
const pt = {x: 0, z: 0, tx: 0, tz: 0};
for (const o of [-.15, -.10, -.05, 0, .05, .10, .15]) for (const st of STARTS) {
  const obsHook = (env, ob) => {
    const S = env.view.state, c = Math.cos(S.yaw), sn = Math.sin(S.yaw), s = env.tracker.s;
    ob[OBS_INDEX.lateral_error] = (env.aux.lateral_error - o) / SCALES.lateral_error;
    for (let j = 0; j < 16; j++) {
      loopPointAt(s + LOOKAHEAD_DS[j], pt);
      const x = pt.x + o * pt.tz, z = pt.z - o * pt.tx, dx = x - S.x, dz = z - S.z;   // R の左 = (tz, −tx)
      ob[8 + 2 * j] = (dx * c - dz * sn) / SCALES.lookahead; ob[9 + 2 * j] = (-dx * sn - dz * c) / SCALES.lookahead;
    }
  };
  res.offset.push({offset_left_m: o, start: st, ...sumRun(await runPolicyEpisode({act, obsHook, start: st}))});
}
// W. 開始を広げる(ゲート 0〜32 × 左右 {−0.3, −0.15, 0, 0.15, 0.3} × 向き {−25, −12, 0, 12, 25})のうち、障害物と重ならない開始
{
  let n = 0, ok = 0, invalid = 0; const fails = [];
  for (let g = 0; g < 33; g++) for (const off of [-.3, -.15, 0, .15, .3]) for (const yaw of [-25, -12, 0, 12, 25]) {
    const r = await runPolicyEpisode({act, start: {gate: g, left_offset_m: off, left_yaw_deg: yaw, seed: 1}});
    if (r.invalid_spawn) { invalid++; continue; }
    n++; const s = sumRun(r);
    const good = s.reason === 'collection_duration_complete' && !s.collisions && !s.gate_miss;
    if (good) ok++; else fails.push({gate: g, off, yaw, reason: s.reason, gate_miss: s.gate_miss, collider: s.collider});
  }
  res.wide_starts = {grid: '33 ゲート × 左右 5 × 向き 5', valid_starts: n, invalid_spawn: invalid, survived_20s_no_gate_miss: ok, rate: ok / n,
    fails_by_offset: Object.fromEntries([-.3, -.15, 0, .15, .3].map(v => [v, fails.filter(f => f.off === v).length])),
    fails_by_yaw: Object.fromEntries([-25, -12, 0, 12, 25].map(v => [v, fails.filter(f => f.yaw === v).length])), fails: fails.slice(0, 60)};
}
const p = writeJson('speed_offset.json', res);
const brief = rows => rows.map(r => `${JSON.stringify(r.start.gate)}/${r.start.left_offset_m}: ${r.reason} v${r.valid_laps} ${r.lap_times.join(',')} c${r.min_clearance} ey${r.mean_ey}`);
console.log('cap'); for (const r of res.speed_cap) console.log(r.cap, brief([r])[0]);
console.log('scale'); for (const r of res.speed_scale) console.log(r.scale, brief([r])[0]);
console.log('offset'); for (const r of res.offset) console.log(r.offset_left_m, brief([r])[0], JSON.stringify(r.ey_q));
console.log('wide', JSON.stringify({...res.wide_starts, fails: undefined}));
console.log('書いた:', p);
