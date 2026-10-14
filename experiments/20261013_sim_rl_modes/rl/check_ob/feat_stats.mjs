// 障害物よけの確認 r1: 障害物の観測(obst95 の後ろ 16 個)を自前で作り直して、ランタイムの obstacleFeatures と比べ、
// 実際の走りでの平均・ばらつきを、方策の正規化(obs_mean・obs_std)と比べる(2026-10-14、確認の段)。
// あわせて F-ID の列(candidates/obstacle/fid_std_validation000.json)を、自前の world(自前の障害物の変換)で走らせた操作と比べる。
//   node feat_stats.mjs → check_obstacle_r1/feat_stats.json
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, worldWith, NOMINAL, RT, policyFor} from './run_ob.mjs';
import {buildSet} from './jobs_ob.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_obstacle_r1';
const core = await import('/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5/dist/sim-core.js');
const {obstacleFeatures} = await import(`${EXP}/web/rl/obs_modes.mjs`);
// 自前: modes.json common.obs_wrappers.obst95 の文言(近い順に 4 個・3 m 以内・左右 100° 以内・前/3・左/3・半径/0.2・有無、箱は芯の線分の一番近い点)
function ownFeat(obstacles, x, z, yaw) {
  const fx = Math.cos(yaw), fz = -Math.sin(yaw), lx = -Math.sin(yaw), lz = -Math.cos(yaw), list = [];
  for (const o of obstacles) {
    let px = o.x, pz = o.z;
    if (o.shape === 'box') {
      const ux = Math.cos(o.yaw), uz = -Math.sin(o.yaw), h = o.length / 2;
      const t = Math.max(-h, Math.min(h, (x - o.x) * ux + (z - o.z) * uz)); px = o.x + t * ux; pz = o.z + t * uz;
    }
    const dx = px - x, dz = pz - z, f = dx * fx + dz * fz, l = dx * lx + dz * lz, d = Math.hypot(dx, dz);
    if (d > 3 || Math.abs(Math.atan2(l, f)) > 100 * Math.PI / 180) continue;
    list.push({d, f, l, r: o.r});
  }
  list.sort((a, b) => a.d - b.d);
  const out = new Array(16).fill(0);
  for (let k = 0; k < 4 && k < list.length; k++) { out[4 * k] = list[k].f / 3; out[4 * k + 1] = list[k].l / 3; out[4 * k + 2] = list[k].r / .2; out[4 * k + 3] = 1; }
  return out;
}
const P = policyFor('obstacle');
const sum = new Array(16).fill(0), sq = new Array(16).fill(0);
let n = 0, maxDiff = 0, nWithObs = 0;
const rt = new Float64Array(95);
// FRESH の配置・開始(障害物よけの方策)を走らせ直す。すべてのフレームで数える
for (const j of buildSet('FRESH')) {
  const sp = j.spec, obs = sp.layout.obstacles;
  const {world: active, variation} = activeWorld(BASE_WORLD, sp.start.gate, sp.start.left_offset_m, sp.start.left_yaw_deg);
  const world = worldWith(active, obs);
  const sim = core.createSimulation(world, {...NOMINAL, seed: sp.start.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed: sp.start.seed, scenario: 'baseline'});
  const t = RT.createModeTeacher(world, snap.config, variation, {mode: 'obstacle', weights: P.weights, policySha256: P.sha, params: {layout_id: sp.layout.id}});
  while (!snap.state.terminal) {
    const st = snap.state, own = ownFeat(obs, st.x, st.z, st.yaw);
    obstacleFeatures(obs, st.x, st.z, st.yaw, rt, 79);
    for (let k = 0; k < 16; k++) { maxDiff = Math.max(maxDiff, Math.abs(own[k] - rt[79 + k])); sum[k] += own[k]; sq[k] += own[k] * own[k]; }
    n++; if (own[3]) nWithObs++;
    const a = t.action({state: st, metrics: snap.metrics, config: snap.config});
    snap = sim.step({steering: a.steering, speed: a.speed});
  }
}
const mean = sum.map(s => s / n), std = sq.map((s, k) => Math.sqrt(Math.max(0, s / n - mean[k] ** 2)));
const pm = P.weights.obs_mean.slice(79), ps = P.weights.obs_std.slice(79);
// 方策の 1 層目で、障害物の 16 列の重みの大きさ(正規化の後の入力 1 あたり)と、元の 79 列の平均
const W = P.weights.layers[0].w, nOut = P.weights.layers[0].n_out, nIn = P.weights.layers[0].n_in;
const colNorm = c => { let s = 0; for (let o = 0; o < nOut; o++) { const v = Array.isArray(W[o]) ? W[o][c] : W[o * nIn + c]; s += v * v; } return Math.sqrt(s); };
const cn = Array.from({length: nIn}, (_, c) => colNorm(c));
// F-ID の列と自前の world での走り
const fid = JSON.parse(fs.readFileSync(`${EXP}/candidates/obstacle/fid_std_validation000.json`, 'utf8'));
const lay = JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/validation.json`, 'utf8')).layouts[0];
const {world: a0, variation: v0} = activeWorld(BASE_WORLD, 0, 0, 0);
const w0 = worldWith(a0, lay.obstacles);
const sim0 = core.createSimulation(w0, {...NOMINAL, seed: fid.config.seed ?? 1, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
let s0 = sim0.reset({seed: fid.config.seed ?? 1, scenario: 'baseline'});
const t0 = RT.createModeTeacher(w0, s0.config, v0, {mode: 'obstacle', weights: P.weights, policySha256: P.sha, params: fid.params});
let fidMax = 0, fidPosMax = 0, k = 0;
while (!s0.state.terminal && k < fid.frames.length) {
  const a = t0.action({state: s0.state, metrics: s0.metrics, config: s0.config}), b = fid.frames[k].action, sb = fid.frames[k].snapshot.state;
  fidMax = Math.max(fidMax, Math.abs(a.steering - b.steering), Math.abs(a.speed - b.speed));
  fidPosMax = Math.max(fidPosMax, Math.hypot(s0.state.x - sb.x, s0.state.z - sb.z));
  s0 = sim0.step({steering: a.steering, speed: a.speed}); k++;
}
const names = ['前/3', '左/3', '半径/0.2', '有無'];
const res = {frames: n, frames_with_obstacle_in_view: nWithObs, own_vs_runtime_max_abs: maxDiff,
  per_dim: mean.map((m, i) => ({slot: Math.floor(i / 4) + 1, name: names[i % 4], real_mean: +m.toFixed(4), real_std: +std[i].toFixed(4), policy_mean: +pm[i].toFixed(4), policy_std: +ps[i].toFixed(4),
    scale_vs_proper: std[i] > 0 ? +(std[i] / ps[i]).toFixed(3) : null, w_col_norm: +cn[79 + i].toFixed(3)})),
  w_col_norm_base79_median: +[...cn.slice(0, 79)].sort((a, b) => a - b)[39].toFixed(3),
  fid: {frames_compared: k, fid_frames: fid.frames.length, max_abs_action_diff: fidMax, max_pos_diff_m: fidPosMax, fid_seed: fid.config.seed, fid_params: fid.params, fid_layout_from_world: null}};
fs.writeFileSync(`${OUT}/feat_stats.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify({...res, per_dim: res.per_dim.slice(0, 8)}));
