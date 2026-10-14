// 障害物よけの確認 r2: 障害物の観測(obst95 の後ろ 16 個)と方策の正規化・感度を自前で調べる(2026-10-14、確認の段 r2)。
//   (1) FRESH の最初の 60 回(30 配置 × 2 開始)を r2 の方策で走らせ直し、方策に渡る観測の 16 個を自前の式(modes.json obst95 の文言)と比べ、実際の平均・ばらつきを
//       方策の obs_mean・obs_std と比べる。3・4 番目の障害物が見えるフレームと、そのときの正規化した値(±5 で切る)も数える
//   (2) 感度: 同じ観測で、一番近い障害物の左右だけを +0.10 m ずらしたときの舵の出力の差(自前の MLP の計算。r2・r1 の方策)
//   (3) 1 層目の列の重みの大きさ(r2・r1)
//   (4) F-ID: candidates/obstacle/fid_std_validation000.json を、自前の world(自前の障害物の変換)で rl/runtime・web/rl の両方で走らせて比べる
//   node feat_stats2.mjs → check_obstacle_r2/feat_stats.json
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, worldWith, NOMINAL, policyFor} from './run_ob2.mjs';
import {buildSet} from './jobs_ob2.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_obstacle_r2';
const core = await import('/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5/dist/sim-core.js');
const RTW = await import(`${EXP}/web/rl/mode_runtime.mjs`), RTN = await import(`${EXP}/rl/runtime/mode_runtime.mjs`);
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
// 自前の MLP(policy.json の文言: x = clip((obs − mean)/std, ±clip) → tanh 256 → tanh 256 → 線形 2 → ±action_clip)
function ownForward(W) {
  const mean = W.obs_mean, std = W.obs_std, clip = W.obs_clip, ac = W.action_clip ?? 1;
  return obs => {
    let x = Array.from(obs, (v, i) => Math.max(-clip, Math.min(clip, (v - mean[i]) / std[i])));
    for (const L of W.layers) {
      const y = new Array(L.n_out);
      for (let o = 0; o < L.n_out; o++) { let s = L.b[o]; for (let i = 0; i < L.n_in; i++) s += L.w[o * L.n_in + i] * x[i]; y[o] = L.act === 'tanh' ? Math.tanh(s) : s; }
      x = y;
    }
    return x.map(v => Math.max(-ac, Math.min(ac, v)));
  };
}
const P2 = policyFor('obstacle'), P1 = policyFor('obstacle_r1');
const f2 = ownForward(P2.weights), f1 = ownForward(P1.weights);
const sum = new Array(16).fill(0), sq = new Array(16).fill(0);
let n = 0, maxDiff = 0, nWithObs = 0, nSlot3 = 0, nSlot4 = 0, slot34ClipFrames = 0, ownVsRtAction = 0;
const sens = {r2: [], r1: []}, sensFwd = {r2: [], r1: []};
const jobs = buildSet('FRESH').slice(0, 60);
for (const j of jobs) {
  const sp = j.spec, obs = sp.layout.obstacles;
  const {world: active, variation} = activeWorld(BASE_WORLD, sp.start.gate, sp.start.left_offset_m, sp.start.left_yaw_deg);
  const world = worldWith(active, obs);
  const sim = core.createSimulation(world, {...NOMINAL, seed: sp.start.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed: sp.start.seed, scenario: 'baseline'});
  let lastObs = null;
  const rtPol = RTW.createModeTeacher(world, snap.config, variation, {mode: 'obstacle', weights: P2.weights, policySha256: P2.sha, params: {layout_id: sp.layout.id}});
  // 同じ観測を見るため、方策の関数を包む(中身は自前の MLP。ランタイムの行動と比べて同じであることも確かめる)
  const t = RTW.createModeTeacher(world, snap.config, variation, {mode: 'obstacle', weights: P2.weights, policySha256: P2.sha, params: {layout_id: sp.layout.id},
    policy: (o, out) => { lastObs = Float64Array.from(o); const a = f2(o); out[0] = a[0]; out[1] = a[1]; return out; }});
  while (!snap.state.terminal) {
    const st = snap.state, own = ownFeat(obs, st.x, st.z, st.yaw);
    const a = t.action({state: st, metrics: snap.metrics, config: snap.config});
    const b = rtPol.action({state: st, metrics: snap.metrics, config: snap.config});
    ownVsRtAction = Math.max(ownVsRtAction, Math.abs(a.steering - b.steering), Math.abs(a.speed - b.speed));
    for (let k = 0; k < 16; k++) { maxDiff = Math.max(maxDiff, Math.abs(own[k] - lastObs[79 + k])); sum[k] += own[k]; sq[k] += own[k] * own[k]; }
    n++; if (own[3]) nWithObs++; if (own[11]) nSlot3++; if (own[15]) nSlot4++;
    if (own[11] || own[15]) {
      let clipped = false;
      for (const k of [8, 9, 10, 11, 12, 13, 14, 15]) if (Math.abs((own[k] - P2.weights.obs_mean[79 + k]) / P2.weights.obs_std[79 + k]) >= P2.weights.obs_clip) clipped = true;
      if (clipped) slot34ClipFrames++;
    }
    // 感度: 一番近い障害物が前 0.3〜2.5 m にあるフレームで、左右 +0.10 m(入力 +0.0333)と前後 −0.10 m
    if (own[3] && own[0] * 3 > .3 && own[0] * 3 < 2.5) {
      for (const [name, F] of [['r2', f2], ['r1', f1]]) {
        const base = F(lastObs), o2 = Float64Array.from(lastObs); o2[80] += .1 / 3;
        const o3 = Float64Array.from(lastObs); o3[79] -= .1 / 3;
        sens[name].push(Math.abs(F(o2)[0] - base[0])); sensFwd[name].push(Math.abs(F(o3)[1] - base[1]));
      }
    }
    snap = sim.step({steering: a.steering, speed: a.speed});
  }
}
const mean = sum.map(s => s / n), std = sq.map((s, k) => Math.sqrt(Math.max(0, s / n - mean[k] ** 2)));
const colNorms = W => { const L = W.layers[0]; return Array.from({length: L.n_in}, (_, c) => { let s = 0; for (let o = 0; o < L.n_out; o++) s += L.w[o * L.n_in + c] ** 2; return Math.sqrt(s); }); };
const cn2 = colNorms(P2.weights), cn1 = colNorms(P1.weights);
const med = a => { const b = [...a].sort((x, y) => x - y); return b[Math.floor(b.length / 2)]; };
const q = (a, p) => { const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * b.length))]; };
// F-ID
const fid = JSON.parse(fs.readFileSync(`${EXP}/candidates/obstacle/fid_std_validation000.json`, 'utf8'));
const lay = JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/validation.json`, 'utf8')).layouts.find(l => l.id === (fid.params?.layout_id ?? 'validation-000'));
const fidRes = {};
for (const [name, RT] of [['web_rl', RTW], ['rl_runtime', RTN]]) {
  const {world: a0, variation: v0} = activeWorld(BASE_WORLD, 0, 0, 0);
  const w0 = worldWith(a0, lay.obstacles);
  const sim0 = core.createSimulation(w0, {...NOMINAL, seed: fid.config.seed ?? 1, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let s0 = sim0.reset({seed: fid.config.seed ?? 1, scenario: 'baseline'});
  const t0 = RT.createModeTeacher(w0, s0.config, v0, {mode: 'obstacle', weights: P2.weights, policySha256: P2.sha, params: fid.params});
  let fidMax = 0, fidPosMax = 0, infoDiff = 0, k = 0;
  while (!s0.state.terminal && k < fid.frames.length) {
    const a = t0.action({state: s0.state, metrics: s0.metrics, config: s0.config}), b = fid.frames[k].action, sb = fid.frames[k].snapshot.state;
    fidMax = Math.max(fidMax, Math.abs(a.steering - b.steering), Math.abs(a.speed - b.speed));
    fidPosMax = Math.max(fidPosMax, Math.hypot(s0.state.x - sb.x, s0.state.z - sb.z));
    if (b.info && JSON.stringify(a.info) !== JSON.stringify(b.info)) infoDiff++;
    s0 = sim0.step({steering: a.steering, speed: a.speed}); k++;
  }
  fidRes[name] = {frames_compared: k, fid_frames: fid.frames.length, max_abs_action_diff: fidMax, max_pos_diff_m: fidPosMax, info_diff_frames: infoDiff};
}
// F-ID の world の障害物と、自前で配置から作った障害物の差
const fidObs = [...fid.world.disks.filter(d => d.kind === 'obstacle'), ...fid.world.segments.filter(s => s.kind === 'obstacle')];
const own = worldWith({disks: [], segments: []}, lay.obstacles), ownObs = [...own.disks, ...own.segments];
const names = ['前/3', '左/3', '半径/0.2', '有無'];
const res = {runs: jobs.length, frames: n, frames_with_obstacle_in_view: nWithObs, frames_slot3: nSlot3, frames_slot4: nSlot4, frames_slot34_clipped: slot34ClipFrames,
  own_vs_runtime_feature_max_abs: maxDiff, own_mlp_vs_runtime_action_max_abs: ownVsRtAction,
  per_dim: mean.map((m, i) => ({slot: Math.floor(i / 4) + 1, name: names[i % 4], real_mean: +m.toFixed(4), real_std: +std[i].toFixed(4),
    policy_mean: +P2.weights.obs_mean[79 + i].toFixed(4), policy_std: +P2.weights.obs_std[79 + i].toFixed(4), r1_policy_std: +P1.weights.obs_std[79 + i].toFixed(4),
    scale_vs_proper_r2: std[i] > 0 ? +(std[i] / P2.weights.obs_std[79 + i]).toFixed(3) : null, w_col_norm_r2: +cn2[79 + i].toFixed(3), w_col_norm_r1: +cn1[79 + i].toFixed(3)})),
  w_col_norm_base79_median: {r2: +med(cn2.slice(0, 79)).toFixed(3), r1: +med(cn1.slice(0, 79)).toFixed(3)},
  sensitivity_steer_per_10cm_lateral: Object.fromEntries(Object.entries(sens).map(([k, v]) => [k, {n: v.length, median: +med(v).toFixed(5), p90: +q(v, .9).toFixed(5), mean: +(v.reduce((a, b) => a + b, 0) / v.length).toFixed(5)}])),
  sensitivity_speed_per_10cm_closer: Object.fromEntries(Object.entries(sensFwd).map(([k, v]) => [k, {n: v.length, median: +med(v).toFixed(5), p90: +q(v, .9).toFixed(5)}])),
  fid: fidRes, fid_layout_id: lay.id, fid_world_obstacles_vs_own_max_abs: (() => { let m = 0; const fo = JSON.stringify(fidObs.map(o => o.x ?? o.a[0])), oo = JSON.stringify(ownObs.map(o => o.x ?? o.a[0]));
    fidObs.forEach((o, i) => { const p = ownObs[i]; if (!p) { m = Infinity; return; } if (o.x != null) m = Math.max(m, Math.abs(o.x - p.x), Math.abs(o.z - p.z), Math.abs(o.radius - p.radius));
      else m = Math.max(m, ...[0, 1].map(k => Math.abs(o.a[k] - p.a[k])), ...[0, 1].map(k => Math.abs(o.b[k] - p.b[k])), Math.abs(o.radius - p.radius)); }); return m; })(),
  fid_world_obstacle_count: fidObs.length, own_obstacle_count: ownObs.length};
fs.writeFileSync(`${OUT}/feat_stats.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify({...res, per_dim: res.per_dim.slice(0, 8)}, null, 1));
