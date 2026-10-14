// 速さいろいろ(speed)の確認 r1: 速さの指令が上限に張り付いている割合(2026-10-14、確認の段)。
//   「速さの違い = 上限で切っているだけか、方策が c_v を見て速さを選んでいるか」を見る。新しい開始 20 個 × c_v 5 段、cond と 最速 + 同じ上限
import fs from 'node:fs';
import {activeWorld, BASE_WORLD, NOMINAL, policyFor} from './run_sp.mjs';
import {freshStarts, LEVELS} from './jobs_sp.mjs';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const core = await import(`${REL}/dist/sim-core.js`);
const RT = await import('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/web/rl/mode_runtime.mjs');
const out = [];
const starts = freshStarts(300).slice(100, 120);
for (const [k, st] of starts.entries()) for (const cv of LEVELS) for (const kind of ['speed', 'fastest']) {
  const P = policyFor(kind), {world, variation} = activeWorld(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  const cfg = {...NOMINAL, seed: 2100000 + k, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
  const sim = core.createSimulation(world, cfg); let snap = sim.reset({seed: cfg.seed, scenario: 'baseline'});
  const cap = 4.5 * cv;
  const t = RT.createModeTeacher(world, snap.config, variation, {mode: kind, weights: P.weights, policySha256: P.sha, params: kind === 'speed' ? {c_v: cv} : {speed_limit: cap}});
  let n = 0, sat = 0, below80 = 0, ratio = 0;
  while (!snap.state.terminal) {
    const s = snap.state, o = t.action({state: s, metrics: snap.metrics, config: snap.config});
    if (s.speed > .3) { n++; if (o.speed >= .98 * cap) sat++; if (o.speed < .8 * cap) below80++; ratio += o.speed / cap; }
    snap = sim.step({steering: o.steering, speed: o.speed});
  }
  const m = sim.report().metrics;
  out.push({k, c_v: cv, kind, n, sat: sat / n, below80: below80 / n, mean_cmd_ratio: ratio / n, collisions: m.collisions});
}
const agg = {};
for (const r of out) { const key = `${r.kind} ${r.c_v}`; (agg[key] ??= []).push(r); }
const res = Object.fromEntries(Object.entries(agg).map(([k, rs]) => [k, {n: rs.length, sat: rs.reduce((a, r) => a + r.sat, 0) / rs.length, below80: rs.reduce((a, r) => a + r.below80, 0) / rs.length,
  mean_cmd_ratio: rs.reduce((a, r) => a + r.mean_cmd_ratio, 0) / rs.length, collisions: rs.filter(r => r.collisions).length}]));
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_speed_r1/saturation.json', JSON.stringify({rows: out, agg: res}, null, 1));
console.log(JSON.stringify(res, null, 1));
