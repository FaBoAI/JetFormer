// 障害物よけの確認 r2: 方策が障害物の入力をどう使うかの下調べ(2026-10-14、確認の段 r2)。学習のコードは使わない。
//   (A) FRESH の 300 回を r2 の方策で走らせ、一番近い障害物が前 0.3〜2.5 m・横 ±0.6 m にあるフレームで、
//       「障害物の 16 入力を全部 0(見えない)にしたとき」との舵の差(切る前の出力)と、障害物の左右の向きの関係を数える(よける向きに切っているか)
//   (B) 3 番目の障害物が見える回(正規化で ±5 に切られる)を、3・4 番目の入力を 0 にして走らせ直し、有効かどうかが変わるか
//   node probe_obs2.mjs → check_obstacle_r2/probe_obs.json
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, worldWith, NOMINAL, policyFor, RT} from './run_ob2.mjs';
import {buildSet} from './jobs_ob2.mjs';

const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_obstacle_r2';
const core = await import('/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5/dist/sim-core.js');
function rawForward(W) {
  const mean = W.obs_mean, std = W.obs_std, clip = W.obs_clip;
  return obs => {
    let x = Array.from(obs, (v, i) => Math.max(-clip, Math.min(clip, (v - mean[i]) / std[i])));
    for (const L of W.layers) {
      const y = new Array(L.n_out);
      for (let o = 0; o < L.n_out; o++) { let s = L.b[o]; for (let i = 0; i < L.n_in; i++) s += L.w[o * L.n_in + i] * x[i]; y[o] = L.act === 'tanh' ? Math.tanh(s) : s; }
      x = y;
    }
    return x;   // 切る前
  };
}
const P2 = policyFor('obstacle'), P1 = policyFor('obstacle_r1');
const F2 = rawForward(P2.weights), F1 = rawForward(P1.weights);
const clip1 = v => Math.max(-1, Math.min(1, v));
function runEp(j, zeroSlots34 = false, collect = null) {
  const sp = j.spec, obs = sp.layout.obstacles;
  const {world: active, variation} = activeWorld(BASE_WORLD, sp.start.gate, sp.start.left_offset_m, sp.start.left_yaw_deg);
  const world = worldWith(active, obs);
  const sim = core.createSimulation(world, {...NOMINAL, seed: sp.start.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed: sp.start.seed, scenario: 'baseline'});
  let slot3 = 0;
  const t = RT.createModeTeacher(world, snap.config, variation, {mode: 'obstacle', weights: P2.weights, policySha256: P2.sha, params: {layout_id: sp.layout.id},
    policy: (o, out) => {
      const ob = Float64Array.from(o);
      if (ob[90]) slot3++;
      if (zeroSlots34) for (let k = 87; k < 95; k++) ob[k] = 0;
      const a = F2(ob);
      if (collect && ob[82] && ob[79] * 3 > .3 && ob[79] * 3 < 2.5 && Math.abs(ob[80] * 3) < .6) {
        const z = Float64Array.from(ob); for (let k = 79; k < 95; k++) z[k] = 0;
        const az = F2(z), az1 = F1(z), a1 = F1(ob);
        collect.push({l: ob[80] * 3, f: ob[79] * 3, d2: a[0] - az[0], v2: a[1] - az[1], d1: a1[0] - az1[0], v1: a1[1] - az1[1], sat: Math.abs(a[0]) > 1});
      }
      out[0] = clip1(a[0]); out[1] = clip1(a[1]); return out;
    }});
  while (!snap.state.terminal) {
    const a = t.action({state: snap.state, metrics: snap.metrics, config: snap.config});
    snap = sim.step({steering: a.steering, speed: a.speed});
  }
  const rep = sim.report();
  return {valid: rep.final_state.reason === 'collection_duration_complete' && rep.final_state.frame === 600 && rep.metrics.collisions === 0, frames: rep.final_state.frame, slot3};
}
const jobs = buildSet('FRESH'), C = [], runs = [];
for (const j of jobs) { const r = runEp(j, false, C); runs.push({id: j.id, ...r}); }
const with3 = runs.filter(r => r.slot3 > 0);
const cf = [];
for (const r of with3) { const j = jobs.find(x => x.id === r.id); const z = runEp(j, true); cf.push({id: r.id, valid: r.valid, valid_zero34: z.valid, frames: r.frames, frames_zero34: z.frames, slot3_frames: r.slot3}); }
// (A) よける向き: 障害物が左(l > 0)なら右へ(舵の差 < 0)
const away = (d, l) => Math.sign(d) === -Math.sign(l);
const sideRows = C.filter(c => Math.abs(c.l) > .05);
const med = a => { const b = [...a].sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : null; };
const bins = [[-.6, -.2], [-.2, -.05], [.05, .2], [.2, .6]].map(([lo, hi]) => {
  const rr = C.filter(c => c.l >= lo && c.l < hi);
  return {l_range_m: [lo, hi], n: rr.length, steer_diff_median_r2: med(rr.map(c => c.d2)), steer_diff_median_r1: med(rr.map(c => c.d1)), speed_diff_median_r2: med(rr.map(c => c.v2)),
    frac_steer_right_r2: rr.length ? rr.filter(c => c.d2 < 0).length / rr.length : null};
});
const res = {runs: runs.length, frames_probe: C.length,
  away_fraction_r2: sideRows.filter(c => away(c.d2, c.l)).length / sideRows.length, away_fraction_r1: sideRows.filter(c => away(c.d1, c.l)).length / sideRows.length,
  steer_diff_abs_median_r2: med(C.map(c => Math.abs(c.d2))), steer_diff_abs_median_r1: med(C.map(c => Math.abs(c.d1))),
  speed_diff_median_r2: med(C.map(c => c.v2)), speed_diff_median_r1: med(C.map(c => c.v1)), steer_raw_saturated_frac_r2: C.filter(c => c.sat).length / C.length,
  by_lateral: bins,
  slot3: {runs_with_slot3: with3.length, valid_with_slot3: with3.filter(r => r.valid).length, valid_without_slot3: runs.filter(r => !r.slot3 && r.valid).length,
    runs_without_slot3: runs.filter(r => !r.slot3).length, counterfactual_zero34_valid: cf.filter(c => c.valid_zero34).length, changed: cf.filter(c => c.valid !== c.valid_zero34), rows: cf}};
fs.writeFileSync(`${OUT}/probe_obs.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify({...res, slot3: {...res.slot3, rows: undefined}}, null, 1));
