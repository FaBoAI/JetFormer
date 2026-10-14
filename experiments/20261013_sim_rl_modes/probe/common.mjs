// 理解の段の小さな確かめ(2026-10-13、sim_rl_modes)。ベストラップの作業のコードは読むだけ(import)で、変えない。
// ここは probe の共通部分: ベストラップの環境・特徴・方策を読み込み、方策で 1 エピソードを走らせる小道具。
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

export const BESTLAP = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap';
export const BESTLAP_REPORTS = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013';
export const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/probe';
export const POLICY_JSON = path.join(BESTLAP_REPORTS, 'rl/best/main/policy.json');

export const envCore = await import(path.join(BESTLAP, 'rl/env_core.mjs'));
export const features = await import(path.join(BESTLAP, 'rl/features.mjs'));
export const policyForward = await import(path.join(BESTLAP, 'rl/policy_forward.mjs'));

export const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export function loadBestPolicy() {
  const w = JSON.parse(fs.readFileSync(POLICY_JSON, 'utf8'));
  return {act: policyForward.createPolicy(w), sha256: sha256File(POLICY_JSON)};
}
export function writeJson(name, obj) {
  fs.mkdirSync(OUT, {recursive: true});
  const p = path.join(OUT, name);
  fs.writeFileSync(p, JSON.stringify(obj, null, 1));
  return p;
}
export function cloneWorld(w) { return JSON.parse(JSON.stringify(w)); }

// 方策で 1 エピソード(eval モード = sim の終了だけ)。hook(env, frame, a) で行動を差し替え・状態を変えられる
//   hook は {a0, a1} を返せば行動を置き換える。null なら方策の行動のまま
export async function runPolicyEpisode({act, baseWorld = null, start = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1},
  episodeMode = 'continuous', timeLimit = 20, coreKind = 'fast', hook = null, obsHook = null, physics = null, delay = 0}) {
  const core = await envCore.loadCore(coreKind);
  const env = new envCore.RlEnv({core, coreKind, baseWorld: baseWorld ?? envCore.loadBaseWorld(), variant: 'main', mode: 'eval', episodeMode, timeLimit});
  const perturb = physics || delay ? {physics: physics ?? {}, delay, steerGain: 1, speedGain: 1} : null;
  if (!env.reset({...start, perturb})) return {invalid_spawn: true};
  const out = [0, 0], obs2 = new Float64Array(features.OBS_DIM);
  const trace = [];
  while (!env.done) {
    let o = env.obs;
    if (obsHook) { obs2.set(env.obs); obsHook(env, obs2); o = obs2; }
    act(o, out);
    let a0 = out[0], a1 = out[1];
    const h = hook ? hook(env, env.view.state.frame, [a0, a1]) : null;
    if (h) { a0 = h.a0; a1 = h.a1; }
    const r = env.step(a0, a1);
    trace.push({f: env.view.state.frame, x: env.view.state.x, z: env.view.state.z, v: env.view.state.speed, ey: env.aux.lateral_error,
      dmin: env.aux.d_min, ri: env.view.metrics.route_index, a0, a1});
    if (r.terminated || r.truncated) break;
  }
  const s = env.summary();
  s.valid_laps = (s.laps ?? []).filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
  return {summary: s, trace, env};
}
export const quantiles = (arr, qs = [0, .05, .25, .5, .75, .95, 1]) => {
  const b = [...arr].sort((x, y) => x - y);
  return Object.fromEntries(qs.map(q => [String(q), b.length ? b[Math.min(b.length - 1, Math.floor(q * (b.length - 1) + .5))] : null]));
};
