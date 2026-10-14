#!/usr/bin/env node
// 経路いろいろ r2 の下調べ(学習しない): 方策を学習の環境(run line = 目標の線 v2 + r2 の報酬・開始の分布)で決定的に走らせ、
// 回ごとの終わり方・最小すき間・追従の誤差(開始から 3 m 以降)・12 cm 越えの割合・開始から 1.5 秒の最小すき間を数える。
// 試験・検証の組とは別の開始(学習の環境の乱数、種は --seed)。
//   node rl/line_r2/probe_env.mjs --policy <policy.json> [--run line|cond] [--episodes 120] [--seed 777] [--out x.json]
import fs from 'node:fs';
import {ModeEnv} from '../env_modes.mjs';
import * as core from '../fast_core_modes.mjs';
import {createRng} from '../runtime/rng.mjs';
import {policyFromWeights} from '../runtime/mode_runtime.mjs';

const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf(k); return i < 0 ? d : args[i + 1]; };
const weights = JSON.parse(fs.readFileSync(opt('--policy'), 'utf8')), run = opt('--run', 'line'), N = +opt('--episodes', 120), seed = +opt('--seed', 777);
const act = policyFromWeights(weights, 97), env = new ModeEnv({core, run, mode: 'eval'}), rng = createRng(seed);
const q = (a, p) => { if (!a.length) return null; const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * (b.length - 1) + .5))]; };
const rows = [], allE = [];
for (let k = 0; k < N; k++) {
  env.resetRandom(rng);
  const a = [0, 0], es = [];
  let startMin = Infinity, t = 0;
  while (!env.done) {
    act(env.obs, a);
    env.step(a[0], a[1]); t++;
    if (t <= 45) startMin = Math.min(startMin, env.aux.d_min);
    if (env.wrapper.tracker.s - env.s0 >= 3) es.push(Math.abs(env.aux.lateral_error - env.aux.e_star));
  }
  const sm = env.summary();
  allE.push(...es);
  rows.push({k, reason: sm.reason, frames: sm.frames, min_clearance: sm.min_clearance, start_min_1p5s: startMin, family: env.line?.family, c_v: env.cV,
    start: {gate: env.start.gate, off: env.start.left_offset_m, yaw: env.start.left_yaw_deg, comp: env.start.component},
    track_median: q(es, .5), track_p95: q(es, .95), gt12: es.length ? es.filter(e => e > .12).length / es.length : null, lap1: sm.laps[0]?.elapsed ?? null});
}
const ok = rows.filter(r => r.reason === 'collection_duration_complete');
const out = {policy: opt('--policy'), run, episodes: N, seed, valid: ok.length, reasons: rows.reduce((m, r) => (m[r.reason] = (m[r.reason] ?? 0) + 1, m), {}),
  min_clearance_valid: Math.min(...ok.map(r => r.min_clearance)), below3cm_valid: ok.filter(r => r.min_clearance < .03).length,
  start_min_1p5s_p5: q(rows.map(r => r.start_min_1p5s), .05), track_median: q(allE, .5), track_p95: q(allE, .95), gt12: allE.filter(e => e > .12).length / allE.length,
  lap1_mean: ok.filter(r => r.lap1).reduce((s, r) => s + r.lap1, 0) / Math.max(1, ok.filter(r => r.lap1).length),
  by_family: Object.fromEntries([...new Set(rows.map(r => r.family))].map(f => { const rs = rows.filter(r => r.family === f); return [f, {n: rs.length, valid: rs.filter(r => r.reason === 'collection_duration_complete').length,
    p95_median: q(rs.map(r => r.track_p95).filter(x => x != null), .5)}]; })), rows};
if (opt('--out')) fs.writeFileSync(opt('--out'), JSON.stringify(out, null, 1));
const {rows: _r, ...head} = out;
console.log(JSON.stringify(head));
