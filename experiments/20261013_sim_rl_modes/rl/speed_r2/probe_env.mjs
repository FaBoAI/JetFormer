#!/usr/bin/env node
// 速さいろいろ r2 の学習の環境の下調べ(2026-10-14)。学習しないで、初期化の候補の方策(決定的)を speed の学習の環境で走らせ、
// 報酬の内訳・終わりの理由・すき間を c_v ごとに数える(すき間の壁と 3 cm の終了が強すぎ・弱すぎでないかを見る)。
//   node rl/speed_r2/probe_env.mjs <policy.json> [回の数] [tuning の JSON]
import fs from 'node:fs';
import {ModeEnv} from '../env_modes.mjs';
import * as core from '../fast_core_modes.mjs';
import {createRng, mix32} from '../runtime/rng.mjs';
import {policyFromWeights} from '../runtime/mode_runtime.mjs';

const [pp, nS = '100', tj = '{}'] = process.argv.slice(2);
const W = JSON.parse(fs.readFileSync(pp, 'utf8')), act = policyFromWeights(W, 97);
const env = new ModeEnv({core, run: 'speed', tuning: JSON.parse(tj)});
const rng = createRng(mix32(4242, 7)), out = [], a = [0, 0];
for (let k = 0; k < +nS; k++) {
  env.resetRandom(rng);
  let barrierFrames = 0, f = 0;
  while (!env.done) {
    act(env.obs, a);
    const r = env.step(a[0], a[1]);
    f++;
    if (env.aux.d_min < env.cfg.reward.clearance_barrier_start_m) barrierFrames++;
  }
  const s = env.summary();
  out.push({c_v: env.cV, start: s.start, reason: s.reason, frames: s.frames, min_clearance: s.min_clearance, ret: s.return, parts: s.reward_parts, barrierFrames, lap1: s.laps[0]?.elapsed ?? null});
}
const by = {};
for (const r of out) { const k = r.c_v.toFixed(3); (by[k] ??= []).push(r); }
const mean = a => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
for (const [k, rs] of Object.entries(by).sort()) {
  const reasons = {}; for (const r of rs) reasons[r.reason] = (reasons[r.reason] ?? 0) + 1;
  const parts = Object.fromEntries(Object.keys(rs[0].parts).map(p => [p, +mean(rs.map(r => r.parts[p])).toFixed(3)]));
  console.log(`c_v ${k} n ${rs.length} 理由 ${JSON.stringify(reasons)} すき間 最小 ${Math.min(...rs.map(r => r.min_clearance)).toFixed(4)} 中央 ${rs.map(r => r.min_clearance).sort((a, b) => a - b)[rs.length >> 1].toFixed(4)}`
    + ` 壁のフレーム ${mean(rs.map(r => r.barrierFrames / r.frames)).toFixed(3)} 報酬 ${mean(rs.map(r => r.ret)).toFixed(2)} 内訳 ${JSON.stringify(parts)}`);
}
const all = out.length, col = out.filter(r => r.reason === 'collision').length, clr = out.filter(r => r.reason === 'clearance').length;
console.log(`全部 ${all} 衝突 ${col} すき間で終了 ${clr} 走り切り ${out.filter(r => r.reason === 'collection_duration_complete').length}`);
