#!/usr/bin/env node
// 障害物よけ r2 の下調べ(判定に使わない): 学習の環境(run obstacle_r2)でチェックポイントの方策を走らせ、回の終わり方を
// 「足した障害物の近く / コースの近く」に分けて数える(2026-10-14 学習の段 obstacle r2)。
//   node rl/ob2/diag_terminations.mjs <policy.json> [回の数 300] [stochastic 1|0]
import fs from 'node:fs';
import * as core from '../fast_core_modes.mjs';
import {ModeEnv} from '../env_modes.mjs';
import {createRng, gaussian} from '../runtime/rng.mjs';
import {createPolicy} from '../runtime/policy_forward.mjs';
import {clearance} from '../runtime/features.mjs';
import {obstacleOnlyGeometry} from '../runtime/layouts.mjs';

const [pp, nArg, stArg] = process.argv.slice(2);
const W = JSON.parse(fs.readFileSync(pp, 'utf8')), pol = createPolicy({...W, schema: 'rl-bestlap-policy/v1'});
const N = +(nArg ?? 300), stoch = (stArg ?? '1') === '1', sig = W.log_std.map(Math.exp);
const env = new ModeEnv({core, run: 'obstacle_r2'}), rng = createRng(777), nz = createRng(31337);
const out = {}, where = {}, speedAtEnd = {}, spd = [], noObs = {n: 0, ok: 0, lap1: []};
const a = [0, 0];
for (let ep = 0; ep < N; ep++) {
  env.resetRandom(rng);
  const og = env.obstacles.length ? obstacleOnlyGeometry(env.obstacles) : null;
  while (!env.done) {
    pol(env.obs, a);
    let a0 = a[0], a1 = a[1];
    if (stoch) { a0 += sig[0] * gaussian(nz); a1 += sig[1] * gaussian(nz); }
    env.step(Math.max(-1, Math.min(1, a0)), Math.max(-1, Math.min(1, a1)));
  }
  const s = env.summary(), st = env.view.state;
  out[s.reason] = (out[s.reason] ?? 0) + 1;
  if (s.reason === 'clearance' || s.reason === 'collision') {
    const dObs = og ? clearance(og, st.x, st.z, st.yaw, .42, .19, .5) : Infinity;
    const k = `${s.reason}:${dObs < .06 ? 'added' : 'course'}`;
    where[k] = (where[k] ?? 0) + 1;
    (speedAtEnd[k] ??= []).push(+st.speed.toFixed(2));
  }
  if (!env.obstacles.length) { noObs.n++; if (s.reason === 'collection_duration_complete') noObs.ok++; if (s.laps[0]) noObs.lap1.push(+s.laps[0].elapsed.toFixed(2)); }
}
const med = x => { const b = [...x].sort((p, q) => p - q); return b.length ? b[Math.floor(b.length / 2)] : null; };
console.log(JSON.stringify({policy: pp.split('/').pop(), stochastic: stoch, N, reasons: out, where, speed_at_end_median: Object.fromEntries(Object.entries(speedAtEnd).map(([k, v]) => [k, med(v)])),
  no_obstacle: {n: noObs.n, survived: noObs.ok, lap1_median: med(noObs.lap1)}}));
