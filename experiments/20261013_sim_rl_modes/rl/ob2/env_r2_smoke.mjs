#!/usr/bin/env node
// 障害物よけ r2 の学習の環境の試し(2026-10-14 学習の段 obstacle r2)。run 'obstacle_r2' で、
//  (1) 開始のすき間 ≥ 4 cm・開始の前後 1.5 m に足した障害物が無い(試験の startClearOfObstacles と同じ)・配置の数の分布、
//  (2) d_min < 3 cm で 'clearance' で終わる・近づく速さの罰・壁の罰が出る、を確かめる。
//   node rl/ob2/env_r2_smoke.mjs [env_modes.mjs のパス(既定 rl/env_modes.mjs)]
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import * as core from '../fast_core_modes.mjs';
import {createRng} from '../runtime/rng.mjs';
import {projectWindow, LOOP_LENGTH} from '../runtime/features.mjs';
import {startClearOfObstacles, obstacleS} from '../runtime/layouts.mjs';

const here = path.dirname(fileURLToPath(import.meta.url)), rl = path.resolve(here, '..');
let modPath = path.join(rl, 'env_modes.mjs'), tmp = null;
if (process.argv[2]) { tmp = path.join(rl, `.env_modes_new_${process.pid}.mjs`); fs.copyFileSync(process.argv[2], tmp); modPath = tmp; }
try {
  const E = await import(pathToFileURL(modPath).href);
  const env = new E.ModeEnv({core, run: process.argv[3] ?? 'obstacle_r2'});
  const rng = createRng(4242), act = createRng(7);
  const counts = [0, 0, 0, 0, 0], reasons = {}, parts = {}, minGap = [];
  let bad = 0, redraws = 0, n = 0, steps = 0;
  for (let ep = 0; ep < 300; ep++) {
    redraws += env.resetRandom(rng);
    n++;
    const sp = projectWindow(env.view.state.x, env.view.state.z, 0, LOOP_LENGTH).s;
    if (!startClearOfObstacles(sp, env.obstacles)) bad++;
    if (env.aux.d_min < .04) bad++;
    counts[env.obstacles.length]++;
    for (const o of env.obstacles) { let d = ((obstacleS(o) - sp) % LOOP_LENGTH + LOOP_LENGTH) % LOOP_LENGTH; if (d > LOOP_LENGTH / 2) d -= LOOP_LENGTH; minGap.push(Math.abs(d)); }
    // 方策の代わりに: まっすぐ速め(衝突・3 cm の終わりが起きる)と、ゆらぐ舵
    const style = ep % 3;
    for (let k = 0; k < 600 && !env.done; k++) {
      const a0 = style === 0 ? 0 : (act.uniform() - .5) * .6, a1 = style === 2 ? -.2 : .6;
      env.step(a0, a1); steps++;
    }
    const s = env.summary();
    reasons[s.reason] = (reasons[s.reason] ?? 0) + 1;
    for (const [k, v] of Object.entries(s.reward_parts)) parts[k] = (parts[k] ?? 0) + v / 300;
  }
  console.log(JSON.stringify({episodes: n, steps, start_rule_violations: bad, redraws, obstacle_count_hist: counts, min_abs_ds_obstacle_start: Math.min(...minGap),
    reasons, reward_parts_mean: Object.fromEntries(Object.entries(parts).map(([k, v]) => [k, +v.toFixed(3)])), cfg: env.cfg}, null, 1));
} finally { if (tmp) fs.rmSync(tmp, {force: true}); }
