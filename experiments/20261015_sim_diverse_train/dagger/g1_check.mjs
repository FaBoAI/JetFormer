#!/usr/bin/env node
// DAgger の G1(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.go_no_go_gates.G1_node):
// 評価(単周)で記録した操作を、Node の元の sim-core.js の「連続周回」(記録と同じ continuous-laps-v1・20 秒)で再生すると、
// フレーム k までの位置が単周の記録と ≤ 1e-6 m で同じか。
//   node g1_check.mjs <走行の JSON(ブラウザのアプリの結果)> [k] [出力.json]
// k を省くと、走行の最後のフレーム(止まる直前)まで比べる。
import fs from 'node:fs';
import {createSimulation} from '../lib/app_emulation.mjs';

const [file, kArg, outPath] = process.argv.slice(2);
const r = JSON.parse(fs.readFileSync(file, 'utf8'));
const n = r.actions.length;
const k = kArg != null && kArg !== '-' ? Math.min(Number(kArg), n) : n;
const cfg = {...r.config, episodeMode: 'continuous-laps-v1', timeLimit: 20};
const sim = createSimulation(r.world, cfg);
let snap = sim.reset({seed: r.seed, scenario: r.scenario});
let maxd = 0, f = 0;
const tr = r.trace;
for (; f < k; f++) {
  if (snap.state.terminal) break;
  // 記録の trace[f] は step(actions[f]) のあとの状態(verify と同じ並び)。step してから比べる
  snap = sim.step({steering: r.actions[f].steering, speed: r.actions[f].speed});
  const t = tr[f];
  maxd = Math.max(maxd, Math.abs(snap.state.x - t.x), Math.abs(snap.state.z - t.z));
}
const out = {schema: 'jf-sim-diverse-dagger-g1/v1', file, frames_single: n, k, compared_frames: f, max_abs_xz_diff_m: maxd,
  pass: f === k && maxd <= 1e-6, continuous_terminal_before_k: snap.state.terminal && f < k, rule: '≤ 1e-6 m(フレーム k まで)'};
if (outPath) fs.writeFileSync(outPath, JSON.stringify(out, null, 1) + '\n');
console.log(JSON.stringify(out));
