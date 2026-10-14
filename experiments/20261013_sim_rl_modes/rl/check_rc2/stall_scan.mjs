// 離脱復活の確認 r2: 方策が開始の所で止まったまま動かない(20 秒・衝突 0 なので valid_episode になる)開始を、記録の開始の分布の格子で探す(2026-10-14)。
//   node rl/check_rc2/stall_scan.mjs <mode recover|fastest> <shard> <nshards> <out.jsonl>
// 乱しなし・元の sim-core.js・ブラウザと同じ mode_runtime.mjs。150 フレーム(5 秒)走らせ、2〜5 秒の最大の速さ < 0.5 m/s なら「止まり」。
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, NOMINAL} from '../check_rc/run_rc.mjs';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const core = await import(`${REL}/dist/sim-core.js`);
const RT = await import(`${EXP}/web/rl/mode_runtime.mjs`);
const [mode, shard, nsh, outPath] = process.argv.slice(2);
const weights = JSON.parse(fs.readFileSync(`${EXP}/bundle/${mode}/policy.json`, 'utf8'));
const fd = fs.openSync(outPath, 'w');
let i = -1;
for (let gate = 0; gate < 33; gate++) for (const off of [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2]) for (const yaw of [-15, -10, -5, 0, 5, 10, 15]) {
  i++; if (i % +nsh !== +shard) continue;
  const seed = 2800000 + i;
  const {world, variation} = activeWorld(BASE_WORLD, gate, off, yaw);
  const sim = core.createSimulation(world, {...NOMINAL, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed, scenario: 'baseline'});
  if (snap.state.terminal) { fs.writeSync(fd, JSON.stringify({gate, off, yaw, invalid_spawn: true}) + '\n'); continue; }
  const t = RT.createModeTeacher(world, snap.config, variation, {mode, weights, policySha256: 'x', overrides: mode === 'recover' ? {perturbation: false} : {}});
  let vmax25 = 0;
  while (!snap.state.terminal && snap.state.frame < 150) {
    const out = t.action({state: snap.state, metrics: snap.metrics, config: snap.config});
    snap = sim.step({steering: out.steering, speed: out.speed});
    if (snap.state.frame >= 60) vmax25 = Math.max(vmax25, snap.state.speed);
  }
  fs.writeSync(fd, JSON.stringify({gate, off, yaw, reason: snap.state.reason, frame: snap.state.frame, vmax25: +vmax25.toFixed(3),
    stall: !snap.state.terminal && vmax25 < 0.5, x: +snap.state.x.toFixed(3), z: +snap.state.z.toFixed(3)}) + '\n');
}
fs.closeSync(fd);
