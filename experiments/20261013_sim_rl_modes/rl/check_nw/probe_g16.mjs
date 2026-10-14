// 確認 r1: 舵の速さ × 遅延 1 の小さな格子(ゲート 16 と STD)。ほかの物理は標準。シールドあり
import fs from 'node:fs';
import {runOne} from './run_nw.mjs';
const out = [];
let n = 0;
for (const sr of [.8, .82, .85, .9, 1.0]) for (const [g, o, y] of [[16, 0, 0], [16, -.05, 0], [16, .05, 0], [16, 0, -3], [16, 0, 3], [15, 0, 0], [17, 0, 0], [0, 0, 0]]) {
  const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 1700000 + n++}, dr: {physics: {steerRate: sr}, delay: 1}});
  out.push({sr, g, o, y, col: r.collisions, t: r.collision?.time ?? null, kind: r.collision?.kind ?? null, min: Math.max(0, r.min_clearance), sh: r.shield_frames, valid: r.valid_episode && r.valid_laps >= 1});
  console.log(JSON.stringify(out.at(-1)));
}
fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 1));
