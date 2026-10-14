#!/usr/bin/env node
// no_wall r3 の報告だけの調べ(2026-10-14): 記録の開始の分布の角(左右 ±0.2 m × 向き ±15°)を全ゲートで走らせ、止まり・衝突を数える
// (確認 r2 の止まりはゲート 17・右 0.2 m・左向き 15° の角だった)。標準の物理・シールド v4.1。開始のすき間 < 0.12 m の角は飛ばす。種 3,403,000 + n。
//   node r3/nw/edge_scan_nw3.mjs --policy p.json --out rows.jsonl
import fs from 'node:fs';
import {runOne, startClearance} from './run_nw3r.mjs';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const fd = fs.openSync(opt('--out'), 'w');
let n = 0;
for (let g = 0; g < 33; g++) for (const o of [-.2, .2]) for (const y of [-15, 15]) {
  const c = startClearance(g, o, y);
  if (!(c >= .12)) continue;
  const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 3403000 + n++}, policyPath: opt('--policy')});
  fs.writeSync(fd, JSON.stringify({id: `E-g${g}o${o}y${y}`, gate: g, o, y, init_clearance: c, reason: r.reason, collisions: r.collisions, valid_laps: r.valid_laps, stall: r.stall,
    stall_max_frames: r.stall_max_frames, valid_episode_r3: r.valid_episode_r3, min_clearance: r.min_clearance, shield_frames: r.shield_frames, frames: r.frames}) + '\n');
}
fs.closeSync(fd);
console.log('回', n);
