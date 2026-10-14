import {runOne} from './run_nw2.mjs';
const out = [];
let n = 0, col = 0, mins = [];
for (const sr of [.8, .82, .85, .9, 1.0]) for (const [g, o, y] of [[16, 0, 0], [16, -.05, 0], [16, .05, 0], [16, 0, -3], [16, 0, 3], [15, 0, 0], [17, 0, 0], [0, 0, 0]]) {
  const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 1700000 + n++}, dr: {physics: {steerRate: sr}, delay: 1}});
  col += r.collisions ? 1 : 0; mins.push(Math.max(0, r.min_clearance));
  console.log(JSON.stringify({sr, g, o, y, col: r.collisions, t: r.collision?.time ?? null, min: +(Math.max(0, r.min_clearance) * 100).toFixed(2), sh: r.shield_frames, valid: r.valid_episode && r.valid_laps >= 1, lap1: r.laps[0]?.elapsed}));
}
console.log('collisions', col, 'min', Math.min(...mins));
