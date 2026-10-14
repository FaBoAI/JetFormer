// 確認 r2 の最初の確かめ: VX・FD3・LONG の 1 回ずつの時間と中身
import {runOne} from './run_nw_r2.mjs';
const specs = [
  {mode: 'no_wall', start: {gate: 16, left_offset_m: 0, left_yaw_deg: 0, seed: 2999001}, dr: {physics: {steerRate: .8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 6, frictionMultiplier: 1.1}, delay: 1, steerGain: 1.06, speedGain: 1.04}},
  {mode: 'no_wall', start: {gate: 5, left_offset_m: .1, left_yaw_deg: -10, seed: 2999002}, dr: {physics: {steerRate: 1.8}, delay: 0, frictionCircle: true}},
  {mode: 'no_wall', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 2999003}, timeLimit: 60},
];
for (const s of specs) { const t = Date.now(); const r = await runOne(s); console.log(JSON.stringify({wall: (Date.now() - t) / 1000, reason: r.reason, frames: r.frames, col: r.collisions, min: r.min_clearance, sh: r.shield_frames, laps: r.valid_laps, lab: r.labels, err: r.audit_pos_err, ov: r.overlap_frames})); }
