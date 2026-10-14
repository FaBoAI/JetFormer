// 図の軌跡(no_wall r2、2026-10-14): ゲート 16・舵 0.8・遅延 1(確認 r1 の衝突の形)と、標準の物理の STD を、
// 方策 × シールドの版ごとに走らせて x, z, v, シールドの作動を残す。NW2_RT・NW2_POLICY で runtime と方策を替える(run_nw2.mjs)
//   node traces_fig.mjs <出力.json> <ラベル>
import fs from 'node:fs';
import {runOne} from './run_nw2.mjs';
const [out, label] = process.argv.slice(2);
const cases = [
  {id: 'g16_s08_d1', start: {gate: 16, left_offset_m: 0, left_yaw_deg: 0, seed: 1700000}, dr: {physics: {steerRate: .8}, delay: 1}},
  {id: 'g12Lp_C2', start: {gate: 12, left_offset_m: .2, left_yaw_deg: 15, seed: 1500000}, dr: {physics: {steerRate: .8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 6.0, frictionMultiplier: .85}, delay: 1, steerGain: .94, speedGain: 1.04}},
  {id: 'std_nominal', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, dr: {}},
];
const res = {label, runs: []};
for (const c of cases) {
  const r = await runOne({mode: 'no_wall', start: c.start, dr: c.dr, keepPath: true});
  res.runs.push({id: c.id, start: c.start, dr: c.dr, collisions: r.collisions, collision: r.collision, min_clearance: r.min_clearance, shield_frames: r.shield_frames,
    valid: r.valid_episode && r.valid_laps >= 1, laps: r.laps.map(l => l.elapsed), policy_sha256: r.policy_sha256, shield: r.shield_params?.version ?? null,
    trace: r.path.map(p => ({x: +p[0].toFixed(4), z: +p[1].toFixed(4), v: +p[2].toFixed(3), sh: p[7], fb: p[9]}))});
  console.log(c.id, r.collisions, r.min_clearance, r.shield_frames);
}
fs.writeFileSync(out, JSON.stringify(res));
