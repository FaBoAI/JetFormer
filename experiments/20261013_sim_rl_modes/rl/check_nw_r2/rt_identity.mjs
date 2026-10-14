// 確認 r2: 書き出したブラウザの組 web/rl(manifest 1f0ba8f1…、試験に使ったもの)と、今の rl/runtime(試験のあとで line r2 が mode_runtime.mjs・obs_modes.mjs・
// line_profiles.mjs を直した)で、no_wall の走り(操作・シールドの作動・軌跡)が同じかを比べる。統合の段が rl/runtime から組を作り直しても no_wall が変わらないことの確かめ
import {runOne} from './run_nw_r2.mjs';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const specs = [
  {start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 2990001}},
  {start: {gate: 16, left_offset_m: .05, left_yaw_deg: 3, seed: 2990002}, dr: {physics: {steerRate: .8}, delay: 1}},
  {start: {gate: 23, left_offset_m: -.07, left_yaw_deg: 12, seed: 2990003}},
  {start: {gate: 9, left_offset_m: .15, left_yaw_deg: -12, seed: 2990004}, dr: {physics: {steerRate: 1.0, brakeDeceleration: .75, frictionMultiplier: .87}, delay: 1, steerGain: .95, speedGain: 1.03, frictionCircle: true}},
  {start: {gate: 30, left_offset_m: 0, left_yaw_deg: -8, seed: 2990005}, dr: {physics: {steerRate: 1.75, driveTau: .3}, delay: 0, steerGain: 1.05}},
];
let worst = 0;
for (const s of specs) {
  const a = await runOne({mode: 'no_wall', ...s, keepPath: true, rtDir: `${EXP}/web/rl`});
  const b = await runOne({mode: 'no_wall', ...s, keepPath: true, rtDir: `${EXP}/rl/runtime`});
  let d = 0;
  for (let i = 0; i < Math.max(a.path.length, b.path.length); i++) { const p = a.path[i], q = b.path[i]; if (!p || !q) { d = Infinity; break; } for (let k = 0; k < 8; k++) d = Math.max(d, Math.abs(p[k] - q[k])); }
  worst = Math.max(worst, d);
  console.log(JSON.stringify({start: s.start, frames: [a.frames, b.frames], sh: [a.shield_frames, b.shield_frames], col: [a.collisions, b.collisions], max_abs_diff: d}));
}
console.log(JSON.stringify({worst}));
