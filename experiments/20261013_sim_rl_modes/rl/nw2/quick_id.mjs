// v3 の見積もりが正しいか(r1 の方策・いろいろな物理で、見積もりの値を最後のフレームで見る)
import {runOne} from './run_nw2.mjs';
const cases = [['nom', {}], ['s08d1', {physics: {steerRate: .8}, delay: 1}], ['s15d0g106', {physics: {steerRate: 1.5}, steerGain: 1.06}], ['s09d1g094b07', {physics: {steerRate: .9, brakeDeceleration: .7}, delay: 1, steerGain: .94, speedGain: 1.04}],
  ['fc', {physics: {}, delay: 0, frictionCircle: true}], ['d2', {physics: {steerRate: 1.0}, delay: 2}]];
for (const [label, dr] of cases) for (const [g, o, y] of [[0, 0, 0], [16, 0, 0], [21, .08, 12]]) {
  const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 1810000 + g}, dr, keepPath: true});
  console.log(JSON.stringify({label, g, col: r.collisions, min: +(r.min_clearance * 100).toFixed(2), sh: r.shield_frames, first: r.first_shield_frame, fb: r.labels.fb}));
}
