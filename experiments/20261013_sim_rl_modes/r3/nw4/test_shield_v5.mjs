#!/usr/bin/env node
// no_wall r4: シールド v5(r3/nw4/runtime_v5/shield.mjs)の試験(2026-10-14)。
//   (1) v4.1 と同じにする鍵の組 {c_nominal 0.08, stop 切, select 'v4', cont_candidates 切, escape 切} を渡すと、r3 の候補の runtime(r3/nw/runtime_v4 = jf-shield/v4.1)と
//       操作・シールドのフレームが全く同じ(標準・遅延 1 の遅い舵・隅 K0・摩擦円・確認 r3 の 2 つの崩れの開始)
//   (2) 近くの障害物の形の作り直し(dyn_r 1.6 m)が、全部の障害物で測るの(dyn_r 100)と操作まで全く同じ
//   (3) 既定の値が protocol_no_wall_r4.json runtime.shield.params と同じ(--protocol-only はこれだけ)
//   (4) 既定(v5)で、確認 r3 の崩れ: ゲート 20・−0.13 m・−13°(v4.1 は衝突)とゲート 20・−0.124 m・−13°(5.63 cm)が衝突せず ≥ 8 cm、
//       ゲート 23・+0.05 m・+12°(止まり)が止まらず有効な周 ≥ 1。隅 K0・K7 の 3 ゲートで衝突なし
//   node r3/nw4/test_shield_v5.mjs [--policy 方策] [--protocol-only]
import fs from 'node:fs';
import {SHIELD_DEFAULTS} from './runtime_v5/shield.mjs';
const E = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const PP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_no_wall_r4.json';
const A = process.argv, PA = A.indexOf('--policy'), POL = PA >= 0 ? A[PA + 1] : `${E}/r3/candidates/no_wall_r3/policy.json`;
let fail = 0;
const ok = (c, msg) => { console.log((c ? 'OK   ' : 'FAIL ') + msg); if (!c) fail++; };
const eq = (x, y) => JSON.stringify(x) === JSON.stringify(y);
// (3)
if (fs.existsSync(PP)) {
  const P = JSON.parse(fs.readFileSync(PP, 'utf8')).runtime.shield;
  const {version, ...D} = SHIELD_DEFAULTS;
  ok(version === P.version, `(3) 版 ${version} = 決まり ${P.version}`);
  ok(eq(JSON.parse(JSON.stringify(D)), P.params), '(3) 既定の値 = protocol_no_wall_r4.json runtime.shield.params(全部の鍵)');
} else ok(false, '(3) protocol_no_wall_r4.json がありません');
if (A.includes('--protocol-only')) { console.log(fail ? `失敗 ${fail}` : '全部合格'); process.exit(fail ? 1 : 0); }
const {runOne} = await import('./run_nw4r.mjs');
const V41 = {c_nominal: .08, stop: {enabled: false}, select: 'v4', cont_candidates: false, escape: {enabled: false}};
const K0 = {physics: {steerRate: 1.8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 3.5, frictionMultiplier: .85}, delay: 0, steerGain: 1.06, speedGain: 1.04, frictionCircle: true};
const cases = [
  {start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}},
  {start: {gate: 9, left_offset_m: .11, left_yaw_deg: -14.8, seed: 5}},
  {start: {gate: 16, left_offset_m: -.1, left_yaw_deg: 10, seed: 7}, dr: {physics: {steerRate: .8, brakeDeceleration: .9}, delay: 1, steerGain: .96, speedGain: 1.02, frictionCircle: false}},
  {start: {gate: 5, left_offset_m: 0, left_yaw_deg: 0, seed: 9}, dr: K0},
  {start: {gate: 20, left_offset_m: .05, left_yaw_deg: 4, seed: 11}, dr: {physics: {}, delay: 0, frictionCircle: true}},
  {start: {gate: 20, left_offset_m: -.13, left_yaw_deg: -13, seed: 1}},
  {start: {gate: 23, left_offset_m: .05, left_yaw_deg: 12, seed: 1}},
];
const samePath = (a, b) => a.path.length === b.path.length && a.path.every((p, k) => p.every((v, j) => v === b.path[k][j]));
for (const [i, c] of cases.entries()) {
  const a = await runOne({mode: 'no_wall', ...c, policyPath: POL, rtDir: `${E}/r3/nw/runtime_v4`, keepPath: true});
  const b = await runOne({mode: 'no_wall', ...c, policyPath: POL, rtDir: `${E}/r3/nw4/runtime_v5`, shieldParams: V41, keepPath: true});
  ok(samePath(a, b) && a.shield_frames === b.shield_frames && a.collisions === b.collisions,
    `(1) 例 ${i}: v4.1 の鍵の v5 = r3 の v4.1(フレーム ${a.path.length}・シールド ${a.shield_frames}/${b.shield_frames}・衝突 ${a.collisions}/${b.collisions})`);
  const x = await runOne({mode: 'no_wall', ...c, policyPath: POL, keepPath: true});
  const y = await runOne({mode: 'no_wall', ...c, policyPath: POL, keepPath: true, shieldParams: {dyn_r: 100}});
  ok(samePath(x, y) && x.shield_frames === y.shield_frames, `(2) 例 ${i}: 近くの形の作り直し = 全部の障害物(シールド ${x.shield_frames}/${y.shield_frames}・最小 ${(100 * x.min_clearance).toFixed(3)} cm)`);
}
for (const [g, o, y, what] of [[20, -.13, -13, '確認 r3 の衝突'], [20, -.124, -13, '確認 r3 の 5.63 cm'], [23, .05, 12, '確認 r3 の止まり(試験 GR-g23o0.05y12)']]) {
  const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 1}, policyPath: POL});
  ok(r.collisions === 0 && r.min_clearance >= .08 && !r.stall && r.valid_episode_r3,
    `(4) ${what}: 衝突 ${r.collisions}・最小 ${(100 * r.min_clearance).toFixed(2)} cm・止まり ${r.stall}・有効な周 ${r.valid_laps}・抜け出し ${r.escapes}・後ろへ ${r.back_frames} フレーム`);
}
for (const [nm, dr] of [['K0', K0], ['K7', {...K0, delay: 1}]]) for (const g of [2, 10, 17]) {
  const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: 0, left_yaw_deg: 0, seed: 2850000 + g}, dr, policyPath: POL});
  ok(r.collisions === 0 && r.min_clearance >= .05, `(4) 隅 ${nm}・ゲート ${g}: 衝突 ${r.collisions}・最小 ${(r.min_clearance * 100).toFixed(2)} cm・作動 ${r.shield_frames}`);
}
console.log(fail ? `失敗 ${fail}` : '全部合格');
process.exit(fail ? 1 : 0);
