#!/usr/bin/env node
// r3 no_wall: シールド v4.1(r3/nw/runtime_v4/shield.mjs)の試験(2026-10-14)。
//   (1) v3 の値 {c_nominal 0.07, worst 切, weak_extra 1, brake_without_known 切, brake_probe 切} を渡すと、web/rl(jf-shield/v3)と操作・シールドのフレームが全く同じ
//   (1b) v4.1 の 2 つの鍵(brake_without_known・brake_probe)を切ると、追補の決まりのとおりの v4(r3/nw/rt_v4_as_frozen)と全く同じ
//   (2) 既定の値が protocol_no_wall_r3.json runtime.shield.params と同じ(v4.1 で足した 2 つの鍵を除く)
//   (3) 既定(v4.1)で、確認 r2 の隅 K0 の 3 ゲートと、K0 + 遅延 1(K7)の 3 ゲートが衝突しない
//   node r3/nw/test_shield_v4.mjs [--policy 方策(3) に使う。既定は bundle/no_wall = r2 の方策)]
import fs from 'node:fs';
import {runOne} from './run_nw3r.mjs';
import {SHIELD_DEFAULTS} from './runtime_v4/shield.mjs';
const E = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const PROT = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_no_wall_r3.json', 'utf8'));
const PA = process.argv.indexOf('--policy'), POL = PA >= 0 ? process.argv[PA + 1] : null;
const V3 = {c_nominal: .07, worst: {enabled: false}, governor: {weak_extra: 1, brake_without_known: false, brake_probe: {enabled: false}}};
const V4 = {governor: {brake_without_known: false, brake_probe: {enabled: false}}};
const K0 = {physics: {steerRate: 1.8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 3.5, frictionMultiplier: .85}, delay: 0, steerGain: 1.06, speedGain: 1.04, frictionCircle: true};
const cases = [
  {start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}},
  {start: {gate: 9, left_offset_m: .11, left_yaw_deg: -14.8, seed: 5}},
  {start: {gate: 16, left_offset_m: -.1, left_yaw_deg: 10, seed: 7}, dr: {physics: {steerRate: .8, brakeDeceleration: .9}, delay: 1, steerGain: .96, speedGain: 1.02, frictionCircle: false}},
  {start: {gate: 5, left_offset_m: 0, left_yaw_deg: 0, seed: 9}, dr: K0},
  {start: {gate: 20, left_offset_m: .05, left_yaw_deg: 4, seed: 11}, dr: {physics: {}, delay: 0, frictionCircle: true}},
];
let fail = 0;
const ok = (c, msg) => { console.log((c ? 'OK   ' : 'FAIL ') + msg); if (!c) fail++; };
for (const [i, c] of cases.entries()) {
  const a = await runOne({mode: 'no_wall', ...c, rtDir: `${E}/web/rl`, keepPath: true});
  const b = await runOne({mode: 'no_wall', ...c, rtDir: `${E}/r3/nw/runtime_v4`, shieldParams: V3, keepPath: true});
  const same = a.path.length === b.path.length && a.path.every((p, k) => p.every((v, j) => v === b.path[k][j]));
  ok(same && a.shield_frames === b.shield_frames && a.collisions === b.collisions, `(1) 例 ${i}: v3 の値の v4.1 = web/rl の v3(フレーム ${a.path.length}・シールド ${a.shield_frames}/${b.shield_frames}・衝突 ${a.collisions}/${b.collisions})`);
  const x = await runOne({mode: 'no_wall', ...c, rtDir: `${E}/r3/nw/rt_v4_as_frozen`, keepPath: true});
  const y = await runOne({mode: 'no_wall', ...c, rtDir: `${E}/r3/nw/runtime_v4`, shieldParams: V4, keepPath: true});
  const same2 = x.path.length === y.path.length && x.path.every((p, k) => p.every((v, j) => v === y.path[k][j]));
  ok(same2 && x.shield_frames === y.shield_frames, `(1b) 例 ${i}: 鍵を切った v4.1 = 決まりの v4(フレーム ${x.path.length}・シールド ${x.shield_frames}/${y.shield_frames}・衝突 ${x.collisions}/${y.collisions})`);
}
const P = PROT.runtime.shield.params;
const D = SHIELD_DEFAULTS;
const eq = (x, y) => JSON.stringify(x) === JSON.stringify(y);
ok(D.version === 'jf-shield/v4.1', `(2) 版 ${D.version}(決まりは ${PROT.runtime.shield.version}、v4.1 は deviations.md)`);
for (const k of ['H', 'c_nominal', 'c_pessimistic', 'keep_nominal']) ok(D[k] === P[k], `(2) ${k} = ${D[k]}`);
ok(eq(D.steer_set, P.steer_set), '(2) steer_set');
for (const k of Object.keys(P.pessimistic)) ok(D.pessimistic[k] === P.pessimistic[k], `(2) pessimistic.${k} = ${D.pessimistic[k]}`);
for (const k of Object.keys(P.identify)) ok(eq(D.identify[k], P.identify[k]), `(2) identify.${k}`);
for (const k of Object.keys(P.governor)) ok(D.governor[k] === P.governor[k], `(2) governor.${k} = ${D.governor[k]}`);
for (const k of Object.keys(P.worst)) ok(D.worst[k] === P.worst[k], `(2) worst.${k} = ${D.worst[k]}`);
ok(D.unstick.enabled === false, '(2) unstick 切');
ok(D.governor.brake_without_known === true && D.governor.brake_probe.enabled === true && D.governor.brake_probe.v_min === 2 && D.governor.brake_probe.factor === .85 && D.governor.brake_probe.every === 15, '(2) v4.1 の鍵');
for (const [nm, dr] of [['K0', K0], ['K7', {...K0, delay: 1}]]) for (const g of [2, 10, 17]) {
  const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: 0, left_yaw_deg: 0, seed: 2850000 + g}, dr, ...(POL ? {policyPath: POL} : {})});
  ok(r.collisions === 0, `(3) 隅 ${nm}・ゲート ${g}: 衝突 ${r.collisions}・最小 ${(r.min_clearance * 100).toFixed(2)} cm・作動 ${r.shield_frames}`);
}
console.log(fail ? `失敗 ${fail}` : '全部合格');
process.exit(fail ? 1 : 0);
