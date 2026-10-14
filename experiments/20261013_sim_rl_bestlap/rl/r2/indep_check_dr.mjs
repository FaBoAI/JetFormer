#!/usr/bin/env node
// r2: r2 の方策(物理のずれのランダム化)の主な値を、確認 r1 の独立な道具(ユーザーの元の dist/sim-core.js・scene.js のコース・app.js の式)で走らせ直す。
// RL の env(env_core.mjs)は通さない。観測の定義(features.mjs)だけは共有。遅延 1 の約束は確認 r1 と同じ(観測の前回の行動 = 方策が今出した値)。
//   node rl/r2/indep_check_dr.mjs --policy <policy.json> --out <json>
import fs from 'node:fs';
import {runClosedLoop, PLAN60} from '../../check_r1/common_r1.mjs';
import {loadPolicy} from '../policy_forward.mjs';
const args = process.argv.slice(2);
const opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const fn = await loadPolicy(opt('--policy'));
const policy = obs => fn(obs);
const rows = [];
const one = (name, o) => {
  const r = {name};
  for (const mode of ['single', 'continuous']) { const x = runClosedLoop(policy, {episodeMode: mode, keepReport: false, ...o}).summary; r[mode] = {valid: x.valid, reason: x.reason, laps: x.laps.map(l => +l.elapsed.toFixed(6))}; }
  rows.push(r); console.log(name, JSON.stringify(r.single), JSON.stringify(r.continuous));
};
one('標準', {});
one('遅延 1(方策の出力の約束)', {delay: 1});
one('遅延 1 + 前進ゲイン 1.01', {delay: 1, speedGain: 1.01});
one('舵の速さ 1.0', {simOverrides: {steerRate: 1.0}});
one('舵の速さ 1.5', {simOverrides: {steerRate: 1.5}});
one('ブレーキ 1.0', {simOverrides: {brakeDeceleration: 1.0}});
one('μ × 0.9', {simOverrides: {frictionMultiplier: .9}});
const rec = [];
for (const delay of [0, 1]) {
  const res = PLAN60.map(c => runClosedLoop(policy, {plan: c, gate: c['start-gate'], offset: c['spawn-offset'], yawDeg: c['spawn-yaw'], episodeMode: 'continuous', delay, keepReport: false}).summary);
  const ok = res.filter(r => r.valid), m = a => a.reduce((x, y) => x + y, 0) / a.length;
  rec.push({delay, valid: ok.length, n: res.length, lap1_mean: m(ok.map(r => r.laps[0].elapsed)), lap2_mean: m(ok.map(r => r.laps[1].elapsed))});
  console.log('REC60 delay', delay, ok.length, '/60', rec.at(-1).lap1_mean, rec.at(-1).lap2_mean);
}
const out = opt('--out');
if (out) fs.writeFileSync(out, JSON.stringify({schema: 'rl-bestlap-r2-indep-check/v1', created_at: new Date().toISOString(), policy: opt('--policy'),
  tools: '確認 r1 の check_r1/common_r1.mjs(ユーザーの元の sim-core.js・scene.js のコース・app.js の式)', rows, rec60: rec}, null, 1) + '\n');
