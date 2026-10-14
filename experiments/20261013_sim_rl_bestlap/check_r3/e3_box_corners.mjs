#!/usr/bin/env node
// 確認 r3 の 3(新しい攻め方): 学習の範囲の「箱の角」をすべて。r3 の段が試したのは 舵 × ブレーキの平面(ほかは標準)と、悪い側をすべて重ねた 1 点だけ。
// 5 つの物理の値(舵の速さ・ブレーキ・時定数・加速の上限・μ)をそれぞれ範囲の下の端か上の端ちょうどにした 32 の角 × 遅延 0/1 = 64 条件を、
// STD の単周と REC60 の開始(既定 60。--every n で n 本おき)で走らせる。どれも学習の範囲の中(r3 の角・辺の回が引く場所)。
//   node check_r3/e3_box_corners.mjs <方策> [--every 1]
import fs from 'node:fs';
import {PLAN, loadPolicy, drive, save} from './common_r3.mjs';

const args = process.argv.slice(2), key = args[0] ?? 'r3_s21';
const every = +(args[args.indexOf('--every') + 1] ?? 1) || 1;
const DR = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/r3-s21/run.json', 'utf8')).env.dr_spec;
const PK = ['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier'];
// 「悪い側」= 舵が遅い・ブレーキが弱い・時定数が長い・加速の上限が低い・μ が低い(= 各値の下の端。時定数だけ上の端)
const BAD = {steerRate: 0, brakeDeceleration: 0, driveTau: 1, driveAccelerationCap: 0, frictionMultiplier: 0};
const pol = loadPolicy(key), rows = PLAN.filter((_, i) => i % every === 0);
const out = {schema: 'rl-check-r3-box-corners/v1', created_at: new Date().toISOString(), policy: key, sha256: pol.sha, every, n_starts: rows.length, dr_spec: DR, cells: []};
const t0 = Date.now();
for (let mask = 0; mask < 32; mask++) for (const delay of [0, 1]) {
  const phys = {}, bad = [];
  PK.forEach((k, b) => { const isBad = !!(mask >> b & 1); phys[k] = DR[k][isBad ? BAD[k] : 1 - BAD[k]]; if (isBad) bad.push(k); });
  const std = drive(pol, {phys, delay}).judge;
  let ok = 0; const fails = [];
  for (const row of rows) { const j = drive(pol, {row, continuous: true, phys, delay}).judge; if (j.ok) ok++; else fails.push({name: row.name, reason: j.reason, collider: j.first_collision?.collider ?? null}); }
  out.cells.push({mask, delay, bad, n_bad: bad.length + delay, phys, std_ok: std.ok, std_lap: std.laps[0] ?? null, std_reason: std.reason, rec_valid: ok, rec_n: rows.length,
    fail_reasons: fails.reduce((a, f) => (a[f.reason] = (a[f.reason] ?? 0) + 1, a), {})});
  const c = out.cells.at(-1);
  console.log(key, `${String(mask).padStart(2)} d${delay}`, bad.join('+') || '(良い側)', c.std_ok ? c.std_lap.toFixed(3) : c.std_reason, `${ok}/${rows.length}`, JSON.stringify(c.fail_reasons));
  save(`e3_box_${key}.json`, out);
}
out.wall_s = (Date.now() - t0) / 1000;
out.summary = {cells_all_valid: out.cells.filter(c => c.rec_valid === c.rec_n).length, cells: out.cells.length,
  total_valid: out.cells.reduce((a, c) => a + c.rec_valid, 0), total_n: out.cells.reduce((a, c) => a + c.rec_n, 0),
  std_ok: out.cells.filter(c => c.std_ok).length, worst: [...out.cells].sort((a, b) => a.rec_valid - b.rec_valid).slice(0, 10).map(c => ({bad: c.bad, delay: c.delay, rec: `${c.rec_valid}/${c.rec_n}`, std: c.std_ok}))};
save(`e3_box_${key}.json`, out);
console.log('summary', JSON.stringify(out.summary));
