#!/usr/bin/env node
// 確認 r2 の 3: 物理のずれへの強さの主張(条件表 39・ランダムな物理 100 × 2)を、自前の道具と自前の乱数で作り直す。
//   node check_r2/e3_physics.mjs <grid|random|rec60x|starts> <方策の名前,…|teacher>
// - grid   : RL の段の条件表(rl/r2/robust_eval.mjs の GRID と同じ 39 条件)を STD の開始で、単周と連続 20 秒。遅延は「適用した行動」の約束。
//            遅延のある条件は「方策の出力」の約束でも走らせる。
// - random : 学習の範囲(run.json の dr_spec)と広い範囲から、自前の乱数(mulberry32、RL の段と別の種)で 100 条件ずつ。連続 20 秒
// - rec60x : 記録の 60 条件の開始 × 学習の範囲のランダムな物理(開始と物理の両方がずれる。RL の段は試していない組み合わせ)
// - starts : 開始の姿勢(ゲート 0 の横 ±15 cm・向き ±10° の 25 条件と、全 33 ゲートからの開始)。標準の物理・連続 20 秒
import fs from 'node:fs';
import {PLAN60, REPORTS, loadMlp, runPolicy, runTeacher, writeJson, rng32} from './common_r2.mjs';

const [set = 'grid', who = 'selected,r2_5M,teacher'] = process.argv.slice(2);
const DR = JSON.parse(fs.readFileSync(REPORTS + '/rl/r2-dr-s11/run.json', 'utf8')).env.dr_spec;   // 学習で使った範囲(run.json の記録)
const WIDE = {steerRate: [.7, 2.0], brakeDeceleration: [.6, 2.5], driveTau: [.25, .6], driveAccelerationCap: [3, 6], frictionMultiplier: [.8, 1.12],
  delay1_fraction: .5, steerGain: [.93, 1.07], speedGain: [.95, 1.05]};
const PK = ['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier'];
const inR = (k, v) => v >= DR[k][0] - 1e-12 && v <= DR[k][1] + 1e-12;
const GRID = [{name: '遅延 1', c: {delay: 1}}, {name: '遅延 1 + 前進ゲイン 1.01', c: {delay: 1, speedGain: 1.01}},
  ...[.8, .9, 1.0, 1.1, 1.3, 1.5, 1.6, 1.8].map(v => ({name: `舵の速さ ${v}`, c: {phys: {steerRate: v}}, in: inR('steerRate', v)})),
  ...[.6, .7, .8, .9, 1.0, 1.1, 1.5, 2.0, 2.5].map(v => ({name: `ブレーキ ${v}`, c: {phys: {brakeDeceleration: v}}, in: inR('brakeDeceleration', v)})),
  ...[.8, .85, .9, .95, 1.05, 1.1].map(v => ({name: `μ × ${v}`, c: {phys: {frictionMultiplier: v}}, in: inR('frictionMultiplier', v)})),
  ...[.25, .3, .5, .6].map(v => ({name: `時定数 ${v}`, c: {phys: {driveTau: v}}, in: inR('driveTau', v)})),
  ...[3, 4].map(v => ({name: `加速の上限 ${v}`, c: {phys: {driveAccelerationCap: v}}, in: inR('driveAccelerationCap', v)})),
  {name: '舵のゲイン 0.95', c: {steerGain: .95}, in: true}, {name: '舵のゲイン 1.05', c: {steerGain: 1.05}, in: true},
  {name: '前進ゲイン 0.97', c: {speedGain: .97}, in: true}, {name: '前進ゲイン 1.03', c: {speedGain: 1.03}, in: true},
  {name: '舵の速さ 1.0 + ブレーキ 1.0', c: {phys: {steerRate: 1.0, brakeDeceleration: 1.0}}, in: true},
  {name: '舵の速さ 1.0 + 遅延 1', c: {phys: {steerRate: 1.0}, delay: 1}, in: true},
  {name: 'μ × 0.9 + ブレーキ 1.0 + 遅延 1', c: {phys: {frictionMultiplier: .9, brakeDeceleration: 1.0}, delay: 1}, in: true},
  {name: '舵 0.9・ブレーキ 0.8・μ 0.9・時定数 0.5・遅延 1', c: {phys: {steerRate: .9, brakeDeceleration: .8, frictionMultiplier: .9, driveTau: .5}, delay: 1}, in: true},
].map(g => ({...g, in: g.in ?? true}));
function sample(rnd, spec) {
  const p = {phys: {}};
  for (const k of PK) p.phys[k] = spec[k][0] + (spec[k][1] - spec[k][0]) * rnd();
  p.delay = rnd() < spec.delay1_fraction ? 1 : 0;
  p.steerGain = spec.steerGain[0] + (spec.steerGain[1] - spec.steerGain[0]) * rnd();
  p.speedGain = spec.speedGain[0] + (spec.speedGain[1] - spec.speedGain[0]) * rnd();
  return p;
}
const ctrls = who.split(',').map(k => k === 'teacher' ? {key: k, run: o => runTeacher(o)} : (pol => ({key: k, run: o => runPolicy(pol, o)}))(loadMlp(k)));
const out = {schema: 'rl-check-r2-physics/v1', created_at: new Date().toISOString(), set, dr_spec_from_run_json: DR, wide: WIDE, results: {}};
const brief = s => ({valid: s.valid, reason: s.reason, laps: s.laps, gates: s.gates, first_collision: s.first_collision});
for (const C of ctrls) {
  const t0 = Date.now(), R = {};
  if (set === 'grid') {
    R.rows = GRID.map(g => {
      const row = {name: g.name, in_range: g.in, single: brief(C.run({mode: 'single', ...g.c}).summary), continuous: brief(C.run({mode: 'continuous', ...g.c}).summary)};
      if (g.c.delay && C.key !== 'teacher') {
        row.single_fresh = brief(C.run({mode: 'single', ...g.c, prev: 'fresh'}).summary);
        row.continuous_fresh = brief(C.run({mode: 'continuous', ...g.c, prev: 'fresh'}).summary);
      }
      return row;
    });
    const both = r => r.single.valid && r.continuous.valid;
    R.both_valid = R.rows.filter(both).length;
    R.both_valid_in_range = R.rows.filter(r => r.in_range && both(r)).length; R.n_in_range = R.rows.filter(r => r.in_range).length;
    R.both_valid_out_of_range = R.rows.filter(r => !r.in_range && both(r)).length; R.n_out_of_range = R.rows.filter(r => !r.in_range).length;
    R.failed = R.rows.filter(r => !both(r)).map(r => `${r.name}${r.in_range ? '' : '(範囲外)'}`);
    R.both_valid_fresh_convention = R.rows.filter(r => r.single_fresh ? (r.single_fresh.valid && r.continuous_fresh.valid) : both(r)).length;
  } else if (set === 'random') {
    for (const [label, spec, seed] of [['in_range', DR, 20261013], ['wide', WIDE, 20261014]]) {
      const rnd = rng32(seed), rows = [];
      for (let i = 0; i < 100; i++) { const p = sample(rnd, spec); const s = C.run({mode: 'continuous', ...p}).summary; rows.push({p, valid: s.valid, reason: s.reason, laps: s.laps}); }
      const ok = rows.filter(r => r.valid), m = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
      R[label] = {n: rows.length, valid: ok.length, lap1_mean: m(ok.map(r => r.laps[0])), lap2_mean: m(ok.map(r => r.laps[1])),
        failures: rows.filter(r => !r.valid).map(r => ({p: r.p, reason: r.reason}))};
    }
  } else if (set === 'rec60x') {
    const rnd = rng32(515151), rows = [];
    for (const c of PLAN60) {
      const p = sample(rnd, DR);
      if (C.key === 'teacher' && (Math.abs(c['spawn-offset']) > .03 + 1e-12 || Math.abs(c['spawn-yaw']) > 3 + 1e-12)) continue;
      const s = C.run({plan: c, gate: c['start-gate'], offset: c['spawn-offset'], yawDeg: c['spawn-yaw'], mode: 'continuous', ...p}).summary;
      rows.push({name: c.name, p, valid: s.valid, reason: s.reason, laps: s.laps});
    }
    R.n = rows.length; R.valid = rows.filter(r => r.valid).length; R.failures = rows.filter(r => !r.valid).map(r => ({name: r.name, p: r.p, reason: r.reason}));
  } else if (set === 'starts') {
    if (C.key === 'teacher') continue;
    const grid = [];
    for (const off of [-.15, -.075, 0, .075, .15]) for (const yaw of [-10, -5, 0, 5, 10]) {
      const s = C.run({mode: 'continuous', offset: off, yawDeg: yaw}).summary; grid.push({off, yaw, valid: s.valid, reason: s.reason, laps: s.laps});
    }
    const gates = [];
    for (let g = 0; g < 33; g++) { const s = C.run({mode: 'continuous', gate: g}).summary; gates.push({gate: g, valid: s.valid, reason: s.reason, laps: s.laps}); }
    R.start_grid = {n: grid.length, valid: grid.filter(r => r.valid).length, failures: grid.filter(r => !r.valid)};
    R.all_gates = {n: gates.length, valid: gates.filter(r => r.valid).length, failures: gates.filter(r => !r.valid)};
  }
  R.wall_s = (Date.now() - t0) / 1000;
  out.results[C.key] = R;
  const short = {...R}; delete short.rows;
  for (const k of ['in_range', 'wide']) if (short[k]) short[k] = `${short[k].valid}/${short[k].n} ${short[k].lap1_mean?.toFixed(3)} ${short[k].lap2_mean?.toFixed(3)}`;
  if (short.failures) short.failures = short.failures.length;
  if (short.start_grid) short.start_grid = `${R.start_grid.valid}/${R.start_grid.n}`;
  if (short.all_gates) short.all_gates = `${R.all_gates.valid}/${R.all_gates.n} ${R.all_gates.failures.map(f => f.gate + ':' + f.reason).join(' ')}`;
  console.log(C.key, set, JSON.stringify(short));
}
writeJson(`e3_${set}_${who.replace(/,/g, '-')}.json`, out);
