#!/usr/bin/env node
// 確認 r3 の 2: r3 の段の「ずれへの強さ」の主張を、新しく書いた道具で作り直す(r3 の段は確認 r2 の道具で測った)
//   node check_r3/e2_ranges.mjs <方策,…> <corner,grid25,rec60x,own,allbad,starts>
// - corner : 確認 r2 と同じ 20 マス(舵の速さ 0.85〜1.1 × ブレーキ 0.8〜1.2、ほかは標準・遅延 0)。STD 単周・連続と REC60 の 60 の開始
// - grid25 : 舵の速さ 0.85・1.0・1.2・1.45・1.7 × ブレーキ 0.8・1.0・1.2・1.6・2.0 を REC60 の 60 の開始で
// - rec60x : REC60 の開始 × 学習の範囲の一様な物理(確認 r2・r3 の段と同じ mulberry32 の種 515151・515152。同じ引き方なので同じ条件になる)
// - own    : 自前の乱数(common_r3 の rand)で、REC60 の開始 × (a) 学習の範囲の一様 8 次元 100 本、(b) 範囲の端(角)だけ 100 本
// - allbad : 範囲の中で悪い側をすべて重ねた組み合わせ(舵 0.85・ブレーキ 0.8・時定数 0.5・加速の上限 4・μ × 0.88・遅延 1)
// - starts : ゲート 0 の横 ±15 cm × 向き ±10° の 25 条件と、全 33 ゲートからの開始(標準の物理・連続 20 秒)
import {PLAN, loadPolicy, drive, driveTeacher, save, rand} from './common_r3.mjs';
import fs from 'node:fs';

const [who = 'r3_s21', setsArg = 'corner,grid25,rec60x,own,allbad,starts'] = process.argv.slice(2);
const sets = setsArg.split(',');
const DR = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/r3-s21/run.json', 'utf8')).env.dr_spec;
const PK = ['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier'];
function mulberry32(seed) {   // 確認 r2 の rec60x と同じ列(公開されている式)
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const uniform = rnd => {
  const p = {phys: {}};
  for (const k of PK) p.phys[k] = DR[k][0] + (DR[k][1] - DR[k][0]) * rnd();
  p.delay = rnd() < DR.delay1_fraction ? 1 : 0;
  p.steerGain = DR.steerGain[0] + (DR.steerGain[1] - DR.steerGain[0]) * rnd();
  p.speedGain = DR.speedGain[0] + (DR.speedGain[1] - DR.speedGain[0]) * rnd();
  return p;
};
// 範囲の端だけ(5 つの値をそれぞれ下の端か上の端ちょうど。遅延 0/1・ゲインも端)
const corner = rnd => ({phys: Object.fromEntries(PK.map(k => [k, DR[k][rnd() < .5 ? 0 : 1]])), delay: rnd() < .5 ? 1 : 0,
  steerGain: DR.steerGain[rnd() < .5 ? 0 : 1], speedGain: DR.speedGain[rnd() < .5 ? 0 : 1]});
const inContract = r => r['start-gate'] === 0 && Math.abs(r['spawn-offset']) <= .03 + 1e-12 && Math.abs(r['spawn-yaw']) <= 3 + 1e-12;

const out = {schema: 'rl-check-r3-e2/v1', created_at: new Date().toISOString(), dr_spec: DR, results: {}};
for (const key of who.split(',')) {
  const teacher = key === 'teacher', pol = teacher ? null : loadPolicy(key);
  const run = o => teacher ? driveTeacher(o) : drive(pol, o);
  const rows = teacher ? PLAN.filter(inContract) : PLAN;
  const rec60 = (o = {}) => rows.filter(row => run({row, continuous: true, ...o}).judge.ok).length;
  const R = {}, t0 = Date.now();
  if (sets.includes('corner')) {
    R.corner = [];
    for (const sr of [.85, .9, .95, 1.0, 1.1]) for (const br of [.8, .9, 1.0, 1.2]) {
      const phys = {steerRate: sr, brakeDeceleration: br};
      R.corner.push({sr, br, std_single: run({phys}).judge.ok, std_cont: run({phys, continuous: true}).judge.ok, rec60: rec60({phys}), n: rows.length});
    }
    R.corner_total = R.corner.reduce((a, c) => a + c.rec60, 0); R.corner_min = Math.min(...R.corner.map(c => c.rec60));
    console.log(key, 'corner', R.corner_total, 'min', R.corner_min, R.corner.map(c => `${c.sr}/${c.br}:${c.std_single ? 'S' : 's'}${c.std_cont ? 'C' : 'c'}${c.rec60}`).join(' '));
  }
  if (sets.includes('grid25')) {
    R.grid25 = [];
    for (const sr of [.85, 1.0, 1.2, 1.45, 1.7]) for (const br of [.8, 1.0, 1.2, 1.6, 2.0]) R.grid25.push({sr, br, rec60: rec60({phys: {steerRate: sr, brakeDeceleration: br}})});
    R.grid25_total = R.grid25.reduce((a, c) => a + c.rec60, 0);
    console.log(key, 'grid25', R.grid25_total, R.grid25.filter(c => c.rec60 < rows.length).map(c => `${c.sr}/${c.br}:${c.rec60}`).join(' '));
  }
  if (sets.includes('rec60x')) {
    R.rec60x = {};
    for (const seed of [515151, 515152]) {
      const rnd = mulberry32(seed), res = [];
      for (const row of PLAN) {
        const p = uniform(rnd);
        if (teacher && !inContract(row)) continue;
        const j = run({row, continuous: true, ...p}).judge;
        res.push({name: row.name, p, ok: j.ok, reason: j.reason});
      }
      const isC = p => p.phys.steerRate < .96 && p.phys.brakeDeceleration < .84;
      R.rec60x[seed] = {valid: res.filter(r => r.ok).length, n: res.length, corner_draws: res.filter(r => isC(r.p)).length, corner_valid: res.filter(r => isC(r.p) && r.ok).length,
        failures: res.filter(r => !r.ok)};
    }
    console.log(key, 'rec60x', JSON.stringify(Object.fromEntries(Object.entries(R.rec60x).map(([s, v]) => [s, `${v.valid}/${v.n} 角 ${v.corner_valid}/${v.corner_draws}`]))));
  }
  if (sets.includes('own')) {
    R.own = {};
    for (const [lab, gen, seed] of [['uniform8', uniform, 909001], ['corners_only', corner, 909002]]) {
      const rnd = rand(seed), res = [];
      for (let i = 0; i < 100; i++) {
        const row = PLAN[Math.floor(rnd() * PLAN.length)], p = gen(rnd);
        if (teacher && !inContract(row)) continue;
        const j = run({row, continuous: true, ...p}).judge;
        res.push({name: row.name, p, ok: j.ok, reason: j.reason, first_collision: j.first_collision});
      }
      R.own[lab] = {valid: res.filter(r => r.ok).length, n: res.length, failures: res.filter(r => !r.ok)};
    }
    console.log(key, 'own', JSON.stringify(Object.fromEntries(Object.entries(R.own).map(([s, v]) => [s, `${v.valid}/${v.n}`]))));
  }
  if (sets.includes('allbad')) {
    const o = {phys: {steerRate: .85, brakeDeceleration: .8, driveTau: .5, driveAccelerationCap: 4, frictionMultiplier: .88}, delay: 1};
    R.allbad = {single: run(o).judge, cont: run({...o, continuous: true}).judge, rec60: rec60(o), rec60_delay0: rec60({phys: o.phys})};
    console.log(key, 'allbad', R.allbad.single.ok, R.allbad.single.laps, R.allbad.cont.ok, R.allbad.rec60, 'delay0', R.allbad.rec60_delay0);
  }
  if (sets.includes('starts') && !teacher) {
    let g25 = 0, g33 = 0, n33 = 0; const fails = [];
    for (const off of [-.15, -.075, 0, .075, .15]) for (const yaw of [-10, -5, 0, 5, 10]) { const j = run({offset: off, yaw, continuous: true}).judge; if (j.ok) g25++; else fails.push({off, yaw, ...j}); }
    for (let g = 0; g < 33; g++) { const r = run({gate: g, continuous: true}); if (r.invalid_spawn) continue; n33++; if (r.judge.ok) g33++; else fails.push({gate: g, ...r.judge}); }
    R.starts = {grid25: g25, gates: g33, gates_n: n33, failures: fails};
    console.log(key, 'starts', g25, `${g33}/${n33}`);
  }
  R.wall_s = (Date.now() - t0) / 1000;
  out.results[key] = R;
  save(`e2_${who.replace(/,/g, '-')}_${setsArg.replace(/,/g, '-')}.json`, out);
}
