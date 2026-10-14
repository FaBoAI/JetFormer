#!/usr/bin/env node
// no_wall の確認 r1(2026-10-14、確認の段)。自前の走らせ方(run_nw.mjs)で次の組を走らせ、1 行 1 回の JSONL に書く。
//   REPLAY   試験の組(判定の 340 回)を同じ条件で走らせ直す(条件の一覧だけ eval_modes.mjs の buildJobs から読む)。NW-OFF(260 回)も
//   FNOM     新しい種・新しい開始(記録の開始の分布から自前の乱数で一様に。初めのすき間 ≥ 0.12 m)・標準の物理・シールドあり 150 回(線も残す)
//   FDR      新しい種・新しい開始・DR-T の範囲から自前の乱数で一様に(遅延 0/1 半々・摩擦円 30%)・シールドあり 400 回
//   FDROFF   FDR をシールドなしで(報告)
//   CORNER   DR-T の悪い隅 4 つ × 全ゲート × 開始の隅(左右 ±0.2 m・向き ±15°、初めのすき間 ≥ 0.12 m)・シールドあり
//   FAST     最速の方策(シールドなし)で FNOM の開始(線の比べ)
//   LOWMU    範囲の外(報告): アプリの「低摩擦」シナリオ(μ × 0.4)で STD と FNOM の最初の 9 回
//   node verify_nw.mjs --sets REPLAY,FNOM,... --jobs 3 --out <dir>
import fs from 'node:fs';
import path from 'node:path';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const here = fileURLToPath(import.meta.url);
const args = process.argv.slice(2);
const opt = (k, d = null) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const DRT = {steerRate: [.8, 1.8], brakeDeceleration: [.7, 2.2], driveTau: [.28, .55], driveAccelerationCap: [3.5, 6.0], frictionMultiplier: [.85, 1.1],
  steerGain: [.94, 1.06], speedGain: [.96, 1.04]};   // modes.json common.dr_specs.DR-T(遅延 0/1 半々、摩擦円 30%)
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

async function buildJobs(sets) {
  const {startClearance} = await import('./run_nw.mjs');
  const J = [];
  const rnd = mulberry32(0x14102026);
  // 新しい開始(記録の開始の分布: ゲート U{0..32}・左 U(−0.2, 0.2) m・向き U(−15°, 15°)・初めのすき間 ≥ 0.12 m)。種は 1,400,000 + i(試験 902xxx・検証 802xxx・記録 702xxx・線 990xxx と重ならない)
  const fresh = [];
  while (fresh.length < 560) {
    const st = {gate: Math.min(32, Math.floor(rnd() * 33)), left_offset_m: -.2 + .4 * rnd(), left_yaw_deg: -15 + 30 * rnd()};
    if (startClearance(st.gate, st.left_offset_m, st.left_yaw_deg) < .12) continue;
    fresh.push({...st, seed: 1400000 + fresh.length});
  }
  const drs = fresh.map(() => {
    const p = k => DRT[k][0] + (DRT[k][1] - DRT[k][0]) * rnd();
    return {physics: {steerRate: p('steerRate'), brakeDeceleration: p('brakeDeceleration'), driveTau: p('driveTau'), driveAccelerationCap: p('driveAccelerationCap'),
      frictionMultiplier: p('frictionMultiplier')}, delay: rnd() < .5 ? 0 : 1, steerGain: p('steerGain'), speedGain: p('speedGain'), frictionCircle: rnd() < .3};
  });
  if (sets.includes('REPLAY')) {
    const {buildJobs: theirJobs} = await import('../eval_modes.mjs');
    for (const j of theirJobs('no_wall', 'test')) {
      if (j.test === 'NW-G-OUT') continue;
      const s = j.spec;
      J.push({set: 'REPLAY', test: j.test, id: j.id, spec: {mode: 'no_wall', start: s.start, episodeMode: s.episodeMode ?? 'continuous', dr: s.dr ?? null,
        shield: !(s.overrides && s.overrides.shield === false)}});
    }
  }
  if (sets.includes('FNOM')) fresh.slice(0, 150).forEach((st, i) => J.push({set: 'FNOM', id: `FN${i}`, spec: {mode: 'no_wall', start: st, keepPath: true}}));
  if (sets.includes('FAST')) fresh.slice(0, 150).forEach((st, i) => J.push({set: 'FAST', id: `FF${i}`, spec: {mode: 'fastest', start: st, keepPath: true}}));
  if (sets.includes('FDR')) fresh.slice(150, 550).forEach((st, i) => J.push({set: 'FDR', id: `FD${i}`, spec: {mode: 'no_wall', start: st, dr: drs[150 + i]}}));
  if (sets.includes('FDROFF')) fresh.slice(150, 550).forEach((st, i) => J.push({set: 'FDROFF', id: `FD${i}`, spec: {mode: 'no_wall', start: st, dr: drs[150 + i], shield: false}}));
  if (sets.includes('CORNER') || sets.includes('CORNEROFF')) {
    const P = {
      C1: {physics: {steerRate: .8, brakeDeceleration: .7, driveTau: .55, driveAccelerationCap: 3.5, frictionMultiplier: .85}, delay: 1, steerGain: .94, speedGain: 1.04},
      C2: {physics: {steerRate: .8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 6.0, frictionMultiplier: .85}, delay: 1, steerGain: .94, speedGain: 1.04},
      C3: {physics: {steerRate: .8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 6.0, frictionMultiplier: .85}, delay: 1, steerGain: .94, speedGain: 1.04, frictionCircle: true},
      C4: {physics: {steerRate: .8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 6.0, frictionMultiplier: .85}, delay: 1, steerGain: 1.06, speedGain: 1.04},
    };
    let n = 0;
    for (let g = 0; g < 33; g++) for (const [o, y] of [[.2, 15], [.2, -15], [-.2, 15], [-.2, -15]]) {
      if (startClearance(g, o, y) < .12) continue;
      for (const [k, dr] of Object.entries(P)) { if (sets.includes('CORNER')) J.push({set: 'CORNER', id: `${k}-g${g}${o > 0 ? 'L' : 'R'}${y > 0 ? '+' : '-'}`, corner: k, spec: {mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 1500000 + n}, dr}});
        if (sets.includes('CORNEROFF')) J.push({set: 'CORNEROFF', id: `${k}-g${g}${o > 0 ? 'L' : 'R'}${y > 0 ? '+' : '-'}`, corner: k, spec: {mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 1500000 + n}, dr, shield: false}}); n++; }
    }
  }
  if (sets.includes('ABL')) {
    // 隅の組の衝突の原因を分ける: C2 の物理から 1 つずつ標準へ戻す・1 つずつだけ悪くする。開始はおだやか(全ゲート・横 0・向き 0)
    const nom = {steerRate: 1.2, brakeDeceleration: 1.2, driveTau: .4, driveAccelerationCap: 6, frictionMultiplier: 1};
    const C2 = {physics: {steerRate: .8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 6.0, frictionMultiplier: .85}, delay: 1, steerGain: .94, speedGain: 1.04};
    const V = {
      A0_C2: C2,
      A1_C2_delay0: {...C2, delay: 0},
      A2_C2_steer12: {...C2, physics: {...C2.physics, steerRate: 1.2}},
      A3_C2_gains1: {...C2, steerGain: 1, speedGain: 1},
      A4_C2_mu1: {...C2, physics: {...C2.physics, frictionMultiplier: 1}},
      A5_C2_brake12: {...C2, physics: {...C2.physics, brakeDeceleration: 1.2}},
      A6_steer08_delay1: {physics: {...nom, steerRate: .8}, delay: 1, steerGain: 1, speedGain: 1},
      A7_steer08: {physics: {...nom, steerRate: .8}, delay: 0, steerGain: 1, speedGain: 1},
      A8_delay1: {physics: {...nom}, delay: 1, steerGain: 1, speedGain: 1},
      A9_steer08_delay1_vg104: {physics: {...nom, steerRate: .8}, delay: 1, steerGain: 1, speedGain: 1.04},
    };
    let n = 0;
    for (const [k, dr] of Object.entries(V)) for (let g = 0; g < 33; g++)
      J.push({set: 'ABL', id: `${k}-g${g}`, corner: k, spec: {mode: 'no_wall', start: {gate: g, left_offset_m: 0, left_yaw_deg: 0, seed: 1600000 + n++}, dr}});
  }
  if (sets.includes('LOWMU')) {
    J.push({set: 'LOWMU', id: 'STD', spec: {mode: 'no_wall', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, dr: {physics: {frictionMultiplier: .4}}}});
    fresh.slice(0, 9).forEach((st, i) => J.push({set: 'LOWMU', id: `LM${i}`, spec: {mode: 'no_wall', start: st, dr: {physics: {frictionMultiplier: .4}}}}));
  }
  return J;
}

async function main() {
  const sets = (opt('--sets') ?? 'REPLAY,FNOM,FAST,FDR,FDROFF,CORNER,LOWMU').split(',');
  const out = opt('--out'), jobs = +(opt('--jobs') ?? 3), shard = opt('--shard');
  const J = await buildJobs(sets);
  if (shard) {
    const {runOne} = await import('./run_nw.mjs');
    const [k, n] = shard.split('/').map(Number);
    const fd = fs.openSync(opt('--rows-out'), 'w');
    for (let j = k; j < J.length; j += n) {
      const t = Date.now();
      let r;
      try { r = await runOne(J[j].spec); } catch (e) { r = {error: String(e && e.stack || e)}; }
      fs.writeSync(fd, JSON.stringify({j, set: J[j].set, test: J[j].test ?? null, id: J[j].id, corner: J[j].corner ?? null, wall_s: (Date.now() - t) / 1000, ...r}) + '\n');
    }
    fs.closeSync(fd);
    return;
  }
  fs.mkdirSync(out, {recursive: true});
  console.error(`回の数 ${J.length}(${sets.join(',')})`);
  const t0 = Date.now();
  await Promise.all(Array.from({length: jobs}, (_, k) => new Promise((res, rej) => {
    const ch = fork(here, ['--sets', sets.join(','), '--shard', `${k}/${jobs}`, '--rows-out', path.join(out, `rows_${sets.join('_')}_${k}.jsonl`)], {stdio: 'inherit'});
    ch.on('exit', c => c === 0 ? res() : rej(new Error('子プロセスが失敗しました ' + c)));
  })));
  const rows = [];
  for (let k = 0; k < jobs; k++) for (const line of fs.readFileSync(path.join(out, `rows_${sets.join('_')}_${k}.jsonl`), 'utf8').split('\n')) if (line) rows.push(JSON.parse(line));
  rows.sort((a, b) => a.j - b.j);
  fs.writeFileSync(path.join(out, `rows_${sets.join('_')}.jsonl`), rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  for (let k = 0; k < jobs; k++) fs.rmSync(path.join(out, `rows_${sets.join('_')}_${k}.jsonl`));
  console.error(`終わり ${rows.length} 回、${(Date.now() - t0) / 1000} 秒`);
}
await main();
