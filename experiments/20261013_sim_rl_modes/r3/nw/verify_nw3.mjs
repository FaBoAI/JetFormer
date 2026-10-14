#!/usr/bin/env node
// r3 の下調べ(2026-10-14、r3 の計画の段): 確認 r2 の verify_nw_r2.mjs(sha256 fee78ea7…)の写し。違いは 3 つだけ:
//   (1) 走らせる仕組みを r3/runtime(シールド v4 の試作)にする(環境変数 JF_RT=web なら元の web/rl)、
//   (2) 環境変数 JF_R3(JSON)を globalThis.JF_R3 に入れる(シールドの値の切り替え)、(3) --corner・--filter・--every で組の一部だけ走らせる。
// 以下は元の説明。
// no_wall の確認 r2(2026-10-14、確認の段 r2)。自前の走らせ方(run_nw_r2.mjs)で次の組を走らせ、1 行 1 回の JSONL に書く。
//   乱数は mulberry32(0x14102031)(確認 r1 の 0x14102026・学習の段 r2 の選びの組 0x22102026 と別)。種は 2,400,000〜2,899,999
//   (試験 902xxx・検証 802xxx・記録 702xxx・線 990xxx・確認 r1 の 1.4〜1.7M・選びの組 1.9M と重ならない)。
//   REPLAY2  試験の組 r2(判定の 340 回 + NW-OFF 260 回)を、試験の JSON に残った条件(開始・物理)で走らせ直す
//   FN3      新しい開始(記録の開始の分布から一様: ゲート U{0..32}・左 U(−0.2, 0.2) m・向き U(−15°, 15°)・初めのすき間 ≥ 0.12 m)・標準の物理・シールドあり 600 回
//   FN3OFF   FN3 をシールドなしで(方策だけ)
//   FAST3    最速の方策(シールドなし)で FN3 の最初の 150 回の開始(線の比べ)
//   FD3      新しい開始・DR-T から一様(遅延 0/1 半々・摩擦円 30%)・シールドあり 600 回。FD3OFF = シールドなし
//   VX       DR-T の頂点(各値を範囲の端のどちらかに乱数で。遅延 0/1・摩擦円 30%)× 新しい開始、シールドあり 480 回
//   NG       標準の物理・開始の格子(33 ゲート × 左右 {−0.2, −0.1, 0, 0.1, 0.2} m × 向き {−15, −7.5, 0, 7.5, 15}°、初めのすき間 ≥ 0.12 m)・シールドあり
//   LONG     120 秒の連続周回: 標準の物理 10 回(STD + 新しい開始 9)・DR-T 一様 20 回、シールドあり
//   D2       範囲の外(報告): 遅延 2 フレーム + 舵の速さ U(0.8, 1.2)、ほかは標準、新しい開始 30 回
//   node verify_nw_r2.mjs --sets FN3,... --jobs 3 --out <dir>
import fs from 'node:fs';
import path from 'node:path';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const here = fileURLToPath(import.meta.url);
const args = process.argv.slice(2);
const opt = (k, d = null) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
const DRT = {steerRate: [.8, 1.8], brakeDeceleration: [.7, 2.2], driveTau: [.28, .55], driveAccelerationCap: [3.5, 6.0], frictionMultiplier: [.85, 1.1],
  steerGain: [.94, 1.06], speedGain: [.96, 1.04]};   // modes.json common.dr_specs.DR-T(遅延 0/1 半々、摩擦円 30%)
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

async function buildJobs(sets) {
  const {startClearance} = await import('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_nw_r2/run_nw_r2.mjs');
  const J = [];
  const rnd = mulberry32(0x14102031);
  const fresh = [];
  while (fresh.length < 1800) {
    const st = {gate: Math.min(32, Math.floor(rnd() * 33)), left_offset_m: -.2 + .4 * rnd(), left_yaw_deg: -15 + 30 * rnd()};
    if (startClearance(st.gate, st.left_offset_m, st.left_yaw_deg) < .12) continue;
    fresh.push({...st, seed: 2400000 + fresh.length});
  }
  const drU = () => {
    const p = k => DRT[k][0] + (DRT[k][1] - DRT[k][0]) * rnd();
    return {physics: {steerRate: p('steerRate'), brakeDeceleration: p('brakeDeceleration'), driveTau: p('driveTau'), driveAccelerationCap: p('driveAccelerationCap'),
      frictionMultiplier: p('frictionMultiplier')}, delay: rnd() < .5 ? 0 : 1, steerGain: p('steerGain'), speedGain: p('speedGain'), frictionCircle: rnd() < .3};
  };
  const drV = () => {
    const e = k => DRT[k][rnd() < .5 ? 0 : 1];
    return {physics: {steerRate: e('steerRate'), brakeDeceleration: e('brakeDeceleration'), driveTau: e('driveTau'), driveAccelerationCap: e('driveAccelerationCap'),
      frictionMultiplier: e('frictionMultiplier')}, delay: rnd() < .5 ? 0 : 1, steerGain: e('steerGain'), speedGain: e('speedGain'), frictionCircle: rnd() < .3};
  };
  const drsFD = Array.from({length: 600}, drU), drsVX = Array.from({length: 480}, drV), drsLONG = Array.from({length: 20}, drU);
  const d2 = Array.from({length: 30}, () => ({physics: {steerRate: .8 + .4 * rnd()}, delay: 2}));
  if (sets.includes('REPLAY2')) {
    const T = JSON.parse(fs.readFileSync(`${REP}/eval/no_wall/no_wall_test_r2.json`, 'utf8')).tests;
    for (const name of ['NW-N60', 'NW-STD', 'NW-G', 'NW-R200', 'NW-FC10', 'NW-OFF']) for (const r of T[name]) {
      const single = /single/.test(r.id) && !/continuous/.test(r.id);
      const dr = r.dr ? {physics: r.dr.physics ?? {}, delay: r.dr.delay ?? 0, steerGain: r.dr.steerGain ?? 1, speedGain: r.dr.speedGain ?? 1, frictionCircle: !!r.dr.frictionCircle} : null;
      J.push({set: 'REPLAY2', test: name, id: r.id, spec: {mode: 'no_wall', start: r.start, episodeMode: single ? 'single' : 'continuous', dr, shield: name !== 'NW-OFF'}});
    }
  }
  if (sets.includes('FN3')) fresh.slice(0, 600).forEach((st, i) => J.push({set: 'FN3', id: `FN${i}`, spec: {mode: 'no_wall', start: st, keepPath: i < 150}}));
  if (sets.includes('FN3OFF')) fresh.slice(0, 600).forEach((st, i) => J.push({set: 'FN3OFF', id: `FN${i}`, spec: {mode: 'no_wall', start: st, shield: false}}));
  if (sets.includes('FAST3')) fresh.slice(0, 150).forEach((st, i) => J.push({set: 'FAST3', id: `FF${i}`, spec: {mode: 'fastest', start: st, keepPath: true}}));
  if (sets.includes('FD3')) fresh.slice(600, 1200).forEach((st, i) => J.push({set: 'FD3', id: `FD${i}`, spec: {mode: 'no_wall', start: st, dr: drsFD[i]}}));
  if (sets.includes('FD3OFF')) fresh.slice(600, 1200).forEach((st, i) => J.push({set: 'FD3OFF', id: `FD${i}`, spec: {mode: 'no_wall', start: st, dr: drsFD[i], shield: false}}));
  if (sets.includes('VX')) fresh.slice(1200, 1680).forEach((st, i) => J.push({set: 'VX', id: `VX${i}`, spec: {mode: 'no_wall', start: {...st, seed: 2500000 + i}, dr: drsVX[i]}}));
  if (sets.includes('VXOFF')) fresh.slice(1200, 1680).forEach((st, i) => J.push({set: 'VXOFF', id: `VX${i}`, spec: {mode: 'no_wall', start: {...st, seed: 2500000 + i}, dr: drsVX[i], shield: false}}));
  if (sets.includes('NG')) {
    let n = 0;
    for (let g = 0; g < 33; g++) for (const o of [-.2, -.1, 0, .1, .2]) for (const y of [-15, -7.5, 0, 7.5, 15]) {
      if (startClearance(g, o, y) < .12) continue;
      J.push({set: 'NG', id: `NG-g${g}o${o}y${y}`, spec: {mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 2600000 + n++}}});
    }
  }
  if (sets.includes('LONG')) {
    J.push({set: 'LONG', id: 'LN-STD', spec: {mode: 'no_wall', start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 2700000}, timeLimit: 120}});
    fresh.slice(1680, 1689).forEach((st, i) => J.push({set: 'LONG', id: `LN${i}`, spec: {mode: 'no_wall', start: {...st, seed: 2700001 + i}, timeLimit: 120}}));
    fresh.slice(1689, 1709).forEach((st, i) => J.push({set: 'LONG', id: `LD${i}`, spec: {mode: 'no_wall', start: {...st, seed: 2700100 + i}, dr: drsLONG[i], timeLimit: 120}}));
  }
  if (sets.includes('D2')) fresh.slice(1709, 1739).forEach((st, i) => J.push({set: 'D2', id: `D2-${i}`, spec: {mode: 'no_wall', start: {...st, seed: 2800000 + i}, dr: d2[i]}}));
  if (sets.includes('ABX') || sets.includes('ABXOFF')) {
    // 確認 r2 で見つけた範囲の中の隅(速い舵 1.8・弱いブレーキ 0.7・低い μ 0.85・摩擦円・前進のゲイン 1.04)の原因分け。開始はおだやか(全ゲート・横 0・向き 0)
    const K = {physics: {steerRate: 1.8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 3.5, frictionMultiplier: .85}, delay: 0, steerGain: 1.06, speedGain: 1.04, frictionCircle: true};
    const V = {
      K0_corner: K,
      K1_fc_off: {...K, frictionCircle: false},
      K2_mu1: {...K, physics: {...K.physics, frictionMultiplier: 1}},
      K3_brake12: {...K, physics: {...K.physics, brakeDeceleration: 1.2}},
      K4_vg1: {...K, speedGain: 1},
      K5_sg1: {...K, steerGain: 1},
      K6_steer12: {...K, physics: {...K.physics, steerRate: 1.2}},
      K7_delay1: {...K, delay: 1},
      K8_sg094: {...K, steerGain: .94},
      K9_inner1: {physics: {steerRate: 1.6, brakeDeceleration: .8, driveTau: .35, driveAccelerationCap: 4.5, frictionMultiplier: .9}, delay: 0, steerGain: 1.03, speedGain: 1.02, frictionCircle: true},
      K10_inner2: {physics: {steerRate: 1.7, brakeDeceleration: .75, driveTau: .3, driveAccelerationCap: 4.0, frictionMultiplier: .87}, delay: 0, steerGain: 1.04, speedGain: 1.03, frictionCircle: true},
      K11_nominal_but_fc_mu085_b07: {physics: {frictionMultiplier: .85, brakeDeceleration: .7}, delay: 0, steerGain: 1, speedGain: 1, frictionCircle: true},
    };
    let n = 0;
    for (const [k, dr] of Object.entries(V)) for (let g = 0; g < 33; g++) {
      const st = {gate: g, left_offset_m: 0, left_yaw_deg: 0, seed: 2850000 + n++};
      if (sets.includes('ABX')) J.push({set: 'ABX', id: `${k}-g${g}`, corner: k, spec: {mode: 'no_wall', start: st, dr}});
      if (sets.includes('ABXOFF') && k === 'K0_corner') J.push({set: 'ABXOFF', id: `${k}-g${g}`, corner: k, spec: {mode: 'no_wall', start: st, dr, shield: false}});
    }
  }
  return J;
}

async function main() {
  const sets = (opt('--sets') ?? 'FN3').split(',');
  const out = opt('--out'), jobs = +(opt('--jobs') ?? 3), shard = opt('--shard'), only = opt('--count');
  let J = await buildJobs(sets);
  const corner = opt('--corner'), filt = opt('--filter'), every = +(opt('--every') ?? 1);
  if (corner) J = J.filter(j => !j.corner || corner.split(',').includes(j.corner));
  if (filt === 'fc') J = J.filter(j => j.spec.dr?.frictionCircle);
  if (every > 1) J = J.filter((_, i) => i % every === 0);
  const RTDIR = process.env.JF_RT === 'web' ? '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/web/rl' : '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/runtime';
  for (const j of J) j.spec.rtDir = RTDIR;
  if (process.env.JF_R3) globalThis.JF_R3 = JSON.parse(process.env.JF_R3);
  if (only) J = J.slice(0, +only);
  if (shard) {
    const {runOne} = await import('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_nw_r2/run_nw_r2.mjs');
    const [k, n] = shard.split('/').map(Number);
    const fd = fs.openSync(opt('--rows-out'), 'w');
    for (let j = k; j < J.length; j += n) {
      const t = Date.now();
      let r;
      try { r = await runOne(J[j].spec); } catch (e) { r = {error: String(e && e.stack || e)}; }
      fs.writeSync(fd, JSON.stringify({j, set: J[j].set, test: J[j].test ?? null, id: J[j].id, wall_s: (Date.now() - t) / 1000, ...r}) + '\n');
    }
    fs.closeSync(fd);
    return;
  }
  fs.mkdirSync(out, {recursive: true});
  const tag = sets.join('_');
  console.error(`[${new Date().toTimeString().slice(0, 8)}] 回の数 ${J.length}(${tag})`);
  const t0 = Date.now();
  await Promise.all(Array.from({length: jobs}, (_, k) => new Promise((res, rej) => {
    const a = ['--sets', sets.join(','), '--shard', `${k}/${jobs}`, '--rows-out', path.join(out, `rows_${tag}_${k}.jsonl`)];
    for (const k2 of ['--corner', '--filter', '--every']) if (opt(k2)) a.push(k2, opt(k2));
    if (only) a.push('--count', only);
    const ch = fork(here, a, {stdio: 'inherit'});
    ch.on('exit', c => c === 0 ? res() : rej(new Error('子プロセスが失敗しました ' + c)));
  })));
  const rows = [];
  for (let k = 0; k < jobs; k++) for (const line of fs.readFileSync(path.join(out, `rows_${tag}_${k}.jsonl`), 'utf8').split('\n')) if (line) rows.push(JSON.parse(line));
  rows.sort((a, b) => a.j - b.j);
  fs.writeFileSync(path.join(out, `rows_${tag}.jsonl`), rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  for (let k = 0; k < jobs; k++) fs.rmSync(path.join(out, `rows_${tag}_${k}.jsonl`));
  console.error(`[${new Date().toTimeString().slice(0, 8)}] 終わり ${rows.length} 回、${(Date.now() - t0) / 1000} 秒`);
}
await main();
