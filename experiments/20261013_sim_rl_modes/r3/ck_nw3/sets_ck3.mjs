// no_wall 確認 r3 の組を作る(2026-10-14)。乱数 = mulberry32(0x0C3B1014)(この段だけ)。開始は no_wall の記録の開始の分布
// (ゲート U{0..32}・左右 U(−0.2, 0.2) m・向き U(−15, 15)°・開始のすき間 ≥ 0.12 m。すき間は own_ck3.mjs の自前の計算)から一様乱数。
// 試験(Halton 26,501〜)・検証(16,001〜)・下調べ(18,001〜)・記録(1〜)・確認 r2(mulberry32(0x14102031))のどれとも別の作り方。
import fs from 'node:fs';
import {mulberry32} from './own_ck3.mjs';
import {startClearance} from './run_ck3.mjs';
const rnd = mulberry32(0x0C3B1014);
const U = (a, b) => a + (b - a) * rnd();
const DRT = {steerRate: [.8, 1.8], brakeDeceleration: [.7, 2.2], driveTau: [.28, .55], driveAccelerationCap: [3.5, 6], frictionMultiplier: [.85, 1.1]};
const GAIN = {sg: [.94, 1.06], vg: [.96, 1.04]};
function start(minC = .12, maxC = Infinity) {
  for (;;) {
    const s = {gate: Math.floor(rnd() * 33), off: U(-.2, .2), yaw: U(-15, 15)};
    const c = startClearance(s.gate, s.off, s.yaw);
    if (c >= minC && c <= maxC) return {...s, c0: +c.toFixed(5)};
  }
}
const jobs = [];
let seed = 4300000;
const add = (set, id, o) => jobs.push({set, id: `${set}:${id}`, seed: seed++, ...o});
// 1) 標準の物理・新しい開始 1000(最初の 150 は軌跡を残す)
const FS = [];
for (let i = 0; i < 1000; i++) { const st = start(); FS.push(st); add('FS', i, {start: st, keepPath: i < 150}); }
// 2) その最初の 300 をシールドなしで(報告)
for (let i = 0; i < 300; i++) add('FSOFF', i, {start: FS[i], shield: false});
// 3) 最速の方策(公開の web/rl と bundle/fastest、シールドなし)で最初の 150
for (let i = 0; i < 150; i++) add('FAST', i, {start: FS[i], mode: 'fastest', policy: '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/bundle/fastest/policy.json',
  rtDir: '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/web/rl', keepPath: true});
// 4) 記録の分布の中の「開始のすき間 12〜15 cm」の層(標準の物理)300。層の重みは別に数える
for (let i = 0; i < 300; i++) add('LOWC', i, {start: start(.12, .15)});
// 層の重み: 記録の分布から 4000 個引いて、すき間 12〜15 cm の割合
let nAll = 0, nLow = 0;
const rW = mulberry32(0x0C3B2014);
const {startClearance: sc} = await import('./run_ck3.mjs');
while (nAll < 4000) { const g = Math.floor(rW() * 33), o = -.2 + .4 * rW(), y = -15 + 30 * rW(), c = sc(g, o, y); if (c >= .12) { nAll++; if (c <= .15) nLow++; } }
// 5) DR-T 一様 400(遅延 0/1 半々・摩擦円 30%)
const drU = () => ({phys: Object.fromEntries(Object.entries(DRT).map(([k, [a, b]]) => [k, U(a, b)])), delay: rnd() < .5 ? 0 : 1, sg: U(...GAIN.sg), vg: U(...GAIN.vg), fc: rnd() < .3});
for (let i = 0; i < 400; i++) add('DRU', i, {start: start(), ...drU()});
// 6) DR-T の頂点の全部の組(7 つの値 × 端 2 × 遅延 2 × 摩擦円 2 = 512)× 開始 1
const keys = [...Object.keys(DRT), 'sg', 'vg'];
const VX = [];
for (let b = 0; b < 512; b++) {
  const e = keys.map((_, j) => (b >> j) & 1), delay = (b >> 7) & 1, fc = !!((b >> 8) & 1);
  const phys = {}; let sg = 1, vg = 1;
  keys.forEach((k, j) => { const r = DRT[k] ?? GAIN[k]; const v = r[e[j]]; if (k === 'sg') sg = v; else if (k === 'vg') vg = v; else phys[k] = v; });
  const o = {start: start(), phys, delay, sg, vg, fc, combo: b};
  VX.push(o); add('VXALL', b, o);
}
// 7) 止まりにくい隅(摩擦円・ブレーキ 0.7・μ 0.85)の全部の組(舵の速さ・時定数・加速の上限・舵のゲイン・前進のゲイン の端 × 遅延 0/1 = 64)× 記録の分布の開始 6
const HK = ['steerRate', 'driveTau', 'driveAccelerationCap', 'sg', 'vg'];
const HC = [];
for (let b = 0; b < 64; b++) for (let r = 0; r < 6; r++) {
  const phys = {brakeDeceleration: .7, frictionMultiplier: .85}; let sg = 1, vg = 1;
  HK.forEach((k, j) => { const rg = DRT[k] ?? GAIN[k]; const v = rg[(b >> j) & 1]; if (k === 'sg') sg = v; else if (k === 'vg') vg = v; else phys[k] = v; });
  const o = {start: start(), phys, delay: (b >> 5) & 1, sg, vg, fc: true, combo: b};
  HC.push(o); add('HCALL', `${b}-${r}`, o);
}
// 8) 隅の近くの内側(摩擦円・ブレーキ U(0.7, 0.8)・μ U(0.85, 0.9)・前進のゲイン U(1.0, 1.04)、ほかは DR-T 一様)300
for (let i = 0; i < 300; i++) add('HCIN', i, {start: start(), phys: {steerRate: U(.8, 1.8), brakeDeceleration: U(.7, .8), driveTau: U(.28, .55), driveAccelerationCap: U(3.5, 6),
  frictionMultiplier: U(.85, .9)}, delay: rnd() < .5 ? 0 : 1, sg: U(...GAIN.sg), vg: U(1.0, 1.04), fc: true});
// 9) 頂点・隅のシールドなし(救った / 起こしたを数える。報告)
VX.forEach((o, b) => add('VXOFF', b, {...o, shield: false}));
HC.forEach((o, i) => add('HCOFF', i, {...o, shield: false}));
// 10) 120 秒の連続周回: 標準 6・隅 6・頂点 6
for (let i = 0; i < 6; i++) add('LONG', `std${i}`, {start: start(), T: 120});
for (let i = 0; i < 6; i++) add('LONG', `hc${i}`, {...HC[Math.floor(rnd() * HC.length)], start: start(), T: 120});
for (let i = 0; i < 6; i++) add('LONG', `vx${i}`, {...VX[Math.floor(rnd() * VX.length)], start: start(), T: 120});
// 11) STD(単周・連続)
add('STD', 'single', {start: {gate: 0, off: 0, yaw: 0}, episode: 'single'});
add('STD', 'cont', {start: {gate: 0, off: 0, yaw: 0}});
// 12) 範囲の外(報告): 遅延 2 + DR-T 一様 20、低摩擦(μ × 0.4)5
for (let i = 0; i < 20; i++) add('OUT-D2', i, {start: start(), ...drU(), delay: 2});
for (let i = 0; i < 5; i++) add('OUT-LOWMU', i, {start: start(), phys: {frictionMultiplier: .4}});
const out = process.argv[2];
fs.writeFileSync(out, jobs.map(j => JSON.stringify(j)).join('\n') + '\n');
fs.writeFileSync(out.replace(/\.jsonl$/, '.meta.json'), JSON.stringify({rng: 'mulberry32(0x0C3B1014)', seeds: [4300000, seed - 1], jobs: jobs.length,
  lowc_stratum: {draws: nAll, in_12_15cm: nLow, weight: nLow / nAll}, counts: jobs.reduce((a, j) => (a[j.set] = (a[j.set] ?? 0) + 1, a), {})}, null, 1));
console.log(jobs.length, nLow / nAll);
