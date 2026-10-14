// no_wall 確認 r4 の組(2026-10-14/15、確認の段)。乱数は mulberry32(0xC4A1_0E01〜)、種 7,400,000 + i。
// 試験(Halton 60,001〜・mulberry32(0x4e4a7e01〜03)・種 6,200,000〜)・検証(Halton 70,001〜・0x4e4a5601/02・種 6,100,000〜)・下調べ(0x4e4a0d01・種 6,900,000〜・
// dev4e/dev4f の格子)・学習の一覧(0x4e4a7a01)・確認 r2(0x14102031)・確認 r3(0x0C3B1014・種 4,300,000〜)のどれとも別。
// 格子の点も、試験の EDGE(±0.16・±0.19 m × ±11.5・±14.5°)・検証の EDGE(±0.18 m × ±13°)・下調べの dev4e(±0.2・±0.15 × ±15・±12°)と同じ点を避けた
// (分布の角に近い所を細かく見る)。
// 使い方: node sets_ck4.mjs <出力 jobs.jsonl>
import fs from 'node:fs';
import {mulberry32, startClear} from './own_ck4.mjs';

const DRT = {steerRate: [.8, 1.8], brakeDeceleration: [.7, 2.2], driveTau: [.28, .55], driveAccelerationCap: [3.5, 6.0], frictionMultiplier: [.85, 1.1], sg: [.94, 1.06], vg: [.96, 1.04]};
const jobs = [];
let seedN = 7400000;
const add = (set, spec) => { jobs.push({id: `${set}:${jobs.filter(j => j.set === set).length}`, set, seed: seedN++, ...spec}); };
const r4 = x => Math.round(x * 1e6) / 1e6;

// 記録の開始の分布(modes.json no_wall.recording.start_distribution)から一様に(開始のすき間 ≥ 0.12 m まで引き直す)
function recStart(R, lo = .12, hi = Infinity, offMax = .2, yawMax = 15) {
  for (let t = 0; t < 100000; t++) {
    const gate = Math.floor(R() * 33), off = r4((R() * 2 - 1) * offMax), yaw = r4((R() * 2 - 1) * yawMax);
    const c = startClear(gate, off, yaw);
    if (c >= lo && c < hi) return {gate, off, yaw, c0: +c.toFixed(5)};
  }
  throw new Error('開始が見つかりません');
}
function drUniform(R) {
  const u = k => DRT[k][0] + (DRT[k][1] - DRT[k][0]) * R();
  const phys = {steerRate: u('steerRate'), brakeDeceleration: u('brakeDeceleration'), driveTau: u('driveTau'), driveAccelerationCap: u('driveAccelerationCap'), frictionMultiplier: u('frictionMultiplier')};
  return {phys, sg: u('sg'), vg: u('vg'), delay: R() < .5 ? 0 : 1, fc: R() < .3};
}
function vertex(bits, delay, fc) {
  const keys = ['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier', 'sg', 'vg'];
  const v = Object.fromEntries(keys.map((k, i) => [k, DRT[k][(bits >> i) & 1]]));
  return {phys: {steerRate: v.steerRate, brakeDeceleration: v.brakeDeceleration, driveTau: v.driveTau, driveAccelerationCap: v.driveAccelerationCap, frictionMultiplier: v.frictionMultiplier},
    sg: v.sg, vg: v.vg, delay, fc, vbits: bits};
}
function hardCorner(bits) {   // 摩擦円・ブレーキ 0.7・μ × 0.85 に固定、舵の速さ・時定数・加速の上限・舵のゲイン・前進のゲイン・遅延の端
  const e = (k, i) => DRT[k][(bits >> i) & 1];
  return {phys: {steerRate: e('steerRate', 0), brakeDeceleration: .7, driveTau: e('driveTau', 1), driveAccelerationCap: e('driveAccelerationCap', 2), frictionMultiplier: .85},
    sg: e('sg', 3), vg: e('vg', 4), delay: (bits >> 5) & 1, fc: true, hbits: bits};
}
const grid = (gates, offs, yaws, set, extra = {}) => {
  for (const gate of gates) for (const off of offs) for (const yaw of yaws) {
    const c = startClear(gate, off, yaw);
    if (c >= .12) add(set, {start: {gate, off, yaw, c0: +c.toFixed(5)}, ...extra});
  }
};

// ---- 標準の物理(判定の形: N1・N2・N5)----
// 1. G17F: ゲート 17 の分布の角(右 0.2 m・左向き 15°)のまわりを細かく + ゲート 16・18
grid([17], [-.2, -.197, -.193, -.188, -.183, -.178, -.172, -.166, -.16], [10.1, 11.2, 12.1, 12.7, 13.3, 13.9, 14.4, 14.8, 15], 'G17F');
grid([16, 18], [-.2, -.194, -.186, -.176, -.165], [11.1, 12.6, 13.6, 14.6, 15], 'G17F');
// 2. G2023: 確認 r3 の崩れの所(ゲート 20 の左 −0.13・−13°、ゲート 23 の柱の足へ向いた開始)のまわり
grid([20, 21], [-.1575, -.1505, -.1435, -.1365, -.1295, -.1225, -.1155, -.1085, -.1015], [-14.8, -14.1, -13.4, -12.7, -11.9, -11.3, -10.6, -9.9, -9.2], 'G2023');   // 確認 r3 の NBR(0.01 m・1° の格子)と重ならない
grid([22, 23], [-.12, -.09, -.06, -.03, 0, .03, .06, .09, .12], [-14.6, -13.2, -11.8, 11.8, 13.2, 14.6], 'G2023');
// 3. CORN: 全ゲートの分布の角(±0.199・±0.184 m × ±14.9・±13.7°)
grid([...Array(33).keys()], [-.199, -.184, .184, .199], [-14.9, -13.7, 13.7, 14.9], 'CORN');
// 4. LOWC: 開始のすき間 12〜15 cm の層 600(mulberry32(0xC4A10E01))
{ const R = mulberry32(0xC4A10E01); for (let i = 0; i < 600; i++) add('LOWC', {start: recStart(R, .12, .15)}); }

// ---- 物理のずれ(判定の形: N1・N3・N5 の DR)----
// 5. HCL: 止まりにくい隅の全部の組 64 × 開始のすき間 12〜15 cm の開始 3
{ const R = mulberry32(0xC4A10E02); for (let b = 0; b < 64; b++) for (let k = 0; k < 3; k++) add('HCL', {start: recStart(R, .12, .15), ...hardCorner(b)}); }
// 6. VXALL: DR-T の頂点の全部の組(7 つの値の端 128 × 遅延 0/1 × 摩擦円あり/なし = 512)× 記録の分布の開始 1
{ const R = mulberry32(0xC4A10E03); for (let b = 0; b < 128; b++) for (const d of [0, 1]) for (const fc of [false, true]) add('VXALL', {start: recStart(R), ...vertex(b, d, fc)}); }
// 7. HC: 止まりにくい隅の全部の組 64 × 記録の分布の開始 4
{ const R = mulberry32(0xC4A10E04); for (let b = 0; b < 64; b++) for (let k = 0; k < 4; k++) add('HC', {start: recStart(R), ...hardCorner(b)}); }
// 8. DIN: 隅の近くの内側(摩擦円・ブレーキ U(0.7, 0.75)・μ U(0.85, 0.88)・前進のゲイン U(1.02, 1.04)・舵の速さは端の近く、ほかは一様)200
{ const R = mulberry32(0xC4A10E05);
  for (let i = 0; i < 200; i++) {
    const d = drUniform(R);
    d.phys.brakeDeceleration = .7 + .05 * R(); d.phys.frictionMultiplier = .85 + .03 * R(); d.vg = 1.02 + .02 * R(); d.fc = true;
    d.phys.steerRate = R() < .5 ? .8 + .1 * R() : 1.7 + .1 * R();
    add('DIN', {start: recStart(R), ...d});
  } }
// 9. DRL: DR-T 一様 × 開始のすき間 12〜15 cm の開始 200
{ const R = mulberry32(0xC4A10E06); for (let i = 0; i < 200; i++) add('DRL', {start: recStart(R, .12, .15), ...drUniform(R)}); }
// 10. VXC: DR-T の頂点(乱数)× 分布の角の開始(全ゲート ±0.19〜0.2 m・±13〜15°)256
{ const R = mulberry32(0xC4A10E07);
  for (let i = 0; i < 256; i++) {
    let s = null;
    for (let t = 0; t < 1000 && !s; t++) {
      const gate = Math.floor(R() * 33), off = r4((R() < .5 ? -1 : 1) * (.19 + .01 * R())), yaw = r4((R() < .5 ? -1 : 1) * (13 + 2 * R()));
      const c = startClear(gate, off, yaw); if (c >= .12) s = {gate, off, yaw, c0: +c.toFixed(5)};
    }
    add('VXC', {start: s, ...vertex(Math.floor(R() * 128), R() < .5 ? 0 : 1, R() < .3)});
  } }

// ---- 標準の物理の広い組・N4 ----
add('STD', {start: {gate: 0, off: 0, yaw: 0}, episode: 'single', T: 60});
add('STD', {start: {gate: 0, off: 0, yaw: 0}});
// 11. U: 記録の開始の分布から一様に 1500(mulberry32(0xC4A10E08))。最初の 150 は軌跡を残す(最速との比べ)
const U = [];
{ const R = mulberry32(0xC4A10E08); for (let i = 0; i < 1500; i++) { const s = recStart(R); U.push(s); add('U', {start: s, keepPath: i < 150}); } }
// 12. DRU: DR-T 一様 400
{ const R = mulberry32(0xC4A10E09); for (let i = 0; i < 400; i++) add('DRU', {start: recStart(R), ...drUniform(R)}); }
// 13. LONG: 120 秒の連続周回: 標準 8・止まりにくい隅 6
{ const R = mulberry32(0xC4A10E0A);
  for (let i = 0; i < 8; i++) add('LONG', {start: recStart(R), T: 120});
  for (let i = 0; i < 6; i++) add('LONG', {start: recStart(R), T: 120, ...hardCorner(Math.floor(R() * 64))}); }

// ---- 報告だけ ----
// 14. UOFF: U の最初の 300 をシールドなしで(対)
for (let i = 0; i < 300; i++) add('UOFF', {start: U[i], shield: false});
// 15. FAST: 最速の方策(公開の web/rl + bundle/fastest、シールドなし)で U の最初の 150
for (let i = 0; i < 150; i++) add('FAST', {start: U[i], mode: 'fastest', keepPath: true});
// 16. APP: アプリが許す開始の全部(左右 ±0.3 m・向き ±25°、開始で重なっていないもの。記録の分布の外を含む)200
{ const R = mulberry32(0xC4A10E0B);
  for (let i = 0; i < 200; i++) {
    let s = null;
    for (let t = 0; t < 1000 && !s; t++) {
      const gate = Math.floor(R() * 33), off = r4((R() * 2 - 1) * .3), yaw = r4((R() * 2 - 1) * 25), c = startClear(gate, off, yaw);
      if (c > 0) s = {gate, off, yaw, c0: +c.toFixed(5)};
    }
    add('APP', {start: s});
  } }
// 17. 範囲の外: 低摩擦(μ × 0.4)10・遅延 2 + DR-T 一様 20
{ const R = mulberry32(0xC4A10E0C);
  for (let i = 0; i < 10; i++) add('LOWMU', {start: recStart(R), phys: {frictionMultiplier: .4}});
  for (let i = 0; i < 20; i++) { const d = drUniform(R); d.delay = 2; add('DELAY2', {start: recStart(R), ...d}); } }

fs.writeFileSync(process.argv[2], jobs.map(j => JSON.stringify(j)).join('\n') + '\n');
const bySet = {};
for (const j of jobs) bySet[j.set] = (bySet[j.set] ?? 0) + 1;
console.log(JSON.stringify({total: jobs.length, bySet, seeds: [7400000, seedN - 1]}));
