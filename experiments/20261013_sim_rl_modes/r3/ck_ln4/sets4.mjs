// 確認 line r4: 新しい組を作る(node sets4.mjs > jobs4.json)。
// 乱数は自前の xoshiro128**(種は組の名前の FNV-1a から splitmix32 で広げる)。回の種は 7,740,000 から(これまでの学習・検証・試験・
// 下調べ・報告・確認 r1〜r3 の種 = 45〜47・8xx,xxx〜9,4xx,xxx・2,1xx,xxx〜5,92x,xxx などのどれとも重ならない。rl/ と reports を grep して確かめた)。
// 開始の分布(記録): ゲート U{0..32}・左右 U(−0.2, 0.2) m・向き U(−15, 15)°・開始のすき間 ≥ 0.12 m。形は family_plan_15 の強さの範囲、c_v U(0.9, 1.0)
import {startInfo} from './run4.mjs';

function fnv(str) { let h = 2166136261; for (const ch of str) h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0; return h; }
function rngOf(label) {
  let s = fnv(label);
  const sm = () => { s = (s + 0x9e3779b9) >>> 0; let z = s; z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0; z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0; return (z ^ (z >>> 16)) >>> 0; };
  let a = sm(), b = sm(), c = sm(), d = sm();
  const rotl = (x, k) => ((x << k) | (x >>> (32 - k))) >>> 0;
  const next = () => { const r = Math.imul(rotl(Math.imul(b, 5) >>> 0, 7), 9) >>> 0; const t = (b << 9) >>> 0; c ^= a; d ^= b; b ^= c; a ^= d; c ^= t; d = rotl(d, 11); return r >>> 0; };
  for (let i = 0; i < 8; i++) next();
  return {u: () => next() / 4294967296, u32: () => next()};
}
const PLAN = ['const_left', 'const_left', 'const_right', 'const_right', 'const_center', 'inside', 'inside', 'outside', 'outside', 'late_apex', 'late_apex', 'early_apex', 'random', 'random', 'random'];
const r4 = x => Math.round(x * 1e4) / 1e4;
function shape(kind, g) {
  const u = g.u();
  if (kind === 'const_left') return {family: 'const', c0: r4(.6 + .3 * u)};
  if (kind === 'const_right') return {family: 'const', c0: r4(-.9 + .3 * u)};
  if (kind === 'const_center') return {family: 'const', c0: r4(-.2 + .4 * u)};
  if (kind === 'random') return {family: 'random', seed: g.u32()};
  return {family: kind, a: r4(.5 + .4 * u)};
}
// 範囲の端の形(強さ最大)。random は種を引く
const EXT = ['c+0.9', 'c-0.9', 'c0', 'inside0.9', 'outside0.9', 'late0.9', 'early0.9', 'random'];
function extShape(k, g) {
  if (k === 'c+0.9') return {family: 'const', c0: .9};
  if (k === 'c-0.9') return {family: 'const', c0: -.9};
  if (k === 'c0') return {family: 'const', c0: 0};
  if (k === 'random') return {family: 'random', seed: g.u32()};
  return {family: {'inside0.9': 'inside', 'outside0.9': 'outside', 'late0.9': 'late_apex', 'early0.9': 'early_apex'}[k], a: .9};
}
const cvU = g => r4(.9 + .1 * g.u());

const jobs = [], meta = {};
let seedNo = 7740000;
const add = (set, st, line, c_v, extra = {}) => {
  const n = jobs.filter(j => j.set === set).length;
  jobs.push({id: `${set}-${n}`, set, shield: true, gate: st.gate, off: st.off, yaw: st.yaw, seed: seedNo++, line, c_v,
    start_c: +st.c.toFixed(4), dpsi: +st.dpsi.toFixed(4), ...extra});
};
// 開始を引く。box = {gates: [...], off: [lo, hi], yaw: [lo, hi]}(無ければ記録の分布)。開始のすき間 ≥ 0.12 m まで引き直す(上限 400)
function draw(g, box = null, accept = null) {
  for (let k = 0; k < 400; k++) {
    const gate = box?.gates ? box.gates[Math.floor(g.u() * box.gates.length)] : Math.floor(33 * g.u());
    const off = box?.off ? box.off[0] + (box.off[1] - box.off[0]) * g.u() : -.2 + .4 * g.u();
    const yaw = box?.yaw ? box.yaw[0] + (box.yaw[1] - box.yaw[0]) * g.u() : -15 + 30 * g.u();
    const si = startInfo(gate, off, yaw);
    if (si.clearance < .12) continue;
    if (accept && !accept(si)) continue;
    return {gate, off, yaw, c: si.clearance, dpsi: si.dpsi, tries: k + 1};
  }
  return null;
}

// U300: 記録の分布から一様に
{ const g = rngOf('ck4-U300'), gs = rngOf('ck4-U300-shape'); let head = 0;
  for (let i = 0; i < 300; i++) { const s = draw(g); if (Math.abs(s.dpsi) >= 1) head++; add('U300', s, shape(PLAN[i % 15], gs), cvU(gs)); }
  meta.U300 = {large_heading: head}; }
// H200: 大きく横を向いた開始だけ(|Δψ| ≥ 1 rad)。形は family_plan_15
{ const g = rngOf('ck4-H200'), gs = rngOf('ck4-H200-shape'); let drawn = 0;
  for (let i = 0; i < 200; i++) { let s; do { s = draw(g); drawn++; } while (Math.abs(s.dpsi) < 1); add('H200', s, shape(PLAN[i % 15], gs), cvU(gs)); }
  meta.H200 = {drawn_including_rejected: drawn}; }
// HX120: 大きく横を向いた開始 × 範囲の端の形 × c_v 0.9 / 1.0
{ const g = rngOf('ck4-HX120'), gs = rngOf('ck4-HX120-shape');
  for (let i = 0; i < 120; i++) { let s; do { s = draw(g); } while (Math.abs(s.dpsi) < 1); add('HX120', s, extShape(EXT[i % 8], gs), i % 2 ? 1.0 : .9); } }
// CB: 33 ゲート × 4 つの角の箱(|左右| 0.16〜0.20・|向き| 12〜15°)× 2(1 回目は family_plan_15 の形、2 回目は範囲の端の形 × c_v 0.9/1.0)
{ const g = rngOf('ck4-CB'), gs = rngOf('ck4-CB-shape'); const skipped = []; let i = 0;
  for (let rep = 0; rep < 2; rep++) for (let gate = 0; gate < 33; gate++) for (const [so, sy] of [[-1, 1], [-1, -1], [1, 1], [1, -1]]) {
    const box = {gates: [gate], off: so > 0 ? [.16, .2] : [-.2, -.16], yaw: sy > 0 ? [12, 15] : [-15, -12]};
    const s = draw(g, box);
    if (!s) { skipped.push(`${rep}:${gate}:${so}${sy}`); continue; }
    add('CB', s, rep === 0 ? shape(PLAN[i % 15], gs) : extShape(EXT[i % 8], gs), rep === 0 ? cvU(gs) : (i % 2 ? 1.0 : .9), {corner: `${so > 0 ? '+' : '-'}${sy > 0 ? '+' : '-'}`});
    i++;
  }
  meta.CB = {skipped}; }
// VX: ちょうどの頂点(左右 ±0.2・向き ±15°)× 33 ゲート × 4。形は範囲の端を順に、c_v 0.9/1.0
{ const gs = rngOf('ck4-VX-shape'); const skipped = []; let i = 0;
  for (let gate = 0; gate < 33; gate++) for (const [so, sy] of [[-1, 1], [-1, -1], [1, 1], [1, -1]]) {
    const si = startInfo(gate, .2 * so, 15 * sy);
    if (si.clearance < .12) { skipped.push(`${gate}:${so}${sy}`); continue; }
    add('VX', {gate, off: .2 * so, yaw: 15 * sy, c: si.clearance, dpsi: si.dpsi}, extShape(EXT[(i + gate) % 8], gs), i % 2 ? 1.0 : .9, {corner: `${so > 0 ? '+' : '-'}${sy > 0 ? '+' : '-'}`});
    i++;
  }
  meta.VX = {skipped}; }
// B1617: 確認 r3 で崩れた箱(ゲート 16・17、左右 −0.20〜−0.16・向き +12〜+15°)を新しい種で 60 回ずつ + 端の形 20 回ずつ
{ const g = rngOf('ck4-B1617'), gs = rngOf('ck4-B1617-shape');
  for (const gate of [16, 17]) {
    for (let i = 0; i < 60; i++) add('B1617', draw(g, {gates: [gate], off: [-.2, -.16], yaw: [12, 15]}), shape(PLAN[i % 15], gs), cvU(gs));
    for (let i = 0; i < 20; i++) add('B1617X', draw(g, {gates: [gate], off: [-.2, -.16], yaw: [12, 15]}), extShape(EXT[i % 8], gs), i % 2 ? 1.0 : .9);
  } }
// G30R: ゲート 28〜32 の (+, +) の角の箱 × 右寄りの形 const c0 U(−0.9, −0.6) × c_v U(0.9, 1.0)、30 回ずつ(ゲート 28 は追補の箱の外の隣)
{ const g = rngOf('ck4-G30R'), gs = rngOf('ck4-G30R-shape');
  for (const gate of [28, 29, 30, 31, 32]) for (let i = 0; i < 30; i++) add('G30R', draw(g, {gates: [gate], off: [.16, .2], yaw: [12, 15]}), {family: 'const', c0: r4(-.9 + .3 * gs.u())}, cvU(gs)); }
// G30RX: 確認 r3 の崩れの条件そのもの(c0 −0.9〜−0.85、c_v 0.95〜1.0)× ゲート 29〜32 の (+, +) の箱
{ const g = rngOf('ck4-G30RX'), gs = rngOf('ck4-G30RX-shape');
  for (const gate of [29, 30, 31, 32]) for (let i = 0; i < 20; i++) add('G30RX', draw(g, {gates: [gate], off: [.16, .2], yaw: [12, 15]}), {family: 'const', c0: r4(-.9 + .05 * gs.u())}, r4(.95 + .05 * gs.u())); }
// G30W: 広めの箱(左右 +0.10〜+0.20・向き +7.5〜+15°)× ゲート 27〜33 の手前(27〜32)× 右寄り
{ const g = rngOf('ck4-G30W'), gs = rngOf('ck4-G30W-shape');
  for (let i = 0; i < 60; i++) add('G30W', draw(g, {gates: [27, 28, 29, 30, 31, 32], off: [.1, .2], yaw: [7.5, 15]}), {family: 'const', c0: r4(-.9 + .3 * gs.u())}, cvU(gs)); }
// MIR: 鏡の組み合わせ — ゲート 29〜32 の (−, −)・(−, +) の角 × 左寄りの端(c0 0.85〜0.9)、ゲート 15〜18 の (+, −)・(+, +) × 右寄りの端
{ const g = rngOf('ck4-MIR'), gs = rngOf('ck4-MIR-shape');
  for (const gate of [29, 30, 31, 32]) for (const sy of [-1, 1]) for (let i = 0; i < 5; i++) {
    const s = draw(g, {gates: [gate], off: [-.2, -.16], yaw: sy > 0 ? [12, 15] : [-15, -12]}); if (s) add('MIR', s, {family: 'const', c0: r4(.85 + .05 * gs.u())}, r4(.95 + .05 * gs.u()), {corner: `-${sy > 0 ? '+' : '-'}`}); }
  for (const gate of [15, 16, 17, 18]) for (const sy of [-1, 1]) for (let i = 0; i < 5; i++) {
    const s = draw(g, {gates: [gate], off: [.16, .2], yaw: sy > 0 ? [12, 15] : [-15, -12]}); if (s) add('MIR', s, {family: 'const', c0: r4(-.9 + .05 * gs.u())}, r4(.95 + .05 * gs.u()), {corner: `+${sy > 0 ? '+' : '-'}`}); } }
// NEAR: 開始のすき間がしきい値のすぐ上(0.12〜0.14 m)の開始。形は family_plan_15
{ const g = rngOf('ck4-NEAR'), gs = rngOf('ck4-NEAR-shape'); let drawn = 0;
  for (let i = 0; i < 100; i++) { const s = draw(g, null, si => { drawn++; return si.clearance < .14; }); add('NEAR', s, shape(PLAN[i % 15], gs), cvU(gs)); }
  meta.NEAR = {accepted_draws: drawn}; }
// RND: random の形(種を変えて)× 一様な開始 60 + 角の箱 60
{ const g = rngOf('ck4-RND'), gs = rngOf('ck4-RND-shape');
  for (let i = 0; i < 60; i++) add('RND', draw(g), {family: 'random', seed: gs.u32()}, cvU(gs));
  for (let i = 0; i < 60; i++) { const so = i % 2 ? 1 : -1, sy = (i >> 1) % 2 ? 1 : -1;
    const s = draw(g, {off: so > 0 ? [.16, .2] : [-.2, -.16], yaw: sy > 0 ? [12, 15] : [-15, -12]}); if (s) add('RNDC', s, {family: 'random', seed: gs.u32()}, cvU(gs)); } }

process.stdout.write(JSON.stringify({created: new Date().toISOString(), seed_first: 7740000, seed_last: seedNo - 1, meta, jobs}));
