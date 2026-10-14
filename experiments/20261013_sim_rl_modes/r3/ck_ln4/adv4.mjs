// 確認 line r4: 攻めの組(node adv4.mjs > advjobs4.json)。新しい組で最小すき間がシールドの 4 cm に張り付いた所・シールドが何度も替えた所
// (ゲート 30 の (+,+) × c0 −0.9、ゲート 32 の (−,−) × c0 +0.9、ゲート 23 の開始のすき間の端、ゲート 21・17・7 の角)のまわりを、
// 範囲の端に寄せて密に引く。崩れ(衝突・3 cm 未満・止まり・抜け出し・brake_first)が出るかを見る。回の種は 7,750,000 から
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
const r4 = x => Math.round(x * 1e4) / 1e4;
const U = (g, a, b) => a + (b - a) * g.u();
const jobs = [];
let seedNo = 7750000;
const skipped = {};
function spot(name, n, gates, off, yaw, shapeFn, cvRange) {
  const g = rngOf('ck4-adv-' + name), gs = rngOf('ck4-adv-shape-' + name);
  let k = 0;
  for (let i = 0; i < n; i++) {
    let st = null;
    for (let t = 0; t < 400 && !st; t++) {
      const gate = gates[Math.floor(g.u() * gates.length)], o = U(g, ...off), y = U(g, ...yaw), si = startInfo(gate, o, y);
      if (si.clearance >= .12) st = {gate, off: o, yaw: y, c: si.clearance, dpsi: si.dpsi};
    }
    if (!st) { skipped[name] = (skipped[name] ?? 0) + 1; continue; }
    jobs.push({id: `ADV-${name}-${k++}`, set: `ADV-${name}`, shield: true, gate: st.gate, off: st.off, yaw: st.yaw, seed: seedNo++, line: shapeFn(gs, i), c_v: r4(U(gs, ...cvRange)),
      start_c: +st.c.toFixed(4), dpsi: +st.dpsi.toFixed(4)});
  }
}
const cst = (a, b) => (g) => ({family: 'const', c0: r4(U(g, a, b))});
const fams = (list) => (g, i) => { const f = list[i % list.length]; if (f === 'random') return {family: 'random', seed: g.u32()}; if (f.startsWith('c')) return {family: 'const', c0: f === 'c+' ? r4(U(g, .85, .9)) : r4(U(g, -.9, -.85))}; return {family: f, a: r4(U(g, .8, .9))}; };
spot('G30', 60, [30], [.16, .2], [12, 15], cst(-.9, -.85), [.95, 1]);
spot('G29G31', 40, [29, 31], [.16, .2], [12, 15], cst(-.9, -.85), [.95, 1]);
spot('G32MM', 50, [32], [-.2, -.16], [-15, -12], cst(.85, .9), [.95, 1]);
spot('G31G0MM', 30, [31, 0], [-.2, -.16], [-15, -12], cst(.85, .9), [.95, 1]);
spot('G32MMF', 20, [32], [-.2, -.16], [-15, -12], fams(['outside', 'inside', 'late_apex', 'early_apex', 'random']), [.9, 1]);
spot('G23', 50, [23], [0, .1], [5, 15], fams(['inside', 'c+', 'random', 'outside', 'late_apex']), [.9, 1]);
spot('G21', 40, [21], [.16, .2], [12, 15], fams(['c+', 'random', 'inside', 'outside']), [.9, 1]);
spot('G17', 40, [17], [-.2, -.16], [12, 15], fams(['c+', 'random', 'outside', 'late_apex']), [.9, 1]);
spot('G7', 30, [7], [.18, .2], [13.5, 15], fams(['late_apex', 'early_apex', 'outside']), [.9, 1]);
process.stdout.write(JSON.stringify({created: new Date().toISOString(), seed_first: 7750000, seed_last: seedNo - 1, skipped, jobs}));
