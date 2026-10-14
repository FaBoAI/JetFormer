// 確認 line r3(再実行): 新しく見つけたゲート 30 の角(左右 +0.16〜+0.20 m・向き +12〜+15°)× 右寄りの形を調べる組
// 回の種 5,920,000〜、乱数 sfc32(0xc3b3004x)。node probe30_b.mjs > probejobs.json
import {startOf} from './run_b.mjs';
function splitmix32(a) { return () => { a |= 0; a = a + 0x9e3779b9 | 0; let t = a ^ a >>> 16; t = Math.imul(t, 0x21f0aaad); t ^= t >>> 15; t = Math.imul(t, 0x735a2d97); return ((t ^= t >>> 15) >>> 0); }; }
function sfc32(seed) {
  const sm = splitmix32(seed); let a = sm(), b = sm(), c = sm(), d = sm();
  const next = () => { a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0; let t = (a + b) | 0; a = b ^ b >>> 9; b = c + (c << 3) | 0; c = (c << 21 | c >>> 11); d = d + 1 | 0; t = t + d | 0; c = c + t | 0; return t >>> 0; };
  for (let i = 0; i < 12; i++) next();
  return {u: () => next() / 4294967296, u32: () => next()};
}
const PLAN = ['const_left', 'const_left', 'const_right', 'const_right', 'const_center', 'inside', 'inside', 'outside', 'outside', 'late_apex', 'late_apex', 'early_apex', 'random', 'random', 'random'];
function shape(kind, g) {
  const u = g.u(), sd = g.u32();
  if (kind === 'const_left') return {family: 'const', c0: +(.6 + .3 * u).toFixed(4)};
  if (kind === 'const_right') return {family: 'const', c0: +(-.9 + .3 * u).toFixed(4)};
  if (kind === 'const_center') return {family: 'const', c0: +(-.2 + .4 * u).toFixed(4)};
  if (kind === 'random') return {family: 'random', seed: sd};
  return {family: kind, a: +(.5 + .4 * u).toFixed(4)};
}
const jobs = [];
let seed = 5920000;
const add = (set, gate, off, yaw, line, c_v) => { const s = startOf(gate, off, yaw); if (s.clearance < .12) return false;
  jobs.push({id: `${set}-${jobs.filter(j => j.set === set).length}`, set, shield: true, gate, off, yaw, seed: seed++, params: {line, c_v}, dpsi: +s.dpsi.toFixed(4), c_start: +s.clearance.toFixed(4)}); return true; };
const box = (g, o0, o1, y0, y1) => [o0 + (o1 - o0) * g.u(), y0 + (y1 - y0) * g.u()];
{ const g = sfc32(0xc3b30040), gs = sfc32(0xc3b30041); for (let n = 0; n < 60;) { const [o, y] = box(g, .16, .2, 12, 15); if (add('G30BOX', 30, o, y, shape(PLAN[n % 15], gs), +(.9 + .1 * gs.u()).toFixed(4))) n++; } }
{ const g = sfc32(0xc3b30042), gs = sfc32(0xc3b30043); for (let n = 0; n < 40;) { const [o, y] = box(g, .16, .2, 12, 15); if (add('G30R', 30, o, y, shape('const_right', gs), +(.9 + .1 * gs.u()).toFixed(4))) n++; } }
for (const gate of [29, 31, 32]) { const g = sfc32(0xc3b30044 + gate), gs = sfc32(0xc3b30064 + gate); for (let n = 0; n < 20;) { const [o, y] = box(g, .16, .2, 12, 15); if (add(`G${gate}R`, gate, o, y, shape('const_right', gs), +(.9 + .1 * gs.u()).toFixed(4))) n++; } }
{ const g = sfc32(0xc3b30050), gs = sfc32(0xc3b30051); for (let n = 0; n < 40;) { const [o, y] = box(g, .1, .2, 7.5, 15); if (add('G30WIDE', 30, o, y, shape('const_right', gs), +(.9 + .1 * gs.u()).toFixed(4))) n++; } }
for (const c0 of [-.9, -.85, -.8, -.75, -.7, -.6]) for (const c_v of [.9, .95, 1]) add('EX213', 30, 0.17034661718644203, 13.814682939089835, {family: 'const', c0}, c_v);
process.stdout.write(JSON.stringify({meta: {what: 'gate 30 corner probe'}, jobs}));
