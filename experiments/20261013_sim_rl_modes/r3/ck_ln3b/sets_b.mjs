// 確認 line r3(再実行): 新しい組を作る。乱数は自前の sfc32(種は splitmix32 で広げる)。
// 回の種は 5,9xx,xxx(学習 45・検証 3,23x,xxx・試験 3,13x,xxx・報告 3,330,000〜・確認 r1/r2 2,1xx,xxx〜2,8xx,xxx・
// 前の確認 r3 4,710,000〜4,725,xxx・line r4 8,4xx,xxx〜8,9xx,xxx のどれとも重ならない)
// 使い方: node sets_b.mjs > jobs.json
import {startOf} from './run_b.mjs';

function splitmix32(a) { return () => { a |= 0; a = a + 0x9e3779b9 | 0; let t = a ^ a >>> 16; t = Math.imul(t, 0x21f0aaad); t ^= t >>> 15; t = Math.imul(t, 0x735a2d97); return ((t ^= t >>> 15) >>> 0); }; }
function sfc32(seed) {
  const sm = splitmix32(seed);
  let a = sm(), b = sm(), c = sm(), d = sm();
  const next = () => { a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0; let t = (a + b) | 0; a = b ^ b >>> 9; b = c + (c << 3) | 0; c = (c << 21 | c >>> 11); d = d + 1 | 0; t = t + d | 0; c = c + t | 0; return t >>> 0; };
  for (let i = 0; i < 12; i++) next();
  return {u: () => next() / 4294967296, u32: () => next()};
}
// family_plan_15 の順(const 左 2・右 2・中央 1・inside 2・outside 2・late 2・early 1・random 3)
const PLAN = ['const_left', 'const_left', 'const_right', 'const_right', 'const_center', 'inside', 'inside', 'outside', 'outside', 'late_apex', 'late_apex', 'early_apex', 'random', 'random', 'random'];
function shape(kind, g, ext = false) {
  const u = g.u(), sd = g.u32();
  if (kind === 'const_left') return {family: 'const', c0: ext ? .9 : +(.6 + .3 * u).toFixed(4)};
  if (kind === 'const_right') return {family: 'const', c0: ext ? -.9 : +(-.9 + .3 * u).toFixed(4)};
  if (kind === 'const_center') return {family: 'const', c0: ext ? 0 : +(-.2 + .4 * u).toFixed(4)};
  if (kind === 'random') return {family: 'random', seed: sd};
  return {family: kind, a: ext ? .9 : +(.5 + .4 * u).toFixed(4)};
}
const cv = (g, ext = null) => ext ?? +(.9 + .1 * g.u()).toFixed(4);

const jobs = [];
let seedNo = 5910000;
const push = (set, st, line, c_v, extra = {}) => jobs.push({id: `${set}-${jobs.filter(j => j.set === set).length}`, set, shield: true, gate: st.gate, off: st.off, yaw: st.yaw,
  seed: seedNo++, params: {line, c_v}, dpsi: +st.dpsi.toFixed(4), c_start: +st.clearance.toFixed(4), ...extra});
function draw(g, box = null) {
  for (let k = 0; k < 2000; k++) {
    let gate, off, yaw;
    if (box) { gate = box.gate; off = box.off[0] + (box.off[1] - box.off[0]) * g.u(); yaw = box.yaw[0] + (box.yaw[1] - box.yaw[0]) * g.u(); }
    else { gate = Math.floor(33 * g.u()); off = -.2 + .4 * g.u(); yaw = -15 + 30 * g.u(); }
    const s = startOf(gate, off, yaw);
    // startOf の yaw は世界の向き [rad] なので、開始の向きのずれ [度] と名前を分ける
    if (s.clearance >= .12) return {gate, off, yaw, wyaw: s.yaw, clearance: s.clearance, dpsi: s.dpsi, s_proj: s.s, tries: k + 1};
  }
  return null;
}

// U360: 開始の分布(ゲート U{0..32}・左右 ±0.2・向き ±15°・開始のすき間 ≥ 0.12 m)から一様に。大きく横を向いた割合もここで数える
{
  const g = sfc32(0xc3b30001), gs = sfc32(0xc3b30002);
  let n = 0, head = 0, tries = 0;
  while (n < 360) { const s = draw(g); tries += s.tries; if (Math.abs(s.dpsi) >= 1) head++; push('U360', s, shape(PLAN[n % 15], gs), cv(gs)); n++; }
  jobs.meta_u = {tries, head};
}
// H180: 大きく横を向いた開始だけ(|Δψ| ≥ 1 rad)
{
  const g = sfc32(0xc3b30003), gs = sfc32(0xc3b30004);
  let n = 0, drawn = 0;
  while (n < 180) { const s = draw(g); drawn++; if (Math.abs(s.dpsi) < 1) continue; push('H180', s, shape(PLAN[n % 15], gs), cv(gs)); n++; }
  jobs.meta_h = {drawn};
}
// CB264: 33 ゲート × 4 つの角の箱(|左右| 0.16〜0.20 m・|向き| 12〜15°、符号の組 4 つ)× 2。形は半分を範囲の端(強さ最大・c_v 0.9 か 1.0)
{
  const g = sfc32(0xc3b30005), gs = sfc32(0xc3b30006);
  let n = 0, skipped = [];
  for (let gate = 0; gate < 33; gate++) for (const [so, sy] of [[-1, 1], [-1, -1], [1, 1], [1, -1]]) for (let r = 0; r < 2; r++) {
    const box = {gate, off: so < 0 ? [-.2, -.16] : [.16, .2], yaw: sy < 0 ? [-15, -12] : [12, 15]};
    const s = draw(g, box);
    if (!s) { skipped.push(`${gate}${so < 0 ? '-' : '+'}${sy < 0 ? '-' : '+'}`); continue; }
    const ext = r === 1;
    push('CB264', s, shape(PLAN[n % 15], gs, ext), cv(gs, ext ? (n % 2 ? 1 : .9) : null), {corner: `${so < 0 ? '-' : '+'}${sy < 0 ? '-' : '+'}`});
    n++;
  }
  jobs.meta_cb = {skipped};
}
// BOX16・BOX17: 前の確認で崩れた箱(左右 −0.20〜−0.16・向き +12〜+15°)を新しい種で 80 回ずつ
for (const gate of [16, 17]) {
  const g = sfc32(0xc3b30010 + gate), gs = sfc32(0xc3b30020 + gate);
  for (let n = 0; n < 80; n++) { const s = draw(g, {gate, off: [-.2, -.16], yaw: [12, 15]}); push(`BOX${gate}`, s, shape(PLAN[n % 15], gs), cv(gs)); }
}
// EDGE132: 範囲の辺(左右 ±0.2 で向き一様、向き ±15° で左右一様)× 33 ゲート
{
  const g = sfc32(0xc3b30030), gs = sfc32(0xc3b30031);
  let n = 0;
  for (let gate = 0; gate < 33; gate++) for (const e of ['off-', 'off+', 'yaw-', 'yaw+']) {
    const box = e === 'off-' ? {gate, off: [-.2, -.2], yaw: [-15, 15]} : e === 'off+' ? {gate, off: [.2, .2], yaw: [-15, 15]} : e === 'yaw-' ? {gate, off: [-.2, .2], yaw: [-15, -15]} : {gate, off: [-.2, .2], yaw: [15, 15]};
    const s = draw(g, box);
    if (!s) continue;
    push('EDGE132', s, shape(PLAN[n % 15], gs), cv(gs), {edge: e});
    n++;
  }
}
const meta = {u: jobs.meta_u, h: jobs.meta_h, cb: jobs.meta_cb};
process.stdout.write(JSON.stringify({meta, jobs}, null, 0));
