// 経路いろいろ(line)の確認 r1 の自前の目標の線(2026-10-14、確認の段)。学習・評価のコード(line_profiles.mjs・obs_modes.mjs・room_table.mjs・
// build_room_table.mjs)を使わず、modes.json common.room_table・line_families の文言から書いた。
//   - 使える幅の表 runtime/room_table.json は「データとして」読むだけ(rhoL・rhoR・kappa の列)。間は線形補間(表の note のとおり)
//   - κ̄(s) は参照ループ R の点の列から自前で作り直し(向きの s ± 0.5 m の差 / 1 m)、表の kappa と比べる
//   - e*(s) = c(s)·ρ_L(s)(c ≥ 0)/ c(s)·ρ_R(s)(c < 0)。c = const: c0、inside: a·tanh(κ̄/0.5)、outside: −a·tanh(κ̄/0.5)、
//     late_apex: a·tanh(κ̄(s − 0.5)/0.5)、early_apex: a·tanh(κ̄(s + 0.5)/0.5)。random は節の作り方が種の乱数なので、ランタイムの値(info.jf.es)を使う
//   - 目標の線の点の列(R + e*·左の法線)と、その線の向き・曲率(自前の差分)
import fs from 'node:fs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const RT = JSON.parse(fs.readFileSync(`${EXP}/rl/runtime/room_table.json`, 'utf8'));
const REF = JSON.parse(fs.readFileSync(`${EXP}/rl/runtime/reference_loop.json`, 'utf8'));
export const L = RT.loop_length_m, N = RT.n, DS = RT.ds_m;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const mod = s => ((s % L) + L) % L;
export function tab(field, s) {
  const a = RT[field], u = mod(s) / DS, i = Math.floor(u), f = u - i, i0 = ((i % N) + N) % N, i1 = (i0 + 1) % N;
  return a[i0] + (a[i1] - a[i0]) * f;
}
// ---- 参照ループ R(点の列)を自前で読む ----
const NP = REF.x.length, SEG = [];
{
  let s = 0;
  for (let i = 0; i < NP; i++) {
    const j = (i + 1) % NP, ax = REF.x[i], az = REF.z[i], bx = REF.x[j], bz = REF.z[j], len = Math.hypot(bx - ax, bz - az);
    SEG.push({ax, az, s0: s, len, tx: (bx - ax) / len, tz: (bz - az) / len});
    s += len;
  }
}
export const L_OWN = SEG.reduce((a, q) => a + q.len, 0);
function segAt(s) {
  const u = ((s % L_OWN) + L_OWN) % L_OWN;
  let lo = 0, hi = NP - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (SEG[m].s0 <= u) lo = m; else hi = m - 1; }
  return {q: SEG[lo], t: u - SEG[lo].s0};
}
// R 上の点・接線・左の法線(左 = (tz, −tx)。own_extra.mjs の e_y と同じ向き)
export function rAt(s) {
  const {q, t} = segAt(s);
  return {x: q.ax + q.tx * t, z: q.az + q.tz * t, tx: q.tx, tz: q.tz, nx: q.tz, nz: -q.tx};
}
// R の向き(x-z 平面の角)。区間ごとに一定なので、なめらかにするため両隣の区間の向きを長さで補間
function headingAt(s) {
  const {q, t} = segAt(s);
  return Math.atan2(q.tz, q.tx) + 0 * t;
}
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
// κ̄(s) = (向き(s + 0.5) − 向き(s − 0.5)) / 1.0。符号: 左曲がりが正にしたいので、x-z 平面の角(z が下向き)の増え方の逆
export function kappaOwn(s) { return -wrap(headingAt(s + .5) - headingAt(s - .5)) / 1.0; }

export function cOf(p, s, kappa = 'table') {
  const K = kappa === 'own' ? kappaOwn : (x => tab('kappa', x));
  const a = p.a ?? .8;
  switch (p.family) {
    case 'const': return p.c0;
    case 'inside': return a * Math.tanh(K(s) / .5);
    case 'outside': return -a * Math.tanh(K(s) / .5);
    case 'late_apex': return a * Math.tanh(K(s - .5) / .5);
    case 'early_apex': return a * Math.tanh(K(s + .5) / .5);
    default: return null;
  }
}
// e*(s)。clipToRoom: 区間 [−ρ_R, ρ_L] に切る(modes.json の文言には無いが、ランタイムの build.md の約束。両方を出して比べる)
export function eStarOwn(p, s, {kappa = 'table', clipToRoom = true} = {}) {
  const c0 = cOf(p, s, kappa);
  if (c0 == null) return null;
  const c = clamp(c0, -1, 1), rL = tab('rhoL', s), rR = tab('rhoR', s);
  let e = c >= 0 ? c * rL : c * rR;
  if (clipToRoom) e = -rR <= rL ? clamp(e, -rR, rL) : (rL - rR) / 2;
  return e;
}
// 目標の線の点の列(s の格子 ds)。eFn(s) → e*。返り値: [{s, x, z, yaw(sim の向き), kappa(線の曲率、左が正)}]
export function targetPath(eFn, s0, s1, ds = .025) {
  const P = [];
  for (let s = s0; s <= s1 + 1e-9; s += ds) {
    const r = rAt(s), e = eFn(s);
    P.push({s, e, x: r.x + e * r.nx, z: r.z + e * r.nz});
  }
  for (let i = 0; i < P.length; i++) {
    const a = P[Math.max(0, i - 1)], b = P[Math.min(P.length - 1, i + 1)];
    // sim の向き: 前 = (cos ψ, −sin ψ) → ψ = atan2(−dz, dx)
    P[i].yaw = Math.atan2(-(b.z - a.z), b.x - a.x);
  }
  for (let i = 0; i < P.length; i++) {
    const a = P[Math.max(0, i - 2)], b = P[Math.min(P.length - 1, i + 2)];
    const len = Math.hypot(b.x - a.x, b.z - a.z) || 1e-9;
    // ψ が増える = 左曲がり(sim-core で舵が正のとき ψ が増える)
    P[i].kappa = wrap(b.yaw - a.yaw) / len;
  }
  return P;
}
export const TABLE = RT;
export const REF_LOOP = REF;
