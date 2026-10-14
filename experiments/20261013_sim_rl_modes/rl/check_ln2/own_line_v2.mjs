// 経路いろいろ(line)の確認 r2 の自前の「目標の線 v2」(2026-10-14、確認の段)。学習・評価のコード(line_profiles.mjs・line_v2_table.json の
// 作り方 build_line_v2_table.mjs・obs_modes.mjs)を使わず、protocol_line_r2.json target_line の文言(steps・params)から書いた。
//   - 使える幅の表 runtime/room_table.json(v1 と同じ。確認 r1 で自前の多角形と照らし合わせ済み)は「データとして」rhoL・rhoR の列だけ読む
//   - R は check_ln/own_line.mjs の自前の点の列(参照ループの点をデータとして読む)
//   - κ(s): R の s − 0.1・s・s + 0.1 m の 3 点の外接円(大きさ = 2|外積|/(3 辺の積)、向き = 進む向きの角の増え方。左曲がりが正)→ σ 0.1 m のガウス
//   - b = max(0, 1/|κ| − 0.7 m) を R の曲がりの内側だけに(左曲がり = 左の上限、右曲がり = 右の上限)。内側でない所は上限なし(cap で代用)
//     → ±0.5 m の最小フィルター → σ 0.25 m のガウス → hi = min(ρ_L, b_L)、lo = max(−ρ_R, −b_R) → |傾き| ≤ 0.25 の円すいの包絡
//   - e*: c(s)·ρ(v1 と同じ)→ [lo, hi] に切る → 5 cm の格子で σ 0.3 m のガウス → もう一度 [lo, hi] に切る(random は開始の 2 m 手前から 5 周分、
//     ほかは 1 周の周期)→ 開始のつなぎ: e0(最初のフレームの自前の e_y)から s0 + 2 m まで smoothstep(s < s0 は e0)
//   - random の c(s)(種から節を引くスプライン)だけはランタイムの c の関数を使う(r1 の確認と同じ扱い。v2 の加工は自前)
import fs from 'node:fs';
import {rAt, L_OWN} from '../check_ln/own_line.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const RT = JSON.parse(fs.readFileSync(`${EXP}/rl/runtime/room_table.json`, 'utf8'));
export const N = RT.n, DS = RT.ds_m, L = RT.loop_length_m;
export const P = Object.freeze({r_min: .7, chord: .1, k_sigma: .1, min_filter: .5, b_sigma: .25, slope: .25, smooth: .3, blend: 2.0, random_laps: 5, random_before: 2.0});
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const ci = i => ((i % N) + N) % N;
function gaussPeriodic(a, sigmaM) {
  const sp = sigmaM / DS, h = Math.ceil(3 * sp), w = [];
  let tot = 0;
  for (let d = -h; d <= h; d++) { const v = Math.exp(-.5 * (d / sp) ** 2); w.push(v); tot += v; }
  return a.map((_, i) => { let s = 0; for (let d = -h; d <= h; d++) s += w[d + h] * a[ci(i + d)]; return s / tot; });
}
function minFilterPeriodic(a, m) { const h = Math.round(m / DS); return a.map((_, i) => { let v = Infinity; for (let d = -h; d <= h; d++) v = Math.min(v, a[ci(i + d)]); return v; }); }
// 3 点の外接円の符号つき曲率(自前の式: 大きさは 2|外積|/(a·b·c)、向きは p0→p1 と p1→p2 の進む向きの角の差の符号。
// sim の向きは ψ = atan2(−dz, dx)(前 = (cos ψ, −sin ψ))で、ψ が増える = 左曲がり)
function curv3(p0, p1, p2) {
  const ax = p1.x - p0.x, az = p1.z - p0.z, bx = p2.x - p1.x, bz = p2.z - p1.z, cx = p2.x - p0.x, cz = p2.z - p0.z;
  const den = Math.hypot(ax, az) * Math.hypot(bx, bz) * Math.hypot(cx, cz);
  if (!(den > 0)) return 0;
  const mag = 2 * Math.abs(ax * bz - az * bx) / den;
  const turn = wrap(Math.atan2(-bz, bx) - Math.atan2(-az, ax));
  return turn >= 0 ? mag : -mag;
}
export function buildOwnTable({cap = 2.0} = {}) {
  const kRaw = [];
  for (let i = 0; i < N; i++) { const s = i * DS; kRaw.push(curv3(rAt(s - P.chord), rAt(s), rAt(s + P.chord))); }
  const kappa = gaussPeriodic(kRaw, P.k_sigma);
  const b = kappa.map(k => Math.abs(k) > 1e-9 ? Math.min(cap, Math.max(0, 1 / Math.abs(k) - P.r_min)) : cap);
  const bL = gaussPeriodic(minFilterPeriodic(kappa.map((k, i) => k > 0 ? b[i] : cap), P.min_filter), P.b_sigma);
  const bR = gaussPeriodic(minFilterPeriodic(kappa.map((k, i) => k < 0 ? b[i] : cap), P.min_filter), P.b_sigma);
  const hi0 = [], lo0 = [];
  for (let i = 0; i < N; i++) { hi0.push(Math.min(RT.rhoL[i], bL[i])); lo0.push(Math.max(-RT.rhoR[i], -bR[i])); }
  const hi = [], lo = [];
  for (let i = 0; i < N; i++) {
    let h = Infinity, l = -Infinity;
    for (let j = 0; j < N; j++) { let d = Math.abs(i - j); d = Math.min(d, N - d) * DS; h = Math.min(h, hi0[j] + P.slope * d); l = Math.max(l, lo0[j] - P.slope * d); }
    hi.push(h); lo.push(l);
  }
  return {kappa, kRaw, hi, lo, bL, bR, cap};
}
export const OWN_TABLE = buildOwnTable();
const lin = (a, s) => { const u = (((s % L) + L) % L) / DS, i = Math.floor(u), f = u - i, i0 = ci(i), i1 = ci(i0 + 1); return a[i0] + (a[i1] - a[i0]) * f; };
export const tabAt = (field, s, T = OWN_TABLE) => lin(T[field], s);
const roomAt = (field, s) => lin(RT[field], s);
function clipLoHi(e, s, T) { const lo = lin(T.lo, s), hi = lin(T.hi, s); return lo <= hi ? clamp(e, lo, hi) : (lo + hi) / 2; }
// c(s)(modes.json line_families の文言。κ̄ は使える幅の表の kappa 列 = v1 と同じ値。確認 r1 で R から作り直して 5e-6 で一致)
export function cOf(p, s) {
  const a = p.a ?? .8;
  switch (p.family) {
    case 'const': return p.c0;
    case 'inside': return a * Math.tanh(roomAt('kappa', s) / .5);
    case 'outside': return -a * Math.tanh(roomAt('kappa', s) / .5);
    case 'late_apex': return a * Math.tanh(roomAt('kappa', s - .5) / .5);
    case 'early_apex': return a * Math.tanh(roomAt('kappa', s + .5) / .5);
    default: return null;
  }
}
function eV1(c, s) { c = clamp(c, -1, 1); const rL = roomAt('rhoL', s), rR = roomAt('rhoR', s); return c >= 0 ? c * rL : c * rR; }
// 目標の線 v2。cFn: random のときだけ外から渡す c(s)(ランタイムの関数)。e0: 開始のつなぎの元(null ならつながない)
export function createOwnV2(p, s0, e0, {cFn = null, T = OWN_TABLE} = {}) {
  const cf = cFn ?? (s => cOf(p, s));
  const sp = P.smooth / DS, h = Math.ceil(3 * sp), w = [];
  let tot = 0;
  for (let d = -h; d <= h; d++) { const v = Math.exp(-.5 * (d / sp) ** 2); w.push(v); tot += v; }
  let grid, g0 = 0, gN, periodic = p.family !== 'random';
  if (periodic) {
    gN = N;
    const raw = []; for (let i = 0; i < N; i++) { const s = i * DS; raw.push(clipLoHi(eV1(cf(s), s), s, T)); }
    grid = raw.map((_, i) => { let a = 0; for (let d = -h; d <= h; d++) a += w[d + h] * raw[ci(i + d)]; return clipLoHi(a / tot, i * DS, T); });
  } else {
    g0 = Math.floor((s0 - P.random_before) / DS) * DS; gN = Math.ceil((P.random_laps * L + P.random_before) / DS) + 2;
    const raw = []; for (let i = 0; i < gN; i++) { const s = g0 + i * DS; raw.push(clipLoHi(eV1(cf(s), s), s, T)); }
    grid = raw.map((_, i) => { let a = 0; for (let d = -h; d <= h; d++) a += w[d + h] * raw[clamp(i + d, 0, gN - 1)]; return clipLoHi(a / tot, g0 + i * DS, T); });
  }
  const lineAt = s => {
    if (periodic) return lin(grid, s);
    const u = (s - g0) / DS; if (u <= 0) return grid[0]; if (u >= gN - 1) return grid[gN - 1];
    const i = Math.floor(u), f = u - i; return grid[i] + (grid[i + 1] - grid[i]) * f;
  };
  const sstep = u => u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u);
  return {lineAt, eStar: s => e0 == null ? lineAt(s) : e0 + (lineAt(s) - e0) * sstep((s - s0) / P.blend)};
}
export {L_OWN};
