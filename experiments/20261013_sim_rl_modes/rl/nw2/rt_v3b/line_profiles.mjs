// 経路の形(目標の横ずれ e*(s))と、使える幅の表の読み出し(2026-10-13、sim_rl_modes)。Node とブラウザで同じファイル(純粋な ESM)。
// modes.json common.room_table・line_families のとおり:
//   e*(s) = c(s)·ρ_L(s)(c ≥ 0)/ c(s)·ρ_R(s)(c < 0)。c(s) ∈ [−1, 1](正が左寄り)。
//   実装の注意: ρ_L・ρ_R は「使える区間 [−ρ_R, ρ_L]」の端(R が自由でない所では片方が負になる)。e* はこの区間に切る(build.md)。
//   const: c = c0。inside: a·tanh(κ̄(s)/0.5)。outside: −a·tanh(κ̄(s)/0.5)。late_apex: a·tanh(κ̄(s − 0.5)/0.5)。early_apex: a·tanh(κ̄(s + 0.5)/0.5)。
//   random: 1.5〜3.0 m おきの節(値 U(−0.9, 0.9))を通る 3 次スプライン(自然スプライン、開始の s から 2.5 周分)。
// s は特徴の追跡器の s(周をまたいで増え続ける値)。κ̄・ρ は s mod L で引く。random だけは開始の s0 からの距離で引く。
import {ROOM_TABLE} from './room_table.mjs';
import {createRng, mix32} from './rng.mjs';

export const LINE_FAMILIES = Object.freeze(['const', 'inside', 'outside', 'late_apex', 'early_apex', 'random']);
const T = ROOM_TABLE, N = T.n, DS = T.ds_m, L = T.loop_length_m;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
// 表の値を s(周をまたいだ値でよい)で線形補間
export function tableAt(field, s) {
  const a = T[field], u = (((s % L) + L) % L) / DS, i = Math.floor(u), f = u - i, i0 = ((i % N) + N) % N, i1 = (i0 + 1) % N;
  return a[i0] + (a[i1] - a[i0]) * f;
}
export function roomAt(s, out = {rhoL: 0, rhoR: 0, emid: 0, width05: 0, kappa: 0}) {
  out.rhoL = tableAt('rhoL', s); out.rhoR = tableAt('rhoR', s); out.emid = tableAt('emid', s); out.width05 = tableAt('width05', s); out.kappa = tableAt('kappa', s);
  return out;
}
// c と s から e*(区間 [−ρ_R, ρ_L] に切る)
export function eStarFromC(c, s) {
  const rL = tableAt('rhoL', s), rR = tableAt('rhoR', s);
  const e = c >= 0 ? c * rL : c * rR;
  const lo = -rR, hi = rL;
  return lo <= hi ? clamp(e, lo, hi) : (lo + hi) / 2;
}
// 自然 3 次スプライン(節 xs は増える順)
function naturalSpline(xs, ys) {
  const n = xs.length, h = [], al = new Float64Array(n), l = new Float64Array(n), mu = new Float64Array(n), z = new Float64Array(n);
  const c = new Float64Array(n), b = new Float64Array(n), d = new Float64Array(n);
  for (let i = 0; i < n - 1; i++) h.push(xs[i + 1] - xs[i]);
  for (let i = 1; i < n - 1; i++) al[i] = 3 / h[i] * (ys[i + 1] - ys[i]) - 3 / h[i - 1] * (ys[i] - ys[i - 1]);
  l[0] = 1;
  for (let i = 1; i < n - 1; i++) { l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]; mu[i] = h[i] / l[i]; z[i] = (al[i] - h[i - 1] * z[i - 1]) / l[i]; }
  l[n - 1] = 1;
  for (let j = n - 2; j >= 0; j--) { c[j] = z[j] - mu[j] * c[j + 1]; b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3; d[j] = (c[j + 1] - c[j]) / (3 * h[j]); }
  return x => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 2;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (xs[m] <= x) lo = m; else hi = m - 1; }
    const t = x - xs[lo];
    return ys[lo] + b[lo] * t + c[lo] * t * t + d[lo] * t * t * t;
  };
}
// ランダムな形の節(種から決まる)。節の間隔 U(1.5, 3.0) m・値 U(−0.9, 0.9)、開始の 3 m 手前から 2.5 周 + 3 m まで
export function randomNodes(seed) {
  const rng = createRng(mix32(seed >>> 0, 0x1ea7));
  const xs = [], ys = [];
  for (let x = -3; x <= 2.5 * L + 3; x += rng.uniform(1.5, 3.0)) { xs.push(x); ys.push(rng.uniform(-.9, .9)); }
  return {xs, ys};
}
// 形の条件 p = {family, c0?, a?, seed?} から c(s)・e*(s) を作る。s0 = 開始の s(random の原点)
export function createLineProfile(p, s0 = 0) {
  const fam = p?.family ?? 'const';
  if (!LINE_FAMILIES.includes(fam)) throw new Error('line_profiles: 知らない経路の形です: ' + fam);
  const a = p.a ?? .8, c0 = p.c0 ?? 0;
  let cf;
  if (fam === 'const') cf = () => c0;
  else if (fam === 'inside') cf = s => a * Math.tanh(tableAt('kappa', s) / .5);
  else if (fam === 'outside') cf = s => -a * Math.tanh(tableAt('kappa', s) / .5);
  else if (fam === 'late_apex') cf = s => a * Math.tanh(tableAt('kappa', s - .5) / .5);
  else if (fam === 'early_apex') cf = s => a * Math.tanh(tableAt('kappa', s + .5) / .5);
  else { const {xs, ys} = randomNodes(p.seed ?? 0), sp = naturalSpline(xs, ys); cf = s => sp(s - s0); }
  const c = s => clamp(cf(s), -1, 1);
  return {family: fam, params: {...p}, c, eStar: s => eStarFromC(c(s), s)};
}
// 学習の形の引き方(modes.json line.training.conditioning.families_p。const の c0 は U(−0.9, 0.9)、ほかの a は U(0.5, 0.9))
export function sampleLineParams(rng, familiesP) {
  const u = rng.uniform(), c0 = rng.uniform(-.9, .9), a = rng.uniform(.5, .9), seed = rng.uint32();
  let acc = 0, family = 'random';
  for (const f of LINE_FAMILIES) { acc += familiesP[f] ?? 0; if (u < acc) { family = f; break; } }
  return family === 'const' ? {family, c0} : family === 'random' ? {family, seed} : {family, a};
}
