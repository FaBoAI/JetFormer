// 経路の形(目標の横ずれ e*(s))と、使える幅の表の読み出し(2026-10-13、sim_rl_modes)。Node とブラウザで同じファイル(純粋な ESM)。
// modes.json common.room_table・line_families のとおり:
//   e*(s) = c(s)·ρ_L(s)(c ≥ 0)/ c(s)·ρ_R(s)(c < 0)。c(s) ∈ [−1, 1](正が左寄り)。
//   実装の注意: ρ_L・ρ_R は「使える区間 [−ρ_R, ρ_L]」の端(R が自由でない所では片方が負になる)。e* はこの区間に切る(build.md)。
//   const: c = c0。inside: a·tanh(κ̄(s)/0.5)。outside: −a·tanh(κ̄(s)/0.5)。late_apex: a·tanh(κ̄(s − 0.5)/0.5)。early_apex: a·tanh(κ̄(s + 0.5)/0.5)。
//   random: 1.5〜3.0 m おきの節(値 U(−0.9, 0.9))を通る 3 次スプライン(自然スプライン、開始の s から 2.5 周分)。
// s は特徴の追跡器の s(周をまたいで増え続ける値)。κ̄・ρ は s mod L で引く。random だけは開始の s0 からの距離で引く。
//
// 目標の線 v2(2026-10-14、学習の段 line r2。protocol_line_r2.json target_line。createLineProfile(p, s0, {version: 'v2'}) のときだけ。既定は v1 のまま):
//   1. c(s) と e = c·ρ は v1 と同じ。それを幅の表 v2(line_v2_table: 使える幅 ∩ R の曲がりの内側の上限 1/|κ| − 0.5 m)の [lo, hi] に切る
//      (目標の点が R の曲率の中心より先になる尖りを無くす。確認 line r1 の指摘 3)
//   2. 5 cm おきの格子で σ 0.3 m のガウスでなめらかにして、もう一度 [lo, hi] に切る(random 以外は 1 周の周期の格子、random は開始から 5 周分)
//   3. 開始のつなぎ: 最初の観測で車の横ずれ e0 を受け取り(anchor)、s0 から 2.0 m の間は e0 から線へ smoothstep でつなぐ(s < s0 は e0)
//      (止まった状態から遠い線へすぐ寄せて壁・柱に近づくのを防ぐ。確認 line r1 の指摘 1)
import {ROOM_TABLE} from './room_table.mjs';
import {LINE_V2_TABLE} from './line_v2_table.mjs';
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
// opts.version: 'v1'(既定。r1 と同じ)| 'v2'(上の目標の線 v2。eStar は anchor(e0) の前は開始のつなぎなし)
export function createLineProfile(p, s0 = 0, opts = {}) {
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
  const version = opts?.version ?? 'v1';
  if (version === 'v1') return {family: fam, params: {...p}, c, eStar: s => eStarFromC(c(s), s), version: 'v1', needsAnchor: false};
  if (version !== 'v2') throw new Error('line_profiles: 知らない目標の線の版です: ' + version);
  return createV2(fam, p, c, s0, opts);
}

// ---- 目標の線 v2 ----
export const LINE_PROFILE_V2 = Object.freeze({id: 'jf-line-profile/v2', smooth_sigma_m: .3, blend_m: 2.0, random_laps: 5, random_before_m: 2.0});
const V2T = LINE_V2_TABLE;
// 幅の表 v2 の [lo, hi](lo > hi なら真ん中に寄せる)
function v2Bounds(s) {
  const u = (((s % L) + L) % L) / V2T.ds_m, i = Math.floor(u), f = u - i, i0 = ((i % V2T.n) + V2T.n) % V2T.n, i1 = (i0 + 1) % V2T.n;
  return [V2T.lo[i0] + (V2T.lo[i1] - V2T.lo[i0]) * f, V2T.hi[i0] + (V2T.hi[i1] - V2T.hi[i0]) * f];
}
function v2Clip(e, s) { const [lo, hi] = v2Bounds(s); return lo <= hi ? clamp(e, lo, hi) : (lo + hi) / 2; }
function gaussWeights(sigmaPts) {
  const h = Math.ceil(3 * sigmaPts), w = new Float64Array(2 * h + 1); let tot = 0;
  for (let d = -h; d <= h; d++) { w[d + h] = Math.exp(-.5 * (d / sigmaPts) ** 2); tot += w[d + h]; }
  for (let k = 0; k < w.length; k++) w[k] /= tot;
  return {h, w};
}
const smoothstep = u => u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u);
function createV2(fam, p, c, s0, opts) {
  const V = LINE_PROFILE_V2, periodic = fam !== 'random';
  let grid, gA, gDs, gN;
  if (periodic) {
    // 1 周の周期の格子(幅の表 v2 と同じ点)。端はつながっている
    gN = V2T.n; gDs = V2T.ds_m; gA = 0;
    const raw = new Float64Array(gN);
    for (let i = 0; i < gN; i++) { const s = i * gDs; raw[i] = v2Clip(eStarFromC(c(s), s), s); }
    const {h, w} = gaussWeights(V.smooth_sigma_m / gDs);
    grid = new Float64Array(gN);
    for (let i = 0; i < gN; i++) { let acc = 0; for (let d = -h; d <= h; d++) acc += w[d + h] * raw[((i + d) % gN + gN) % gN]; const s = i * gDs; grid[i] = v2Clip(acc, s); }
  } else {
    // random: 開始の 2 m 手前から 5 周分(20 秒 × 4.5 m/s = 90 m より長い)。端の外は格子の端の値
    // 格子の点は幅の表 v2 の点にそろえる(線形補間した線が、線形補間した [lo, hi] の外に出ないように)
    gDs = V2T.ds_m; gA = Math.floor((s0 - V.random_before_m) / gDs) * gDs; gN = Math.ceil((V.random_laps * L + V.random_before_m) / gDs) + 2;
    const raw = new Float64Array(gN);
    for (let i = 0; i < gN; i++) { const s = gA + i * gDs; raw[i] = v2Clip(eStarFromC(c(s), s), s); }
    const {h, w} = gaussWeights(V.smooth_sigma_m / gDs);
    grid = new Float64Array(gN);
    for (let i = 0; i < gN; i++) { let acc = 0; for (let d = -h; d <= h; d++) acc += w[d + h] * raw[Math.min(gN - 1, Math.max(0, i + d))]; grid[i] = v2Clip(acc, gA + i * gDs); }
  }
  const lineAt = s => {
    if (periodic) { const u = (((s % L) + L) % L) / gDs, i = Math.floor(u), f = u - i, i0 = i % gN, i1 = (i0 + 1) % gN; return grid[i0] + (grid[i1] - grid[i0]) * f; }
    const u = (s - gA) / gDs; if (u <= 0) return grid[0]; if (u >= gN - 1) return grid[gN - 1];
    const i = Math.floor(u), f = u - i; return grid[i] + (grid[i + 1] - grid[i]) * f;
  };
  let e0 = null;
  const prof = {family: fam, params: {...p}, c, version: 'v2', s0, lineAt,
    get needsAnchor() { return e0 === null; }, get e0() { return e0; },
    // 開始のつなぎの元(最初の観測の車の横ずれ)。2 回目からは無視する
    anchor(e) { if (e0 === null && Number.isFinite(e)) e0 = e; return prof; },
    eStar(s) {
      const e = lineAt(s);
      if (e0 === null) return e;
      const w = smoothstep((s - s0) / V.blend_m);
      return e0 + (e - e0) * w;
    }};
  if (opts?.e0 != null) prof.anchor(opts.e0);
  return prof;
}
// 学習の形の引き方(modes.json line.training.conditioning.families_p。const の c0 は U(−0.9, 0.9)、ほかの a は U(0.5, 0.9))
export function sampleLineParams(rng, familiesP) {
  const u = rng.uniform(), c0 = rng.uniform(-.9, .9), a = rng.uniform(.5, .9), seed = rng.uint32();
  let acc = 0, family = 'random';
  for (const f of LINE_FAMILIES) { acc += familiesP[f] ?? 0; if (u < acc) { family = f; break; } }
  return family === 'const' ? {family, c0} : family === 'random' ? {family, seed} : {family, a};
}
