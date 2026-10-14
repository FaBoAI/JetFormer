// 決定的な乱数と Halton 列(2026-10-13、sim_rl_modes)。Node とブラウザで同じファイル(純粋な ESM。時刻・Math.random を使わない)。
// mix32・createRng はベストラップの rl/env_core.mjs と同じ式(protocol.json seeds.episode_rng)。
export function mix32(a, b = 0) {
  let h = (Math.imul(a >>> 0, 0x9e3779b1) ^ Math.imul((b >>> 0) + 0x7f4a7c15, 0x85ebca6b)) >>> 0;
  h ^= h >>> 16; h = Math.imul(h, 0x7feb352d) >>> 0; h ^= h >>> 15; h = Math.imul(h, 0x846ca68b) >>> 0; h ^= h >>> 16;
  return h >>> 0;
}
export function createRng(seed) {
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state; z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0; z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0; z = (z ^ (z >>> 15)) >>> 0;
    return z;
  };
  return {uint32: next, uniform: (lo = 0, hi = 1) => lo + (hi - lo) * (next() / 4294967296), get state() { return state; }};
}
// 標準正規(Box–Muller。1 回に一様を 2 つ使う)
export function gaussian(rng) {
  const u1 = Math.max(1e-12, rng.uniform()), u2 = rng.uniform();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
// Halton 列(基数 b、番号 i ≥ 1)
export function halton(i, b) {
  let f = 1, r = 0, n = i;
  while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); }
  return r;
}
// 開始の Halton(protocol seeds.halton: 基数 2・3・5 を ゲート・左右・向きに)。dist = {gate: 'U{0..32}' | 数, left_offset_m: [lo, hi], left_yaw_deg: [lo, hi]}
export function haltonStart(i, dist) {
  const g = typeof dist.gate === 'number' ? dist.gate : Math.min(32, Math.floor(halton(i, 2) * 33));
  const [o0, o1] = dist.left_offset_m, [y0, y1] = dist.left_yaw_deg;
  return {gate: g, left_offset_m: o0 + (o1 - o0) * halton(i, 3), left_yaw_deg: y0 + (y1 - y0) * halton(i, 5), halton_index: i};
}
