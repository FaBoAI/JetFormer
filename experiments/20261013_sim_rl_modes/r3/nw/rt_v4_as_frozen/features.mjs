// RL の方策の観測(79 次元)を、正解の状態から計算する(2026-10-13、RL 環境の段)
//
// Node の環境(rl/env_core.mjs)とブラウザの差し込みモジュール(記録の段の rl-teacher-bestlap.js)が同じこのファイルを import する。
// そのため node: の import や Buffer を使わない(純粋な ESM)。乱数・時刻も使わない(決定的)。
// 中身は design.md 2.3 節 / protocol.json observation のとおり:
//
//   番号(1 始まり) 中身                                                     割る値
//   1–3    速さ v・実際の舵角 δ・ヨーレート ω(sim-core と同じ式で μg/|v| の頭打ち)  4.5・0.52・4
//   4–5    前回の行動 a0, a1(フレーム 0 は 0, 0)                               1
//   6–8    参照ループ R からの横ずれ e_y(左が正)、向きのずれ e_ψ の sin・cos       0.5・1・1
//   9–40   R 上の Δs = 0.25, 0.5, …, 4.0 m の 16 点(車の座標系の 前・左 を交互)     4
//   41–56  その 16 点の路面 μ                                                  0.8
//   57–75  車体の中心から −90°(右)〜+90°(左)を 10° おきの 19 本の距離(全障害物・外枠、最大 3 m) 3
//   76     車体の中心の路面 μ                                                  0.8
//   77–78  今の目標ゲートの線に対する along(線の手前が負)・lateral(符号つき)     2・0.65
//   79     車体の長方形と全障害物の最小距離 d_min(0.5 m で切る)                 0.5
//
// 座標の約束(sim-core と同じ): yaw = 0 は +x 向き、向き yaw の進行方向は (cos yaw, −sin yaw)。
// 正の舵・正のヨーレートは「左」に曲がり、左の向きは (−sin yaw, −cos yaw)。レイの角度も observe() と同じで −90° が右。
// sim.observe() のレイは高さ 8 cm 未満の障害物(柱の脚・ゲートの脚)を見ないので使わない。ここでは高さに関係なく全部を見る。
import {REFERENCE_LOOP, REFERENCE_LOOP_SHA256} from './reference_loop.mjs';

export const FEATURES_VERSION = 'rl-bestlap-features-v1';
export {REFERENCE_LOOP_SHA256};
export const OBS_DIM = 79;
export const G = 9.80665;
export const LOOKAHEAD_DS = Object.freeze(Array.from({length: 16}, (_, i) => .25 * (i + 1)));
export const RAY_ANGLES_DEG = Object.freeze(Array.from({length: 19}, (_, i) => -90 + 10 * i));
export const RAY_RANGE = 3;
export const DMIN_CLIP = .5;
export const SCALES = Object.freeze({speed: 4.5, steering: .52, yaw_rate: 4, lateral_error: .5, lookahead: 4, mu: .8, ray: 3,
  gate_along: 2, gate_lateral: .65, d_min: .5});
// 観測の並び(番号は 0 始まり)。報告・試験で使う
export const OBS_INDEX = Object.freeze({speed: 0, steering: 1, yaw_rate: 2, prev_a0: 3, prev_a1: 4, lateral_error: 5, heading_sin: 6, heading_cos: 7,
  lookahead_xy: 8, lookahead_mu: 40, rays: 56, center_mu: 75, gate_along: 76, gate_lateral: 77, d_min: 78});
// 進み s の射影の探索幅(前のフレームの s から −0.5〜+1.5 m)
export const PROJECTION_WINDOW = Object.freeze({behind: .5, ahead: 1.5});

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const finite = v => typeof v === 'number' && Number.isFinite(v);

// ---------------- 参照ループ R ----------------
const RL = REFERENCE_LOOP;
const LOOP_N = RL.n, LOOP_L = RL.length_m;
const LX = Float64Array.from(RL.x), LZ = Float64Array.from(RL.z), LS = Float64Array.from(RL.s);
const SEG_LEN = new Float64Array(LOOP_N), SEG_TX = new Float64Array(LOOP_N), SEG_TZ = new Float64Array(LOOP_N);
for (let i = 0; i < LOOP_N; i++) {
  const j = (i + 1) % LOOP_N, dx = LX[j] - LX[i], dz = LZ[j] - LZ[i], len = Math.hypot(dx, dz);
  SEG_LEN[i] = (i + 1 < LOOP_N ? LS[j] : LOOP_L) - LS[i];   // s の差(= 点の間の距離)
  SEG_TX[i] = dx / len; SEG_TZ[i] = dz / len;
}
export const LOOP_LENGTH = LOOP_L;
export const CANONICAL_ROUTE = RL.route;
export const GATE_S = Object.freeze([...RL.gate_s]);   // 元の targets[i](route[i+1])の R 上の s

// s(0〜L)を含む区間の番号(二分探索)
function segmentAt(sLocal) {
  let lo = 0, hi = LOOP_N - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (LS[mid] <= sLocal) lo = mid; else hi = mid - 1; }
  return lo;
}
const modL = s => ((s % LOOP_L) + LOOP_L) % LOOP_L;

// R 上の s(周をまたいだ値でよい)の位置と接線
export function loopPointAt(s, out = {x: 0, z: 0, tx: 0, tz: 0}) {
  const local = modL(s), i = segmentAt(local), u = clamp((local - LS[i]) / SEG_LEN[i], 0, 1), j = (i + 1) % LOOP_N;
  out.x = LX[i] + (LX[j] - LX[i]) * u; out.z = LZ[i] + (LZ[j] - LZ[i]) * u; out.tx = SEG_TX[i]; out.tz = SEG_TZ[i];
  return out;
}

// 周をまたいだ s の範囲 [a, b] の中で (x, z) にいちばん近い R 上の点。同じ距離なら s の小さいほう
export function projectWindow(x, z, a, b, out = {s: 0, d: 0, px: 0, pz: 0, tx: 1, tz: 0}) {
  let lap = Math.floor(a / LOOP_L), i = segmentAt(a - lap * LOOP_L), best = Infinity;
  for (let guard = 0; guard <= LOOP_N + 1; guard++) {
    const start = lap * LOOP_L + LS[i], len = SEG_LEN[i];
    if (start > b) break;
    const j = (i + 1) % LOOP_N, dx = LX[j] - LX[i], dz = LZ[j] - LZ[i];
    const uLo = Math.max(0, (a - start) / len), uHi = Math.min(1, (b - start) / len);
    if (uHi >= uLo) {
      const u = clamp(((x - LX[i]) * dx + (z - LZ[i]) * dz) / (dx * dx + dz * dz), uLo, uHi);
      const px = LX[i] + dx * u, pz = LZ[i] + dz * u, d2 = (x - px) ** 2 + (z - pz) ** 2;
      if (d2 < best) { best = d2; out.s = start + u * len; out.px = px; out.pz = pz; out.tx = SEG_TX[i]; out.tz = SEG_TZ[i]; }
    }
    i++; if (i === LOOP_N) { i = 0; lap++; }
  }
  out.d = Math.sqrt(best);
  return out;
}

// ---------------- コースの形(sim-core の normalizedWorld と同じ正規化) ----------------
// 線分(半径つき)・円(柱)・軸に平行な四角(柱の脚 0.16×0.15 m)・外枠。高さ 0 のものは sim と同じく除く。
const geometryCache = new Map();
export function prepareGeometry(worldInput) {
  const key = JSON.stringify([worldInput.segments, worldInput.disks, worldInput.surfaces, worldInput.bounds]);
  const cached = geometryCache.get(key);
  if (cached) return cached;
  const num = (v, fallback, lo, hi) => finite(v) ? clamp(v, lo, hi) : fallback;
  const point = v => Array.isArray(v) && v.length >= 2 && finite(v[0]) && finite(v[1]) ? [v[0], v[1]] : null;
  const segments = [], circles = [], rects = [];
  for (const s of Array.isArray(worldInput.segments) ? worldInput.segments : []) {
    if (!s || typeof s !== 'object') continue;
    const a = point(s.a), b = point(s.b);
    if (!a || !b || !(num(s.height, .1, 0, 10) > 0)) continue;
    const r = num(s.radius, .01, 0, 2), mx = (a[0] + b[0]) / 2, mz = (a[1] + b[1]) / 2;
    segments.push({ax: a[0], az: a[1], bx: b[0], bz: b[1], r, cx: mx, cz: mz, reach: Math.hypot(b[0] - a[0], b[1] - a[1]) / 2 + r, kind: String(s.kind || 'rail')});
  }
  for (const d of Array.isArray(worldInput.disks) ? worldInput.disks : []) {
    if (!d || typeof d !== 'object' || !finite(d.x) || !finite(d.z) || !(num(d.height, .1, 0, 10) > 0)) continue;
    const kind = String(d.kind || 'post');
    if (kind === 'post-foot') rects.push({x: d.x, z: d.z, hw: .08, hd: .075, reach: Math.hypot(.16, .15) / 2, kind});
    else { const r = num(d.radius, .02, 0, 2); circles.push({x: d.x, z: d.z, r, reach: r, kind}); }
  }
  const bi = worldInput.bounds || {};
  const bounds = {minX: num(bi.minX, -20, -1e6, 1e6), maxX: num(bi.maxX, 20, -1e6, 1e6), minZ: num(bi.minZ, -20, -1e6, 1e6), maxZ: num(bi.maxZ, 20, -1e6, 1e6)};
  if (bounds.maxX <= bounds.minX) { bounds.minX = -20; bounds.maxX = 20; }
  if (bounds.maxZ <= bounds.minZ) { bounds.minZ = -20; bounds.maxZ = 20; }
  const surfaces = (Array.isArray(worldInput.surfaces) ? worldInput.surfaces : []).filter(s => s && finite(s.x) && finite(s.z) && finite(s.w) && finite(s.d) && s.w > 0 && s.d > 0)
    .map(s => ({id: String(s.id), x: s.x, z: s.z, hw: s.w / 2, hd: s.d / 2, mu: num(s.mu, .8, .01, 3)}));
  const boundSegments = [[bounds.minX, bounds.minZ, bounds.maxX, bounds.minZ], [bounds.maxX, bounds.minZ, bounds.maxX, bounds.maxZ],
    [bounds.maxX, bounds.maxZ, bounds.minX, bounds.maxZ], [bounds.minX, bounds.maxZ, bounds.minX, bounds.minZ]]
    .map(([ax, az, bx, bz]) => ({ax, az, bx, bz, r: 0}));
  const geometry = Object.freeze({segments, circles, rects, bounds, surfaces, boundSegments,
    counts: {segments: segments.length, circles: circles.length, rects: rects.length}});
  geometryCache.set(key, geometry);
  return geometry;
}

// 路面 μ(sim-core の surfaceAt と同じ「最後に当たった領域」、既定 0.8)
export function surfaceMu(geometry, x, z, frictionMultiplier = 1) {
  let mu = .8;
  const s = geometry.surfaces;
  for (let i = 0; i < s.length; i++) if (Math.abs(x - s[i].x) <= s[i].hw && Math.abs(z - s[i].z) <= s[i].hd) mu = s[i].mu;
  return mu * frictionMultiplier;
}

// ---- 距離の小道具(sim-core の同名の関数と同じ式。こちらは自前の実装) ----
function pointSegmentDistanceSquared(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az, len2 = dx * dx + dz * dz;
  const t = len2 ? clamp(((px - ax) * dx + (pz - az) * dz) / len2, 0, 1) : 0;
  return (px - ax - t * dx) ** 2 + (pz - az - t * dz) ** 2;
}
function pointRectDistanceSquared(px, pz, hx, hz) {
  return Math.max(0, Math.abs(px) - hx) ** 2 + Math.max(0, Math.abs(pz) - hz) ** 2;
}
function segmentIntersectsRect(ax, az, bx, bz, hx, hz) {
  let low = 0, high = 1;
  for (const [origin, delta, half] of [[ax, bx - ax, hx], [az, bz - az, hz]]) {
    if (Math.abs(delta) < 1e-12) { if (Math.abs(origin) > half) return false; }
    else {
      let t0 = (-half - origin) / delta, t1 = (half - origin) / delta;
      if (t0 > t1) { const t = t0; t0 = t1; t1 = t; }
      low = Math.max(low, t0); high = Math.min(high, t1);
      if (low > high) return false;
    }
  }
  return true;
}
// 車の座標系(原点が車体の中心、x 前・z 右 = sim の localPoint と同じ)での線分と長方形 [−hx,hx]×[−hz,hz] の距離の 2 乗
function segmentRectDistanceSquared(ax, az, bx, bz, hx, hz) {
  if (segmentIntersectsRect(ax, az, bx, bz, hx, hz)) return 0;
  let m = Math.min(pointRectDistanceSquared(ax, az, hx, hz), pointRectDistanceSquared(bx, bz, hx, hz));
  m = Math.min(m, pointSegmentDistanceSquared(-hx, -hz, ax, az, bx, bz), pointSegmentDistanceSquared(-hx, hz, ax, az, bx, bz),
    pointSegmentDistanceSquared(hx, -hz, ax, az, bx, bz), pointSegmentDistanceSquared(hx, hz, ax, az, bx, bz));
  return m;
}

// 車体の長方形と全障害物の最小距離 [m](当たっていれば 0)。clip より遠いものは clip を返す。
// 車の座標系への変換は sim の localPoint と同じ: [dx·c − dz·s, dx·s + dz·c]
export function clearance(geometry, x, z, yaw, vehicleLength = .42, vehicleWidth = .19, clip = DMIN_CLIP) {
  const c = Math.cos(yaw), s = Math.sin(yaw), hx = vehicleLength / 2, hz = vehicleWidth / 2, reach = Math.hypot(hx, hz) + clip;
  let best = clip;
  const local = (px, pz, o) => { const dx = px - x, dz = pz - z; o[0] = dx * c - dz * s; o[1] = dx * s + dz * c; return o; };
  const A = [0, 0], B = [0, 0];
  for (const g of geometry.segments) {
    if (Math.sqrt(pointSegmentDistanceSquared(x, z, g.ax, g.az, g.bx, g.bz)) - g.r > reach) continue;
    local(g.ax, g.az, A); local(g.bx, g.bz, B);
    const d = Math.sqrt(segmentRectDistanceSquared(A[0], A[1], B[0], B[1], hx, hz)) - g.r;
    if (d < best) best = d;
  }
  for (const g of geometry.circles) {
    if (Math.hypot(g.x - x, g.z - z) - g.r > reach) continue;
    local(g.x, g.z, A);
    const d = Math.sqrt(pointRectDistanceSquared(A[0], A[1], hx, hz)) - g.r;
    if (d < best) best = d;
  }
  for (const g of geometry.rects) {
    if (Math.hypot(g.x - x, g.z - z) - g.reach > reach) continue;
    // 脚の四角の 4 辺を線分として、車体の長方形との距離の最小(重なれば 0)
    const corners = [[g.x - g.hw, g.z - g.hd], [g.x + g.hw, g.z - g.hd], [g.x + g.hw, g.z + g.hd], [g.x - g.hw, g.z + g.hd]];
    let m = Infinity;
    for (let k = 0; k < 4; k++) {
      local(corners[k][0], corners[k][1], A); local(corners[(k + 1) % 4][0], corners[(k + 1) % 4][1], B);
      m = Math.min(m, segmentRectDistanceSquared(A[0], A[1], B[0], B[1], hx, hz));
    }
    // 脚の中に車体の中心が入っている(ありえないが)ときも 0
    if (Math.abs(x - g.x) <= g.hw && Math.abs(z - g.z) <= g.hd) m = 0;
    const d = Math.sqrt(m);
    if (d < best) best = d;
  }
  // 外枠: 車体の 4 隅(sim の carCorners と同じ式)から枠までの距離
  const b = geometry.bounds;
  for (const [px, pz] of [[-hx, -hz], [-hx, hz], [hx, -hz], [hx, hz]]) {
    const wx = x + px * c + pz * s, wz = z - px * s + pz * c;
    const d = Math.min(wx - b.minX, b.maxX - wx, wz - b.minZ, b.maxZ - wz);
    if (d < best) best = d;
  }
  return Math.max(0, best);
}

// ---- レイ(sim-core の rayCircle / rayRectangle / rayCapsule と同じ式) ----
function rayCircle(ox, oz, dx, dz, cx, cz, radius) {
  const x = ox - cx, z = oz - cz, q = x * dx + z * dz, c = x * x + z * z - radius * radius;
  if (c <= 0) return 0;
  const disc = q * q - c;
  if (disc < 0) return Infinity;
  const t = -q - Math.sqrt(disc);
  return t >= 0 ? t : Infinity;
}
function rayRectangle(ox, oz, dx, dz, r) {
  let low = 0, high = Infinity;
  for (let k = 0; k < 2; k++) {
    const offset = k ? oz - r.z : ox - r.x, delta = k ? dz : dx, half = k ? r.hd : r.hw;
    if (Math.abs(delta) < 1e-12) { if (Math.abs(offset) > half) return Infinity; }
    else {
      let t0 = (-half - offset) / delta, t1 = (half - offset) / delta;
      if (t0 > t1) { const t = t0; t0 = t1; t1 = t; }
      low = Math.max(low, t0); high = Math.min(high, t1);
      if (low > high) return Infinity;
    }
  }
  return low;
}
function rayCapsule(ox, oz, dx, dz, g) {
  const {ax, az, bx, bz, r} = g;
  if (pointSegmentDistanceSquared(ox, oz, ax, az, bx, bz) <= r * r) return 0;
  let closest = Math.min(rayCircle(ox, oz, dx, dz, ax, az, r), rayCircle(ox, oz, dx, dz, bx, bz, r));
  const length = Math.hypot(bx - ax, bz - az);
  if (length > 1e-12) {
    const tx = (bx - ax) / length, tz = (bz - az) / length;
    const along = (ox - ax) * tx + (oz - az) * tz, across = -(ox - ax) * tz + (oz - az) * tx;
    const alongD = dx * tx + dz * tz, acrossD = -dx * tz + dz * tx;
    if (Math.abs(acrossD) > 1e-12) for (const edge of [-r, r]) {
      const t = (edge - across) / acrossD, u = along + t * alongD;
      if (t >= 0 && u >= 0 && u <= length) closest = Math.min(closest, t);
    }
  }
  return closest;
}
const RAY_COS = new Float64Array(19), RAY_SIN = new Float64Array(19);
for (let i = 0; i < 19; i++) { const a = RAY_ANGLES_DEG[i] * Math.PI / 180; RAY_COS[i] = Math.cos(a); RAY_SIN[i] = Math.sin(a); }
const RAY_STEP = Math.PI / 18;

// 19 本の距離 [m](最大 RAY_RANGE)を out[offset..offset+18] に書く(すべての障害物を全部のレイで調べる素直な版。試験の基準)
export function castRaysReference(geometry, x, z, yaw, out, offset = 0, range = RAY_RANGE) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  for (let i = 0; i < 19; i++) {
    const ca = cy * RAY_COS[i] - sy * RAY_SIN[i], sa = sy * RAY_COS[i] + cy * RAY_SIN[i], dx = ca, dz = -sa;
    let best = range;
    for (const g of geometry.segments) best = Math.min(best, rayCapsule(x, z, dx, dz, g));
    for (const g of geometry.boundSegments) best = Math.min(best, rayCapsule(x, z, dx, dz, g));
    for (const g of geometry.circles) best = Math.min(best, rayCircle(x, z, dx, dz, g.x, g.z, g.r));
    for (const g of geometry.rects) best = Math.min(best, rayRectangle(x, z, dx, dz, g));
    out[offset + i] = best;
  }
  return out;
}

// 速い版(結果は castRaysReference と同じ。試験で確かめる): 障害物を包む円が 3 m 以内のものだけを、
// その円が見える角度の範囲に入るレイだけで調べる(包む円の外のレイは当たり得ないので、結果は変わらない)
const rayDirX = new Float64Array(19), rayDirZ = new Float64Array(19), rayBest = new Float64Array(19);
let sectorLo = 0, sectorHi = -1, sectorNear = 0;
// 包む円(中心 cx, cz・半径 reach)が見える角度に入るレイの番号の範囲を sectorLo..sectorHi に入れる(無ければ hi < lo)
function raySector(x, z, yaw, cx, cz, reach, range) {
  const ddx = cx - x, ddz = cz - z, D = Math.hypot(ddx, ddz);
  sectorNear = D - reach; sectorLo = 0; sectorHi = -1;
  if (sectorNear > range) return;
  if (D <= reach + 1e-9) { sectorHi = 18; return; }
  // 中心の向き(sim の角度の約束: 向き (cos φ, −sin φ))と、包む円の見かけの半幅
  const phi = Math.atan2(-ddz, ddx), half = Math.asin(Math.min(1, reach / D)) + 1e-6;
  const rel = wrap(phi - yaw);   // −π〜π、レイ i の角度は −π/2 + i·π/18
  let lo = Math.ceil((rel - half + Math.PI / 2) / RAY_STEP - 1e-9), hi = Math.floor((rel + half + Math.PI / 2) / RAY_STEP + 1e-9);
  if (lo < 0) lo = 0; if (hi > 18) hi = 18;
  sectorLo = lo; sectorHi = hi;
}
export function castRays(geometry, x, z, yaw, out, offset = 0, range = RAY_RANGE) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  for (let i = 0; i < 19; i++) {
    const ca = cy * RAY_COS[i] - sy * RAY_SIN[i], sa = sy * RAY_COS[i] + cy * RAY_SIN[i];
    rayDirX[i] = ca; rayDirZ[i] = -sa; rayBest[i] = range;
  }
  for (const g of geometry.segments) {
    raySector(x, z, yaw, g.cx, g.cz, g.reach, range);
    for (let i = sectorLo; i <= sectorHi; i++) if (sectorNear < rayBest[i]) { const t = rayCapsule(x, z, rayDirX[i], rayDirZ[i], g); if (t < rayBest[i]) rayBest[i] = t; }
  }
  for (const g of geometry.circles) {
    raySector(x, z, yaw, g.x, g.z, g.r, range);
    for (let i = sectorLo; i <= sectorHi; i++) if (sectorNear < rayBest[i]) { const t = rayCircle(x, z, rayDirX[i], rayDirZ[i], g.x, g.z, g.r); if (t < rayBest[i]) rayBest[i] = t; }
  }
  for (const g of geometry.rects) {
    raySector(x, z, yaw, g.x, g.z, g.reach, range);
    for (let i = sectorLo; i <= sectorHi; i++) if (sectorNear < rayBest[i]) { const t = rayRectangle(x, z, rayDirX[i], rayDirZ[i], g); if (t < rayBest[i]) rayBest[i] = t; }
  }
  // 外枠(半径 0 の線分 4 本)は広いので全部のレイで調べる(枠までの距離が 3 m を超えるものは飛ばす)
  for (const g of geometry.boundSegments) {
    if (Math.sqrt(pointSegmentDistanceSquared(x, z, g.ax, g.az, g.bx, g.bz)) > range) continue;
    for (let i = 0; i < 19; i++) { const t = rayCapsule(x, z, rayDirX[i], rayDirZ[i], g); if (t < rayBest[i]) rayBest[i] = t; }
  }
  for (let i = 0; i < 19; i++) out[offset + i] = rayBest[i];
  return out;
}

// 今の目標ゲートの線に対する位置(sim-core の advanceGates と同じ向きの定義)
// along = (p − target)·n(n は前のゲート → 目標の向き、線の手前が負)、lateral = −(x − tx)·nz + (z − tz)·nx(符号つき)
export function gateStatus(route, routeIndex, x, z, out = {along: 0, lateral: 0, index: 0}) {
  const n = route.length, i = Math.min(Math.max(0, routeIndex | 0), n - 1);
  const target = route[(i + 1) % n], before = route[i];
  const len = Math.hypot(target[0] - before[0], target[1] - before[1]), nx = (target[0] - before[0]) / len, nz = (target[1] - before[1]) / len;
  out.along = (x - target[0]) * nx + (z - target[1]) * nz;
  out.lateral = -(x - target[0]) * nz + (z - target[1]) * nx;
  out.index = i;
  return out;
}

// 回した経路(アプリの reset で開始ゲート k から並べ直したもの)が、元の ROUTE の何番から始まるか
export function routeOffset(route) {
  const R = CANONICAL_ROUTE, n = R.length;
  if (!Array.isArray(route) || route.length !== n) throw new Error('features: 経路のゲート数が標準(33)と違います');
  for (let k = 0; k < n; k++) {
    let same = true;
    for (let i = 0; i < n && same; i++) { const a = route[i], b = R[(k + i) % n]; same = a[0] === b[0] && a[1] === b[1]; }
    if (same) return k;
  }
  throw new Error('features: 経路が標準の ROUTE を回したものではありません');
}

// ---------------- 観測を作る追跡器 ----------------
// 1 エピソードに 1 つ。resetProgress(フレーム 0)→ 毎フレーム advanceProgress → observe の順に呼ぶ。
// 状態として持つのは進み s(周をまたいで増え続ける)・その最大値・最後の射影だけ。
export function createFeatureTracker(worldInput, config = {}, {geometry = null} = {}) {
  const geom = geometry ?? prepareGeometry(worldInput);
  const route = worldInput.route.map(p => [p[0], p[1]]);
  const k = routeOffset(route), n = route.length;
  const wheelbase = config.wheelbase ?? .26, fm = config.frictionMultiplier ?? 1;
  const vehicleLength = config.vehicleLength ?? .42, vehicleWidth = config.vehicleWidth ?? .19;
  // 回した経路の目標 i(= route[i+1])の R 上の s。元の targets[(k+i) % n]
  const targetS = Array.from({length: n}, (_, i) => GATE_S[(k + i) % n]);
  const startS = GATE_S[(k + n - 1) % n];   // route[0](= 元の targets[k−1])
  const proj = {s: 0, d: 0, px: 0, pz: 0, tx: 1, tz: 0}, pt = {x: 0, z: 0, tx: 0, tz: 0}, gs = {along: 0, lateral: 0, index: 0};
  let s = 0, sMax = 0;

  // リセット時: 今の目標ゲートの前後の区間(前のゲートの s − 0.5 〜 目標の s + 0.5)で最も近い点
  function resetProgress(x, z, routeIndex = 0) {
    const i = Math.min(Math.max(0, routeIndex | 0), n - 1);
    const prev = i ? targetS[i - 1] : startS;
    let target = targetS[i]; while (target < prev) target += LOOP_L;
    projectWindow(x, z, prev - .5, target + .5, proj);
    s = proj.s; sMax = s;
    return s;
  }
  // 毎フレーム: 前のフレームの s の −0.5〜+1.5 m だけを探す
  function advanceProgress(x, z) {
    projectWindow(x, z, s - PROJECTION_WINDOW.behind, s + PROJECTION_WINDOW.ahead, proj);
    s = proj.s; if (s > sMax) sMax = s;
    return s;
  }
  // 79 次元(割る値で割ったもの。移動平均での正規化は学習側)を out に書く。aux に報酬・判定用の生の値を入れる
  function observe(state, routeIndex, prevA0, prevA1, out, aux = null) {
    const x = state.x, z = state.z, yaw = state.yaw, v = state.speed, delta = state.steering;
    const muC = surfaceMu(geom, x, z, fm);
    const limit = muC * G / Math.max(Math.abs(v), .01);
    const omega = clamp(v / wheelbase * Math.tan(delta), -limit, limit);
    out[0] = v / SCALES.speed; out[1] = delta / SCALES.steering; out[2] = omega / SCALES.yaw_rate;
    out[3] = prevA0; out[4] = prevA1;
    // 横ずれ(R の左が正)と向きのずれ
    const leftX = proj.tz, leftZ = -proj.tx;   // R の進行方向 (tx, tz) の左 = (tz, −tx)(yaw 表記の (−sin, −cos) と同じ)
    const ey = (x - proj.px) * leftX + (z - proj.pz) * leftZ;
    const yawR = Math.atan2(-proj.tz, proj.tx), epsi = wrap(yaw - yawR);
    out[5] = ey / SCALES.lateral_error; out[6] = Math.sin(epsi); out[7] = Math.cos(epsi);
    // 先読みの 16 点(車の座標系: 前 = (cos yaw, −sin yaw)、左 = (−sin yaw, −cos yaw))
    const c = Math.cos(yaw), sn = Math.sin(yaw);
    for (let j = 0; j < 16; j++) {
      loopPointAt(s + LOOKAHEAD_DS[j], pt);
      const dx = pt.x - x, dz = pt.z - z;
      out[8 + 2 * j] = (dx * c - dz * sn) / SCALES.lookahead;
      out[9 + 2 * j] = (-dx * sn - dz * c) / SCALES.lookahead;
      out[40 + j] = surfaceMu(geom, pt.x, pt.z, fm) / SCALES.mu;
    }
    castRays(geom, x, z, yaw, out, 56);
    for (let j = 56; j < 75; j++) out[j] /= SCALES.ray;
    out[75] = muC / SCALES.mu;
    gateStatus(route, routeIndex, x, z, gs);
    out[76] = gs.along / SCALES.gate_along; out[77] = gs.lateral / SCALES.gate_lateral;
    const dmin = clearance(geom, x, z, yaw, vehicleLength, vehicleWidth, DMIN_CLIP);
    out[78] = dmin / SCALES.d_min;
    if (aux) { aux.mu = muC; aux.omega = omega; aux.d_min = dmin; aux.lateral_error = ey; aux.heading_error = epsi;
      aux.gate_along = gs.along; aux.gate_lateral = gs.lateral; aux.s = s; aux.s_max = sMax; aux.projection_distance = proj.d; }
    return out;
  }
  return {geometry: geom, route, routeOffset: k, targetS, resetProgress, advanceProgress, observe,
    get s() { return s; }, get sMax() { return sMax; }, get projection() { return proj; }};
}

// 方策の行動 a ∈ [−1, 1]² → アプリが適用する単位 {steering [rad], speed [m/s]}(design.md 2.4 節)
export function actionToCommand(a0, a1, speedCap = 4.5, maxSteer = .52) {
  const c0 = clamp(Number.isFinite(a0) ? a0 : 0, -1, 1), c1 = clamp(Number.isFinite(a1) ? a1 : -1, -1, 1);
  return {steering: maxSteer * c0, speed: speedCap * (c1 + 1) / 2, a0: c0, a1: c1};
}
// 逆向き(教師などの操作を観測の「前回の行動」にするとき)
export function commandToAction(steering, speed, speedCap = 4.5, maxSteer = .52) {
  return {a0: clamp(steering / maxSteer, -1, 1), a1: clamp(2 * speed / speedCap - 1, -1, 1)};
}
