// no_wall 確認 r4(2026-10-14/15、確認の段)の自前の部品。学習・評価のコード(r3/nw4/*・own_geom.mjs・fast_core_*.mjs・features.mjs の判定)と、
// 確認 r3 のコード(r3/ck_nw3/*。学習の段が回帰で使った)を使わず、この段で新しく書いたもの:
//   - 物理: ユーザーの元の dist/sim-core.js を読み、(1) 小区間の姿勢を受け取る引っかけ 1 行と (2) 摩擦円(設定の __ck4fc が true の回だけ)を
//     文字列の置き換えで入れた写し `coreH`。摩擦円でない回は元の sim-core.js を同時に動かし、毎フレーム状態がビット単位で同じことを確かめる
//   - 開始: ユーザーの dist/app.js の reset() の式を読んで書き直したもの
//   - 障害物の形: sim-core の normalizedWorld の決まり(高さ > 0、post-foot は 0.16 × 0.15 m の軸にそろった箱、外枠)を読んで書き直したもの
//   - すき間: 車体の長方形(0.42 × 0.19 m)を多角形として、線分どうしの距離だけで全障害物との正確な距離を出す(確認 r3 や features.mjs とは別の作り)
import fs from 'node:fs';
import crypto from 'node:crypto';

export const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const SRC = fs.readFileSync(`${REL}/dist/sim-core.js`, 'utf8');
export const SIM_CORE_SHA = crypto.createHash('sha256').update(SRC).digest('hex');
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います: ' + SIM_CORE_SHA);
export const coreO = await import(`${REL}/dist/sim-core.js`);

function replaceOnce(text, from, to) {
  const i = text.indexOf(from);
  if (i < 0 || text.indexOf(from, i + 1) >= 0) throw new Error('置き換えの場所が 1 つではありません: ' + from.slice(0, 60));
  return text.slice(0, i) + to + text.slice(i + from.length);
}
// (1) 引っかけ: 受け入れた小区間の姿勢(衝突した小区間は sim が受け入れないので呼ばない)
let H = SRC;
H = replaceOnce(H, 'state = trial; metrics.distance', 'state = trial; __ck4pose && __ck4pose(state.x, state.z, state.yaw); metrics.distance');
// (2) 摩擦円(modes.json dr_specs.conventions: 縦の加減速を √((μg)² − a_lat²) まで。a_lat = |v·ω|、ω は頭打ちのあとの値。超えたら v を作り直し、ω も作り直す)
H = replaceOnce(H, `      const rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
      const limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01);
      const yawRate = clamp(rawRate, -limit, limit);`,
`      let rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
      let limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01);
      let yawRate = clamp(rawRate, -limit, limit);
      if (__ck4fc) {
        const g4 = surface.mu*9.80665, lat4 = Math.abs(nextSpeed*yawRate), lon4 = (nextSpeed-state.speed)/h;
        const avail4 = Math.sqrt(Math.max(0, g4*g4-lat4*lat4));
        if (Math.abs(lon4) > avail4) {
          nextSpeed = state.speed+Math.sign(lon4)*avail4*h;
          rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
          limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01);
          yawRate = clamp(rawRate, -limit, limit);
        }
      }`);
H = replaceOnce(H, 'export function createSimulation(worldInput, configInput = {}) {',
  'export function createSimulation(worldInput, configInput = {}) {\n  const __ck4fc = !!(configInput && configInput.__ck4fc === true);');
H = 'let __ck4pose = null;\nexport function __ck4SetPose(f) { __ck4pose = f; }\n' + H;
export const coreH = await import(`data:text/javascript;base64,${Buffer.from(H).toString('base64')}`);
export const CORE_H_SHA = crypto.createHash('sha256').update(H).digest('hex');

// ---- コース ----
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();

// 開始(app.js reset(): startIndex = trunc(start-gate) を 0..n−1 に、route を回す、heading = atan2(−(next.z − first.z), next.x − first.x)、
//   offset は ±0.3 m・turn は ±25° に丸め、spawn = first + offset·(sin h, −cos h)、yaw = heading + turn。variation の向きは turn·180/π)
export function makeStart(gate, offset, yawDeg, world = BASE_WORLD) {
  const n = world.route.length;
  const k = Math.max(0, Math.min(n - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(k), ...world.route.slice(0, k)];
  const a = route[0], b = route[1];
  const heading = Math.atan2(-(b[1] - a[1]), b[0] - a[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0));
  const turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {world: {...world, route, spawn: {x: a[0] + off * Math.sin(heading), z: a[1] - off * Math.cos(heading), yaw: heading + turn}},
    variation: {gate: k, left_offset_m: off, left_yaw_deg: turn * 180 / Math.PI}};
}

// ---- 障害物 ----
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const lim = (v, d, lo, hi) => isNum(v) ? Math.max(lo, Math.min(hi, v)) : d;
export function obstacles(world) {
  const caps = [], dots = [], boxes = [];
  for (const s of world.segments ?? []) {
    if (!s || typeof s !== 'object' || !Array.isArray(s.a) || !Array.isArray(s.b)) continue;
    if (!isNum(s.a[0]) || !isNum(s.a[1]) || !isNum(s.b[0]) || !isNum(s.b[1])) continue;
    if (!(lim(s.height, .1, 0, 10) > 0)) continue;
    const r = lim(s.radius, .01, 0, 2);
    caps.push({a: [s.a[0], s.a[1]], b: [s.b[0], s.b[1]], r, kind: String(s.kind || 'rail'),
      c: [(s.a[0] + s.b[0]) / 2, (s.a[1] + s.b[1]) / 2], reach: Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]) / 2 + r});
  }
  for (const d of world.disks ?? []) {
    if (!d || typeof d !== 'object' || !isNum(d.x) || !isNum(d.z)) continue;
    if (!(lim(d.height, .1, 0, 10) > 0)) continue;
    const kind = String(d.kind || 'post');
    if (kind === 'post-foot') {
      const hx = .08, hz = .075;
      boxes.push({c: [d.x, d.z], kind, reach: Math.hypot(hx, hz),
        poly: [[d.x - hx, d.z - hz], [d.x + hx, d.z - hz], [d.x + hx, d.z + hz], [d.x - hx, d.z + hz]]});
    } else dots.push({c: [d.x, d.z], r: lim(d.radius, .02, 0, 2), kind});
  }
  const bi = world.bounds ?? {};
  const B = {minX: lim(bi.minX, -20, -1e6, 1e6), maxX: lim(bi.maxX, 20, -1e6, 1e6), minZ: lim(bi.minZ, -20, -1e6, 1e6), maxZ: lim(bi.maxZ, 20, -1e6, 1e6)};
  if (B.maxX <= B.minX) { B.minX = -20; B.maxX = 20; }
  if (B.maxZ <= B.minZ) { B.minZ = -20; B.maxZ = 20; }
  const surf = (world.surfaces ?? []).filter(s => s && isNum(s.x) && isNum(s.z) && isNum(s.w) && isNum(s.d) && s.w > 0 && s.d > 0)
    .map(s => ({x: s.x, z: s.z, w: s.w, d: s.d, mu: lim(s.mu, .8, .01, 3)}));
  return {caps, dots, boxes, B, surf};
}
export function muOf(O, x, z, fm) {
  let mu = .8;
  for (const s of O.surf) if (Math.abs(x - s.x) <= s.w / 2 && Math.abs(z - s.z) <= s.d / 2) mu = s.mu;
  return mu * fm;
}

// ---- 線分の幾何 ----
function cross(ox, oz, ax, az, bx, bz) { return (ax - ox) * (bz - oz) - (az - oz) * (bx - ox); }
function onSeg(px, pz, ax, az, bx, bz) { return Math.min(ax, bx) - 1e-15 <= px && px <= Math.max(ax, bx) + 1e-15 && Math.min(az, bz) - 1e-15 <= pz && pz <= Math.max(az, bz) + 1e-15; }
function segsCross(p, q, r, s) {
  const d1 = cross(r[0], r[1], s[0], s[1], p[0], p[1]), d2 = cross(r[0], r[1], s[0], s[1], q[0], q[1]);
  const d3 = cross(p[0], p[1], q[0], q[1], r[0], r[1]), d4 = cross(p[0], p[1], q[0], q[1], s[0], s[1]);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
  if (d1 === 0 && onSeg(p[0], p[1], r[0], r[1], s[0], s[1])) return true;
  if (d2 === 0 && onSeg(q[0], q[1], r[0], r[1], s[0], s[1])) return true;
  if (d3 === 0 && onSeg(r[0], r[1], p[0], p[1], q[0], q[1])) return true;
  if (d4 === 0 && onSeg(s[0], s[1], p[0], p[1], q[0], q[1])) return true;
  return false;
}
function ptSegD(p, a, b) {
  const dx = b[0] - a[0], dz = b[1] - a[1], L2 = dx * dx + dz * dz;
  let t = L2 > 0 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / L2 : 0;
  if (t < 0) t = 0; else if (t > 1) t = 1;
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dz);
}
function segSegD(p, q, r, s) {
  if (segsCross(p, q, r, s)) return 0;
  return Math.min(ptSegD(p, r, s), ptSegD(q, r, s), ptSegD(r, p, q), ptSegD(s, p, q));
}
// 凸多角形(反時計回りでも時計回りでも)の中に点があるか
function inPoly(p, P) {
  let pos = false, neg = false;
  for (let i = 0; i < P.length; i++) {
    const a = P[i], b = P[(i + 1) % P.length], c = cross(a[0], a[1], b[0], b[1], p[0], p[1]);
    if (c > 0) pos = true; else if (c < 0) neg = true;
    if (pos && neg) return false;
  }
  return true;
}
function polySegD(P, a, b) {
  if (inPoly(a, P) || inPoly(b, P)) return 0;
  let m = Infinity;
  for (let i = 0; i < P.length; i++) m = Math.min(m, segSegD(P[i], P[(i + 1) % P.length], a, b));
  return m;
}
function polyPtD(P, p) {
  if (inPoly(p, P)) return 0;
  let m = Infinity;
  for (let i = 0; i < P.length; i++) m = Math.min(m, ptSegD(p, P[i], P[(i + 1) % P.length]));
  return m;
}
function polyPolyD(P, Q) {
  for (const p of P) if (inPoly(p, Q)) return 0;
  for (const q of Q) if (inPoly(q, P)) return 0;
  let m = Infinity;
  for (let i = 0; i < P.length; i++) for (let j = 0; j < Q.length; j++) m = Math.min(m, segSegD(P[i], P[(i + 1) % P.length], Q[j], Q[(j + 1) % Q.length]));
  return m;
}
// 車体の長方形の 4 隅(世界座標)。sim の決まり: 前 = (cos ψ, −sin ψ)、左 = (sin ψ, cos ψ)… 隅 = c + u·(cos ψ, −sin ψ) + w·(sin ψ, cos ψ)
export const CAR_L = .42, CAR_W = .19;
export function carPoly(x, z, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw), hx = CAR_L / 2, hz = CAR_W / 2;
  return [[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]].map(([u, w]) => [x + u * c + w * s, z - u * s + w * c]);
}
// 車体と全障害物の最小の距離(負は重なり。外枠は隅が外へ出たぶんの負)。cap より遠いものは cap
export function clearOwn(O, x, z, yaw, cap = .4, who = null) {
  const P = carPoly(x, z, yaw), R = Math.hypot(CAR_L, CAR_W) / 2;
  let best = cap, kind = null;
  for (const g of O.caps) {
    if (Math.hypot(g.c[0] - x, g.c[1] - z) - g.reach - R > best) continue;
    const d = polySegD(P, g.a, g.b) - g.r;
    if (d < best) { best = d; kind = g.kind; }
  }
  for (const g of O.dots) {
    if (Math.hypot(g.c[0] - x, g.c[1] - z) - g.r - R > best) continue;
    const d = polyPtD(P, g.c) - g.r;
    if (d < best) { best = d; kind = g.kind; }
  }
  for (const g of O.boxes) {
    if (Math.hypot(g.c[0] - x, g.c[1] - z) - g.reach - R > best) continue;
    const d = polyPolyD(P, g.poly);
    if (d < best) { best = d; kind = g.kind; }
  }
  for (const p of P) {
    const d = Math.min(p[0] - O.B.minX, O.B.maxX - p[0], p[1] - O.B.minZ, O.B.maxZ - p[1]);
    if (d < best) { best = d; kind = 'bounds'; }
  }
  if (who) who.kind = kind;
  return best;
}

// mulberry32(この段だけの種)
export function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export const NOMINAL_CFG = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0, longitudinalModel: 'real-drive-response-estimate-v1',
  driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2});
export function startClear(gate, off, yaw) {
  const {world} = makeStart(gate, off, yaw);
  return clearOwn(obstacles(world), world.spawn.x, world.spawn.z, world.spawn.yaw, 2.0);
}
