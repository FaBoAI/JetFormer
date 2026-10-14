// no_wall 確認 r3(2026-10-14、確認の段)の自前の部品。学習・評価・確認 r1/r2 のコード(own_geom.mjs・run_nw*.mjs・fast_core_*.mjs・
// features.mjs・lap_audit.mjs・episode.mjs)を使わず、この段で新しく書いたもの:
//   - 開始の作り方(ユーザーの dist/app.js の reset の式を読んで書き直したもの)
//   - 障害物の形(sim-core の normalizedWorld の決まり: 高さ > 0・post-foot は 0.16 × 0.15 m の箱・外枠)と、車体の長方形との正確な距離
//   - 1 フレームの積分(sim-core の step() の式を書き直したもの。小区間の姿勢を返す)。摩擦円は protocol の式(縦の加減速を √((μg)² − a_lat²) まで)
//   - 摩擦円つきの物理: ユーザーの元の sim-core.js に、この段で書いた置き換えを 1 か所だけ入れたもの(近くの障害物だけを見る高速化は入れない = 衝突は全障害物で判定)
import fs from 'node:fs';
import crypto from 'node:crypto';

export const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const SRC = fs.readFileSync(`${REL}/dist/sim-core.js`, 'utf8');
export const SIM_CORE_SHA = crypto.createHash('sha256').update(SRC).digest('hex');
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います');
export const core = await import(`${REL}/dist/sim-core.js`);

// ---- 摩擦円つきの物理(自前の置き換え) ----
const FROM = `      const rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
      const limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01);
      const yawRate = clamp(rawRate, -limit, limit);`;
const TO = `      let rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
      let limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01);
      let yawRate = clamp(rawRate, -limit, limit);
      if (__ckFc) {
        const __g = surface.mu*9.80665, __lat = Math.abs(nextSpeed*yawRate), __av = Math.sqrt(Math.max(0, __g*__g-__lat*__lat)), __al = (nextSpeed-state.speed)/h;
        if (Math.abs(__al) > __av) {
          nextSpeed = state.speed+Math.sign(__al)*__av*h;
          rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering); limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01); yawRate = clamp(rawRate, -limit, limit);
        }
      }`;
const HEAD_FROM = 'export function createSimulation(worldInput, configInput = {}) {';
const HEAD_TO = 'export function createSimulation(worldInput, configInput = {}) {\n  const __ckFc = !!(configInput && configInput.ckFrictionCircle === true);';
function once(text, a, b) {
  const i = text.indexOf(a);
  if (i < 0 || text.indexOf(a, i + 1) >= 0) throw new Error('置き換えの場所が 1 つではありません');
  return text.slice(0, i) + b + text.slice(i + a.length);
}
const FC_SRC = once(once(SRC, FROM, TO), HEAD_FROM, HEAD_TO);
export const coreFc = await import(`data:text/javascript;base64,${Buffer.from(FC_SRC).toString('base64')}`);

// ---- コース(ブラウザと同じ scene.js から) ----
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();

// 開始(app.js: startIndex = trunc(gate)、経路を回す、heading = atan2(−(next.z − first.z), next.x − first.x)、
//   spawn = first + offset·(sin h, −cos h)、yaw = heading + turn。offset は ±0.3 m、turn は ±25° に丸める)
export function startWorld(world, gate, offset, yawDeg) {
  const n = world.route.length, g = Math.max(0, Math.min(n - 1, Math.trunc(gate) || 0));
  const route = world.route.map((_, i) => world.route[(i + g) % n]);
  const p0 = route[0], p1 = route[1], hd = Math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), tr = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {world: {...world, route, spawn: {x: p0[0] + off * Math.sin(hd), z: p0[1] - off * Math.cos(hd), yaw: hd + tr}},
    variation: {gate: g, left_offset_m: off, left_yaw_deg: tr * 180 / Math.PI}};
}

// ---- 障害物の形(sim-core の normalizedWorld と同じ決まりを自分で書いたもの) ----
const fin = v => typeof v === 'number' && Number.isFinite(v);
const clampN = (v, d, lo, hi) => fin(v) ? Math.max(lo, Math.min(hi, v)) : d;
export function shapes(world) {
  const segs = [], circ = [], boxes = [];
  for (const s of world.segments ?? []) {
    if (!s || !Array.isArray(s.a) || !Array.isArray(s.b)) continue;
    const r = clampN(s.radius, .01, 0, 2), h = clampN(s.height, .1, 0, 10);
    if (h > 0) segs.push({ax: s.a[0], az: s.a[1], bx: s.b[0], bz: s.b[1], r, kind: String(s.kind || 'rail'),
      cx: (s.a[0] + s.b[0]) / 2, cz: (s.a[1] + s.b[1]) / 2, half: Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]) / 2 + r});
  }
  for (const d of world.disks ?? []) {
    if (!d || !fin(d.x) || !fin(d.z)) continue;
    const r = clampN(d.radius, .02, 0, 2), h = clampN(d.height, .1, 0, 10), kind = String(d.kind || 'post');
    if (!(h > 0)) continue;
    if (kind === 'post-foot') boxes.push({x: d.x, z: d.z, hx: .08, hz: .075, kind, half: Math.hypot(.08, .075)});
    else circ.push({x: d.x, z: d.z, r, kind, half: r});
  }
  const b = world.bounds ?? {};
  let bounds = {minX: clampN(b.minX, -20, -1e6, 1e6), maxX: clampN(b.maxX, 20, -1e6, 1e6), minZ: clampN(b.minZ, -20, -1e6, 1e6), maxZ: clampN(b.maxZ, 20, -1e6, 1e6)};
  if (bounds.maxX <= bounds.minX) { bounds.minX = -20; bounds.maxX = 20; }
  if (bounds.maxZ <= bounds.minZ) { bounds.minZ = -20; bounds.maxZ = 20; }
  const surf = (world.surfaces ?? []).filter(s => s && fin(s.x) && fin(s.z) && fin(s.w) && fin(s.d) && s.w > 0 && s.d > 0)
    .map(s => ({x: s.x, z: s.z, w: s.w, d: s.d, mu: clampN(s.mu, .8, .01, 3)}));
  return {segs, circ, boxes, bounds, surf};
}
export function muAt(S, x, z, fm) {
  let mu = .8;
  for (const s of S.surf) if (Math.abs(x - s.x) <= s.w / 2 && Math.abs(z - s.z) <= s.d / 2) mu = s.mu;
  return mu * fm;
}

// 車体の座標: 前 f = (cos ψ, −sin ψ)、横 n = (sin ψ, cos ψ)。点 p の車体座標 u = (p − c)·f、w = (p − c)·n
const HX = .21, HZ = .095;
function ptRect(u, w) {   // 点と長方形 [−HX, HX] × [−HZ, HZ] の符号つき距離(中は負)
  const ex = Math.abs(u) - HX, ez = Math.abs(w) - HZ;
  if (ex <= 0 && ez <= 0) return Math.max(ex, ez);
  return Math.hypot(Math.max(ex, 0), Math.max(ez, 0));
}
function ptSeg(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
  let t = L2 > 0 ? ((px - ax) * dx + (pz - az) * dz) / L2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(px - ax - t * dx, pz - az - t * dz);
}
function segHitsRect(u0, w0, u1, w1) {   // 線分(車体座標)が長方形と交わるか(パラメーターの区間を両方の軸で切る)
  let lo = 0, hi = 1;
  for (const [o, d, h] of [[u0, u1 - u0, HX], [w0, w1 - w0, HZ]]) {
    if (Math.abs(d) < 1e-15) { if (o < -h || o > h) return false; continue; }
    let a = (-h - o) / d, b = (h - o) / d; if (a > b) [a, b] = [b, a];
    if (a > lo) lo = a; if (b < hi) hi = b; if (lo > hi) return false;
  }
  return true;
}
const CORN = [[-HX, -HZ], [-HX, HZ], [HX, -HZ], [HX, HZ]];
// 車体(x, z, ψ)とすべての障害物の最小距離。cap より遠いものは cap を返す(近いものだけ正確)
export function clearanceAt(S, x, z, yaw, cap = .3, out = null) {
  const c = Math.cos(yaw), s = Math.sin(yaw), R = Math.hypot(HX, HZ);
  let best = cap, kind = null;
  const loc = (px, pz) => { const dx = px - x, dz = pz - z; return [dx * c - dz * s, dx * s + dz * c]; };
  for (const g of S.segs) {
    if (Math.hypot(g.cx - x, g.cz - z) - g.half - R > best) continue;
    const [u0, w0] = loc(g.ax, g.az), [u1, w1] = loc(g.bx, g.bz);
    let d;
    if (segHitsRect(u0, w0, u1, w1)) d = 0;
    else {
      d = Math.min(ptRect(u0, w0), ptRect(u1, w1));
      for (const [cu, cw] of CORN) d = Math.min(d, ptSeg(cu, cw, u0, w0, u1, w1));
    }
    d -= g.r;
    if (d < best) { best = d; kind = g.kind; }
  }
  for (const k of S.circ) {
    if (Math.hypot(k.x - x, k.z - z) - k.half - R > best) continue;
    const [u, w] = loc(k.x, k.z), d = ptRect(u, w) - k.r;
    if (d < best) { best = d; kind = k.kind; }
  }
  // 車体の角(世界座標)
  const cw = CORN.map(([u, w]) => [x + u * c + w * s, z - u * s + w * c]);
  for (const b of S.boxes) {
    if (Math.hypot(b.x - x, b.z - z) - b.half - R > best) continue;
    // 凸多角形どうし: 重なり(分離軸)なら 0、そうでなければ 頂点 → 相手の多角形 の距離の最小
    const sep = (() => {
      const dx = b.x - x, dz = b.z - z, ac = Math.abs(c), as = Math.abs(s);
      return !(Math.abs(dx * c - dz * s) <= HX + b.hx * ac + b.hz * as && Math.abs(dx * s + dz * c) <= HZ + b.hx * as + b.hz * ac
        && Math.abs(dx) <= b.hx + HX * ac + HZ * as && Math.abs(dz) <= b.hz + HX * as + HZ * ac);
    })();
    let d = 0;
    if (sep) {
      d = Infinity;
      for (const [px, pz] of cw) { const ex = Math.abs(px - b.x) - b.hx, ez = Math.abs(pz - b.z) - b.hz; d = Math.min(d, Math.hypot(Math.max(ex, 0), Math.max(ez, 0))); }
      for (const [sx, sz] of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) { const [u, w] = loc(b.x + sx * b.hx, b.z + sz * b.hz); d = Math.min(d, Math.max(ptRect(u, w), 0)); }
    }
    if (d < best) { best = d; kind = b.kind; }
  }
  const B = S.bounds;
  for (const [px, pz] of cw) { const d = Math.min(px - B.minX, B.maxX - px, pz - B.minZ, B.maxZ - pz); if (d < best) { best = d; kind = 'bounds'; } }
  if (out) out.kind = kind;
  return best;
}

// ---- 1 フレームの積分(sim-core の step() の式を自分で書いたもの。衝突は見ない) ----
// st = {x, z, yaw, speed, steering}(書き換えない。新しいものを返す)、cmd = sim が受け取った指令(actionLog の値)、remaining = このフレームの時間
const wrapA = a => Math.atan2(Math.sin(a), Math.cos(a));
const toward = (v, t, d) => v + Math.max(-d, Math.min(d, t - v));
export function integrateFrame(S, cfg, st0, cmd, remaining, fc, poses) {
  const wb = Math.min(cfg.wheelbase, cfg.vehicleLength), sub = Math.min(.01, cfg.dt);
  const vr = Math.hypot(cfg.vehicleLength, cfg.vehicleWidth) / 2;
  const maxRate = cfg.maxSpeed + vr * cfg.maxSpeed / wb * Math.tan(cfg.maxSteer);
  let {x, z, yaw, speed, steering} = st0, t = 0;
  const G = 9.80665;
  while (t < remaining - 1e-12) {
    const h = Math.min(sub, .012 / maxRate, remaining - t);
    const mu = muAt(S, x, z, cfg.frictionMultiplier);
    let v1;
    if (cfg.longitudinalModel === 'real-drive-response-estimate-v1') {
      const rate = speed >= 0 && cmd.speed > speed ? Math.min(cfg.driveAccelerationCap, mu * G, (cmd.speed - speed) / cfg.driveTau) : cfg.brakeDeceleration;
      v1 = toward(speed, cmd.speed, rate * h);
    } else v1 = toward(speed, cmd.speed, cfg.acceleration * h);
    const d1 = toward(steering, cmd.steering, cfg.steerRate * h);
    const lim = v => mu * G / Math.max(Math.abs(v), .01);
    let yr = Math.max(-lim(v1), Math.min(lim(v1), v1 / wb * Math.tan(d1)));
    if (fc) {
      const aLat = Math.abs(v1 * yr), av = Math.sqrt(Math.max(0, (mu * G) ** 2 - aLat ** 2)), aL = (v1 - speed) / h;
      if (Math.abs(aL) > av) { v1 = speed + Math.sign(aL) * av * h; yr = Math.max(-lim(v1), Math.min(lim(v1), v1 / wb * Math.tan(d1))); }
    }
    const ym = yaw + yr * h / 2;
    x += v1 * Math.cos(ym) * h; z -= v1 * Math.sin(ym) * h; yaw = wrapA(yaw + yr * h); speed = v1; steering = d1;
    if (poses) poses.push([x, z, yaw]);
    t += h;
  }
  return {x, z, yaw, speed, steering};
}

// mulberry32(この段の乱数。種はこの段だけのもの)
export function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
