// 追従の運転手 PP-k(2026-10-16、シミュレーター較正 S4)。Node だけ。protocol cap_test.drivers.PP-k のとおり。
//
//   - 経路: コース図の基準線(course_map.json centerline)をシミュレーターの座標に写し(x = X − 5.15, z = 2.8 − Z)、
//     周期のなめらかにした線。(a) 33 ゲートを 0.5 m 以内で通る、(b) ショートカットの開口を車幅ごと通る、
//     (c) 車の矩形で静的なすき間 ≥ 0.05 m、のどれかが成り立たなければ参照ループ R を使う。
//     さらに (a') sim の周の数え方でゲートを順に通ること を足した。基準線も R も (a') を満たさない(LH の入口のゲート
//     (3.85, 1.93) の面を越えない)ので、sim のゲート 33 点の折れ線をなめらかにした線を使う(deviations D20)。
//   - 舵: 純追従。先読み L_d = 0.30 + 0.20 v、δ_des = atan(2 L sin α / L_d)。指令 u_s は使う物理の舵の対応の逆(P_default は恒等)、±0.52 で切る。
//   - 速さ: 経路の曲率 κ(±0.3 m で平滑)から v_lim = k sqrt(0.85 μ g / max(|κ|, 0.05))、上限 4.3 m/s。
//     後ろ向きは惰性 c0 だけ(ブレーキなし)、前向きは 3 m/s²。指令 u_v = 速さの対応の逆(v_ref(s + 0.15 v))。持ち上げるだけ(u_v ≥ 0)。
// ユーザーのファイル(course_map・world・reference_loop・features.mjs)は読むだけ。
import fs from 'node:fs';
import {prepareGeometry, surfaceMu, clearance} from '/home/jetson/jetformer/jetformer/sim/static/rl/features.mjs';

export const PP_VERSION = 'jf-pp-driver/v1';
export const COURSE_MAP = '/home/jetson/jetformer_reports/overnight_20261011/course/course_map.json';
export const REFERENCE_LOOP_JSON = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/rl/reference_loop.json';
const G = 9.80665, DS = .05;
const STEER_XS = [-1, -.9, -.75, -.5, -.25, 0, .25, .5, .75, .9, 1];
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

// 閉じた折れ線を等間隔 ds に取り直す
function resampleClosed(pts, ds) {
  const n = pts.length, cum = [0];
  for (let i = 0; i < n; i++) { const a = pts[i], b = pts[(i + 1) % n]; cum.push(cum[i] + Math.hypot(b[0] - a[0], b[1] - a[1])); }
  const L = cum[n], m = Math.round(L / ds), out = [];
  let j = 0;
  for (let k = 0; k < m; k++) {
    const s = k * L / m;
    while (cum[j + 1] < s) j++;
    const a = pts[j % n], b = pts[(j + 1) % n], u = (s - cum[j]) / (cum[j + 1] - cum[j] || 1);
    out.push([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]);
  }
  return {pts: out, L};
}
// 周期のガウスの平滑(σ [m])
function smoothClosed(pts, sigmaM, ds) {
  const n = pts.length, h = Math.ceil(3 * sigmaM / ds), w = [];
  for (let k = -h; k <= h; k++) w.push(Math.exp(-.5 * (k * ds / sigmaM) ** 2));
  const ws = w.reduce((a, b) => a + b, 0);
  return pts.map((_, i) => {
    let x = 0, z = 0;
    for (let k = -h; k <= h; k++) { const p = pts[((i + k) % n + n) % n], c = w[k + h]; x += c * p[0]; z += c * p[1]; }
    return [x / ws, z / ws];
  });
}
// 点列 → 経路(s・向き・曲率)。向きは sim の約束(進行方向 (cos yaw, −sin yaw))、左回りが正の曲率
export function pathFromPoints(pts0, ds = DS) {
  const {pts, L} = resampleClosed(pts0, ds);
  const n = pts.length, h = L / n, yaw = new Float64Array(n), kap = new Float64Array(n);
  for (let i = 0; i < n; i++) { const a = pts[(i - 1 + n) % n], b = pts[(i + 1) % n]; yaw[i] = Math.atan2(-(b[1] - a[1]), b[0] - a[0]); }
  for (let i = 0; i < n; i++) kap[i] = wrap(yaw[(i + 1) % n] - yaw[(i - 1 + n) % n]) / (2 * h);
  return {x: Float64Array.from(pts, p => p[0]), z: Float64Array.from(pts, p => p[1]), yaw, kappa: kap, n, L, ds: h};
}

// 経路の確かめ (a)(b)(c)
export function checkPath(path, world, {vehicleLength = .42, vehicleWidth = .19} = {}) {
  const geom = prepareGeometry(world);
  const gateD = world.route.map(g => { let best = Infinity; for (let i = 0; i < path.n; i++) best = Math.min(best, Math.hypot(path.x[i] - g[0], path.z[i] - g[1])); return best; });
  const a = gateD.every(d => d <= .5);
  // (b) 必須の横断(z = 0 を −z 向き)を、sim と同じ halfX で開口の中に通す
  const rc = world.requiredCrossings[0];
  let crossing = null;
  for (let i = 0; i < path.n; i++) {
    const j = (i + 1) % path.n, za = (path.z[i] - rc.z) * rc.direction, zb = (path.z[j] - rc.z) * rc.direction;
    if (za < 0 && zb >= 0) {
      const f = -za / (zb - za), x = path.x[i] + (path.x[j] - path.x[i]) * f, yaw = path.yaw[i];
      const halfX = vehicleLength / 2 * Math.abs(Math.cos(yaw)) + vehicleWidth / 2 * Math.abs(Math.sin(yaw));
      crossing = {x, halfX, ok: x - halfX >= rc.minX && x + halfX <= rc.maxX};
    }
  }
  const b = !!crossing?.ok;
  let minClear = Infinity, minAt = -1;
  for (let i = 0; i < path.n; i++) {
    const c = clearance(geom, path.x[i], path.z[i], path.yaw[i], vehicleLength, vehicleWidth, .5);
    if (c < minClear) { minClear = c; minAt = i; }
  }
  const c = minClear >= .05;
  // (a') sim の数え方(advanceGates: 目標ゲートの面を負 → 正へ、横 ≤ gateHalfWidth 0.65 m)で、経路をなぞると 33 ゲートを順に通るか。
  //      開始ゲート 0・8・16・24 から 1 周ずつ(protocol の (a) は距離だけで、これが無いと周が数えられない経路を選んでしまう。deviations D20)
  const simGates = [0, 8, 16, 24].map(g => simGatePass(path, [...world.route.slice(g), ...world.route.slice(0, g)]));
  const a2 = simGates.every(r => r.passed === world.route.length);
  return {a_gates_within_0_5m: a, gate_max_dist_m: Math.max(...gateD), a2_sim_gate_rule: a2, sim_gate_check: simGates, b_shortcut: b, crossing,
    c_clearance_ge_0_05: c, min_clearance_m: minClear, min_clearance_at: minAt >= 0 ? [path.x[minAt], path.z[minAt]] : null, ok: a && a2 && b && c};
}
function simGatePass(path, route, halfWidth = .65) {
  const n = path.n; let i0 = 0, best = Infinity;
  for (let i = 0; i < n; i++) { const d = Math.hypot(path.x[i] - route[0][0], path.z[i] - route[0][1]); if (d < best) { best = d; i0 = i; } }
  const targets = [...route.slice(1), route[0]]; let ri = 0, maxLat = 0;
  for (let k = 0; k < 2 * n && ri < targets.length; k++) {
    const a = (i0 + k) % n, b = (i0 + k + 1) % n, target = targets[ri], before = ri ? targets[ri - 1] : route[0];
    const len = Math.hypot(target[0] - before[0], target[1] - before[1]), nx = (target[0] - before[0]) / len, nz = (target[1] - before[1]) / len;
    const sa = (path.x[a] - target[0]) * nx + (path.z[a] - target[1]) * nz, sb = (path.x[b] - target[0]) * nx + (path.z[b] - target[1]) * nz;
    if (sa < 0 && sb >= 0) {
      const f = -sa / (sb - sa), x = path.x[a] + (path.x[b] - path.x[a]) * f, z = path.z[a] + (path.z[b] - path.z[a]) * f;
      const lat = Math.abs(-(x - target[0]) * nz + (z - target[1]) * nx);
      if (lat <= halfWidth) { ri++; maxLat = Math.max(maxLat, lat); }
    }
  }
  return {passed: ri, max_lateral_m: +maxLat.toFixed(4), stuck_at_target: ri < targets.length ? ri : null};
}

// 経路を選ぶ。protocol: コース図の基準線 → だめなら R。どちらも (a') を満たさないときは、sim のゲート 33 点の折れ線を
// 周期でなめらかにした線(σ 0.3 m)を使う(deviations D20)。σ はなめらかにする幅
export function choosePath(world, {sigma = .15, sigmaGates = .3} = {}) {
  const cm = JSON.parse(fs.readFileSync(COURSE_MAP, 'utf8'));
  const pts = cm.centerline.xz.map(([X, Z]) => [X - 5.15, 2.8 - Z]);
  const course = pathFromPoints(smoothClosed(resampleClosed(pts, DS).pts, sigma, DS));
  const rl = JSON.parse(fs.readFileSync(REFERENCE_LOOP_JSON, 'utf8'));
  const R = pathFromPoints(rl.x.map((x, i) => [x, rl.z[i]]));
  const gates = pathFromPoints(smoothClosed(resampleClosed(world.route, DS).pts, sigmaGates, DS));
  const cands = [['course_centerline', course], ['reference_loop_R', R], ['sim_gates_smoothed', gates]];
  const checks = Object.fromEntries(cands.map(([k, p]) => [k, checkPath(p, world)]));
  const pick = cands.find(([k]) => checks[k].ok) ?? cands[1];
  return {path: pick[1], use: pick[0], checks, sigma_m: sigma, sigma_gates_m: sigmaGates};
}

// 物理ごとの対応(PP が知っている物理)。cal: compileProfile の値、default: 恒等
export function physicsMaps(kind, compiled = null) {
  if (kind === 'default') return {kind, mu: {carpet: .8, shortcut: .7, turf: .6, ptfe: .22, 'blue-mat': .65}, muDefault: .8, coast: 1.2,
    steerInv: d => Math.max(-.52, Math.min(.52, d)), speedInv: v => Math.max(0, v)};
  const P = compiled;
  const ys = P.steerYs;   // n = STEER_XS → δ(単調)
  const steerInv = d => {
    if (d >= ys[ys.length - 1]) return .52;
    if (d <= ys[0]) return -.52;
    for (let i = 0; i < ys.length - 1; i++) if (ys[i] <= d && d <= ys[i + 1]) {
      const n = ys[i + 1] > ys[i] ? STEER_XS[i] + (STEER_XS[i + 1] - STEER_XS[i]) * (d - ys[i]) / (ys[i + 1] - ys[i]) : STEER_XS[i];
      return Math.max(-.52, Math.min(.52, n * .52));
    }
    return 0;
  };
  // T(u) の逆(k_batt = 1 の名目。PP は電池の状態を知らない)。不感帯の上 u_db から
  const xs = P.speedXs, vs = P.speedYs;
  const speedInv = v => {
    if (v <= 0) return 0;
    for (let i = 1; i < vs.length - 1; i++) if (vs[i] <= v && v <= vs[i + 1] && vs[i + 1] > vs[i]) return xs[i] + (xs[i + 1] - xs[i]) * (v - vs[i]) / (vs[i + 1] - vs[i]);
    return xs[xs.length - 1];
  };
  return {kind, mu: {...P.mu}, muDefault: P.muDefault, coast: P.c0, steerInv, speedInv};
}

// 速さの表 v_ref(s)(経路の各点)
export function speedProfile(path, world, maps, k, {vmax = 4.3, aFwd = 3, frac = .85, kappaFloor = .05, smoothM = .3} = {}) {
  const geom = prepareGeometry({...world, surfaces: world.surfaces.map(s => ({...s, mu: Object.hasOwn(maps.mu, s.id) ? maps.mu[s.id] : maps.muDefault}))});
  const n = path.n, h = Math.round(smoothM / path.ds), ks = new Float64Array(n), v = new Float64Array(n), mu = new Float64Array(n);
  for (let i = 0; i < n; i++) { let a = 0; for (let j = -h; j <= h; j++) a += path.kappa[((i + j) % n + n) % n]; ks[i] = a / (2 * h + 1); }
  for (let i = 0; i < n; i++) { mu[i] = surfaceMu(geom, path.x[i], path.z[i], 1); v[i] = Math.min(vmax, k * Math.sqrt(frac * mu[i] * G / Math.max(Math.abs(ks[i]), kappaFloor))); }
  for (let rep = 0; rep < 3; rep++) {
    for (let i = n - 1; i >= 0; i--) { const j = (i + 1) % n; v[i] = Math.min(v[i], Math.sqrt(v[j] * v[j] + 2 * maps.coast * path.ds)); }
    for (let i = 0; i < n; i++) { const j = (i - 1 + n) % n; v[i] = Math.min(v[i], Math.sqrt(v[j] * v[j] + 2 * aFwd * path.ds)); }
  }
  return {v, kappaSmooth: ks, mu};
}

// 経路の上の進み(窓つきの射影。リセットでは向きのそろった最も近い点)
function createPathTracker(path) {
  let i0 = -1;
  function nearest(x, z, yaw, lo, hi) {
    let best = Infinity, bi = 0;
    for (let k = lo; k <= hi; k++) {
      const i = ((k % path.n) + path.n) % path.n;
      if (lo === 0 && hi === path.n - 1 && Math.cos(wrap(path.yaw[i] - yaw)) < .3) continue;
      const d = (path.x[i] - x) ** 2 + (path.z[i] - z) ** 2;
      if (d < best) { best = d; bi = k; }
    }
    return bi;
  }
  return {
    reset(x, z, yaw) { i0 = nearest(x, z, yaw, 0, path.n - 1); return i0; },
    advance(x, z, yaw) { i0 = nearest(x, z, yaw, i0 - Math.round(.5 / path.ds), i0 + Math.round(1.5 / path.ds)); return i0; },
    get i() { return i0; },
  };
}

export function createPPDriver(path, profile, maps, {k = 1.0, wheelbase = .26} = {}) {
  const tr = createPathTracker(path), n = path.n;
  const at = i => ((i % n) + n) % n;
  function action(state) {
    const {x, z, yaw} = state, v = Math.max(0, state.speed);
    const i = state.frame === 0 ? tr.reset(x, z, yaw) : tr.advance(x, z, yaw);
    const Ld = .30 + .20 * v;
    const ti = at(i + Math.round(Ld / path.ds));
    const dx = path.x[ti] - x, dz = path.z[ti] - z;
    const fx = Math.cos(yaw), fz = -Math.sin(yaw), lx = -Math.sin(yaw), lz = -Math.cos(yaw);
    const alpha = Math.atan2(dx * lx + dz * lz, dx * fx + dz * fz);
    const delta = Math.atan(2 * wheelbase * Math.sin(alpha) / Ld);
    const steering = maps.steerInv(delta);
    const vref = profile.v[at(i + Math.round(.15 * v / path.ds))];
    const speed = maps.speedInv(vref);
    return {steering, speed, info: {i, vref, delta}};
  }
  return {action, k, version: PP_VERSION};
}
