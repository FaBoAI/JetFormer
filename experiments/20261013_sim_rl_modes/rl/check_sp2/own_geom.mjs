// no_wall の確認 r1 の自前の形と自前の積分(2026-10-14、確認の段)。学習・評価のコード(features.mjs・lap_audit.mjs・own_physics_r1.mjs)を使わずに書いた。
//   - すき間: 車体の長方形(世界の座標の多角形)と、コースのカプセル(線分 + 半径)・柱(円)・柱の足(軸にそろった 0.16 × 0.15 m の長方形)・外枠の距離。
//     重なれば 0 以下(外枠は外へ出た分だけ負)。凸多角形どうしの距離 = 交わらなければ「頂点と辺」の距離の最小。
//   - 積分: sim-core の step() の小区間の式を読み直して書いたもの(摩擦円は modes.json common.dr_specs.conventions の式)。
//     sim-core のフレームの軌跡と比べて、ずれを返す(合っていることを確かめてから、小区間ごとのすき間を測る)。
const G = 9.80665;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const approach = (v, t, d) => v + clamp(t - v, -d, d);

// 世界の形(sim-core の normalizedWorld と同じ読み方: 高さ 0 以下は捨てる、post-foot は長方形)
export function buildObstacles(world) {
  const caps = [], circles = [], rects = [];
  for (const s of world.segments ?? []) {
    const h = typeof s.height === 'number' ? s.height : .1;
    if (!(h > 0)) continue;
    const r = typeof s.radius === 'number' ? Math.max(0, Math.min(2, s.radius)) : .01;
    caps.push({ax: s.a[0], az: s.a[1], bx: s.b[0], bz: s.b[1], r, kind: s.kind ?? 'rail',
      cx: (s.a[0] + s.b[0]) / 2, cz: (s.a[1] + s.b[1]) / 2, reach: Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]) / 2 + r});
  }
  for (const d of world.disks ?? []) {
    const h = typeof d.height === 'number' ? d.height : .1;
    if (!(h > 0)) continue;
    if ((d.kind ?? 'post') === 'post-foot') {
      const hw = .08, hd = .075;
      rects.push({x: d.x, z: d.z, poly: [[d.x - hw, d.z - hd], [d.x + hw, d.z - hd], [d.x + hw, d.z + hd], [d.x - hw, d.z + hd]], reach: Math.hypot(hw, hd), kind: 'post-foot'});
    } else {
      const r = typeof d.radius === 'number' ? Math.max(0, Math.min(2, d.radius)) : .02;
      circles.push({x: d.x, z: d.z, r, reach: r, kind: d.kind ?? 'post'});
    }
  }
  return {caps, circles, rects, bounds: {...world.bounds}, surfaces: (world.surfaces ?? []).map(s => ({...s}))};
}
export function nearObstacles(ob, x, z, radius) {
  const near = (cx, cz, reach) => Math.hypot(cx - x, cz - z) - reach <= radius;
  return {caps: ob.caps.filter(c => near(c.cx, c.cz, c.reach)), circles: ob.circles.filter(c => near(c.x, c.z, c.reach)),
    rects: ob.rects.filter(c => near(c.x, c.z, c.reach)), bounds: ob.bounds};
}
// 車体の 4 隅(前 = (cos ψ, −sin ψ)、横 = (sin ψ, cos ψ))
export function carPoly(x, z, yaw, L = .42, W = .19) {
  const fx = Math.cos(yaw), fz = -Math.sin(yaw), lx = Math.sin(yaw), lz = Math.cos(yaw), a = L / 2, b = W / 2;
  return [[x + a * fx + b * lx, z + a * fz + b * lz], [x + a * fx - b * lx, z + a * fz - b * lz],
    [x - a * fx - b * lx, z - a * fz - b * lz], [x - a * fx + b * lx, z - a * fz + b * lz]];
}
function ptSeg(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz;
  const t = l2 > 0 ? clamp(((px - ax) * dx + (pz - az) * dz) / l2, 0, 1) : 0;
  return Math.hypot(px - ax - t * dx, pz - az - t * dz);
}
function cross(ax, az, bx, bz, cx, cz) { return (bx - ax) * (cz - az) - (bz - az) * (cx - ax); }
function segsIntersect(p1, p2, q1, q2) {
  const d1 = cross(q1[0], q1[1], q2[0], q2[1], p1[0], p1[1]), d2 = cross(q1[0], q1[1], q2[0], q2[1], p2[0], p2[1]);
  const d3 = cross(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1]), d4 = cross(p1[0], p1[1], p2[0], p2[1], q2[0], q2[1]);
  return ((d1 > 0) !== (d2 > 0)) && ((d3 > 0) !== (d4 > 0)) && d1 !== 0 && d2 !== 0 && d3 !== 0 && d4 !== 0;
}
// 凸多角形(反時計回りでも時計回りでもよい)の中に点があるか
function inPoly(poly, px, pz) {
  let sgn = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length], c = cross(a[0], a[1], b[0], b[1], px, pz);
    if (c === 0) continue;
    const s = c > 0 ? 1 : -1;
    if (!sgn) sgn = s; else if (s !== sgn) return false;
  }
  return true;
}
// 点と凸多角形の距離(中なら 0)
function ptPoly(poly, px, pz) {
  if (inPoly(poly, px, pz)) return 0;
  let m = Infinity;
  for (let i = 0; i < poly.length; i++) { const a = poly[i], b = poly[(i + 1) % poly.length]; m = Math.min(m, ptSeg(px, pz, a[0], a[1], b[0], b[1])); }
  return m;
}
// 線分と凸多角形の距離(交われば 0)
function segPoly(poly, ax, az, bx, bz) {
  if (inPoly(poly, ax, az) || inPoly(poly, bx, bz)) return 0;
  let m = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    if (segsIntersect([ax, az], [bx, bz], p, q)) return 0;
    m = Math.min(m, ptSeg(p[0], p[1], ax, az, bx, bz), ptSeg(ax, az, p[0], p[1], q[0], q[1]), ptSeg(bx, bz, p[0], p[1], q[0], q[1]));
  }
  return m;
}
function polyPoly(A, B) {
  let m = Infinity;
  for (const p of A) if (inPoly(B, p[0], p[1])) return 0;
  for (const p of B) if (inPoly(A, p[0], p[1])) return 0;
  for (let i = 0; i < A.length; i++) {
    const a = A[i], b = A[(i + 1) % A.length];
    for (let j = 0; j < B.length; j++) { const c = B[j], d = B[(j + 1) % B.length]; if (segsIntersect(a, b, c, d)) return 0; }
    for (const q of B) m = Math.min(m, ptSeg(q[0], q[1], a[0], a[1], b[0], b[1]));
  }
  for (let j = 0; j < B.length; j++) { const c = B[j], d = B[(j + 1) % B.length]; for (const p of A) m = Math.min(m, ptSeg(p[0], p[1], c[0], c[1], d[0], d[1])); }
  return m;
}
// すき間 [m](重なれば ≤ 0)。which に一番近いものの種類を入れる
export function ownClearance(ob, x, z, yaw, L = .42, W = .19, which = null) {
  const P = carPoly(x, z, yaw, L, W);
  let m = Infinity, k = null;
  for (const c of ob.caps) { const d = segPoly(P, c.ax, c.az, c.bx, c.bz) - c.r; if (d < m) { m = d; k = c.kind; } }
  for (const c of ob.circles) { const d = ptPoly(P, c.x, c.z) - c.r; if (d < m) { m = d; k = c.kind; } }
  for (const r of ob.rects) { const d = polyPoly(P, r.poly); if (d < m) { m = d; k = r.kind; } }
  const b = ob.bounds;
  for (const p of P) { const d = Math.min(p[0] - b.minX, b.maxX - p[0], p[1] - b.minZ, b.maxZ - p[1]); if (d < m) { m = d; k = 'bounds'; } }
  if (which) which.kind = k;
  return m;
}
export function surfaceMuAt(surfaces, x, z, fm) {
  let mu = .8;
  for (const s of surfaces) if (Math.abs(x - s.x) <= s.w / 2 && Math.abs(z - s.z) <= s.d / 2) mu = s.mu;
  return mu * fm;
}

// 自前の積分(1 フレーム)。cfg = sim の正規化した config(snapshot.config)。st を書き換え、小区間の姿勢を返す
export function ownStepFrame(st, cmd, cfg, surfaces, {frictionCircle = false, timeLeft = Infinity} = {}) {
  const wb = Math.min(cfg.wheelbase, cfg.vehicleLength), vr = Math.hypot(cfg.vehicleLength, cfg.vehicleWidth) / 2;
  const maxRate = cfg.maxSpeed + vr * cfg.maxSpeed / cfg.wheelbase * Math.tan(cfg.maxSteer);
  const steer = clamp(Number.isFinite(cmd.steering) ? cmd.steering : 0, -cfg.maxSteer, cfg.maxSteer);
  const speed = clamp(Number.isFinite(cmd.speed) ? cmd.speed : 0, -cfg.maxSpeed, cfg.maxSpeed);
  const total = Math.min(cfg.dt, timeLeft), poses = [];
  let el = 0;
  while (el < total - 1e-12) {
    const h = Math.min(Math.min(.01, cfg.dt), .012 / maxRate, total - el);
    const mu = surfaceMuAt(surfaces, st.x, st.z, cfg.frictionMultiplier);
    let v;
    if (cfg.longitudinalModel === 'real-drive-response-estimate-v1') {
      const rate = st.speed >= 0 && speed > st.speed ? Math.min(cfg.driveAccelerationCap, mu * G, (speed - st.speed) / cfg.driveTau) : cfg.brakeDeceleration;
      v = approach(st.speed, speed, rate * h);
    } else v = approach(st.speed, speed, cfg.acceleration * h);
    const d = approach(st.steering, steer, cfg.steerRate * h);
    let lim = mu * G / Math.max(Math.abs(v), .01), w = clamp(v / wb * Math.tan(d), -lim, lim);
    if (frictionCircle) {
      const aLat = Math.abs(v * w), avail = Math.sqrt(Math.max(0, (mu * G) ** 2 - aLat ** 2)), aLong = (v - st.speed) / h;
      if (Math.abs(aLong) > avail) { v = st.speed + Math.sign(aLong) * avail * h; lim = mu * G / Math.max(Math.abs(v), .01); w = clamp(v / wb * Math.tan(d), -lim, lim); }
    }
    const ym = st.yaw + w * h / 2;
    const prev = {x: st.x, z: st.z, yaw: st.yaw};
    st.x += v * Math.cos(ym) * h; st.z -= v * Math.sin(ym) * h; st.yaw = wrap(st.yaw + w * h); st.speed = v; st.steering = d;
    poses.push({x: st.x, z: st.z, yaw: st.yaw, px: prev.x, pz: prev.z, pyaw: prev.yaw});
    el += h;
  }
  return poses;
}

// 保存した操作列(sim.report().actions)を自前で積分し直し、小区間ごと(近いときは間も 4 分割)のすき間を測る
export function auditEpisode(world, cfg, spawn, actions, trace, {frictionCircle = false, skipLastErr = false} = {}) {
  const ob = buildObstacles(world);
  const st = {x: spawn.x, z: spawn.z, yaw: spawn.yaw, speed: 0, steering: 0};
  let minC = ownClearance(ob, st.x, st.z, st.yaw, cfg.vehicleLength, cfg.vehicleWidth), minFrame = -1, maxErr = 0, minKind = null, t = 0;
  const which = {kind: null}, perFrame = [];
  let clipFrames = 0;   // 小区間で重なったのに sim が衝突と言わないフレーム(すり抜け)
  for (let f = 0; f < actions.length; f++) {
    const near = nearObstacles(ob, st.x, st.z, 1.0);
    const poses = ownStepFrame(st, actions[f], cfg, ob.surfaces, {frictionCircle, timeLeft: cfg.timeLimit - t});
    t += cfg.dt;
    let fMin = Infinity, fKind = null;
    for (const p of poses) {
      let c = ownClearance(near, p.x, p.z, p.yaw, cfg.vehicleLength, cfg.vehicleWidth, which);
      if (c < fMin) { fMin = c; fKind = which.kind; }
      if (c < .12) for (let k = 1; k < 4; k++) {   // 小区間の間(直線と角の補間)
        const u = k / 4, x = p.px + (p.x - p.px) * u, z = p.pz + (p.z - p.pz) * u, y = p.pyaw + wrap(p.yaw - p.pyaw) * u;
        c = ownClearance(near, x, z, y, cfg.vehicleLength, cfg.vehicleWidth, which);
        if (c < fMin) { fMin = c; fKind = which.kind; }
      }
    }
    perFrame.push(fMin);
    if (fMin < minC) { minC = fMin; minFrame = f; minKind = fKind; }
    const tr = trace[f];
    if (tr && !(skipLastErr && f === actions.length - 1)) { const e = Math.hypot(tr.x - st.x, tr.z - st.z); if (e > maxErr) maxErr = e; }
    if (fMin <= 0) clipFrames++;
  }
  return {min_clearance: minC, min_frame: minFrame, min_kind: minKind, max_pos_err: maxErr, overlap_frames: clipFrames, per_frame: perFrame};
}
