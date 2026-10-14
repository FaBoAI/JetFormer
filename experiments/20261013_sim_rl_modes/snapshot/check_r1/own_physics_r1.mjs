// 自前の積分と、小区間ごとの判定(2026-10-13、確認の段 r1)
//
// sim-core.js の step() の式(1 次遅れの加速・一定のブレーキ・舵の速さの上限・μg/|v| のヨーレートの頭打ち・中点の積分)を
// 読んで書き直したもの。判定(衝突・すき間・ゲート・黒シート・路面)は別の方法で書いた:
//   - 車体は 4 隅の多角形。障害物との距離は「辺と辺の線分の距離」と「中に入っているか」で求める(sim の局所座標の式は使わない)
//   - 小区間の間も、位置・向きを線形に補間した姿勢を細かく調べる(フレームの間・小区間の間のすり抜けを探す)
//   - 黒シートは、z = 0 の線で切った車体の多角形の x の範囲(実際の幅)でも見る
//   - 路面は、車体の面積のうち各路面に乗っている割合も出す(sim は車体の中心だけで決める)
// refine > 1 なら小区間を refine 分の 1 にする(離散化に頼った走りでないかを見る)。
const G = 9.80665;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const finite = v => typeof v === 'number' && Number.isFinite(v);
const num = (v, fb, lo, hi) => finite(v) ? clamp(v, lo, hi) : fb;
const approach = (v, t, d) => v + clamp(t - v, -d, d);

// ---------- 幾何(自前) ----------
export function carPolygon(x, z, yaw, L, W) {
  // 進行方向 f = (cos, −sin)、左 l = (−sin, −cos)。隅は 前左・前右・後右・後左 の順(凸、周回順)
  const c = Math.cos(yaw), s = Math.sin(yaw), fx = c, fz = -s, lx = -s, lz = -c, h = L / 2, w = W / 2;
  return [[x + h * fx + w * lx, z + h * fz + w * lz], [x + h * fx - w * lx, z + h * fz - w * lz],
    [x - h * fx - w * lx, z - h * fz - w * lz], [x - h * fx + w * lx, z - h * fz + w * lz]];
}
function segSegDist(p1, p2, q1, q2) {
  // 2 つの線分の距離(交われば 0)
  const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const d1 = cross(q1, q2, p1), d2 = cross(q1, q2, p2), d3 = cross(p1, p2, q1), d4 = cross(p1, p2, q2);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return 0;
  return Math.min(ptSeg(p1, q1, q2), ptSeg(p2, q1, q2), ptSeg(q1, p1, p2), ptSeg(q2, p1, p2));
}
function ptSeg(p, a, b) {
  const dx = b[0] - a[0], dz = b[1] - a[1], l2 = dx * dx + dz * dz;
  const t = l2 ? clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / l2, 0, 1) : 0;
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dz);
}
function insideConvex(p, poly) {
  let sign = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (Math.abs(c) < 1e-15) continue;
    if (sign === 0) sign = Math.sign(c); else if (Math.sign(c) !== sign) return false;
  }
  return true;
}
function polySegDist(poly, a, b) {
  if (insideConvex(a, poly) || insideConvex(b, poly)) return 0;
  let m = Infinity;
  for (let i = 0; i < poly.length; i++) m = Math.min(m, segSegDist(poly[i], poly[(i + 1) % poly.length], a, b));
  return m;
}
function polyPolyDist(P, Q) {
  if (P.some(p => insideConvex(p, Q)) || Q.some(q => insideConvex(q, P))) return 0;
  let m = Infinity;
  for (let i = 0; i < P.length; i++) for (let j = 0; j < Q.length; j++)
    m = Math.min(m, segSegDist(P[i], P[(i + 1) % P.length], Q[j], Q[(j + 1) % Q.length]));
  return m;
}
// コースの障害物(sim の normalizedWorld と同じ取捨: 高さ 0 は除く、post-foot は 0.16 × 0.15 の四角)
export function obstacles(world) {
  const out = [];
  for (const s of world.segments ?? []) {
    const h = num(s.height, .1, 0, 10); if (!(h > 0)) continue;
    out.push({type: 'capsule', a: s.a, b: s.b, r: num(s.radius, .01, 0, 2), kind: s.kind, id: `seg ${s.kind} (${s.a.map(v => v.toFixed(2))})-(${s.b.map(v => v.toFixed(2))})`,
      cx: (s.a[0] + s.b[0]) / 2, cz: (s.a[1] + s.b[1]) / 2, reach: Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]) / 2 + num(s.radius, .01, 0, 2)});
  }
  for (const d of world.disks ?? []) {
    const h = num(d.height, .1, 0, 10); if (!(h > 0)) continue;
    if (d.kind === 'post-foot') out.push({type: 'box', poly: [[d.x - .08, d.z - .075], [d.x + .08, d.z - .075], [d.x + .08, d.z + .075], [d.x - .08, d.z + .075]],
      kind: d.kind, id: `foot (${d.x.toFixed(2)},${d.z.toFixed(2)})`, cx: d.x, cz: d.z, reach: .11});
    else out.push({type: 'disk', c: [d.x, d.z], r: num(d.radius, .02, 0, 2), kind: d.kind, id: `${d.kind} (${d.x.toFixed(2)},${d.z.toFixed(2)})`,
      cx: d.x, cz: d.z, reach: num(d.radius, .02, 0, 2)});
  }
  return out;
}
// 車体と障害物の最小距離(重なれば負ではなく 0)と、いちばん近いもの
export function clearanceOwn(poly, obs, bounds, x, z) {
  let best = Infinity, who = null;
  for (const o of obs) {
    if (Math.hypot(o.cx - x, o.cz - z) - o.reach > .5 + best && best < Infinity) continue;
    let d;
    if (o.type === 'capsule') d = polySegDist(poly, o.a, o.b) - o.r;
    else if (o.type === 'disk') d = (insideConvex(o.c, poly) ? 0 : Math.min(...poly.map((p, i) => ptSeg(o.c, p, poly[(i + 1) % 4])))) - o.r;
    else d = polyPolyDist(poly, o.poly);
    if (d < best) { best = d; who = o.id; }
  }
  for (const [px, pz] of poly) {
    const d = Math.min(px - bounds.minX, bounds.maxX - px, pz - bounds.minZ, bounds.maxZ - pz);
    if (d < best) { best = d; who = 'bounds'; }
  }
  return {d: best, who};
}
// 多角形と軸に平行な四角の重なりの面積(Sutherland–Hodgman)
function clipArea(poly, minX, maxX, minZ, maxZ) {
  let P = poly;
  const clipEdge = (P, inside, inter) => {
    const out = [];
    for (let i = 0; i < P.length; i++) {
      const a = P[i], b = P[(i + 1) % P.length], ia = inside(a), ib = inside(b);
      if (ia) out.push(a);
      if (ia !== ib) out.push(inter(a, b));
    }
    return out;
  };
  const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  P = clipEdge(P, p => p[0] >= minX, (a, b) => lerp(a, b, (minX - a[0]) / (b[0] - a[0]))); if (!P.length) return 0;
  P = clipEdge(P, p => p[0] <= maxX, (a, b) => lerp(a, b, (maxX - a[0]) / (b[0] - a[0]))); if (!P.length) return 0;
  P = clipEdge(P, p => p[1] >= minZ, (a, b) => lerp(a, b, (minZ - a[1]) / (b[1] - a[1]))); if (!P.length) return 0;
  P = clipEdge(P, p => p[1] <= maxZ, (a, b) => lerp(a, b, (maxZ - a[1]) / (b[1] - a[1]))); if (!P.length) return 0;
  let A = 0;
  for (let i = 0; i < P.length; i++) { const a = P[i], b = P[(i + 1) % P.length]; A += a[0] * b[1] - b[0] * a[1]; }
  return Math.abs(A) / 2;
}
// z = zc の線で切った多角形の x の範囲(線にかからなければ null)
function sliceX(poly, zc) {
  const xs = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    if ((a[1] - zc) * (b[1] - zc) <= 0 && a[1] !== b[1]) xs.push(a[0] + (b[0] - a[0]) * (zc - a[1]) / (b[1] - a[1]));
  }
  return xs.length ? [Math.min(...xs), Math.max(...xs)] : null;
}

// ---------- 自前の積分(sim-core の式を書き直したもの)+ 小区間ごとの判定 ----------
export function createOwnSim(world, cfgIn, {refine = 1, interp = 4, collisionEps = 0, frictionCircle = false} = {}) {
  // frictionCircle: 反事実の確認用。縦の加減速を √((μg)² − a_lat²) までにする(sim-core は縦と横を別々に頭打ちにする)
  const cfg = {...cfgIn};
  const L = cfg.vehicleLength, W = cfg.vehicleWidth, wb = Math.min(cfg.wheelbase, L);
  const obs = obstacles(world), bounds = world.bounds;
  const surfaces = (world.surfaces ?? []).map(s => ({id: s.id, x: s.x, z: s.z, w: s.w, d: s.d, mu: num(s.mu, .8, .01, 3)}));
  const route = world.route, targets = [...route.slice(1), route[0]], N = targets.length;
  const routeLength = targets.reduce((s, p, i) => s + Math.hypot(p[0] - (i ? targets[i - 1][0] : route[0][0]), p[1] - (i ? targets[i - 1][1] : route[0][1])), 0);
  const req = world.requiredCrossings ?? [];
  const continuous = cfg.episodeMode === 'continuous-laps-v1';
  const vehicleRadius = Math.hypot(L, W) / 2;
  const maxRate = cfg.maxSpeed + vehicleRadius * cfg.maxSpeed / wb * Math.tan(cfg.maxSteer);
  const substep = Math.min(.01, cfg.dt);
  const surfaceAt = (x, z) => { let s = {id: 'default', mu: .8}; for (const q of surfaces) if (Math.abs(x - q.x) <= q.w / 2 && Math.abs(z - q.z) <= q.d / 2) s = q; return {id: s.id, mu: s.mu * cfg.frictionMultiplier}; };
  let st = {x: world.spawn.x, z: world.spawn.z, yaw: wrap(world.spawn.yaw), speed: 0, steering: 0, t: 0, frame: 0};
  let terminal = false, reason = null, distance = 0, ri = 0, passed = 0, lapGates = 0, lapStartT = 0, lapStartD = 0, lapCross = [];
  const lapEvents = [], gateEvents = [], sheetEvents = [], collisionsOwn = [], trace = [];
  const stats = {substeps: 0, traction_limited: 0, minClear: Infinity, minClearWho: null, minClearAt: null, minClearInterp: Infinity,
    surfaceTimeCenter: {}, surfaceTimeArea: {}, ptfeCenterOff_areaOn: 0, maxLatAcc: 0, maxSpeed: 0};
  let completed = 0, success = false;
  const c0 = clearanceOwn(carPolygon(st.x, st.z, st.yaw, L, W), obs, bounds, st.x, st.z);
  if (c0.d <= collisionEps) { terminal = true; reason = 'invalid_spawn'; }

  function judge(prev, next, tEnd, h) {
    // ---- すき間(小区間の終わりと、間の補間した姿勢) ----
    const poly = carPolygon(next.x, next.z, next.yaw, L, W), c = clearanceOwn(poly, obs, bounds, next.x, next.z);
    if (c.d < stats.minClear) { stats.minClear = c.d; stats.minClearWho = c.who; stats.minClearAt = {t: tEnd, x: next.x, z: next.z}; }
    if (c.d < .03) for (let k = 1; k < interp; k++) {   // 近いときだけ間も調べる
      const u = k / interp, x = prev.x + (next.x - prev.x) * u, z = prev.z + (next.z - prev.z) * u, yaw = prev.yaw + wrap(next.yaw - prev.yaw) * u;
      const ci = clearanceOwn(carPolygon(x, z, yaw, L, W), obs, bounds, x, z);
      if (ci.d < stats.minClearInterp) stats.minClearInterp = ci.d;
    }
    // ---- 路面(中心 = sim の決め方、面積の割合 = 実際の車体) ----
    const sc = surfaceAt(next.x, next.z).id;
    stats.surfaceTimeCenter[sc] = (stats.surfaceTimeCenter[sc] ?? 0) + h;
    const area = L * W;
    for (const q of surfaces) {
      const a = clipArea(poly, q.x - q.w / 2, q.x + q.w / 2, q.z - q.d / 2, q.z + q.d / 2) / area;
      if (a > 0) stats.surfaceTimeArea[q.id] = (stats.surfaceTimeArea[q.id] ?? 0) + h * a;
      if (q.id === 'ptfe' && sc !== 'ptfe' && a > .25) stats.ptfeCenterOff_areaOn += h;
    }
    // ---- 黒シート: z の線を越えるたびに全部数える(向き・実際の幅・sim の幅) ----
    for (const g of req) {
      const a = (prev.z - g.z) * g.direction, b = (next.z - g.z) * g.direction;
      const fwd = a < 0 && b >= 0, back = a >= 0 && b < 0 && a > 0;
      if (!fwd && !back) continue;
      const f = (prev.z - g.z) / (prev.z - next.z), x = prev.x + (next.x - prev.x) * f, yaw = prev.yaw + wrap(next.yaw - prev.yaw) * f;
      const halfX = L / 2 * Math.abs(Math.cos(yaw)) + W / 2 * Math.abs(Math.sin(yaw));
      const sl = sliceX(carPolygon(x, g.z, yaw, L, W), g.z);
      const sheet = surfaces.find(s => s.id === g.surface_id);
      sheetEvents.push({t: tEnd - (1 - f) * h, lap: completed + 1, dir: fwd ? 'required' : 'reverse', x, yaw,
        sim_rule_ok: fwd && x - halfX >= g.minX - 1e-12 && x + halfX <= g.maxX + 1e-12 && (!sheet || (Math.abs(x - sheet.x) <= sheet.w / 2 && Math.abs(g.z - sheet.z) <= sheet.d / 2)),
        margin_min_m: Math.min(x - halfX - g.minX, g.maxX - x - halfX), slice: sl, slice_margin_min_m: sl ? Math.min(sl[0] - g.minX, g.maxX - sl[1]) : null});
      if (fwd && sheetEvents.at(-1).sim_rule_ok && !lapCross.includes(g.id)) lapCross.push(g.id);
    }
    // ---- ゲート(sim と同じ規則: 目標の線を手前から越え、|横| ≤ gateHalfWidth) ----
    if (ri < N) {
      const target = targets[ri], before = ri ? targets[ri - 1] : route[0];
      const len = Math.hypot(target[0] - before[0], target[1] - before[1]), nx = (target[0] - before[0]) / len, nz = (target[1] - before[1]) / len;
      const sa = (prev.x - target[0]) * nx + (prev.z - target[1]) * nz, sb = (next.x - target[0]) * nx + (next.z - target[1]) * nz;
      if (sa < 0 && sb >= 0) {
        const f = -sa / (sb - sa), x = prev.x + (next.x - prev.x) * f, z = prev.z + (next.z - prev.z) * f;
        const lat = -(x - target[0]) * nz + (z - target[1]) * nx;
        gateEvents.push({lap: completed + 1, gate: ri, t: tEnd - (1 - f) * h, lateral: lat, ok: Math.abs(lat) <= cfg.gateHalfWidth});
        if (Math.abs(lat) <= cfg.gateHalfWidth) {
          passed++; ri++; lapGates++;
          if (continuous && lapGates === N) {
            const endT = tEnd - (1 - f) * h, endD = distance - (1 - f) * Math.hypot(next.x - prev.x, next.z - prev.z);
            const okCross = req.every(g => lapCross.includes(g.id)), okDist = endD - lapStartD >= Math.max(L, routeLength * .5);
            if (!okCross || !okDist) { terminal = true; reason = okCross ? 'lap_distance_insufficient' : 'required_crossing_missing'; next.speed = 0; return; }
            completed++;
            lapEvents.push({lap: completed, elapsed: endT - lapStartT, start: lapStartT, end: endT, crossings: lapCross.length, distance: endD - lapStartD});
            lapStartT = endT; lapStartD = endD; ri = 0; lapGates = 0; lapCross = [];
          } else if (!continuous && passed === N && distance >= Math.max(L, routeLength * .5)) {
            const okCross = req.every(g => lapCross.includes(g.id));
            terminal = true; success = okCross; reason = okCross ? 'route_complete' : 'required_crossing_missing'; next.speed = 0;
            if (okCross) { completed = 1; lapEvents.push({lap: 1, elapsed: tEnd, elapsed_interpolated: tEnd - (1 - f) * h, crossings: lapCross.length, distance}); }
          }
        }
      }
    }
  }

  function step(action) {
    if (terminal) return;
    const cmdSteer = num(action?.steering, 0, -cfg.maxSteer, cfg.maxSteer), cmdSpeed = num(action?.speed, 0, -cfg.maxSpeed, cfg.maxSpeed);
    const remaining = Math.min(cfg.dt, cfg.timeLimit - st.t);
    let el = 0;
    while (el < remaining - 1e-12 && !terminal) {
      const h = Math.min(substep, .012 / maxRate, remaining - el) / refine;
      const prev = {...st}, sf = surfaceAt(st.x, st.z);
      let v;
      if (cfg.longitudinalModel === 'real-drive-response-estimate-v1') {
        const rate = st.speed >= 0 && cmdSpeed > st.speed ? Math.min(cfg.driveAccelerationCap, sf.mu * G, (cmdSpeed - st.speed) / cfg.driveTau) : cfg.brakeDeceleration;
        v = approach(st.speed, cmdSpeed, rate * h);
      } else v = approach(st.speed, cmdSpeed, cfg.acceleration * h);
      const d = approach(st.steering, cmdSteer, cfg.steerRate * h);
      let raw = v / wb * Math.tan(d), lim = sf.mu * G / Math.max(Math.abs(v), .01), w = clamp(raw, -lim, lim);
      if (frictionCircle) {
        const aLat = Math.abs(v * w), avail = Math.sqrt(Math.max(0, (sf.mu * G) ** 2 - aLat ** 2)), aLong = (v - st.speed) / h;
        if (Math.abs(aLong) > avail) {
          v = st.speed + Math.sign(aLong) * avail * h;
          raw = v / wb * Math.tan(d); lim = sf.mu * G / Math.max(Math.abs(v), .01); w = clamp(raw, -lim, lim);
          stats.friction_circle_limited = (stats.friction_circle_limited ?? 0) + 1;
        }
      }
      if (Math.abs(raw) > lim) stats.traction_limited++;
      stats.substeps++;
      stats.maxLatAcc = Math.max(stats.maxLatAcc, Math.abs(v * w));
      const ym = st.yaw + w * h / 2;
      const next = {x: st.x + v * Math.cos(ym) * h, z: st.z - v * Math.sin(ym) * h, yaw: wrap(st.yaw + w * h), speed: v, steering: d, t: st.t, frame: st.frame};
      const c = clearanceOwn(carPolygon(next.x, next.z, next.yaw, L, W), obs, bounds, next.x, next.z);
      if (c.d <= collisionEps) {
        collisionsOwn.push({t: st.t + el + h, who: c.who, x: next.x, z: next.z});
        st.speed = 0; st.steering = d;
        if (cfg.stopOnCollision) { terminal = true; reason = 'collision'; }
      } else {
        distance += Math.hypot(next.x - st.x, next.z - st.z);
        st = next;
        stats.maxSpeed = Math.max(stats.maxSpeed, v);
        judge(prev, st, st.t + el + h, h);
      }
      el += h;
      if (c.d <= collisionEps && !cfg.stopOnCollision) break;
    }
    st.t = Math.min(cfg.timeLimit, st.t + el); st.frame++;
    if (!terminal && st.t >= cfg.timeLimit - 1e-10) { terminal = true; reason = continuous ? 'collection_duration_complete' : 'time_limit'; }
    trace.push({frame: st.frame, time: st.t, x: st.x, z: st.z, yaw: st.yaw, speed: st.speed, steering: st.steering});
  }
  return {step, get state() { return {...st, terminal, reason, success, route_index: ri}; },
    result: () => ({reason, success, terminal, completed_laps: completed, lap_events: lapEvents, passed_gates: passed, gate_events: gateEvents,
      sheet_events: sheetEvents, collisions_own: collisionsOwn, distance, trace, stats: {...stats,
        traction_limited_frac: stats.traction_limited / Math.max(1, stats.substeps)}})};
}
