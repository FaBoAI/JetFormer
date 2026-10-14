// 確認 line r3 の自前の幾何(2026-10-14、確認の段)。学習・評価のコード(features.mjs clearance・lap_audit.mjs・shield.mjs)と、
// 前の確認の道具(check_ln/own_geom.mjs・own_extra.mjs)のどちらも使わずに書いた。
//   - 障害物: ユーザーのリリースの world(load-course-world.mjs の結果)を sim-core.js normalizedWorld の文言どおりに読み直す
//       segments: 半径つきの線分(カプセル)。radius 既定 0.01・[0, 2]、height 既定 0.1、height > 0 のものだけ
//       disks: kind 'post-foot' は軸にそろった 0.16 × 0.15 m の長方形、ほかは円(radius 既定 0.02)
//       bounds: 外枠(車体の角が外へ出たら衝突)
//   - 車体: 中心 (x, z)・向き yaw の 0.42 × 0.19 m の長方形。前 = (cos yaw, −sin yaw)、横 = (sin yaw, cos yaw)(sim-core carCorners と同じ向き)
//   - 距離: 凸多角形どうしの距離を「分離軸で重なりを調べ、重ならなければ頂点と辺の全部の組の最小」で出す(多角形の一般の式。線分は頂点 2 つの多角形)
import fs from 'node:fs';

const num = (v, fb, lo, hi) => (typeof v === 'number' && Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : fb);
const pt = v => (Array.isArray(v) && v.length >= 2 && Number.isFinite(v[0]) && Number.isFinite(v[1]) ? [v[0], v[1]] : null);

export function buildObstacles(world, {L = .42, W = .19} = {}) {
  const polys = [], circles = [];
  (world.segments ?? []).forEach((s, i) => {
    if (!s) return;
    const a = pt(s.a), b = pt(s.b);
    if (!a || !b) return;
    const height = num(s.height, .1, 0, 10);
    if (!(height > 0)) return;
    const r = num(s.radius, .01, 0, 2);
    polys.push({id: `segment-${i}`, kind: String(s.kind || 'rail'), v: [a, b], r, ...bsph([a, b], r)});
  });
  (world.disks ?? []).forEach((s, i) => {
    if (!s || !Number.isFinite(s.x) || !Number.isFinite(s.z)) return;
    const height = num(s.height, .1, 0, 10);
    if (!(height > 0)) return;
    const kind = String(s.kind || 'post');
    if (kind === 'post-foot') {
      const hw = .08, hd = .075, v = [[s.x - hw, s.z - hd], [s.x + hw, s.z - hd], [s.x + hw, s.z + hd], [s.x - hw, s.z + hd]];
      polys.push({id: `disk-${i}`, kind, v, r: 0, ...bsph(v, 0)});
    } else {
      const r = num(s.radius, .02, 0, 2);
      circles.push({id: `disk-${i}`, kind, x: s.x, z: s.z, r, cx: s.x, cz: s.z, reach: r});
    }
  });
  const b = world.bounds ?? {};
  let bounds = {minX: num(b.minX, -20, -1e6, 1e6), maxX: num(b.maxX, 20, -1e6, 1e6), minZ: num(b.minZ, -20, -1e6, 1e6), maxZ: num(b.maxZ, 20, -1e6, 1e6)};
  if (bounds.maxX <= bounds.minX) { bounds.minX = -20; bounds.maxX = 20; }
  if (bounds.maxZ <= bounds.minZ) { bounds.minZ = -20; bounds.maxZ = 20; }
  return {polys, circles, bounds, L, W, carR: Math.hypot(L, W) / 2};
}
function bsph(v, r) {
  let cx = 0, cz = 0;
  for (const p of v) { cx += p[0]; cz += p[1]; }
  cx /= v.length; cz /= v.length;
  let reach = 0;
  for (const p of v) reach = Math.max(reach, Math.hypot(p[0] - cx, p[1] - cz));
  return {cx, cz, reach: reach + r};
}
export function carPoly(x, z, yaw, L = .42, W = .19) {
  const c = Math.cos(yaw), s = Math.sin(yaw), hx = L / 2, hz = W / 2, out = [];
  for (const [lx, lz] of [[hx, hz], [hx, -hz], [-hx, -hz], [-hx, hz]]) out.push([x + lx * c + lz * s, z - lx * s + lz * c]);
  return out;
}
function ptSeg(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz;
  let t = l2 > 0 ? ((px - ax) * dx + (pz - az) * dz) / l2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(px - ax - t * dx, pz - az - t * dz);
}
// 分離軸: 多角形 A・B の辺の法線(線分は自分の法線 1 本)に射影して、すき間があれば分離
function separated(A, B) {
  for (const P of [A, B]) {
    const n = P.length === 2 ? 1 : P.length;
    for (let i = 0; i < n; i++) {
      const p = P[i], q = P[(i + 1) % P.length], nx = -(q[1] - p[1]), nz = q[0] - p[0];
      let aMin = Infinity, aMax = -Infinity, bMin = Infinity, bMax = -Infinity;
      for (const v of A) { const d = v[0] * nx + v[1] * nz; if (d < aMin) aMin = d; if (d > aMax) aMax = d; }
      for (const v of B) { const d = v[0] * nx + v[1] * nz; if (d < bMin) bMin = d; if (d > bMax) bMax = d; }
      if (aMax < bMin || bMax < aMin) return true;
    }
  }
  return false;
}
export function polyDist(A, B) {
  if (!separated(A, B)) return 0;
  let m = Infinity;
  const eA = A.length === 2 ? 1 : A.length, eB = B.length === 2 ? 1 : B.length;
  for (const v of A) for (let i = 0; i < eB; i++) { const p = B[i], q = B[(i + 1) % B.length]; m = Math.min(m, ptSeg(v[0], v[1], p[0], p[1], q[0], q[1])); }
  for (const v of B) for (let i = 0; i < eA; i++) { const p = A[i], q = A[(i + 1) % A.length]; m = Math.min(m, ptSeg(v[0], v[1], p[0], p[1], q[0], q[1])); }
  return m;
}
function ptPolyDist(px, pz, P) {
  // 点が中に入っていれば 0(凸多角形、頂点は回り順)
  let inside = true, sgn = 0;
  for (let i = 0; i < P.length; i++) {
    const p = P[i], q = P[(i + 1) % P.length], cr = (q[0] - p[0]) * (pz - p[1]) - (q[1] - p[1]) * (px - p[0]);
    if (cr !== 0) { const sg = Math.sign(cr); if (!sgn) sgn = sg; else if (sg !== sgn) { inside = false; break; } }
  }
  if (inside) return 0;
  let m = Infinity;
  for (let i = 0; i < P.length; i++) { const p = P[i], q = P[(i + 1) % P.length]; m = Math.min(m, ptSeg(px, pz, p[0], p[1], q[0], q[1])); }
  return m;
}
// 車体の長方形と全障害物の距離(重なっていれば 0 以下)。best より遠い障害物は外接円で飛ばす。what に一番近いものの id・kind を書く
export function clearance(ob, x, z, yaw, what = null) {
  const C = carPoly(x, z, yaw, ob.L, ob.W);
  let best = Infinity, id = null, kind = null;
  const b = ob.bounds;
  for (const [px, pz] of C) {
    const d = Math.min(px - b.minX, b.maxX - px, pz - b.minZ, b.maxZ - pz);
    if (d < best) { best = d; id = 'bounds'; kind = 'bounds'; }
  }
  for (const o of ob.polys) {
    if (Math.hypot(o.cx - x, o.cz - z) - o.reach - ob.carR > best) continue;
    const d = polyDist(C, o.v) - o.r;
    if (d < best) { best = d; id = o.id; kind = o.kind; }
  }
  for (const o of ob.circles) {
    if (Math.hypot(o.x - x, o.z - z) - o.r - ob.carR > best) continue;
    const d = ptPolyDist(o.x, o.z, C) - o.r;
    if (d < best) { best = d; id = o.id; kind = o.kind; }
  }
  if (what) { what.id = id; what.kind = kind; }
  return best;
}
// 近くだけの障害物(フレームごとの計算を速くする。外枠は残す)
export function nearSet(ob, x, z, radius) {
  return {...ob, polys: ob.polys.filter(o => Math.hypot(o.cx - x, o.cz - z) - o.reach <= radius), circles: ob.circles.filter(o => Math.hypot(o.x - x, o.z - z) - o.r <= radius)};
}
// 4 mm おきの点で車体の外周を細かく置き、各点から障害物までの距離の最小(正確な値の「上からの」確かめ: 点の距離 ≥ 正確な距離)
export function sampledClearance(ob, x, z, yaw, step = .004) {
  const C = carPoly(x, z, yaw, ob.L, ob.W), pts = [];
  for (let i = 0; i < 4; i++) {
    const p = C[i], q = C[(i + 1) % 4], n = Math.max(1, Math.ceil(Math.hypot(q[0] - p[0], q[1] - p[1]) / step));
    for (let k = 0; k < n; k++) pts.push([p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n]);
  }
  let m = Infinity;
  for (const [px, pz] of pts) {
    for (const o of ob.polys) {
      if (o.v.length === 2) m = Math.min(m, ptSeg(px, pz, o.v[0][0], o.v[0][1], o.v[1][0], o.v[1][1]) - o.r);
      else m = Math.min(m, ptPolyDist(px, pz, o.v) - o.r);
    }
    for (const o of ob.circles) m = Math.min(m, Math.hypot(px - o.x, pz - o.z) - o.r);
    const b = ob.bounds;
    m = Math.min(m, px - b.minX, b.maxX - px, pz - b.minZ, b.maxZ - pz);
  }
  return m;
}

// ---- 参照ループ R(点の列はデータとして読む)と自前の射影 ----
export function loadLoop(path) {
  const R = JSON.parse(fs.readFileSync(path, 'utf8'));
  const n = R.x.length, seg = [];
  let s = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, ax = R.x[i], az = R.z[i], bx = R.x[j], bz = R.z[j], len = Math.hypot(bx - ax, bz - az);
    seg.push({ax, az, len, s0: s, tx: (bx - ax) / len, tz: (bz - az) / len});
    s += len;
  }
  const Lp = s;
  const at = sAbs => {
    const u = ((sAbs % Lp) + Lp) % Lp;
    let lo = 0, hi = n - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (seg[m].s0 <= u) lo = m; else hi = m - 1; }
    const q = seg[lo], t = u - q.s0;
    return {x: q.ax + q.tx * t, z: q.az + q.tz * t, tx: q.tx, tz: q.tz};
  };
  // R の向き(sim の角 ψ = atan2(−dz, dx))を s ± h の点の差で
  const psiAt = (sAbs, h = .05) => { const a = at(sAbs - h), b = at(sAbs + h); return Math.atan2(-(b.z - a.z), b.x - a.x); };
  const near = (i, x, z) => {
    const q = seg[i];
    let t = (x - q.ax) * q.tx + (z - q.az) * q.tz;
    t = t < 0 ? 0 : t > q.len ? q.len : t;
    const px = q.ax + q.tx * t, pz = q.az + q.tz * t;
    // 左の法線: 前 (tx, tz) を左へ 90°。sim の左 = 舵が正で曲がる向き(ψ が増える)= (−sin ψ, −cos ψ)。前が (cos ψ, −sin ψ) なので左 = (tz, −tx)
    return {d: Math.hypot(x - px, z - pz), s: q.s0 + t, ey: (x - px) * q.tz - (z - pz) * q.tx, i};
  };
  function tracker() {
    let idx = 0, prev = 0;
    return {
      // 最初: 車の前の向きと区間の向きの内積が 0 以上の区間のうち一番近いもの(向きが反対の枝を取らない)
      start(x, z, yaw) {
        const fx = Math.cos(yaw), fz = -Math.sin(yaw);
        let best = null;
        for (let i = 0; i < n; i++) { if (seg[i].tx * fx + seg[i].tz * fz < 0) continue; const p = near(i, x, z); if (!best || p.d < best.d) best = p; }
        idx = best.i; prev = best.s;
        return {s: best.s, ey: best.ey, d: best.d};
      },
      // 次: 前の区間から −3〜+40 の窓(1 フレームで動けるのは高々 0.25 m = 10 区間)。後ろへは 7 cm まで
      //   (R の急な曲がりの所で、車が R の別の枝に近いと後ろの枝へ戻って迷子になるのを防ぐ。確認の段の H48 などで見つけた)
      step(x, z) {
        let best = null;
        for (let k = -3; k <= 40; k++) { const p = near(((idx + k) % n + n) % n, x, z); if (!best || p.d < best.d) best = p; }
        // 窓の中の一番近い点が 0.25 m より遠い(R の急な曲がりの内側を車が横切った)ときは、前へ 3 m まで広げて探し直す
        if (best.d > .25) for (let k = 41; k <= 125; k++) { const p = near(((idx + k) % n + n) % n, x, z); if (p.d < best.d) best = p; }
        idx = best.i;
        let sAbs = prev - (((prev % Lp) + Lp) % Lp) + best.s;
        if (sAbs - prev > Lp / 2) sAbs -= Lp; else if (prev - sAbs > Lp / 2) sAbs += Lp;
        prev = sAbs;
        return {s: sAbs, ey: best.ey, d: best.d};
      },
    };
  }
  return {L: Lp, n, at, psiAt, tracker, gate_s: R.gate_s};
}
