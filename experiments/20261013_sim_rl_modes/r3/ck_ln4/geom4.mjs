// 確認 line r4(2026-10-14 夜): 自前のすき間の幾何と、R(参照ループ)への自前の射影。
// 学習・評価のコード(r3/line*/、rl/*)も、前の確認のコード(r3/ck_ln3*/)も使わない。sim-core.js の normalizedWorld・collisionAt の
// 「何が障害物か」(高さ > 0 の segments = 半径 r のカプセル、kind 'post-foot' の disks = 回さない 0.16 × 0.15 m の箱、
// ほかの disks = 半径 r の円、bounds = 車の 4 隅が枠の外へ出たら衝突)だけを読んで、距離の出し方はこの段で別に書いた:
//   - 障害物を車の座標(前 = +u、横 = +w)へ移し、車体は軸にそろった長方形 [−hx, hx] × [−hw, hw] として扱う
//   - 線分と長方形: 長方形の 4 辺との線分どうしの距離の最小(交われば 0)、線分が長方形の中なら 0
//   - 箱と長方形: 辺どうし 16 組の線分の距離の最小、どちらかの頂点がもう一方の中なら 0
//   - 円: 点と長方形の距離 − r
import fs from 'node:fs';

export const HX = .42 / 2, HW = .19 / 2;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

// 2 つの線分 p1p2・q1q2 が交わるか(端の接触も交わりとする)
function cross(ax, az, bx, bz, cx, cz) { return (bx - ax) * (cz - az) - (bz - az) * (cx - ax); }
function onSeg(ax, az, bx, bz, px, pz) { return Math.min(ax, bx) - 1e-15 <= px && px <= Math.max(ax, bx) + 1e-15 && Math.min(az, bz) - 1e-15 <= pz && pz <= Math.max(az, bz) + 1e-15; }
function segsIntersect(p1x, p1z, p2x, p2z, q1x, q1z, q2x, q2z) {
  const d1 = cross(q1x, q1z, q2x, q2z, p1x, p1z), d2 = cross(q1x, q1z, q2x, q2z, p2x, p2z);
  const d3 = cross(p1x, p1z, p2x, p2z, q1x, q1z), d4 = cross(p1x, p1z, p2x, p2z, q2x, q2z);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
  if (d1 === 0 && onSeg(q1x, q1z, q2x, q2z, p1x, p1z)) return true;
  if (d2 === 0 && onSeg(q1x, q1z, q2x, q2z, p2x, p2z)) return true;
  if (d3 === 0 && onSeg(p1x, p1z, p2x, p2z, q1x, q1z)) return true;
  if (d4 === 0 && onSeg(p1x, p1z, p2x, p2z, q2x, q2z)) return true;
  return false;
}
function ptSegD(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz;
  let t = l2 > 0 ? ((px - ax) * dx + (pz - az) * dz) / l2 : 0;
  if (t < 0) t = 0; else if (t > 1) t = 1;
  return Math.hypot(ax + t * dx - px, az + t * dz - pz);
}
function segSegD(p1x, p1z, p2x, p2z, q1x, q1z, q2x, q2z) {
  if (segsIntersect(p1x, p1z, p2x, p2z, q1x, q1z, q2x, q2z)) return 0;
  return Math.min(ptSegD(p1x, p1z, q1x, q1z, q2x, q2z), ptSegD(p2x, p2z, q1x, q1z, q2x, q2z),
    ptSegD(q1x, q1z, p1x, p1z, p2x, p2z), ptSegD(q2x, q2z, p1x, p1z, p2x, p2z));
}
// 車の座標の長方形の 4 辺
const RE = [[-HX, -HW, HX, -HW], [HX, -HW, HX, HW], [HX, HW, -HX, HW], [-HX, HW, -HX, -HW]];
const inRect = (u, w) => Math.abs(u) <= HX && Math.abs(w) <= HW;
// 点(車の座標)と長方形の距離
function ptRect(u, w) { const du = Math.max(0, Math.abs(u) - HX), dw = Math.max(0, Math.abs(w) - HW); return Math.hypot(du, dw); }
// 線分(車の座標)と長方形の距離
function segRect(au, aw, bu, bw) {
  if (inRect(au, aw) || inRect(bu, bw)) return 0;
  let m = Infinity;
  for (const e of RE) { const d = segSegD(au, aw, bu, bw, e[0], e[1], e[2], e[3]); if (d < m) m = d; if (m === 0) return 0; }
  return m;
}
// 凸多角形(車の座標の頂点の列)と長方形の距離
function polyRect(P) {
  for (const [u, w] of P) if (inRect(u, w)) return 0;
  // 長方形の頂点が多角形の中か(多角形は凸。向きによらないよう符号がそろうかで見る)
  for (const [cu, cw] of [[-HX, -HW], [HX, -HW], [HX, HW], [-HX, HW]]) {
    let pos = 0, neg = 0;
    for (let i = 0; i < P.length; i++) { const a = P[i], b = P[(i + 1) % P.length], c = cross(a[0], a[1], b[0], b[1], cu, cw); if (c > 0) pos++; else if (c < 0) neg++; }
    if (pos === 0 || neg === 0) return 0;
  }
  let m = Infinity;
  for (let i = 0; i < P.length; i++) { const a = P[i], b = P[(i + 1) % P.length]; const d = segRect(a[0], a[1], b[0], b[1]); if (d < m) m = d; }
  return m;
}

// world(loadCourseWorld の出力)から障害物の表
export function buildObstacles(world) {
  const list = [];
  (world.segments ?? []).forEach((s, i) => {
    const h = Number.isFinite(s.height) ? s.height : .1;
    if (!(h > 0)) return;
    const r = Number.isFinite(s.radius) ? s.radius : .01;
    list.push({id: `segment-${i}`, k: 'cap', ax: s.a[0], az: s.a[1], bx: s.b[0], bz: s.b[1], r,
      cx: (s.a[0] + s.b[0]) / 2, cz: (s.a[1] + s.b[1]) / 2, ext: Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]) / 2 + r});
  });
  (world.disks ?? []).forEach((d, i) => {
    const h = Number.isFinite(d.height) ? d.height : .1;
    if (!(h > 0)) return;
    if (String(d.kind || 'post') === 'post-foot') {
      const a = .08, b = .075;
      list.push({id: `disk-${i}`, k: 'box', V: [[d.x - a, d.z - b], [d.x + a, d.z - b], [d.x + a, d.z + b], [d.x - a, d.z + b]], cx: d.x, cz: d.z, ext: Math.hypot(a, b)});
    } else {
      const r = Number.isFinite(d.radius) ? d.radius : .02;
      list.push({id: `disk-${i}`, k: 'circ', x: d.x, z: d.z, r, cx: d.x, cz: d.z, ext: r});
    }
  });
  const b = world.bounds ?? {};
  return {list, bounds: {minX: b.minX ?? -20, maxX: b.maxX ?? 20, minZ: b.minZ ?? -20, maxZ: b.maxZ ?? 20}};
}
// 車の中心から rad 以内に入りうるものだけ
export function nearList(OB, x, z, rad) { return {list: OB.list.filter(o => Math.hypot(o.cx - x, o.cz - z) - o.ext <= rad), bounds: OB.bounds}; }

// 世界の点 → 車の座標(sim: 前 = (cos ψ, −sin ψ)。横はどちら向きでも長方形は対称なので (sin ψ, cos ψ) とする)
function toCar(x, z, yaw, px, pz) { const c = Math.cos(yaw), s = Math.sin(yaw), dx = px - x, dz = pz - z; return [dx * c - dz * s, dx * s + dz * c]; }
// すき間(負は枠の外だけ。重なりは 0)。out.id に一番近いもの
export function clearance(OB, x, z, yaw, out = null) {
  let m = Infinity, id = null;
  for (const o of OB.list) {
    let d;
    if (o.k === 'cap') { const a = toCar(x, z, yaw, o.ax, o.az), b = toCar(x, z, yaw, o.bx, o.bz); d = segRect(a[0], a[1], b[0], b[1]) - o.r; }
    else if (o.k === 'circ') { const p = toCar(x, z, yaw, o.x, o.z); d = ptRect(p[0], p[1]) - o.r; }
    else d = polyRect(o.V.map(v => toCar(x, z, yaw, v[0], v[1])));
    if (d < m) { m = d; id = o.id; }
  }
  const c = Math.cos(yaw), s = Math.sin(yaw), B = OB.bounds;
  for (const [u, w] of [[-HX, -HW], [-HX, HW], [HX, -HW], [HX, HW]]) {
    const px = x + u * c + w * s, pz = z - u * s + w * c;
    const d = Math.min(px - B.minX, B.maxX - px, pz - B.minZ, B.maxZ - pz);
    if (d < m) { m = d; id = 'bounds'; }
  }
  if (out) out.id = id;
  return m;
}

// 障害物を margin だけ太らせた world(sim-core.js にそのまま渡せる形)。箱の柱の足は、元の箱 + 4 辺に半径 margin のカプセル(= 箱と円のミンコフスキー和)
export function inflatedWorld(world, margin) {
  const segs = [], disks = [];
  (world.segments ?? []).forEach(s => segs.push({...s, radius: (Number.isFinite(s.radius) ? s.radius : .01) + margin}));
  (world.disks ?? []).forEach(d => {
    if (String(d.kind || 'post') === 'post-foot') {
      disks.push({...d});
      const a = .08, b = .075, h = Number.isFinite(d.height) ? d.height : .1;
      const V = [[d.x - a, d.z - b], [d.x + a, d.z - b], [d.x + a, d.z + b], [d.x - a, d.z + b]];
      for (let i = 0; i < 4; i++) segs.push({a: V[i], b: V[(i + 1) % 4], radius: margin, height: h, kind: 'inflated-foot'});
    } else disks.push({...d, radius: (Number.isFinite(d.radius) ? d.radius : .02) + margin});
  });
  const b = world.bounds;
  return {...world, segments: segs, disks, bounds: {minX: b.minX + margin, maxX: b.maxX - margin, minZ: b.minZ + margin, maxZ: b.maxZ - margin}};
}

// R(参照ループ)への自前の射影。点の列はデータとして読む(閉じたループ)。前の確認とは別の作り方:
//   各フレームは前の線分の番号の前後 ±K 本だけを見る(K は動いた距離から)。開始だけ全体から。
export function loadLoop(path) {
  const d = JSON.parse(fs.readFileSync(path, 'utf8'));
  const X = Float64Array.from(d.x), Z = Float64Array.from(d.z), n = X.length;
  const S = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) S[i + 1] = S[i] + Math.hypot(X[(i + 1) % n] - X[i], Z[(i + 1) % n] - Z[i]);
  const L = S[n], segLen = L / n;
  function seg(i, x, z) {
    i = ((i % n) + n) % n;
    const j = (i + 1) % n, dx = X[j] - X[i], dz = Z[j] - Z[i], len = Math.hypot(dx, dz);
    let t = ((x - X[i]) * dx + (z - Z[i]) * dz) / (len * len); t = t < 0 ? 0 : t > 1 ? 1 : t;
    const px = X[i] + t * dx, pz = Z[i] + t * dz;
    // 左 = 進む向き (tx, tz) を左へ 90°(sim の向き: yaw の正は左回り。前 (cos ψ, −sin ψ) の左は (−sin ψ, −cos ψ) = (tz, −tx))
    const ey = (x - px) * (dz / len) - (z - pz) * (dx / len);
    return {i, s: S[i] + t * len, d: Math.hypot(x - px, z - pz), ey};
  }
  function proj(x, z) { let b = null; for (let i = 0; i < n; i++) { const r = seg(i, x, z); if (!b || r.d < b.d) b = r; } return b; }
  function pointAt(s) {
    s = ((s % L) + L) % L;
    let lo = 0, hi = n; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S[m] <= s) lo = m; else hi = m; }
    const u = (s - S[lo]) / (S[lo + 1] - S[lo]), j = (lo + 1) % n;
    return [X[lo] + u * (X[j] - X[lo]), Z[lo] + u * (Z[j] - Z[lo])];
  }
  function heading(s) { const a = pointAt(s - .05), b = pointAt(s + .05); return Math.atan2(-(b[1] - a[1]), b[0] - a[0]); }
  // 続けて追う: lap(周の数)と線分の番号を持つ。s は周をまたいで増え続ける値
  function follower() {
    let idx = 0, lap = 0, last = null;
    return {
      first(x, z) { const r = proj(x, z); idx = r.i; lap = 0; last = {s: r.s, ey: r.ey, d: r.d}; return last; },
      next(x, z) {
        const K = 20;
        let b = null;
        for (let k = -6; k <= K; k++) { const r = seg(idx + k, x, z); if (!b || r.d < b.d - 1e-12) b = {...r, k}; }
        if (b.d > .35) { const g = proj(x, z); b = {...g, k: null}; }
        // 周のまたぎ: 番号が大きく戻ったら次の周、大きく進んだら前の周
        const ni = b.i, di = ni - idx;
        if (di < -n / 2) lap++; else if (di > n / 2) lap--;
        idx = ni;
        last = {s: lap * L + b.s, ey: b.ey, d: b.d};
        return last;
      },
    };
  }
  return {L, n, proj, pointAt, heading, follower, wrap};
}
