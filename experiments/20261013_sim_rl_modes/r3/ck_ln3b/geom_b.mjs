// 確認 line r3(再実行、2026-10-14 夜): 自前のすき間の幾何と R への射影。
// 学習・評価・前の確認のコード(r3/line/*、rl/*、r3/ck_ln3/*)を使わず、sim-core.js の normalizedWorld の文言だけから書いた。
//   障害物: segments(height > 0)= 半径 r のカプセル、disks(height > 0)の kind 'post-foot' = 回さない 0.16 × 0.15 m の箱、
//           それ以外の disks = 半径 r の円、bounds = 外枠(車の角が枠の外へ出たら衝突)
//   距離: 車体の長方形(長さ 0.42・幅 0.19)と各障害物の距離。凸多角形どうしは「分離軸で重なりを見て、重ならなければ頂点と辺の全部の組の最小」
import fs from 'node:fs';

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

// 点 p と線分 ab の距離
function ptSeg(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
  let t = L2 > 0 ? ((px - ax) * dx + (pz - az) * dz) / L2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const ex = ax + t * dx - px, ez = az + t * dz - pz;
  return Math.sqrt(ex * ex + ez * ez);
}
// 凸多角形(頂点の列、2 点なら線分)の分離軸の重なり判定
function axesOf(P) {
  const out = [];
  const n = P.length;
  for (let i = 0; i < (n === 2 ? 1 : n); i++) {
    const a = P[i], b = P[(i + 1) % n];
    const ex = b[0] - a[0], ez = b[1] - a[1];
    out.push([-ez, ex]);
    if (n === 2) out.push([ex, ez]);
  }
  return out;
}
function overlap(P, Q) {
  for (const ax of [...axesOf(P), ...axesOf(Q)]) {
    let p0 = Infinity, p1 = -Infinity, q0 = Infinity, q1 = -Infinity;
    for (const v of P) { const d = v[0] * ax[0] + v[1] * ax[1]; if (d < p0) p0 = d; if (d > p1) p1 = d; }
    for (const v of Q) { const d = v[0] * ax[0] + v[1] * ax[1]; if (d < q0) q0 = d; if (d > q1) q1 = d; }
    if (p1 < q0 || q1 < p0) return false;
  }
  return true;
}
function polyDist(P, Q) {
  if (overlap(P, Q)) return 0;
  let m = Infinity;
  const edges = A => A.length === 2 ? [[A[0], A[1]]] : A.map((v, i) => [v, A[(i + 1) % A.length]]);
  for (const [a, b] of edges(Q)) for (const v of P) m = Math.min(m, ptSeg(v[0], v[1], a[0], a[1], b[0], b[1]));
  for (const [a, b] of edges(P)) for (const v of Q) m = Math.min(m, ptSeg(v[0], v[1], a[0], a[1], b[0], b[1]));
  return m;
}
// 点と凸多角形の距離(中なら 0)
function ptPoly(px, pz, P) {
  let inside = true;
  const n = P.length;
  for (let i = 0; i < n; i++) {
    const a = P[i], b = P[(i + 1) % n];
    const cr = (b[0] - a[0]) * (pz - a[1]) - (b[1] - a[1]) * (px - a[0]);
    if (cr < 0) { inside = false; break; }
  }
  // 向きに頼らないよう、内外は両方の向きで見る
  if (!inside) {
    let inside2 = true;
    for (let i = 0; i < n; i++) {
      const a = P[i], b = P[(i + 1) % n];
      const cr = (b[0] - a[0]) * (pz - a[1]) - (b[1] - a[1]) * (px - a[0]);
      if (cr > 0) { inside2 = false; break; }
    }
    inside = inside2;
  }
  if (inside) return 0;
  let m = Infinity;
  for (let i = 0; i < n; i++) { const a = P[i], b = P[(i + 1) % n]; m = Math.min(m, ptSeg(px, pz, a[0], a[1], b[0], b[1])); }
  return m;
}

export const CAR = {L: .42, W: .19};
// 車体の 4 隅(sim の向き: 前 = (cos ψ, −sin ψ)、左 = (−sin ψ, −cos ψ))。周回の順に並べる
export function carPoly(x, z, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw), hx = CAR.L / 2, hw = CAR.W / 2;
  const fx = c, fz = -s, lx = -s, lz = -c;
  const P = (a, b) => [x + a * fx + b * lx, z + a * fz + b * lz];
  return [P(hx, hw), P(-hx, hw), P(-hx, -hw), P(hx, -hw)];
}

// コースの world(loadCourseWorld の出力)から障害物の表を作る
export function obstaclesOf(world) {
  const ob = [];
  (world.segments ?? []).forEach((s, i) => {
    const h = Number.isFinite(s.height) ? s.height : .1;
    if (!(h > 0)) return;
    ob.push({id: `segment-${i}`, type: 'cap', a: [s.a[0], s.a[1]], b: [s.b[0], s.b[1]], r: Number.isFinite(s.radius) ? s.radius : .01,
      cx: (s.a[0] + s.b[0]) / 2, cz: (s.a[1] + s.b[1]) / 2, ext: Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]) / 2 + (s.radius ?? .01)});
  });
  (world.disks ?? []).forEach((d, i) => {
    const h = Number.isFinite(d.height) ? d.height : .1;
    if (!(h > 0)) return;
    if (d.kind === 'post-foot') {
      const w = .16 / 2, e = .15 / 2;
      ob.push({id: `disk-${i}`, type: 'box', P: [[d.x - w, d.z - e], [d.x + w, d.z - e], [d.x + w, d.z + e], [d.x - w, d.z + e]], cx: d.x, cz: d.z, ext: Math.hypot(w, e)});
    } else ob.push({id: `disk-${i}`, type: 'circ', x: d.x, z: d.z, r: Number.isFinite(d.radius) ? d.radius : .02, cx: d.x, cz: d.z, ext: d.radius ?? .02});
  });
  const b = world.bounds;
  return {list: ob, bounds: b};
}
// 車の中心から rad m 以内に何かがあり得る障害物だけ(速くするため)
export function nearOf(OB, x, z, rad) {
  return {list: OB.list.filter(o => Math.hypot(o.cx - x, o.cz - z) - o.ext <= rad), bounds: OB.bounds};
}
// すき間 = 車体と全障害物の最小距離(負にはしない。重なりは 0)。who に一番近いものの id を入れる
export function clr(OB, x, z, yaw, who = null) {
  const C = carPoly(x, z, yaw);
  let m = Infinity, id = null;
  for (const o of OB.list) {
    let d;
    if (o.type === 'cap') d = polyDist(C, [o.a, o.b]) - o.r;
    else if (o.type === 'box') d = polyDist(C, o.P);
    else d = ptPoly(o.x, o.z, C) - o.r;
    if (d < m) { m = d; id = o.id; }
  }
  const b = OB.bounds;
  for (const [px, pz] of C) {
    const d = Math.min(px - b.minX, b.maxX - px, pz - b.minZ, b.maxZ - pz);
    if (d < m) { m = d; id = 'bounds'; }
  }
  if (who) who.id = id;
  return m;
}

// 自前の R(参照ループ)への射影。点の列はデータとして読む(閉じたループ)
export function loadR(path) {
  const d = JSON.parse(fs.readFileSync(path, 'utf8'));
  const n = d.x.length, X = d.x, Z = d.z;
  const S = new Float64Array(n + 1);
  for (let i = 1; i <= n; i++) S[i] = S[i - 1] + Math.hypot(X[i % n] - X[i - 1], Z[i % n] - Z[i - 1]);
  const L = S[n];
  function at(s) {
    s = ((s % L) + L) % L;
    let lo = 0, hi = n;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S[m] <= s) lo = m; else hi = m; }
    const u = (s - S[lo]) / (S[lo + 1] - S[lo]), j = (lo + 1) % n;
    return [X[lo] + u * (X[j] - X[lo]), Z[lo] + u * (Z[j] - Z[lo])];
  }
  // R の向き(s ± 0.05 m の点から)
  function psi(s) { const a = at(s - .05), b = at(s + .05); return Math.atan2(-(b[1] - a[1]), b[0] - a[0]); }
  // 線分 i への射影(s と、左が正の横ずれ)
  function onSeg(i, x, z) {
    const j = (i + 1) % n, ax = X[i], az = Z[i], dx = X[j] - ax, dz = Z[j] - az, L2 = dx * dx + dz * dz;
    let t = ((x - ax) * dx + (z - az) * dz) / L2; t = t < 0 ? 0 : t > 1 ? 1 : t;
    const px = ax + t * dx, pz = az + t * dz, len = Math.sqrt(L2);
    // 左の法線: 進む向き (tx, tz) に対して (tz, −tx)
    const tx = dx / len, tz = dz / len;
    const ey = (x - px) * tz + (z - pz) * -tx;
    return {s: S[i] + t * len, d: Math.hypot(x - px, z - pz), ey};
  }
  function globalProj(x, z) {
    let best = null;
    for (let i = 0; i < n; i++) { const r = onSeg(i, x, z); if (!best || r.d < best.d) best = r; }
    return best;
  }
  // 続けて追う射影: 前の s(連続の値、周をまたいで増える)の −0.1〜+1.2 m の窓。窓の一番近い点が 0.3 m より遠ければ全体から探して、前の s に一番近い周に合わせる
  function tracker() {
    let sc = null;
    return {
      start(x, z) { const r = globalProj(x, z); sc = r.s; return {s: sc, ey: r.ey, d: r.d}; },
      step(x, z) {
        const base = ((sc % L) + L) % L, k = Math.floor(sc / L);
        let best = null;
        for (let ds = -.1; ds <= 1.2; ds += .02) {
          const ss = base + ds, sm = ((ss % L) + L) % L;
          let lo = 0, hi = n; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S[m] <= sm) lo = m; else hi = m; }
          const r = onSeg(lo, x, z);
          const lap = k * L + (ss >= L ? L : ss < 0 ? -L : 0);
          const sAbs = lap + r.s;
          if (!best || r.d < best.d - 1e-12) best = {...r, sAbs};
        }
        if (best.d > .3) {
          const g = globalProj(x, z);
          let cand = Math.floor(sc / L) * L + g.s;
          if (cand - sc > L / 2) cand -= L; else if (sc - cand > L / 2) cand += L;
          best = {...g, sAbs: cand};
        }
        // 窓の継ぎ目で s が 1 周ずれないよう、前の s の近くへそろえる
        while (best.sAbs - sc > L / 2) best.sAbs -= L;
        while (sc - best.sAbs > L / 2) best.sAbs += L;
        sc = best.sAbs;
        return {s: sc, ey: best.ey, d: best.d};
      },
    };
  }
  return {L, n, at, psi, globalProj, tracker, wrap};
}
