// 経路いろいろ(line)の確認 r1: 目標の線そのものの確かめ(2026-10-14、確認の段)。学習・評価のコードを使わない。
//   1. κ̄: 表の kappa と、R の点の列から自前で作った κ̄ の差
//   2. 使える幅の表の生の値(left08_raw・right08_raw)を自前の形(own_geom.mjs)で作り直して比べる(R の接線の向きの車体を横へ 1 cm おきにずらし、すき間 ≥ 0.08 m の
//      R を含むひと続きの区間)
//   3. 形ごとの目標の線(R + e*·左の法線)を、線の向きの車体で置いたときの静的なすき間
//   node target_audit.mjs > 結果.json
import {buildObstacles, ownClearance, nearObstacles} from './own_geom.mjs';
import {tab, kappaOwn, eStarOwn, targetPath, rAt, L, N, DS, TABLE} from './own_line.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
const W = await loadCourseWorld();
const OB = buildObstacles(W);
const out = {};
// 1. κ̄
{
  let maxd = 0, at = 0, sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < N; i++) {
    const s = i * DS, a = TABLE.kappa[i], b = kappaOwn(s), d = Math.abs(a - b);
    if (d > maxd) { maxd = d; at = s; }
    sxy += a * b; sxx += a * a; syy += b * b;
  }
  out.kappa_check = {max_abs_diff: maxd, at_s: at, corr: sxy / Math.sqrt(sxx * syy), note: '表の kappa と自前の κ̄(R の向きの s ± 0.5 m の差 / 1 m)'};
}
// 2. 使える幅の生の値
{
  const rows = [];
  let maxdL = 0, maxdR = 0, nBig = 0, wideL = 0, wideR = 0;
  for (let i = 0; i < N; i++) {
    const s = i * DS, r = rAt(s), yaw = Math.atan2(-r.tz, r.tx);
    const near = nearObstacles(OB, r.x, r.z, 1.6);
    const ok = e => ownClearance(near, r.x + e * r.nx, r.z + e * r.nz, yaw) >= .08;
    let left = null, right = null;
    if (ok(0)) {
      let e = 0; while (e < 1 - 1e-9 && ok(+(e + .01).toFixed(2))) e = +(e + .01).toFixed(2); left = e;
      e = 0; while (e < 1 - 1e-9 && ok(-(+(e + .01).toFixed(2)))) e = +(e + .01).toFixed(2); right = e;
    }
    // 表の right08_raw は符号つき(右 = 負)。表はゲート・黒シートで切るので、表 ≤ 自前 になるはず(表が広ければ危ない向き)
    const tL = TABLE.left08_raw[i], tR = -TABLE.right08_raw[i];
    if (left != null) { const d = Math.abs(left - tL); maxdL = Math.max(maxdL, d); if (d > .015) nBig++; wideL = Math.max(wideL, tL - left); }
    if (right != null) { const d = Math.abs(right - tR); maxdR = Math.max(maxdR, d); if (d > .015) nBig++; wideR = Math.max(wideR, tR - right); }
    rows.push([+s.toFixed(3), left, right, tL, tR]);
  }
  out.room_raw_check = {n: N, max_abs_diff_left: maxdL, max_abs_diff_right: maxdR, bins_diff_gt_1_5cm: nBig, table_wider_than_own_left_max: wideL, table_wider_than_own_right_max: wideR, not_free_at_R_own: rows.filter(r => r[1] == null).length,
    not_free_at_R_table: rows.filter(r => r[3] == null).length, sample: rows.filter((_, i) => i % 40 === 0)};
}
// 3. 目標の線(静的なすき間だけ。曲率の数値微分は尖り(e*·κ_R ≥ 1)の所で発散するので、走れる速さの見積もりはしない。尖りは cusp_scan.mjs)
{
  const fams = [['c+0.9', {family: 'const', c0: .9}], ['c+0.8', {family: 'const', c0: .8}], ['c+0.6', {family: 'const', c0: .6}], ['c0', {family: 'const', c0: 0}],
    ['c-0.6', {family: 'const', c0: -.6}], ['c-0.8', {family: 'const', c0: -.8}], ['c-0.9', {family: 'const', c0: -.9}],
    ['inside0.8', {family: 'inside', a: .8}], ['inside0.9', {family: 'inside', a: .9}], ['outside0.8', {family: 'outside', a: .8}], ['outside0.9', {family: 'outside', a: .9}],
    ['late0.8', {family: 'late_apex', a: .8}], ['late0.9', {family: 'late_apex', a: .9}], ['early0.9', {family: 'early_apex', a: .9}]];
  out.targets = {};
  for (const [nm, p] of fams) {
    const P = targetPath(s => eStarOwn(p, s), 0, L, .025);
    let minC = Infinity, minAt = null;
    for (const q of P) {
      const c = ownClearance(nearObstacles(OB, q.x, q.z, 1.0), q.x, q.z, q.yaw);
      if (c < minC) { minC = c; minAt = q.s; }
    }
    out.targets[nm] = {static_min_clearance_on_target: minC, at_s: minAt, note: '車体を目標の線の上に、線の向きで置いたときの静的なすき間の最小'};
  }
}
console.log(JSON.stringify(out));
