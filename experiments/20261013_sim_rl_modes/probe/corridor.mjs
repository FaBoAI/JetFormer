// 参照ループ R に沿って、車体(向きは R の接線)を左右にずらしたときに障害物とのすき間が c 以上になる横の範囲を測る。
// 「いろいろな経路(線)」を作れる余地(R からどれだけ横にずらせるか)の目安。ゲートの窓(±0.65 m)との重なりも見る。
//   node probe/corridor.mjs
import {features, envCore, writeJson, quantiles} from './common.mjs';

const {prepareGeometry, clearance, loopPointAt, LOOP_LENGTH, GATE_S, CANONICAL_ROUTE, gateStatus} = features;
const world = envCore.loadBaseWorld();
const geom = prepareGeometry(world);
const STEP_S = .05, OFF_MAX = 1.0, OFF_STEP = .01, CLEARS = [0, .03, .05, .10];
const ok = (v, c) => v >= c && v > 1e-9;   // すき間 0 は当たり(clearance は当たると 0 を返す)
const rows = [];
const pt = {x: 0, z: 0, tx: 0, tz: 0};
for (let s = 0; s < LOOP_LENGTH; s += STEP_S) {
  loopPointAt(s, pt);
  const yaw = Math.atan2(-pt.tz, pt.tx), lx = pt.tz, lz = -pt.tx;   // R の左
  const offs = [], cl = [];
  for (let o = -OFF_MAX; o <= OFF_MAX + 1e-9; o += OFF_STEP) {
    const x = pt.x + o * lx, z = pt.z + o * lz;
    offs.push(o); cl.push(clearance(geom, x, z, yaw, .42, .19, .5));
  }
  const i0 = Math.round(OFF_MAX / OFF_STEP);
  const row = {s: +s.toFixed(3), x: pt.x, z: pt.z, clear_at_R: cl[i0]};
  // この s で車体を R の接線に向けたときの、横にずらして得られる最大のすき間(0.5 m で頭打ち)と、そのずれ(最小の |o|)
  // R を含む「当たらない」連続区間(となりの通路へ飛ばない)の中だけで探す
  let lo0 = i0, hi0 = i0;
  while (lo0 - 1 >= 0 && ok(cl[lo0 - 1], 0)) lo0--;
  while (hi0 + 1 < offs.length && ok(cl[hi0 + 1], 0)) hi0++;
  let bestC = -1, bestO = 0;
  for (let k = lo0; k <= hi0; k++) if (cl[k] > bestC + 1e-12 || (Math.abs(cl[k] - bestC) <= 1e-12 && Math.abs(offs[k]) < Math.abs(bestO))) { bestC = cl[k]; bestO = offs[k]; }
  row.max_clear = bestC; row.max_clear_offset = +bestO.toFixed(2);
  for (const c of CLEARS) {
    // 0 を含む連続した区間(0 が c 未満なら、いちばん近い自由な点を含む区間)
    let center = i0;
    if (!ok(cl[i0], c)) {
      let best = -1;
      for (let d = 1; d < offs.length; d++) {
        if (i0 + d < offs.length && ok(cl[i0 + d], c)) { best = i0 + d; break; }
        if (i0 - d >= 0 && ok(cl[i0 - d], c)) { best = i0 - d; break; }
      }
      if (best < 0) { row[`c${c}`] = null; continue; }
      center = best;
    }
    let lo = center, hi = center;
    while (lo - 1 >= 0 && ok(cl[lo - 1], c)) lo--;
    while (hi + 1 < offs.length && ok(cl[hi + 1], c)) hi++;
    row[`c${c}`] = {right: +offs[lo].toFixed(2), left: +offs[hi].toFixed(2), width: +(offs[hi] - offs[lo]).toFixed(2), R_inside: center === i0};
  }
  rows.push(row);
}
// ゲートごと: ゲートの s の近くの自由な幅と、ゲートの線上の窓(±0.65 m)との重なり
const gates = GATE_S.map((gs, i) => {
  const r = rows.reduce((b, q) => Math.abs(q.s - gs) < Math.abs(b.s - gs) ? q : b, rows[0]);
  return {gate_target_index: i, route_gate: (i + 1) % GATE_S.length, s: +gs.toFixed(3), free_c0: r.c0, free_c003: r['c0.03'], free_c005: r['c0.05']};
});
const widths = c => rows.map(r => r[`c${c}`]?.width ?? 0);
const summary = {};
for (const c of CLEARS) {
  const w = widths(c);
  const lefts = rows.map(r => r[`c${c}`]?.left ?? 0), rights = rows.map(r => r[`c${c}`]?.right ?? 0);
  summary[`clear_${c}`] = {width_q: quantiles(w), left_room_q: quantiles(lefts), right_room_q: quantiles(rights.map(x => -x)),
    frac_width_ge_0_2: w.filter(x => x >= .2).length / w.length, frac_width_ge_0_4: w.filter(x => x >= .4).length / w.length,
    frac_width_ge_0_6: w.filter(x => x >= .6).length / w.length,
    frac_R_inside: rows.filter(r => r[`c${c}`]?.R_inside).length / rows.length};
}
// 狭いところ(clear 0.05 の幅が小さい順に、1 m 以上離れた 10 か所)
const narrow = [];
for (const r of [...rows].sort((a, b) => (a['c0.05']?.width ?? 0) - (b['c0.05']?.width ?? 0))) {
  if (narrow.every(q => Math.abs(q.s - r.s) > 1)) narrow.push({s: r.s, x: +r.x.toFixed(2), z: +r.z.toFixed(2), c005: r['c0.05'], c0: r.c0,
    clear_at_R: +r.clear_at_R.toFixed(3)});
  if (narrow.length >= 10) break;
}
// 区間(ゲートの番号でまとめる)ごとの 0.05 の幅の中央値
const sections = [];
for (let i = 0; i < GATE_S.length; i++) {
  // 目標 i(= route[i+1]、報告の「ゲート i+1」)の手前の区間 (前の目標の s, この目標の s]
  const a = GATE_S[(i + GATE_S.length - 1) % GATE_S.length]; let b = GATE_S[i]; if (b <= a) b += LOOP_LENGTH;
  const inSec = rows.filter(r => (r.s > a && r.s <= b) || (r.s + LOOP_LENGTH > a && r.s + LOOP_LENGTH <= b));
  const w = inSec.map(r => r['c0.05']?.width ?? 0);
  sections.push({gate_from: i, gate_to: (i + 1) % GATE_S.length, s_from: +a.toFixed(2), s_to: +(b % LOOP_LENGTH).toFixed(2), n: inSec.length,
    width005_min: w.length ? Math.min(...w) : null, width005_median: quantiles(w)['0.5'],
    left005_min: inSec.length ? Math.min(...inSec.map(r => r['c0.05']?.left ?? 0)) : null,
    right005_min: inSec.length ? Math.min(...inSec.map(r => -(r['c0.05']?.right ?? 0))) : null});
}
// 安全な線の目安: 各 s で取れる最大のすき間(車体は R の接線向き)。その最小 = コースの一番きついところで取れるすき間
const maxc = rows.map(r => r.max_clear);
const bottleneck = [...rows].sort((a, b) => a.max_clear - b.max_clear).slice(0, 5).map(r => ({s: r.s, x: +r.x.toFixed(2), z: +r.z.toFixed(2),
  max_clear: +r.max_clear.toFixed(3), at_offset: r.max_clear_offset}));
summary.max_achievable_clearance = {q: quantiles(maxc), bottleneck,
  note: '車体を R の接線の向きにしたときの値(曲がりで車体を傾ければ少し変わる)。0.5 m で頭打ち'};
const out = {schema: 'sim-rl-modes-probe-corridor/v1', note: '車体の向きは R の接線、横に 1 cm 刻み、R に沿って 5 cm 刻み。width は c 以上のすき間で動ける横の幅 [m](0 を含む区間)',
  loop_length_m: LOOP_LENGTH, n: rows.length, summary, narrow, sections, gates};
const p = writeJson('corridor.json', out);
console.log(JSON.stringify({summary}, null, 0)); for (const q of narrow) console.log(JSON.stringify(q)); for (const q of sections) console.log(JSON.stringify(q));
console.log('書いた:', p);
