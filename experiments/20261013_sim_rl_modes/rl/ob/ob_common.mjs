// 障害物よけ(obstacle)の学習の段の共通の道具(2026-10-14、obstacle の学習の段 r1)。報告・図にだけ使う(判定の数字は rl/eval_modes.mjs)。
//   passageStats(r, obstacles): 1 回の走り(rl/episode.mjs の結果。keepLog)で、足した障害物の横を通った「通過」ごとに
//     通った向き(障害物の左 / 右)・その障害物とのすき間(フレームの値、車体の長方形と障害物の正確な距離)・通過のときの速さを数える。
//     通過 = 車の R 上の位置 s(mod L)が障害物の s ±0.6 m に入ってから出るまで(layouts.mjs の通り道の検査と同じ幅)。
//   obstacleEy(o): 障害物の R からの横の位置(左が正。layouts.mjs の置き方と同じ向き)
import {projectWindow, clearance, LOOP_LENGTH} from '../runtime/features.mjs';
import {obstacleOnlyGeometry, obstacleS} from '../runtime/layouts.mjs';

const L = LOOP_LENGTH, HALF = .6;
const modS = s => ((s % L) + L) % L;
const dS = (a, b) => { let d = modS(a - b); if (d > L / 2) d -= L; return d; };

export function obstacleEy(o) {
  const p = projectWindow(o.x, o.z, 0, L);
  return (o.x - p.px) * p.tz - (o.z - p.pz) * p.tx;
}

export function passageStats(r, obstacles) {
  const out = [];
  if (!r?.log || !obstacles?.length) return out;
  // r.log[f] はフレーム f の操作の前の状態、r.trace[f] は f の操作の後の状態(= フレーム f + 1 の前)。同じ時刻にそろえて r.trace[f − 1] と r.log[f] を使う
  const n = Math.min(r.log.length, r.trace.length + 1);
  obstacles.forEach((o, k) => {
    const so = o.s_ref ?? obstacleS(o), eyo = obstacleEy(o), g = obstacleOnlyGeometry([o]);
    let cur = null;
    const close = () => {
      if (cur) {
        cur.side = cur.ey_at_min > eyo ? 'left' : 'right';   // 車が障害物の左(R の左)を通った = 'left'
        cur.lateral_gap_m = +(cur.ey_at_min - eyo).toFixed(4);
        out.push(cur); cur = null;
      }
    };
    for (let f = 1; f < n; f++) {
      const q = r.log[f], tr = r.trace[f - 1];
      if (Math.abs(dS(q.s, so)) <= HALF) {
        const d = clearance(g, tr.x, tr.z, tr.yaw, .42, .19, .5);
        if (!cur) cur = {obstacle: k, shape: o.shape, s: +modS(so).toFixed(3), ey_obstacle: +eyo.toFixed(4), f0: f, f1: f, min_clearance_m: d, ey_at_min: q.ey,
          v_at_min: q.v, v_min: q.v};
        cur.f1 = f;
        if (d < cur.min_clearance_m) { cur.min_clearance_m = d; cur.ey_at_min = q.ey; cur.v_at_min = q.v; }
        cur.v_min = Math.min(cur.v_min, q.v);
      } else close();
    }
    close();
  });
  // 回が障害物の手前で終わった(衝突など)通過は「通り切っていない」印
  for (const p of out) {
    p.completed = p.f1 < n - 1 || r.reason === 'collection_duration_complete';
    p.min_clearance_m = +p.min_clearance_m.toFixed(4); p.ey_at_min = +p.ey_at_min.toFixed(4); p.v_at_min = +p.v_at_min.toFixed(3); p.v_min = +p.v_min.toFixed(3);
  }
  return out.sort((a, b) => a.f0 - b.f0);
}

export const quantile = (arr, p) => { const b = [...arr].sort((x, y) => x - y); if (!b.length) return null; const i = p * (b.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };

// 通過の集計(報告「障害物の横を通った回数とすき間」)
export function summarizePassages(list) {
  const done = list.filter(p => p.completed), cl = done.map(p => p.min_clearance_m);
  return {passes: done.length, left: done.filter(p => p.side === 'left').length, right: done.filter(p => p.side === 'right').length,
    clearance_min_m: cl.length ? Math.min(...cl) : null, clearance_p05_m: quantile(cl, .05), clearance_median_m: quantile(cl, .5),
    speed_at_pass_median_mps: quantile(done.map(p => p.v_at_min), .5), lateral_gap_abs_median_m: quantile(done.map(p => Math.abs(p.lateral_gap_m)), .5)};
}
