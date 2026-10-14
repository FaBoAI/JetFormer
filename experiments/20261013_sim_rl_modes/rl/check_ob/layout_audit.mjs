// 障害物よけの確認 r1: 配置の組を自前で調べる(2026-10-14、確認の段)。方策は走らせない(試験の組を見るだけ = 試験の 2 回の枠は使わない)。
// modes.json common.obstacle_layouts の文言をそのまま自前の形(check_ln/own_geom.mjs)で確かめる:
//   形と大きさ(柱 r 0.05〜0.10・高さ 0.30 / 箱 長さ 0.15〜0.35・r 0.05・高さ 0.25)、数 1〜4、s で 2 m 以上離れる、黒シートの区間・奥の直線に無い、
//   最速のふだんの線 ē_y(s) ± 0.15 m、ゲートの柱から 0.3 m、STD(ゲート 0)の前後 1.5 m(試験・検証)、
//   通り道: 障害物の s ±0.6 m のどこでも(0.02 m おき)、車体を R の接線の向きに置いて全障害物とのすき間 ≥ 0.08 m の横の範囲(1 cm おき)が
//   0.10 m 以上残り、ゲートの窓(±0.65 m)と重なる
//   node layout_audit.mjs → check_obstacle_r1/layout_audit.json
import fs from 'node:fs';
import {buildObstacles, ownClearance, nearObstacles} from '../check_ln/own_geom.mjs';
import {BASE_WORLD, worldWith, projectGlobal} from './run_ob.mjs';
import {LOOP_L_OWN} from '../check_ln/own_extra.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_obstacle_r1';
const REF = JSON.parse(fs.readFileSync(`${EXP}/rl/runtime/reference_loop.json`, 'utf8'));
const NOM = JSON.parse(fs.readFileSync(`${EXP}/rl/nominal_line_fastest.json`, 'utf8'));
const N = REF.x.length, SEG = [];
{ let s = 0; for (let i = 0; i < N; i++) { const j = (i + 1) % N, len = Math.hypot(REF.x[j] - REF.x[i], REF.z[j] - REF.z[i]); SEG.push({x: REF.x[i], z: REF.z[i], tx: (REF.x[j] - REF.x[i]) / len, tz: (REF.z[j] - REF.z[i]) / len, len, s0: s}); s += len; } }
const modL = s => ((s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN;
const dS = (a, b) => { let d = modL(a - b); if (d > LOOP_L_OWN / 2) d -= LOOP_L_OWN; return d; };
function pointAt(s) {
  const m = modL(s);
  let lo = 0, hi = N - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (SEG[mid].s0 <= m) lo = mid; else hi = mid - 1; }
  const q = SEG[lo], t = m - q.s0;
  return {x: q.x + q.tx * t, z: q.z + q.tz * t, tx: q.tx, tz: q.tz};
}
const nomAt = s => { const i = Math.round(modL(s) / NOM.ds_m) % NOM.n; return NOM.ey_mean[i] ?? 0; };
const BS = [5.9, 8.0], BACK = [14.6, 19.3];
const gatePosts = BASE_WORLD.disks.filter(d => d.kind === 'gate-post');
function audit(lay, stdExcluded) {
  const obs = lay.obstacles, issues = [];
  const ob = buildObstacles(worldWith(BASE_WORLD, obs));
  const per = obs.map(o => {
    const p = projectGlobal(o.x, o.z), s = modL(p.s), off = p.ey - nomAt(s);
    const ext = o.shape === 'box' ? o.length / 2 + o.r : o.r;
    const gp = Math.min(...gatePosts.map(g => Math.hypot(g.x - o.x, g.z - o.z) - ext));
    if (o.shape === 'post' && !(o.r >= .05 - 1e-6 && o.r <= .10 + 1e-6 && Math.abs((o.h ?? .3) - .3) < 1e-9)) issues.push('柱の大きさ');
    if (o.shape === 'box' && !(o.length >= .15 - 1e-6 && o.length <= .35 + 1e-6 && Math.abs(o.r - .05) < 1e-9 && Math.abs((o.h ?? .25) - .25) < 1e-9)) issues.push('箱の大きさ');
    if ((s >= BS[0] && s <= BS[1]) || (s >= BACK[0] && s <= BACK[1])) issues.push(`置けない区間 s=${s.toFixed(2)}`);
    if (Math.abs(off) > .15 + .02) issues.push(`ふだんの線からのずれ ${off.toFixed(3)}`);
    if (gp < .3 - 1e-6) issues.push(`ゲートの柱から ${gp.toFixed(3)} m`);
    // 通り道
    let minWidth = Infinity, minWidthS = null;
    const near = nearObstacles(ob, pointAt(s).x, pointAt(s).z, 2.5);
    for (let ds = -.6; ds <= .6 + 1e-9; ds += .02) {
      const q = pointAt(s + ds), yaw = Math.atan2(-q.tz, q.tx);
      let best = 0, run = 0, lo = 0;
      for (let k = 0; k <= 200; k++) {
        const e = -1 + k * .01, x = q.x + e * q.tz, z = q.z - e * q.tx;
        const free = ownClearance(near, x, z, yaw) >= .08;
        if (free) { if (!run) lo = e; run++; }
        if (!free || k === 200) {
          if (run) { const hi = lo + (run - 1) * .01, olo = Math.max(lo, -.65), ohi = Math.min(hi, .65); if (ohi - olo > best) best = ohi - olo; }
          run = 0;
        }
      }
      if (best < minWidth) { minWidth = best; minWidthS = +(s + ds).toFixed(2); }
    }
    if (minWidth < .10 - 1e-6) issues.push(`通り道 ${minWidth.toFixed(3)} m(s=${minWidthS})`);
    return {shape: o.shape, s: +s.toFixed(3), s_ref: o.s_ref ?? null, off_nominal: +off.toFixed(4), gate_post_m: +gp.toFixed(3), passage_min_width_m: +minWidth.toFixed(3),
      std_ds: +dS(s, 0).toFixed(3)};
  });
  for (let i = 0; i < per.length; i++) for (let j = i + 1; j < per.length; j++) if (Math.abs(dS(per[i].s, per[j].s)) < 2.0 - .05) issues.push(`s の間 ${Math.abs(dS(per[i].s, per[j].s)).toFixed(2)} m`);
  if (!(obs.length >= 1 && obs.length <= 4)) issues.push('数');
  const stdMin = per.length ? Math.min(...per.map(q => Math.abs(q.std_ds))) : null;
  if (stdExcluded && stdMin != null && stdMin < 1.5) issues.push(`STD の前後 ${stdMin.toFixed(2)} m`);
  // STD の車体と重ならない
  const stdClear = ownClearance(ob, BASE_WORLD.spawn.x, BASE_WORLD.spawn.z, BASE_WORLD.spawn.yaw);
  return {id: lay.id, n: obs.length, per, std_min_ds: stdMin, std_start_clearance: +stdClear.toFixed(4), issues};
}
const sets = {
  validation: [JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/validation.json`, 'utf8')).layouts, true],
  test: [JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/test.json`, 'utf8')).layouts, true],
  recording: [JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/recording.json`, 'utf8')).layouts, false],
  fresh: [JSON.parse(fs.readFileSync(`${OUT}/fresh_layouts.json`, 'utf8')).layouts, true],
};
const res = {};
for (const [name, [lays, stdEx]] of Object.entries(sets)) {
  const rows = lays.map(l => audit(l, stdEx));
  const widths = rows.flatMap(r => r.per.map(q => q.passage_min_width_m)).sort((a, b) => a - b);
  const stdD = rows.map(r => r.std_min_ds).filter(x => x != null).sort((a, b) => a - b);
  res[name] = {layouts: rows.length, obstacles: widths.length, with_issues: rows.filter(r => r.issues.length).map(r => ({id: r.id, issues: r.issues})),
    passage_min_width: {min: widths[0], p10: widths[Math.floor(.1 * widths.length)], median: widths[Math.floor(widths.length / 2)]},
    std_min_ds: {min: stdD[0], p10: stdD[Math.floor(.1 * stdD.length)], under_1_8: stdD.filter(x => x < 1.8).length, under_2_5: stdD.filter(x => x < 2.5).length},
    std_start_clearance_min: Math.min(...rows.map(r => r.std_start_clearance)), rows};
  console.log(name, JSON.stringify({layouts: rows.length, issues: res[name].with_issues.length, passage: res[name].passage_min_width, std: res[name].std_min_ds}));
}
fs.writeFileSync(`${OUT}/layout_audit.json`, JSON.stringify(res, null, 1));
