// REC60 の 60 条件を閉ループで走らせ、その操作列を自前の積分 + 多角形の距離で数え直す(すき間・ゲート・黒シート・衝突)(確認の段 r1)
import {loadOwnPolicy, runClosedLoop, PLAN60, activeWorld, SCENE_WORLD, writeJson} from './common_r1.mjs';
import {createOwnSim} from './own_physics_r1.mjs';
const policy = loadOwnPolicy();
const rows = [];
for (const c of PLAN60) {
  const r = runClosedLoop(policy, {plan: c, gate: c['start-gate'], offset: c['spawn-offset'], yawDeg: c['spawn-yaw'], episodeMode: 'continuous'});
  const own = createOwnSim(activeWorld(SCENE_WORLD, c['start-gate'], c['spawn-offset'], c['spawn-yaw']), r.report.config, {refine: 1, interp: 6});
  for (const a of r.report.actions) own.step(a);
  const o = own.result();
  const sim = r.summary.laps.map(l => l.elapsed), mine = o.lap_events.map(l => l.elapsed);
  rows.push({name: c.name, sim_laps: sim, own_laps: mine, lap_diff_max: Math.max(...sim.map((v, i) => Math.abs(v - (mine[i] ?? Infinity)))),
    own_collisions: o.collisions_own.length, min_clearance_m: Math.min(o.stats.minClear, o.stats.minClearInterp), who: o.stats.minClearWho,
    required_ok: o.sheet_events.filter(e => e.dir === 'required' && e.sim_rule_ok).length, required_bad: o.sheet_events.filter(e => e.dir === 'required' && !e.sim_rule_ok).length,
    sheet_margin_min_m: Math.min(...o.sheet_events.filter(e => e.dir === 'required').map(e => e.margin_min_m)),
    gate_lateral_max_m: Math.max(...o.gate_events.filter(g => g.ok).map(g => Math.abs(g.lateral))), gate_line_misses: o.gate_events.filter(g => !g.ok).length,
    traction_limited_frac: o.stats.traction_limited_frac});
}
const s = {n: rows.length, own_collisions: rows.reduce((a, r) => a + r.own_collisions, 0), lap_diff_max: Math.max(...rows.map(r => r.lap_diff_max)),
  min_clearance_m: Math.min(...rows.map(r => r.min_clearance_m)), min_clearance_who: rows.reduce((a, r) => r.min_clearance_m < a.m ? {m: r.min_clearance_m, who: r.who, name: r.name} : a, {m: Infinity}),
  required_bad: rows.reduce((a, r) => a + r.required_bad, 0), sheet_margin_min_m: Math.min(...rows.map(r => r.sheet_margin_min_m)),
  gate_lateral_max_m: Math.max(...rows.map(r => r.gate_lateral_max_m)), gate_line_misses: rows.reduce((a, r) => a + r.gate_line_misses, 0),
  traction_limited_frac_mean: rows.reduce((a, r) => a + r.traction_limited_frac, 0) / rows.length};
console.error(JSON.stringify(s));
writeJson('rec60_own_audit.json', {schema: 'rl-check-r1-rec60-own/v1', summary: s, rows});
