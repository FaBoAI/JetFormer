// 下調べ(obstacle の学習の段 r1、2026-10-14): 「障害物を見て目標の線を決める計画 + 経路いろいろの方策 cond(線に沿って走る)」で、
// 障害物よけの配置を走り切れるか。protocol の外(判定・選びには使わない)。RL の obstacle が学習で伸びないときの別の作り方の見込みを見る。
//   node rl/ob/probe_planned_line.mjs --policy <cond の policy.json(obs 97)> [--set validation|recording] [--cv 0.85] [--margin 0.10] --out <結果.json>
// 計画: 基準の線 e*(s) = 0(参照の線 R)。障害物ごとに、R から見た横の広がり [lo, hi](箱は両端を R に射影)と、使える幅の表の区間 [−ρ_R, ρ_L] から
//   広いほうの側を選び、通る位置 e_o = hi + 0.095 + margin(左)/ lo − 0.095 − margin(右)を区間に切る。
//   e*(s) は s_o − 1.6 m から e_o へなめらかに寄せ(余弦)、s_o ± 0.4 m は e_o のまま、s_o + 1.2 m までに 0 へ戻す。重なる所は 0 から遠いほう。
// 走らせ方: 元の sim-core.js・標準の物理・連続 20 秒。方策は cond97 の観測(先読み 16 点の e*・追従の誤差・c_v)で、mode_runtime と同じ行動の換算。
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {readJson, BASE_WORLD, loadCore} from '../episode.mjs';
import {activeWorldFor, simConfig} from '../../snapshot/rl/env_core.mjs';
import {createObsWrapper} from '../runtime/obs_modes.mjs';
import {policyFromWeights, BASE_SPEED_CAP} from '../runtime/mode_runtime.mjs';
import {actionToCommand, prepareGeometry, clearance, projectWindow, LOOP_LENGTH} from '../runtime/features.mjs';
import {worldWithObstacles, boxEnds} from '../runtime/layouts.mjs';
import {tableAt} from '../runtime/line_profiles.mjs';
import {buildJobs} from '../eval_modes.mjs';
import {haltonStart} from '../runtime/rng.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';
import {startClearOfObstacles, geometryWithObstacles} from '../runtime/layouts.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const W = readJson(opt('--policy')), set = opt('--set', 'validation'), cV = +opt('--cv', '0.85'), margin = +opt('--margin', '0.10');
const L = LOOP_LENGTH, HW = .095, PRE = +opt('--pre', '1.6'), HOLD = +opt('--hold', '0.4');
const modS = s => ((s % L) + L) % L;
const dS = (a, b) => { let d = modS(a - b); if (d > L / 2) d -= L; return d; };
const eyOf = (x, z) => { const p = projectWindow(x, z, 0, L); return {s: p.s, ey: (x - p.px) * p.tz - (z - p.pz) * p.tx}; };
function planFor(obstacles) {
  const plan = obstacles.map(o => {
    const pts = o.shape === 'box' ? boxEnds(o).map(([x, z]) => eyOf(x, z)) : [eyOf(o.x, o.z)];
    const c = eyOf(o.x, o.z), lo = Math.min(...pts.map(p => p.ey)) - o.r, hi = Math.max(...pts.map(p => p.ey)) + o.r;
    const rL = tableAt('rhoL', c.s), rR = tableAt('rhoR', c.s);
    const left = rL - hi >= lo + rR;   // 左の余り ≥ 右の余り
    let e = left ? hi + HW + margin : lo - HW - margin;
    e = Math.max(-rR, Math.min(rL, e));
    return {s: c.s, e, left, lo, hi};
  });
  const sm = t => .5 - .5 * Math.cos(Math.PI * Math.max(0, Math.min(1, t)));
  return s => {
    let best = 0;
    for (const p of plan) {
      const d = dS(s, p.s);
      let w = 0;
      if (d >= -PRE && d < -HOLD) w = sm((d + PRE) / (PRE - HOLD)); else if (d >= -HOLD && d <= HOLD) w = 1; else if (d > HOLD && d <= HOLD + .8) w = 1 - sm((d - HOLD) / .8);
      const v = w * p.e;
      if (Math.abs(v) > Math.abs(best)) best = v;
    }
    return best;
  };
}
const core = await loadCore('original');
const act = policyFromWeights(W, 97);
let jobs;
if (set === 'recording') {
  const lays = readJson(path.join(here, '../obstacle_sets/recording.json')).layouts, dist = CATALOGUE.modes.find(m => m.key === 'obstacle').recording.start_distribution;
  const geom = prepareGeometry(BASE_WORLD);
  jobs = []; let h = 1;
  for (const lay of lays) {
    for (; h < 10000; h++) {
      const c = haltonStart(h, dist), {world} = activeWorldFor(BASE_WORLD, c.gate, c.left_offset_m, c.left_yaw_deg);
      if (!(clearance(geometryWithObstacles(geom, lay.obstacles), world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5) > 1e-9)) continue;
      if (!startClearOfObstacles(projectWindow(world.spawn.x, world.spawn.z, 0, L).s, lay.obstacles)) continue;
      jobs.push({id: `${lay.id}-H${h}`, spec: {start: {...c, seed: 706000 + jobs.length}, layout: lay}}); h++; break;
    }
  }
} else jobs = buildJobs('obstacle', set).filter(j => j.test === 'OB-T100');
const rows = [];
for (const j of jobs) {
  const {start, layout} = j.spec;
  const {world: active} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
  const world = worldWithObstacles(active, layout.obstacles), geom = prepareGeometry(world);
  const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: start.seed});
  const sim = core.createSimulation(world, cfg);
  let snap = sim.reset({seed: cfg.seed});
  if (snap.state.terminal) { rows.push({id: j.id, invalid_spawn: true}); continue; }
  const eStar = planFor(layout.obstacles);
  const wr = createObsWrapper({wrapper: 'cond97', world, config: snap.config, geometry: geom, lineFactory: () => ({eStar}), cV, obstacles: null});
  const obs = new Float64Array(97), aux = {}, pa = [0, 0];
  let prev = [0, 0], dmin = Infinity;
  const dbg = opt('--debug') === j.id ? [] : null;
  while (!snap.state.terminal) {
    const st = snap.state;
    if (dbg && st.frame > 0) dbg.push([st.frame, +aux.s?.toFixed?.(3), +aux.lateral_error?.toFixed?.(3), +aux.e_star?.toFixed?.(3), +st.speed.toFixed(2)]);
    if (st.frame === 0) wr.reset(st, snap.metrics.route_index ?? 0); else wr.advance(st, snap.metrics.route_index ?? 0);
    wr.observe(st, prev[0], prev[1], obs, aux);
    act(obs, pa);
    const c = actionToCommand(pa[0], pa[1], BASE_SPEED_CAP * cV, snap.config.maxSteer);
    prev = [c.a0 ?? pa[0], c.a1 ?? pa[1]];
    snap = sim.step({steering: c.steering, speed: c.speed});
    const s2 = snap.state;
    dmin = Math.min(dmin, clearance(geom, s2.x, s2.z, s2.yaw, .42, .19, .5));
  }
  if (dbg) console.error(JSON.stringify({plan: layout.obstacles.map(o => ({shape: o.shape, s_ref: o.s_ref, r: o.r, len: o.length ?? null})), tail: dbg.slice(-45)}));
  const st = snap.state, m = snap.metrics;
  const laps = (m.lap_events ?? []).map(l => l.elapsed);
  rows.push({id: j.id, layout: layout.id, n: layout.obstacles.length, reason: st.reason, frames: st.frame, collisions: m.collisions,
    valid: st.reason === 'collection_duration_complete' && st.frame === 600 && m.collisions === 0, laps, frame_min_clearance: dmin,
    collider: m.collision_events?.[0]?.collider ?? null, kind: m.collision_events?.[0]?.kind ?? null});
}
const ok = rows.filter(r => r.valid), l1 = ok.map(r => r.laps[0]).filter(Number.isFinite).sort((a, b) => a - b);
const res = {schema: 'jf-ob-probe-planned-line/v1', created_at: new Date().toISOString(), policy: opt('--policy'), set, c_v: cV, margin_m: margin, pre_m: PRE, hold_m: HOLD,
  runs: rows.length, valid: ok.length, with_lap: ok.filter(r => r.laps.length).length, lap1_median: l1.length ? l1[Math.floor(l1.length / 2)] : null,
  frame_min_clearance_valid: ok.length ? Math.min(...ok.map(r => r.frame_min_clearance)) : null, rows};
console.log(JSON.stringify({set, c_v: cV, margin, pre: PRE, hold: HOLD, runs: res.runs, valid: res.valid, with_lap: res.with_lap, lap1_median: res.lap1_median, min_cl: res.frame_min_clearance_valid,
  fails: rows.filter(r => !r.valid).map(r => `${r.id}:${r.reason}@${r.frames}:${r.kind}`)}));
fs.writeFileSync(opt('--out'), JSON.stringify(res));
