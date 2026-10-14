#!/usr/bin/env node
// no_wall r3 の診断(2026-10-14): 1 回を走らせ、フレームごとのシールドの記録(速さ・指令・上限・替え方・E/Q/P の予想・ブレーキの見積もり)を出す。
// 判定には使わない(残った衝突の仕組みを調べるため)。走らせ方は run_nw3r.mjs と同じ(元の sim-core.js / 摩擦円は高速版の写し・runtime_v4)。
//   node r3/nw/diag_corner_nw3.mjs --policy p.json --spec '{"start":{...},"dr":{...}}' [--from 0] [--every 1] [--params '{"governor":{...}}']
import fs from 'node:fs';
import crypto from 'node:crypto';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const {activeWorld, BASE_WORLD, NOMINAL} = await import('./run_nw3r.mjs');
const core = await import(`${REL}/dist/sim-core.js`);
const fcCore = await import('./fast_core_nw3.mjs');
const RT = await import(opt('--rt', `${EXP}/r3/nw/runtime_v4`) + '/mode_runtime.mjs');
const pp = opt('--policy'), weights = JSON.parse(fs.readFileSync(pp, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(pp)).digest('hex');
const spec = JSON.parse(opt('--spec')), params = opt('--params') ? JSON.parse(opt('--params')) : null;
const st0 = spec.start, seed = st0.seed >>> 0, dr = spec.dr ?? {};
const {world, variation} = activeWorld(BASE_WORLD, st0.gate, st0.left_offset_m, st0.left_yaw_deg);
const base = {...NOMINAL, seed, timeLimit: spec.timeLimit ?? 20, episodeMode: 'continuous-laps-v1'};
const fc = !!dr.frictionCircle;
const sim = (fc ? fcCore : core).createSimulation(world, {...base, ...(dr.physics ?? {}), ...(fc ? {frictionCircle: true} : {})});
let snap = sim.reset({seed, scenario: 'baseline'});
if (fc) snap = JSON.parse(JSON.stringify(sim.peek()));
const cfgN = core.createSimulation(world, base).reset({seed, scenario: 'baseline'}).config;
const t = RT.createModeTeacher(world, cfgN, variation, {mode: 'no_wall', weights, policySha256: sha, ...(params ? {shieldParams: params} : {})});
const delay = dr.delay ?? 0, sg = dr.steerGain ?? 1, vg = dr.speedGain ?? 1, q = [];
let applied = [0, 0];
const from = +opt('--from', '0'), every = +opt('--every', '1');
const rows = [];
while (!snap.state.terminal) {
  const st = snap.state;
  const out = t.action({state: st, metrics: snap.metrics, config: cfgN}, delay ? {applied} : null);
  const lg = t.log.at(-1), S = lg.shield ?? {};
  rows.push({f: st.frame, v: +st.speed.toFixed(3), d: +lg.dmin.toFixed(4), cmdV: +out.speed.toFixed(3), cmdS: +out.steering.toFixed(3), sh: lg.sh, fb: S.fb ?? null,
    cap: S.cap != null && Number.isFinite(S.cap) ? +S.cap.toFixed(3) : null, nom: S.nom != null ? +S.nom.toFixed(4) : null, wq: S.wq != null ? +S.wq.toFixed(4) : null,
    pess: S.pess != null ? +S.pess.toFixed(4) : null, b: S.id ? +S.id.b.toFixed(3) : null, known: S.id?.known ?? null, s: +lg.s.toFixed(2)});
  let ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
  if (delay) { q.push(ap); ap = q.length > delay ? q.shift() : {steering: 0, speed: 0, a: [0, -1]}; applied = ap.a; }
  if (fc) { sim.step({steering: ap.steering * sg, speed: ap.speed * vg}, true); snap = sim.peek(); }
  else snap = sim.step({steering: ap.steering * sg, speed: ap.speed * vg});
}
const m = snap.metrics;
for (const r of rows) if (r.f >= from && (r.f % every === 0 || r.sh)) console.log(JSON.stringify(r));
console.log(JSON.stringify({end: snap.state.reason, frame: snap.state.frame, collisions: m.collisions, collision: m.collision_events?.[0] ?? null,
  first_brake_known_frame: rows.find(r => r.b != null && r.b < 1.19)?.f ?? null, max_v: Math.max(...rows.map(r => r.v)),
  max_v_before_brake_known: Math.max(0, ...rows.filter(r => !(r.b != null && r.b < 1.19)).map(r => r.v)), shield_frames: rows.filter(r => r.sh).length}));
