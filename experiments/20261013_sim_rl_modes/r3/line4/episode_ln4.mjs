// line r4(2026-10-14): r3/line/episode_ln3.mjs の写し。読み込む先を ./rt4(シールド v5・mode.json の runtime で切り替える mode_runtime)にし、modeRuntime を渡せるようにした。
// r3 の経路いろいろ(line_r3、2026-10-14 学習の段): 共有の rl/ の写し。違いは読み込む先(走行中の仕組みは r3/line/rt = r3/runtime の固定の写し、
// snapshot は絶対パス)だけ。元の sha256 は r3/line/sources_sha256_line_r3.txt。
// 1 回(エピソード)を「アプリと同じ形」で走らせる(Node。2026-10-13、sim_rl_modes)。試験・物差し・記録の前の予想に使う。
//   - コース: snapshot/sim/world-v3.json を開始ゲートで回し(アプリの reset と同じ)、障害物よけなら配置の障害物を足す
//   - 物理: core 'original'(ユーザーの sim-core.js の写し。判定の数字はこれ)/ 'fast'(高速版。状態のキック)/ 'fastfc'(摩擦円)
//   - 操作: mode_runtime.mjs(ブラウザと同じファイル)の action(snapshot)。物理のずれ(遅延・ゲイン)は sim に渡す前にかける
//     (観測は標準の物理の値で作る。観測の「前回の行動」は適用した行動 = r2 の約束)
//   - すき間: 保存した操作列を lap_audit.mjs(写し)の自前の積分で小区間ごとに測り直す(protocol definitions.clearance)。摩擦円は own_physics_r1 の写し
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {activeWorldFor, simConfig, PHYSICS, PERTURB_KEYS} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/env_core.mjs';
import {independentAudit} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/lap_audit.mjs';
import {createOwnSim} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/check_r1/own_physics_r1.mjs';
import {createModeTeacher, MODE_KEYS} from './rt4/mode_runtime.mjs';
import {worldWithObstacles} from './rt4/layouts.mjs';
import {prepareGeometry, surfaceMu, G} from './rt4/features.mjs';
import {realGateMisses} from './gate_miss.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
export const BASE_WORLD = JSON.parse(fs.readFileSync(path.join(EXP, 'snapshot/sim/world-v3.json'), 'utf8'));
const cores = {};
export async function loadCore(kind) {
  if (!cores[kind]) {
    if (kind === 'original') cores[kind] = await import(pathToFileURL(path.join(EXP, 'snapshot/sim/dist/sim-core.js')).href);
    else if (kind === 'fast' || kind === 'fastfc') cores[kind] = await import(pathToFileURL(path.join(here, 'train/fast_core_modes.mjs')).href);
    else throw new Error('episode: core は original・fast・fastfc です');
  }
  return cores[kind];
}
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const policyCache = new Map();
export function readJson(p) { return JSON.parse(fs.readFileSync(p, 'utf8')); }

// start: {gate, left_offset_m, left_yaw_deg, seed, scenario}。dr: {physics: {...}, delay: 0|1|2, steerGain, speedGain, frictionCircle}
// kick: {frame, yaw?, lateral?, speed_factor?}(高速版だけ)。layout: {id, obstacles}。overrides: {shield: false} など
export async function runModeEpisode({mode, weights, policySha256 = null, policy = null, start, params = {}, episodeMode = 'continuous', coreKind = 'original',
  dr = null, kick = null, layout = null, shieldParams = {}, overrides = {}, timeLimit = null, audit = true, keepLog = true, modeRuntime = null}) {
  if (!MODE_KEYS.includes(mode)) throw new Error('episode: 知らないモード ' + mode);
  const fc = !!dr?.frictionCircle;
  const ck = fc ? 'fastfc' : kick ? 'fast' : coreKind;
  const core = await loadCore(ck);
  const {world: active, variation} = activeWorldFor(BASE_WORLD, start.gate ?? 0, start.left_offset_m ?? 0, start.left_yaw_deg ?? 0);
  const world = layout?.obstacles?.length ? worldWithObstacles(active, layout.obstacles) : active;
  const tl = timeLimit ?? (episodeMode === 'single' ? 20 : 20);
  const nominal = simConfig({episodeMode, timeLimit: tl, seed: start.seed ?? 1});
  const physics = dr?.physics ?? {};
  for (const k of Object.keys(physics)) if (!PERTURB_KEYS.includes(k)) throw new Error('episode: 変えられない物理の値 ' + k);
  const simCfg = {...nominal, ...physics, ...(fc ? {frictionCircle: true} : {})};
  const sim = core.createSimulation(world, simCfg);
  const fast = ck !== 'original';
  let snap = sim.reset({seed: simCfg.seed, scenario: start.scenario ?? 'baseline'});
  if (fast) snap = sim.peek();
  if (snap.state.terminal) return {invalid_spawn: true, start, variation};
  // 教師に渡す config は標準の物理(方策・シールドは今の物理を知らない)
  const cfgN = {...JSON.parse(JSON.stringify(snap.config)), ...Object.fromEntries(PERTURB_KEYS.map(k => [k, PHYSICS[k]]))};
  delete cfgN.frictionCircle;
  const teacher = createModeTeacher(world, cfgN, variation, {mode, weights, params, policySha256, policy, shieldParams, overrides, modeRuntime});
  const delay = dr?.delay ?? 0, sg = dr?.steerGain ?? 1, vg = dr?.speedGain ?? 1;
  const queue = [];
  let appliedPrev = null, kicked = null;
  const frames = [];
  // 本当のゲートの外しを数えるため、各フレーム(操作の前)の sim の数え(passed_gates・completed_laps)を残す(2026-10-14 recover r2)
  const gateCount = [];
  while (!snap.state.terminal) {
    const st = snap.state;
    gateCount.push(`${snap.metrics.completed_laps ?? 0}:${snap.metrics.passed_gates ?? 0}`);
    const out = teacher.action({state: st, metrics: snap.metrics, config: cfgN}, delay ? {applied: appliedPrev ?? [0, 0]} : null);
    // 遅延: n フレーム前の指令(最初は中立 = 舵 0・速さ 0、観測の前回の行動は (0, −1))
    const lg = teacher.log.at(-1);
    let ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
    if (delay) { queue.push(ap); ap = queue.length > delay ? queue.shift() : {steering: 0, speed: 0, a: [0, -1]}; appliedPrev = ap.a; }
    if (kick && st.frame === kick.frame && !kicked) {
      const s = sim.peek().state;
      if (kick.yaw) s.yaw = wrap(s.yaw + kick.yaw);
      if (kick.lateral) { s.x += kick.lateral * -Math.sin(s.yaw); s.z += kick.lateral * -Math.cos(s.yaw); }
      if (kick.speed_factor) s.speed *= kick.speed_factor;
      kicked = {frame: st.frame, x: s.x, z: s.z, yaw: s.yaw};
    }
    if (fast) { sim.step({steering: ap.steering * sg, speed: ap.speed * vg}, true); snap = sim.peek(); }
    else snap = sim.step({steering: ap.steering * sg, speed: ap.speed * vg});
  }
  const st = snap.state, m = snap.metrics, rep = {...sim.report(), world};
  const laps = (m.lap_events ?? []).map(l => ({lap: l.lap_index, elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length,
    collisions: l.collisions}));
  const res = {mode, start, variation, core: ck, episode_mode: episodeMode, dr, kick: kicked ? {...kick, applied: kicked} : null, layout_id: layout?.id ?? null,
    params: teacher.params, reason: st.reason, frames: st.frame, sim_time: st.sim_time, collisions: m.collisions,
    collision: m.collision_events?.[0] ? {t: m.collision_events[0].time, collider: m.collision_events[0].collider, kind: m.collision_events[0].kind} : null,
    laps, completed_laps: m.completed_laps ?? (st.reason === 'route_complete' ? 1 : 0), passed_gates: m.passed_gates,
    shield_frames: teacher.shieldFrames, teacher_metadata: teacher.metadata, events: teacher.events.map(e => ({n: e.n, type: e.type, cls: e.cls, ps: e.ps, pe: e.pe, aborted: e.aborted, s: e.s})),
    gate_misses_virtual: teacher.wrapper.misses};
  // 本当のゲートの外し(sim の数えが止まったまま s がゲートの間の最大 + 0.5 m を越えて進んだ。rl/gate_miss.mjs)。
  // gate_misses_virtual(ランタイムの仮想の外し)は面を越えるのが遅れただけでも数えるので、外しの数には gate_miss を使う
  const gm = realGateMisses(teacher.log.map(q => q.s), gateCount);
  Object.assign(res, {gate_miss: gm.misses > 0, gate_misses: gm.misses, gate_stuck_max_m: gm.stuck_max_m, gate_miss_frames: gm.frames});
  res.valid_episode = episodeMode === 'continuous' && st.reason === 'collection_duration_complete' && st.frame === Math.round(tl * 30) && m.collisions === 0;
  res.valid_laps = laps.filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
  if (episodeMode === 'single') {
    res.lap_complete = st.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1;
    res.lap_from_stop = res.lap_complete ? m.elapsed : null;
  }
  // すき間(小区間ごと)。摩擦円は own_physics_r1 の写し、ほかは lap_audit の写し(どちらも保存した操作列を積分し直す)
  if (audit && !kicked) {
    if (fc) {
      const own = createOwnSim(world, rep.config, {frictionCircle: true, interp: 4});
      for (const a of rep.actions) own.step(a);
      const r = own.result();
      res.min_clearance = Math.min(r.stats.minClear, r.stats.minClearInterp);
      res.audit = {method: 'own_physics_r1(frictionCircle)', reason: r.reason, frames: r.trace.length,
        max_pos_err: Math.max(0, ...r.trace.map((q, i) => rep.trace[i] ? Math.hypot(q.x - rep.trace[i].x, q.z - rep.trace[i].z) : Infinity))};
    } else {
      const own = independentAudit(rep);
      res.min_clearance = own.min_clearance_m;
      res.min_clearance_at = own.min_clearance_at;   // r3 line: 最小すき間の時刻(報告用)
      res.audit = {method: 'lap_audit.independentAudit', reason: own.reason, frames: own.frames.length,
        max_pos_err: Math.max(0, ...own.frames.map((q, i) => rep.trace[i] ? Math.hypot(q.x - rep.trace[i].x, q.z - rep.trace[i].z) : Infinity))};
    }
  }
  // ρ(no_wall の N6): ステップ後の実際の v・δ、μ = 車体の中心の路面 × frictionMultiplier
  const geom = prepareGeometry(world), fm = rep.config.frictionMultiplier, wb = rep.config.wheelbase, dt = rep.config.dt;
  const rhoReq = [], rhoComb = [];
  let prev = null;
  for (const q of rep.trace) {
    const mu = surfaceMu(geom, q.x, q.z, fm);
    if (q.speed > .3) {
      rhoReq.push(q.speed * q.speed * Math.abs(Math.tan(q.steering)) / (wb * mu * G));
      if (prev) { const al = (q.speed - prev.speed) / dt, om = wrap(q.yaw - prev.yaw) / dt, alat = q.speed * om; rhoComb.push(Math.hypot(al, alat) / (mu * G)); }
    }
    prev = q;
  }
  res.rho = {n: rhoReq.length, req_gt1: rhoReq.filter(x => x > 1).length, comb_n: rhoComb.length, comb_gt1: rhoComb.filter(x => x > 1).length,
    req_max: rhoReq.length ? Math.max(...rhoReq) : null, comb_max: rhoComb.length ? Math.max(...rhoComb) : null};
  if (keepLog) {
    res.log = teacher.log.map(q => ({f: q.f, s: q.s, ey: q.ey, epsi: q.epsi, dmin: q.dmin, es: q.es, v: q.v, ph: q.ph, l: q.l, ev: q.ev, sh: q.sh,
      a0: q.a0, a1: q.a1, st: q.steering, sp: q.speed, shield: q.shield}));
    res.trace = rep.trace.map(q => ({x: q.x, z: q.z, yaw: q.yaw, v: q.speed, d: q.steering}));
    res.actions = rep.actions.map(a => ({steering: a.steering, speed: a.speed}));
  }
  res.metadata = teacher.metadata;
  return res;
}
