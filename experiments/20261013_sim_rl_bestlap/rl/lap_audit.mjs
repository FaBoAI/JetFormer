// 周回の独立な確認(2026-10-13、RL 環境の段)。rl/verify_lap.mjs と verify_recording.mjs から使う
//
// sim のレポート(course-simulator-report/v1: world・config・actions・trace・metrics・outcome)を受け取り、
//  1) 保存した操作列を **元の sim-core.js** で再生し、保存した軌跡・集計と一致するか(差 ≤ 1e-9)を見る
//  2) さらに **自前のコード** で同じ操作列を積分し直し(sim-core の文書どおりの式: 1 次遅れの加速・一定のブレーキ・
//     舵の速さの上限・μg/|v| のヨーレートの頭打ち・中点の積分・小区間)、小区間ごとに
//       - ゲートの通過(順番どおり、線を手前から越え、|横| ≤ 0.65 m)
//       - 黒シートの横断(z = 0 を −z 向きに越えるときに車幅全体が x 1.26〜2.36 m の中。**周ごとに全部数える**。sim の台帳は 1 周に 1 回までしか残さない)
//       - 衝突(車体の長方形と全障害物の距離 ≤ 1e-9 m)、最小すき間
//       - 周の完了(33 ゲート + 黒シート + 距離の条件)と周回時間(小区間の中で線形補間)
//     を数え直して、sim の集計・軌跡と一致するか(差 ≤ 1e-9)を見る
// 自前の積分が元のコアと一致すれば、物理・判定の両方を別の実装で確かめたことになる。
import fs from 'node:fs';
import {prepareGeometry, surfaceMu, clearance, CANONICAL_ROUTE} from './features.mjs';
import {PHYSICS, loadBaseWorld, loadCore, activeWorldFor} from './env_core.mjs';

export const TOLERANCE = 1e-9;
const G = 9.80665;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const finite = v => typeof v === 'number' && Number.isFinite(v);
const num = (v, fallback, lo, hi) => finite(v) ? clamp(v, lo, hi) : fallback;
const approach = (value, target, d) => value + clamp(target - value, -d, d);
const stable = v => JSON.stringify(v, (_, x) => x && typeof x === 'object' && !Array.isArray(x) ? Object.fromEntries(Object.keys(x).sort().map(k => [k, x[k]])) : x);

// ---- 1) 元のコアで再生 ----
export async function replayOriginal(report) {
  const core = await loadCore('original');
  const sim = core.createSimulation(report.world, report.config);
  sim.reset({seed: report.seed, scenario: report.scenario});
  for (const a of report.actions) sim.step({steering: a.steering, speed: a.speed});
  if (report.final_state?.paused) sim.setPaused(true);
  return sim.report();
}

// ---- 2) 自前の積分と数え直し ----
export function independentAudit(report) {
  const cfg = report.config, world = report.world, geom = prepareGeometry(world);
  const route = world.route.map(p => [p[0], p[1]]), n = route.length, targets = [...route.slice(1), route[0]];
  const continuous = cfg.episodeMode === 'continuous-laps-v1';
  const L = cfg.vehicleLength, W = cfg.vehicleWidth, wb = cfg.wheelbase;
  const routeLength = targets.reduce((s, p, i) => s + Math.hypot(p[0] - (i ? targets[i - 1][0] : route[0][0]), p[1] - (i ? targets[i - 1][1] : route[0][1])), 0);
  const crossing = (world.requiredCrossings ?? [])[0];
  const sheet = crossing && (world.surfaces ?? []).find(s => s.id === crossing.surface_id);
  const vehicleRadius = Math.hypot(L, W) / 2;
  const maxRate = cfg.maxSpeed + vehicleRadius * cfg.maxSpeed / wb * Math.tan(cfg.maxSteer);
  let st = {x: world.spawn.x, z: world.spawn.z, yaw: wrap(world.spawn.yaw), speed: 0, steering: 0, sim_time: 0, frame: 0};
  let terminal = false, reason = null, distance = 0, routeIndex = 0, passed = 0, lapPassed = 0;
  let lapStartTime = 0, lapStartDistance = 0, lapCrossings = [], lapCollisions = 0;
  const laps = [], gateEvents = [], crossings = [], misses = [], collisions = [];
  let minClear = Infinity, minClearAt = null;
  const frames = [];
  const c0 = clearance(geom, st.x, st.z, st.yaw, L, W, 10);
  if (c0 <= TOLERANCE) { terminal = true; reason = 'invalid_spawn'; }
  for (const action of report.actions) {
    if (terminal) break;
    const steering = num(action?.steering, 0, -cfg.maxSteer, cfg.maxSteer), speed = num(action?.speed, 0, -cfg.maxSpeed, cfg.maxSpeed);
    const remaining = Math.min(cfg.dt, cfg.timeLimit - st.sim_time);
    let elapsed = 0;
    while (elapsed < remaining - 1e-12 && !terminal) {
      const h = Math.min(cfg.substep, .012 / maxRate, remaining - elapsed);
      const prev = st, mu = surfaceMu(geom, st.x, st.z, cfg.frictionMultiplier);
      let v;
      if (cfg.longitudinalModel === 'real-drive-response-estimate-v1') {
        const rate = st.speed >= 0 && speed > st.speed ? Math.min(cfg.driveAccelerationCap, mu * G, (speed - st.speed) / cfg.driveTau) : cfg.brakeDeceleration;
        v = approach(st.speed, speed, rate * h);
      } else v = approach(st.speed, speed, cfg.acceleration * h);
      const d = approach(st.steering, steering, cfg.steerRate * h);
      const raw = v / wb * Math.tan(d), limit = mu * G / Math.max(Math.abs(v), .01), w = clamp(raw, -limit, limit);
      const yawMid = st.yaw + w * h / 2;
      const trial = {x: st.x + v * Math.cos(yawMid) * h, z: st.z - v * Math.sin(yawMid) * h, yaw: wrap(st.yaw + w * h), speed: v, steering: d,
        sim_time: st.sim_time, frame: st.frame};
      const t = st.sim_time + elapsed + h;
      const clr = clearance(geom, trial.x, trial.z, trial.yaw, L, W, 10);
      if (clr <= TOLERANCE) {
        collisions.push({time: t, frame: st.frame, x: trial.x, z: trial.z}); lapCollisions++;
        st = {...st, speed: 0, steering: d};
        if (cfg.stopOnCollision) { terminal = true; reason = 'collision'; }
      } else {
        if (clr < minClear) { minClear = clr; minClearAt = {time: t, x: trial.x, z: trial.z}; }
        st = trial; distance += Math.hypot(st.x - prev.x, st.z - prev.z);
        // 黒シート(全部数える)
        if (crossing) {
          const a = (prev.z - crossing.z) * crossing.direction, b = (st.z - crossing.z) * crossing.direction;
          if (a < 0 && b >= 0) {
            const f = -a / (b - a), x = prev.x + (st.x - prev.x) * f, yaw = prev.yaw + wrap(st.yaw - prev.yaw) * f;
            const half = L / 2 * Math.abs(Math.cos(yaw)) + W / 2 * Math.abs(Math.sin(yaw));
            const onSheet = !sheet || (Math.abs(x - sheet.x) <= sheet.w / 2 && Math.abs(crossing.z - sheet.z) <= sheet.d / 2);
            const marginLeft = x - half - crossing.minX, marginRight = crossing.maxX - (x + half);
            const valid = marginLeft >= -1e-12 && marginRight >= -1e-12 && onSheet;
            const ev = {time: t - (1 - f) * h, x, yaw, half_width_x: half, margin_min_m: Math.min(marginLeft, marginRight), valid, lap: laps.length + 1};
            crossings.push(ev);
            if (valid) lapCrossings.push(ev);
          }
        }
        // ゲート(順番どおり)
        if (routeIndex < n) {
          const target = targets[routeIndex], before = routeIndex ? targets[routeIndex - 1] : route[0];
          const len = Math.hypot(target[0] - before[0], target[1] - before[1]), nx = (target[0] - before[0]) / len, nz = (target[1] - before[1]) / len;
          const a = (prev.x - target[0]) * nx + (prev.z - target[1]) * nz, b = (st.x - target[0]) * nx + (st.z - target[1]) * nz;
          if (a < 0 && b >= 0) {
            const f = -a / (b - a), x = prev.x + (st.x - prev.x) * f, z = prev.z + (st.z - prev.z) * f;
            const lateral = Math.abs(-(x - target[0]) * nz + (z - target[1]) * nx);
            if (lateral > cfg.gateHalfWidth) misses.push({gate: routeIndex, time: t, lateral});
            else {
              passed++; routeIndex++; lapPassed++;
              gateEvents.push({lap: laps.length + 1, gate: routeIndex - 1, time: t - (1 - f) * h, margin_m: cfg.gateHalfWidth - lateral});
              if (continuous) {
                if (lapPassed === n) {
                  const endTime = t - (1 - f) * h, endDistance = distance - (1 - f) * Math.hypot(st.x - prev.x, st.z - prev.z);
                  const crossedAll = lapCrossings.length >= 1, enough = endDistance - lapStartDistance >= Math.max(L, routeLength * .5);
                  if (!crossedAll || !enough) { terminal = true; reason = crossedAll ? 'lap_distance_insufficient' : 'required_crossing_missing'; st = {...st, speed: 0}; }
                  else {
                    laps.push({lap: laps.length + 1, start_time: lapStartTime, end_time: endTime, elapsed: endTime - lapStartTime, gates: lapPassed,
                      valid_crossings: lapCrossings.length, collisions: lapCollisions, distance: endDistance - lapStartDistance});
                    lapStartTime = endTime; lapStartDistance = endDistance; lapCrossings = []; lapCollisions = 0; routeIndex = 0; lapPassed = 0;
                  }
                }
              } else if (passed === n && distance >= Math.max(L, routeLength * .5)) {
                terminal = true; reason = lapCrossings.length >= 1 ? 'route_complete' : 'required_crossing_missing'; st = {...st, speed: 0};
              }
            }
          }
        }
      }
      elapsed += h;
    }
    st = {...st, sim_time: Math.min(cfg.timeLimit, st.sim_time + elapsed), frame: st.frame + 1};
    if (!terminal && st.sim_time >= cfg.timeLimit - 1e-10) {
      terminal = true;
      if (continuous) { st.sim_time = cfg.timeLimit; reason = 'collection_duration_complete'; } else { reason = 'time_limit'; st.speed = 0; }
    }
    if (!continuous && reason === 'route_complete')
      laps.push({lap: 1, start_time: 0, end_time: st.sim_time, elapsed: st.sim_time, gates: passed, valid_crossings: lapCrossings.length,
        collisions: lapCollisions, distance});
    frames.push({frame: st.frame, time: st.sim_time, x: st.x, z: st.z, yaw: st.yaw, speed: st.speed, steering: st.steering});
  }
  return {reason, terminal, frames, laps, passed_gates: passed, gate_events: gateEvents, gate_misses: misses, crossings, collisions,
    min_clearance_m: minClear, min_clearance_at: minClearAt, current_lap_gates: continuous ? lapPassed : null,
    current_lap_valid_crossings: lapCrossings.length, route_length_m: routeLength};
}

// ---- コース・物理・開始の照合 ----
export function checkContract(report, failures) {
  const base = loadBaseWorld(), world = report.world, cfg = report.config;
  for (const k of ['segments', 'disks', 'surfaces', 'bounds', 'requiredCrossings'])
    if (stable(world[k]) !== stable(base[k])) failures.push(`world.${k} が v3 コース(world-v3.json)と違います`);
  // 経路は標準の ROUTE を開始ゲートで回したもの
  const variation = report.context?.start_variation ?? {gate: 0, left_offset_m: 0, left_yaw_deg: 0};
  const expected = activeWorldFor(base, variation.gate ?? 0, variation.left_offset_m ?? 0, variation.left_yaw_deg ?? 0).world;
  if (stable(world.route) !== stable(expected.route)) failures.push('world.route が標準の経路を開始ゲートで回したものではありません');
  for (const k of ['x', 'z', 'yaw']) if (!(Math.abs(world.spawn[k] - expected.spawn[k]) <= 1e-9)) failures.push(`開始姿勢 spawn.${k} が開始ずれの記録と違います`);
  if (stable(CANONICAL_ROUTE) !== stable(base.route)) failures.push('参照ループの ROUTE と world-v3.json の経路が違います');
  for (const [k, v] of Object.entries({...PHYSICS, gateHalfWidth: .65, substep: .01}))
    if (!['noiseStd'].includes(k) && cfg[k] !== v) failures.push(`物理の設定 ${k} が手順と違います: ${cfg[k]}`);
  return variation;
}

// 全部の確認。minLaps: 連続周回で要る完走の周の数、minClearance: 要る最小すき間 [m]
export async function verifyReport(report, {minLaps = null, minClearance = 0, requireSuccess = true} = {}) {
  const failures = [], errors = {};
  const note = (label, a, b) => {
    const e = Math.abs(a - b); errors[label] = Math.max(errors[label] ?? 0, Number.isFinite(e) ? e : Infinity);
    if (!(e <= TOLERANCE)) failures.push(`${label}: ${a} と ${b}(差 ${e})`);
  };
  if (report.schema !== 'course-simulator-report/v1') failures.push('レポートの形式が course-simulator-report/v1 ではありません');
  const variation = checkContract(report, failures);
  const continuous = report.config.episodeMode === 'continuous-laps-v1';
  // 1) 元のコアで再生
  const replay = await replayOriginal(report);
  if (Array.isArray(report.trace)) {
    if (report.trace.length !== replay.trace.length) failures.push(`軌跡のフレーム数が違います: ${report.trace.length} と ${replay.trace.length}`);
    for (let i = 0; i < Math.min(report.trace.length, replay.trace.length); i++)
      for (const k of ['time', 'x', 'z', 'yaw', 'speed', 'steering']) note('replay_trace.' + k, report.trace[i][k], replay.trace[i][k]);
  }
  if (stable(replay.outcome) !== stable(report.outcome)) failures.push(`終了の理由が再生と違います: ${stable(report.outcome)} と ${stable(replay.outcome)}`);
  for (const k of ['collisions', 'passed_gates', 'route_index']) if (replay.metrics[k] !== report.metrics[k]) failures.push(`metrics.${k} が再生と違います`);
  note('replay_elapsed', report.metrics.elapsed, replay.metrics.elapsed);
  const simLaps = continuous ? replay.metrics.lap_events.map(l => ({lap: l.lap_index, elapsed: l.elapsed, gates: l.passed_gates, collisions: l.collisions,
    crossings: l.required_crossing_events.length})) : (replay.outcome.reason === 'route_complete' ? [{lap: 1, elapsed: replay.metrics.elapsed,
    gates: replay.metrics.passed_gates, collisions: replay.metrics.collisions, crossings: replay.metrics.required_crossing_events.length}] : []);
  if (continuous) {
    const saved = report.metrics.lap_events ?? [];
    if (saved.length !== simLaps.length) failures.push('周の数が再生と違います');
    saved.forEach((l, i) => simLaps[i] && note('replay_lap_elapsed', l.elapsed, simLaps[i].elapsed));
  }
  // 2) 自前の積分と数え直し
  const own = independentAudit(report);
  if (own.frames.length !== replay.trace.length) failures.push(`自前の積分のフレーム数 ${own.frames.length} が再生 ${replay.trace.length} と違います`);
  for (let i = 0; i < Math.min(own.frames.length, replay.trace.length); i++)
    for (const k of ['time', 'x', 'z', 'yaw', 'speed', 'steering']) note('own_physics.' + k, own.frames[i][k], replay.trace[i][k]);
  if (own.reason !== replay.outcome.reason) failures.push(`自前の終了の理由 ${own.reason} が sim の ${replay.outcome.reason} と違います`);
  if (own.passed_gates !== replay.metrics.passed_gates) failures.push(`自前のゲート数 ${own.passed_gates} が sim の ${replay.metrics.passed_gates} と違います`);
  if (own.collisions.length !== replay.metrics.collisions) failures.push(`自前の衝突 ${own.collisions.length} が sim の ${replay.metrics.collisions} と違います`);
  if (own.laps.length !== simLaps.length) failures.push(`自前の周の数 ${own.laps.length} が sim の ${simLaps.length} と違います`);
  own.laps.forEach((l, i) => { if (simLaps[i]) note('own_lap_elapsed', l.elapsed, simLaps[i].elapsed); });
  // 3) ベストラップとしての条件
  const lapChecks = own.laps.map(l => ({lap: l.lap, elapsed: l.elapsed, gates_ok: l.gates === 33, crossing_exactly_once: l.valid_crossings === 1,
    collisions: l.collisions}));
  for (const l of lapChecks) {
    if (!l.gates_ok) failures.push(`${l.lap} 周目のゲートが 33 ではありません`);
    if (!l.crossing_exactly_once) failures.push(`${l.lap} 周目の黒シートの有効な横断がちょうど 1 回ではありません`);
    if (l.collisions) failures.push(`${l.lap} 周目に衝突があります`);
  }
  if (requireSuccess) {
    if (continuous) {
      if (replay.outcome.reason !== 'collection_duration_complete') failures.push(`連続周回が 20 秒走り切っていません(${replay.outcome.reason})`);
      if (minLaps !== null && own.laps.length < minLaps) failures.push(`完走した周が ${own.laps.length} で、${minLaps} 周に足りません`);
    } else if (replay.outcome.reason !== 'route_complete') failures.push(`単周が完走していません(${replay.outcome.reason})`);
    if (replay.metrics.collisions) failures.push('衝突があります');
  }
  if (!(own.min_clearance_m >= minClearance)) failures.push(`最小すき間 ${own.min_clearance_m} m が ${minClearance} m より小さい`);
  const crossingMargins = own.crossings.filter(c => c.valid).map(c => c.margin_min_m);
  return {schema: 'rl-bestlap-lap-verification/v1', pass: failures.length === 0, failures, tolerance: TOLERANCE,
    episode_mode: continuous ? 'continuous-laps-v1' : 'single-lap-v1', start_variation: variation, reason: replay.outcome.reason,
    frames: replay.trace.length, laps_sim: simLaps, laps_own: own.laps, lap_checks: lapChecks, passed_gates: own.passed_gates,
    gate_margin_min_m: own.gate_events.length ? Math.min(...own.gate_events.map(g => g.margin_m)) : null, gate_misses: own.gate_misses.length,
    crossings_total: own.crossings.length, crossings_valid: crossingMargins.length, crossing_margin_min_m: crossingMargins.length ? Math.min(...crossingMargins) : null,
    collisions_sim: replay.metrics.collisions, collisions_own: own.collisions.length, collision_events_own: own.collisions,
    min_clearance_m: own.min_clearance_m, min_clearance_at: own.min_clearance_at, max_abs_error: errors};
}

export function loadReport(p) { return JSON.parse(fs.readFileSync(p, 'utf8')); }
