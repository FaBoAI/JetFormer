// 障害物よけ(obstacle)の確認 r2: 1 回を走らせる(2026-10-14、確認の段 r2)。check_ob/run_ob.mjs(確認 r1)の写しに、r2 の方策・はう回の物差し・
// 衝突の速さを足したもの。学習・評価のコード(episode.mjs・eval_modes.mjs・env_core.mjs・env_modes.mjs・lap_audit.mjs・variation_metrics.mjs・
// gate_miss.mjs・rl/ob/*・rl/ob2/*)を使わない。
//   コース: ユーザーのリリースの qa/load-course-world.mjs(ブラウザと同じ scene.js から作る)を app.js の reset と同じ式で開始ゲートへ回す
//   障害物: 配置の {shape, x, z, r, length, yaw, h} を、modes.json common.obstacle_layouts.shapes の文言から自前で world の項目にする
//           (柱 = disks の kind 'obstacle' の円、箱 = segments の kind 'obstacle' のカプセル。向きは sim の向き = 前 (cos ψ, −sin ψ))
//   物理: ユーザーのリリースの dist/sim-core.js(元のファイル、sha256 aaeb2379…549e)。標準の物理だけ(obstacle の判定と記録は標準の物理)
//   操作: ブラウザに配る組 web/rl/mode_runtime.mjs + 候補 candidates/obstacle/policy.json(r2、3cffd1d2…)。比べる相手は bundle/fastest(7c43cb53…)と
//         r1 の候補 candidates/obstacle_r1/policy.json(4ba77f44…)
//   すき間: check_ln/own_geom.mjs(確認の段の自前の多角形の距離・自前の積分。小区間ごと、12 cm 未満なら間を 4 分割)
//   s・e_y: check_ln/own_extra.mjs の自前の R への射影 + この下の自前の全体の射影(障害物の位置)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {auditEpisode, buildObstacles, ownClearance} from '../check_ln/own_geom.mjs';
import {createProjector, binFirstPass, LOOP_L_OWN} from '../check_ln/own_extra.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export const SIM_CORE_SHA = sha(`${REL}/dist/sim-core.js`);
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います');
const core = await import(`${REL}/dist/sim-core.js`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();
export const RT = await import(`${EXP}/web/rl/mode_runtime.mjs`);
const LAYOUTS_RT = await import(`${EXP}/web/rl/layouts.mjs`);   // obstaclesFromWorld だけ(ランタイムが world から障害物を読む所の確かめ)
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const POLICY_PATH = {obstacle: `${EXP}/candidates/obstacle/policy.json`, obstacle_r1: `${EXP}/candidates/obstacle_r1/policy.json`, fastest: `${EXP}/bundle/fastest/policy.json`};
const EXPECT_SHA = {obstacle: '3cffd1d233fc206756917cc9db10466d98088cca7bc85ea136ffb437aad1f2fa', obstacle_r1: '4ba77f44a77c06a93c9f5c53ba3027091748e2ad45cf55ad1758c55480a906bb',
  fastest: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee'};
const MODE_OF = {obstacle: 'obstacle', obstacle_r1: 'obstacle', fastest: 'fastest'};
const policies = {};
export function policyFor(kind) {
  const p = POLICY_PATH[kind] ?? kind;
  if (!policies[p]) policies[p] = {weights: JSON.parse(fs.readFileSync(p, 'utf8')), sha: sha(p), path: p};
  if (EXPECT_SHA[kind] && policies[p].sha !== EXPECT_SHA[kind]) throw new Error(`${kind} の方策の sha256 が違います`);
  return policies[p];
}
export const NOMINAL = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0, longitudinalModel: 'real-drive-response-estimate-v1',
  driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2});
// app.js の reset と同じ(dist/app.js 111〜114 行。check_ln/run_ln.mjs と同じ写し)
export function activeWorld(world, gate, offset, yawDeg) {
  const startIndex = Math.max(0, Math.min(world.route.length - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(startIndex), ...world.route.slice(0, startIndex)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {world: {...world, route, spawn: {x: first[0] + off * Math.sin(heading), z: first[1] - off * Math.cos(heading), yaw: heading + turn}},
    variation: {gate: startIndex, left_offset_m: off, left_yaw_deg: turn * 180 / Math.PI}};
}
// 自前: 配置の障害物 → world の項目(modes.json の文言: 柱 = 円柱 半径 r・高さ 0.30、箱 = カプセル 長さ length・半径 0.05・高さ 0.25、向きは sim の向き)
export function ownObstacleEntries(obstacles) {
  const disks = [], segments = [];
  for (const o of obstacles ?? []) {
    if (o.shape === 'post') disks.push({x: o.x, z: o.z, radius: o.r, height: o.h ?? .3, kind: 'obstacle'});
    else if (o.shape === 'box') {
      const fx = Math.cos(o.yaw), fz = -Math.sin(o.yaw), h = o.length / 2;
      segments.push({a: [o.x - h * fx, o.z - h * fz], b: [o.x + h * fx, o.z + h * fz], radius: o.r ?? .05, height: o.h ?? .25, kind: 'obstacle'});
    } else throw new Error('知らない形 ' + o.shape);
  }
  return {disks, segments};
}
export function worldWith(world, obstacles) {
  const e = ownObstacleEntries(obstacles);
  return {...world, disks: [...(world.disks ?? []), ...e.disks], segments: [...(world.segments ?? []), ...e.segments]};
}
// 足した障害物だけの world(外枠は遠く。路面は元のまま = 自前の積分の摩擦に使う)
function addedOnlyWorld(world, obstacles) {
  const e = ownObstacleEntries(obstacles), far = 1e5;
  return {segments: e.segments, disks: e.disks, bounds: {minX: -far, maxX: far, minZ: -far, maxZ: far}, surfaces: world.surfaces ?? []};
}
const BASE_OB = buildObstacles(BASE_WORLD);
export function startClearance(gate, offset, yawDeg, obstacles = null) {
  const {world} = activeWorld(BASE_WORLD, gate, offset, yawDeg);
  const ob = obstacles?.length ? buildObstacles(worldWith(BASE_WORLD, obstacles)) : BASE_OB;
  return ownClearance(ob, world.spawn.x, world.spawn.z, world.spawn.yaw);
}
// ---- 自前の全体の射影(参照ループ R の点の列はデータとして読む) ----
const REF = JSON.parse(fs.readFileSync(`${EXP}/rl/runtime/reference_loop.json`, 'utf8'));
const SEGS = [];
{
  let s = 0;
  for (let i = 0; i < REF.x.length; i++) {
    const j = (i + 1) % REF.x.length, ax = REF.x[i], az = REF.z[i], bx = REF.x[j], bz = REF.z[j], len = Math.hypot(bx - ax, bz - az);
    SEGS.push({ax, az, len, s0: s, tx: (bx - ax) / len, tz: (bz - az) / len}); s += len;
  }
}
export const GATE_S = REF.gate_s;
export function projectGlobal(x, z) {
  let best = null;
  for (const q of SEGS) {
    const t = Math.max(0, Math.min(q.len, (x - q.ax) * q.tx + (z - q.az) * q.tz)), px = q.ax + q.tx * t, pz = q.az + q.tz * t, d = Math.hypot(x - px, z - pz);
    if (!best || d < best.d) best = {d, s: q.s0 + t, ey: (x - px) * q.tz + (z - pz) * (-q.tx)};
  }
  return best;
}
const quant = (arr, p) => { const b = [...arr].sort((x, y) => x - y); if (!b.length) return null; const i = p * (b.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };
const steerUs = a0 => 1442 + a0 * (a0 >= 0 ? 558 : 442);
const r4 = x => x == null || !Number.isFinite(x) ? null : +x.toFixed(4);
const modL = s => ((s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN;

// spec: {policy: 'obstacle'|'fastest', start: {gate, left_offset_m, left_yaw_deg, seed}, layout: {id, obstacles} | null, single: bool, keepPath}
export async function runOne(spec) {
  const kind = spec.policy ?? 'obstacle', P = policyFor(kind), mode = MODE_OF[kind];
  const st0 = spec.start, seed = (st0.seed ?? 1) >>> 0, obstacles = spec.layout?.obstacles ?? [];
  const {world: active, variation} = activeWorld(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const world = obstacles.length ? worldWith(active, obstacles) : active;
  const base = {...NOMINAL, seed, timeLimit: 20, ...(spec.single ? {} : {episodeMode: 'continuous-laps-v1'})};
  const sim = core.createSimulation(world, base);
  let snap = sim.reset({seed, scenario: 'baseline'});
  if (snap.state.terminal) return {invalid_spawn: true, start: st0, reason0: snap.state.reason};
  const cfgN = snap.config;
  const params = mode === 'obstacle' && spec.layout?.id ? {layout_id: spec.layout.id} : {};
  const teacher = RT.createModeTeacher(world, cfgN, variation, {mode, weights: P.weights, policySha256: P.sha, params,
    overrides: mode !== 'obstacle' && obstacles.length ? {allow_obstacles: true} : {}});
  // ランタイムが world から読み直した障害物と、配置の値の差(ブラウザの重ね描きが同じ形で足せば、方策は同じ障害物を見る)
  const seen = LAYOUTS_RT.obstaclesFromWorld(world);
  let seenErr = 0;
  if (obstacles.length) {
    if (seen.length !== obstacles.length) seenErr = Infinity;
    for (const o of obstacles) {
      const m = seen.filter(q => q.shape === o.shape).map(q => Math.max(Math.abs(q.x - o.x), Math.abs(q.z - o.z), Math.abs((q.r ?? 0) - (o.r ?? 0)),
        o.shape === 'box' ? Math.max(Math.abs(q.length - o.length), Math.abs(Math.sin(q.yaw - o.yaw))) : 0));
      seenErr = Math.max(seenErr, m.length ? Math.min(...m) : Infinity);
    }
  }
  const OBW = buildObstacles(world);
  const labels = {frames: 0, l0: 0, ph: {}, ev_nonzero: 0, sh_key: 0, es_key: 0, bad_keys: 0, p0: null, bytes_max: 0, cl_diff_mm_max: 0};
  const pre = [], gateCnt = [];
  while (!snap.state.terminal) {
    const st = snap.state;
    gateCnt.push(snap.metrics.passed_gates ?? 0);
    const out = teacher.action({state: st, metrics: snap.metrics, config: cfgN});
    const jf = out.info.jf, js = JSON.stringify(out.info);
    labels.frames++; if (jf.l !== 1) labels.l0++; labels.ph[jf.ph] = (labels.ph[jf.ph] ?? 0) + 1; if ('sh' in jf) labels.sh_key++; if ('es' in jf) labels.es_key++;
    if (jf.ev) labels.ev_nonzero++;
    if (st.frame === 0) labels.p0 = jf.p ?? null;
    if (js.match(/steering_pwm|throttle_pwm|"\//)) labels.bad_keys++;
    if (st.frame > 0) labels.bytes_max = Math.max(labels.bytes_max, js.length);
    // info.jf.cl(ランタイムの d_min [mm])と自前のすき間(足した障害物を含む全部)
    const own = ownClearance(OBW, st.x, st.z, st.yaw);
    // ランタイムの d_min は 0.5 m で切る(features.mjs DMIN_CLIP)ので、自前も 0.5 m で切って比べる
    labels.cl_diff_mm_max = Math.max(labels.cl_diff_mm_max, Math.abs(jf.cl - Math.min(own, .5) * 1000));
    pre.push({f: st.frame, x: st.x, z: st.z, yaw: st.yaw, v: st.speed, st_cmd: out.steering, sp_cmd: out.speed});
    snap = sim.step({steering: out.steering, speed: out.speed});
  }
  const rep = sim.report(), m = rep.metrics, fs_ = rep.final_state;
  // 最後のフレームは sim が途中で止める(衝突・周の終わり)ので、位置の照合から外す(check_ln と同じ)
  const skipLastErr = fs_.reason !== 'collection_duration_complete';
  const au = auditEpisode(world, rep.config, world.spawn, rep.actions, rep.trace, {skipLastErr});
  const auO = obstacles.length ? auditEpisode(addedOnlyWorld(world, obstacles), rep.config, world.spawn, rep.actions, rep.trace, {skipLastErr}) : null;
  // 自前の R への射影(操作の前の状態)
  const pj = createProjector();
  const frames = pre.map((q, k) => {
    const r = k === 0 ? pj.reset(q.x, q.z, q.yaw) : pj.advance(q.x, q.z);
    return {f: q.f, s: r.s, ey: r.ey, v: q.v, x: q.x, z: q.z, yaw: q.yaw, st: steerUs(q.st_cmd / .52), sp: q.sp_cmd, clr: au.per_frame[k] ?? null, clrO: auO ? auO.per_frame[k] : null, t: k / 30};
  });
  const s0 = frames[0].s, sEnd = frames.at(-1).s;
  // 足した障害物の横の通過(s ±0.6 m の窓に入って出た = 通り切った)。向き = 車の e_y − 障害物の e_y の符号(左が正)
  const passages = [];
  obstacles.forEach((o, oi) => {
    const pg = projectGlobal(o.x, o.z), so = pg.s, eo = pg.ey;
    const one = buildObstacles(addedOnlyWorld(world, [o]));
    for (let n = Math.floor((s0 - 1) / LOOP_L_OWN); n <= Math.floor(sEnd / LOOP_L_OWN) + 1; n++) {
      const c = n * LOOP_L_OWN + so;
      if (c - .6 < s0) continue;             // 開始のときにもう窓の中・後ろ
      const inWin = frames.filter(fr => fr.s >= c - .6 && fr.s <= c + .6);
      if (!inWin.length) continue;
      const completed = sEnd > c + .6 && !(m.collisions && frames.length && frames.at(-1).s <= c + .6);
      let near = inWin[0];
      for (const fr of inWin) if (Math.abs(fr.s - c) < Math.abs(near.s - c)) near = fr;
      let cmin = Infinity;
      for (const fr of inWin) cmin = Math.min(cmin, ownClearance(one, fr.x, fr.z, fr.yaw));
      passages.push({obstacle: oi, shape: o.shape, s_obst: r4(so), ey_obst: r4(eo), lap_n: n, completed, side: Math.sign(near.ey - eo), d_lat: r4(near.ey - eo),
        clearance_frame_min: r4(cmin), v: r4(near.v), t: r4(near.t)});
    }
  });
  // 周・ゲート(自前: R 上の s で通ったゲートの数と sim の passed_gates)
  const gatesCrossedS = [];
  for (let n = Math.floor((s0 - 1) / LOOP_L_OWN); n <= Math.floor(sEnd / LOOP_L_OWN) + 1; n++)
    for (const gs of GATE_S) { const a = n * LOOP_L_OWN + gs; if (a > s0 + .3 && a <= sEnd) gatesCrossedS.push(a); }
  gatesCrossedS.sort((a, b) => a - b);
  let stuck = 0;
  const cum = [];   // sim の数え(連続周回は周ごとに 0 に戻るので、足し上げる)
  { let tot = 0, prev = 0; for (const g of gateCnt) { if (g < prev) tot += prev; prev = g; cum.push(tot + g); } }
  for (let k = 0; k < frames.length; k++) {
    const need = gatesCrossedS.filter(a => a + .5 < frames[k].s).length;
    stuck = Math.max(stuck, need - cum[k]);
  }
  // 黒シートの横断(自前: 軌跡の z = 0 の下向きの横断、車幅の分の開口 1.26〜2.36 m)
  const ownCross = [];
  for (let i = 1; i < rep.trace.length; i++) {
    const a = rep.trace[i - 1], b = rep.trace[i];
    if (a.z > 0 && b.z <= 0) {
      const u = a.z / (a.z - b.z), x = a.x + (b.x - a.x) * u, y = a.yaw + wrap(b.yaw - a.yaw) * u;
      const half = .21 * Math.abs(Math.cos(y)) + .095 * Math.abs(Math.sin(y));
      if (x > 0) ownCross.push({frame: i, x: +x.toFixed(4), inside: x - half >= 1.26 - 1e-9 && x + half <= 2.36 + 1e-9});
    }
  }
  const laps = (m.lap_events ?? []).map(l => ({elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions}));
  const run = frames.filter(fr => fr.v > .3).map(fr => fr.v);
  const minFr = au.min_frame >= 0 ? frames[Math.min(au.min_frame, frames.length - 1)] : frames[0];
  const ce = m.collision_events?.[0] ?? null;
  const res = {policy: kind, mode, policy_sha256: P.sha, start: {...st0}, variation, layout_id: spec.layout?.id ?? null, n_obstacles: obstacles.length,
    single: !!spec.single, reason: fs_.reason, frames: fs_.frame, collisions: m.collisions,
    collision: ce ? {t: ce.time, frame: ce.frame, collider: ce.collider, kind: ce.kind, added: /^(disk|segment)-/.test(ce.collider) && isAdded(world, ce.collider)} : null,
    laps, passed_gates: m.passed_gates, gates_by_s: gatesCrossedS.length, gate_stuck_max: stuck,
    min_clearance: au.min_clearance, min_frame: au.min_frame, min_t: au.min_frame >= 0 ? au.min_frame / 30 : 0, min_kind: au.min_kind,
    min_s_from_start: minFr.s - s0, min_clear_added: auO ? auO.min_clearance : null, min_clear_added_t: auO && auO.min_frame >= 0 ? auO.min_frame / 30 : null,
    audit_pos_err: au.max_pos_err, overlap_frames: au.overlap_frames, overlap_frames_added: auO ? auO.overlap_frames : 0,
    start_clearance: ownClearance(OBW, world.spawn.x, world.spawn.z, world.spawn.yaw),
    obstacle_ahead_m: obstacles.length ? Math.min(...obstacles.map(o => { const d = modL(projectGlobal(o.x, o.z).s - modL(s0)); return d; })) : null,
    seen_obstacle_err: seenErr, speed: {mean_running: run.length ? run.reduce((a, b) => a + b, 0) / run.length : 0, p95: quant(run, .95), max: Math.max(0, ...rep.trace.map(t => t.speed))},
    passages, own_crossings: ownCross, labels, distance: m.distance, s_travel: sEnd - s0};
  if (spec.single) {
    res.lap_complete = fs_.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33 && (m.required_crossing_events?.length ?? 0) === 1;
    res.lap_from_stop = res.lap_complete ? m.elapsed : null;
    res.elapsed = m.elapsed;
  }
  res.valid_episode = !spec.single && fs_.reason === 'collection_duration_complete' && fs_.frame === 600 && m.collisions === 0;
  // はう・止まる回の物差し(確認 r2 で足した): 2 秒(60 フレーム)の窓で R の進みの最小(1.5 秒以降)、0.3 m/s 未満のフレームの数、周の数
  let prog2sMin = Infinity;
  for (let k = 45 + 60; k < frames.length; k++) prog2sMin = Math.min(prog2sMin, frames[k].s - frames[k - 60].s);
  res.crawl = {progress_2s_min_m: Number.isFinite(prog2sMin) ? r4(prog2sMin) : null, slow_frames: frames.filter(fr => fr.v < .3).length,
    laps_n: laps.length, mean_speed_all: r4(frames.reduce((a, fr) => a + fr.v, 0) / Math.max(1, frames.length))};
  if (ce) { const fr = frames[Math.min(ce.frame ?? frames.length - 1, frames.length - 1)]; res.collision.v = r4(rep.trace[Math.min(ce.frame ?? 0, rep.trace.length - 1)]?.speed ?? fr.v);
    res.collision.s_from_start = r4(frames.at(-1).s - s0); }
  res.valid_laps = laps.filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
  res.prof = {ey: binFirstPass(frames, 'ey').map(r4), v: binFirstPass(frames, 'v').map(r4), st: binFirstPass(frames, 'st').map(x => x == null ? null : +x.toFixed(2))};
  res.st_time = frames.map(fr => +fr.st.toFixed(1));
  // フレームの列(解析用): [t, s − s0, s mod L, e_y, すき間(全部), すき間(足した障害物), v, 舵 µs, 速さの指令]
  if (spec.keepFrames !== false) res.fr = frames.map(fr => [+fr.t.toFixed(4), r4(fr.s - s0), r4(modL(fr.s)), r4(fr.ey), r4(fr.clr), r4(fr.clrO), r4(fr.v), +fr.st.toFixed(1), r4(fr.sp)]);
  if (spec.keepPath) res.path = pre.filter((_, i) => i % 2 === 0).map(q => [+q.x.toFixed(4), +q.z.toFixed(4)]);
  return res;
}
// sim-core の id(disk-i・segment-i の i は world の配列の元の番号。sim-core.js normalizedWorld の flatMap の i)
function isAdded(world, id) {
  const [k, i] = id.split('-'), list = k === 'disk' ? world.disks : world.segments;
  return list?.[+i]?.kind === 'obstacle';
}
