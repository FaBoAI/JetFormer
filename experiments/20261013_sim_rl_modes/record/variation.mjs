#!/usr/bin/env node
// 記録の段: 記録して JetFormer に取り込んだデータセットの「ばらつきの物差し」VM1〜VM7(protocol.json variation_metrics)を、
// モードごと・全モードを合わせた群・前の教師のデータ(jetformer-simulator-id8-20min_…、舵のばらつき 7.6 µs)で計算する(2026-10-14、sim_rl_modes)。
//
//   node record/variation.mjs --dataset fastest=<data/datasets/名前> --dataset no_wall=... --teacher <教師のデータセット> --out variation_raw.json [--figdata fig.json]
//
// 入力はどれも「記録したもの」(frames.jsonl の操作の前の姿勢 pose_pre_action・前のフレームの state_post_action の速さと舵角・command_si の舵)。
// Node の予想ではなく、ブラウザで記録した走りそのものを物差しにかける(protocol variation_metrics.where_judged)。
//   s・e_y・向きのずれ: rl/runtime/features.mjs の createFeatureTracker(方策の観測と同じ射影。シーンの開始ゲートで回したコース)
//   すき間(フレームごと): features.mjs の clearance(車体の長方形 0.42 × 0.19 m と全障害物の距離。ここでは 3 m で切る)
//   シーンの最小すき間: replay_check.mjs の正確な値(lap_audit の小区間ごと)があればそれ(--replay)、無ければフレームの最小
//   乱しの出来事・学習から外す割合・立て直し: action_info.jf(ph = perturb の続いた区間 = 1 つの出来事)、mask_rule は JetFormer と同じ k ∈ [j − 10, j]、
//   立て直しは protocol definitions.recovered(乱しなしの線 ē_y(s)・ψ̄(s) から 0.10 m・0.15 rad 以内が 15 フレーム続く最初のフレーム)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {activeWorldFor} from '../snapshot/rl/env_core.mjs';
import {BASE_WORLD, readJson} from '../rl/episode.mjs';
import {createFeatureTracker, clearance, LOOP_LENGTH} from '../rl/runtime/features.mjs';
import {VM1, VM2, VM3, VM4, VM5, VM6, VM7, nominalAt, quantile, steerUs, binProfile} from '../rl/variation_metrics.mjs';
import {createRng} from '../rl/runtime/rng.mjs';

const args = process.argv.slice(2);
const multi = n => args.flatMap((a, i) => a === n ? [args[i + 1]] : []);
const opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const L = LOOP_LENGTH;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;

// ---------------------------------------------------------------- 1 シーンを読む
function readScene(root, scene, coll) {
  const dir = path.join(root, 'simulation/scenes', scene);
  const sess = readJson(path.join(dir, 'session.json'));
  const rep = readJson(path.join(dir, 'evaluation-report.json'));
  const rows = fs.readFileSync(path.join(dir, 'frames.jsonl'), 'utf8').split('\n').filter(Boolean).map(JSON.parse);
  const cs = coll?.scenes?.find(s => s.scene === scene) ?? null;
  const st0 = cs?.plan_start ?? sess.context.start_variation;
  const {world} = activeWorldFor(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const cfg = {...sess.config};
  const tr = createFeatureTracker(world, cfg), obs = new Float64Array(79), aux = {};
  const frames = [];
  let prev = {speed: 0, steering: 0};
  rows.forEach((r, i) => {
    const p = r.pose_pre_action;
    if (i === 0) tr.resetProgress(p.x, p.z, 0); else tr.advanceProgress(p.x, p.z);
    tr.observe({x: p.x, z: p.z, yaw: p.yaw, speed: prev.speed, steering: prev.steering}, 0, 0, 0, obs, aux);
    const jf = r.action_info?.jf ?? null;
    frames.push({f: i, x: p.x, z: p.z, yaw: p.yaw, s: aux.s, ey: aux.lateral_error, epsi: aux.heading_error, v: prev.speed, st: r.command_si.steering,
      cmd_v: r.command_si.speed, dmin: clearance(tr.geometry, p.x, p.z, p.yaw, cfg.vehicleLength ?? .42, cfg.vehicleWidth ?? .19, 3),
      l: jf ? jf.l : 1, ph: jf?.ph ?? null, sh: jf?.sh ?? 0});
    prev = r.state_post_action;
  });
  const m = rep.metrics ?? {};
  const laps = (m.lap_events ?? []).filter(l => l.collisions === 0).map(l => l.elapsed);
  const p0 = rows[0].action_info?.jf?.p ?? null;
  return {scene, frames, s0: frames[0].s, laps, start: st0, seed: sess.config.seed, scenario: rep.scenario ?? null, params: cs?.params ?? p0 ?? {},
    collisions: m.collisions ?? null, reason: rep.outcome?.reason ?? null, completed_laps: m.completed_laps ?? null,
    teacher_id: sess.context?.teacher?.id ?? null, world_route_offset: tr.routeOffset};
}

function readDataset(root, {limit = null} = {}) {
  const card = readJson(path.join(root, 'dataset_card.json'));
  const coll = fs.existsSync(path.join(root, 'simulation/collection.json')) ? readJson(path.join(root, 'simulation/collection.json')) : null;
  // シーンの順番: collection.json(記録の順)。無ければ source_scenes
  let scenes = coll?.scenes?.map(s => s.scene) ?? card.source_scenes;
  if (limit) scenes = scenes.slice(0, limit);
  return {name: path.basename(root), card, coll, scenes: scenes.map(s => readScene(root, s, coll))};
}

// ---------------------------------------------------------------- 乱しの出来事・学習から外す・立て直し
function maskFromLearn(learn, horizon = 10) {
  const mask = new Array(learn.length).fill(0);
  learn.forEach((l, j) => { if (l === 0) for (let k = Math.max(0, j - horizon); k <= j; k++) mask[k] = 1; });
  return mask;
}
function eventsOf(sc) {
  const ev = [];
  let cur = null;
  sc.frames.forEach((q, f) => {
    const on = q.ph === 'perturb' || q.l === 0;
    if (on && !cur) cur = {ps: f};
    if (!on && cur) { cur.pe = f; ev.push(cur); cur = null; }
  });
  if (cur) { cur.pe = sc.frames.length; ev.push(cur); }
  return ev;
}
function recoveryOf(sc, nl) {
  const F = sc.frames, T = F.length, ev = sc.events;
  const stallAt = [], wrongAt = [];
  let sf = 0, smax = -Infinity;
  for (let f = 0; f < T; f++) {
    const q = F[f];
    sf = f / 30 >= 1 && q.v < .2 ? sf + 1 : 0; stallAt.push(sf >= 60);
    smax = Math.max(smax, q.s); wrongAt.push(q.s < smax - 1);
  }
  return ev.map((e, k) => {
    const pe = e.pe, next = ev[k + 1]?.ps ?? T;
    let rec = null, run = 0;
    for (let f = pe; f < T; f++) {
      const q = F[f], nm = nominalAt(nl, q.s);
      const ok = Math.abs(q.ey - nm.ey) <= .10 && Math.abs(wrap(q.epsi - nm.psi)) <= .15;
      run = ok ? run + 1 : 0;
      if (run >= 15) { rec = f - 14; break; }
    }
    let bad = false;
    for (let f = pe; f < Math.min(next, T) && !bad; f++) if (stallAt[f] || wrongAt[f]) bad = true;
    const recS = rec == null ? null : (rec - pe) / 30;
    return {ps: e.ps, pe, frames: pe - e.ps, recovered_s: recS, success: recS != null && recS <= 3 && !bad, bad_after: bad};
  });
}

// ---------------------------------------------------------------- 群の物差し
// VM7 を、群ごとに同じフレーム数ずつ(合計 N)取って数える(protocol headline_judged の vm7_text)
function vm7Equal(groups, N = 9000, seed = 12345) {
  const per = Math.floor(N / groups.length), pts = [];
  groups.forEach((scenes, gi) => {
    const p = [];
    for (const sc of scenes) for (const q of sc.frames) if (q.s - sc.s0 >= 3) p.push(q);
    const rng = createRng(seed + gi), idx = Array.from(p.keys());
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (const i of idx.slice(0, per)) pts.push(p[i]);
  });
  const cells = new Set();
  for (const q of pts) {
    const sm = ((q.s % L) + L) % L;
    cells.add(`${Math.floor(sm / .25)}|${Math.floor(q.ey / .05)}|${Math.floor(q.epsi / .1)}|${Math.floor(q.v / .5)}`);
  }
  return {cells: cells.size, frames_used: pts.length, per_group: per};
}
function dist(a) {
  if (!a.length) return {n: 0};
  return {n: a.length, min: Math.min(...a), p05: quantile(a, .05), p25: quantile(a, .25), median: quantile(a, .5), p75: quantile(a, .75), p95: quantile(a, .95),
    max: Math.max(...a), mean: mean(a)};
}
function groupMetrics(scenes, {exactMin = null} = {}) {
  // VM5 のシーンの最小すき間は、正確な値(replay_check)があればそれ
  const withMin = scenes.map(sc => ({...sc, min_clearance: exactMin?.[sc.scene] ?? Math.min(...sc.frames.map(q => q.dmin))}));
  const lap1 = scenes.map(sc => sc.laps[0]).filter(Number.isFinite);
  const allV = scenes.flatMap(sc => sc.frames.filter(q => q.s - sc.s0 >= 3).map(q => q.v));
  const out = {scenes: scenes.length, frames: scenes.reduce((a, sc) => a + sc.frames.length, 0),
    VM1_us: VM1(scenes), VM2_us: VM2(scenes), VM3_m: VM3(scenes), VM4: VM4(scenes), VM5: VM5(withMin), VM6: VM6(scenes), VM7: VM7(scenes, 9000)};
  // 足したもの(報告): 開始の広がり・速さ・すき間の分布・周の時間・シールド
  out.extra = {
    start_gates: [...new Set(scenes.map(sc => sc.start.gate))].sort((a, b) => a - b),
    start_offset_m: dist(scenes.map(sc => sc.start.left_offset_m ?? 0)), start_yaw_deg: dist(scenes.map(sc => sc.start.left_yaw_deg ?? 0)),
    scenarios: Object.fromEntries([...new Set(scenes.map(sc => sc.scenario))].map(s => [s, scenes.filter(x => x.scenario === s).length])),
    speed_mps_after3m: dist(allV), mean_speed_per_scene: dist(scenes.map(sc => mean(sc.frames.map(q => q.v)))),
    ey_abs_m_after3m: dist(scenes.flatMap(sc => sc.frames.filter(q => q.s - sc.s0 >= 3).map(q => Math.abs(q.ey)))),
    clearance_frame_m: dist(scenes.flatMap(sc => sc.frames.map(q => q.dmin))),
    scene_min_clearance_m: dist(withMin.map(sc => sc.min_clearance)),
    lap1_s: dist(lap1), laps_per_scene: dist(scenes.map(sc => sc.laps.length)),
    shield_frames: scenes.reduce((a, sc) => a + sc.frames.filter(q => q.sh === 1).length, 0),
    collisions: scenes.reduce((a, sc) => a + (sc.collisions ?? 0), 0),
    speed_levels: Object.fromEntries([...new Set(scenes.map(sc => sc.params?.c_v).filter(x => x != null))].sort().map(c => [c, scenes.filter(x => x.params?.c_v === c).length])),
    steer_cmd_us_after3m: dist(scenes.flatMap(sc => sc.frames.filter(q => q.s - sc.s0 >= 3).map(q => steerUs(q.st / .52)))),
  };
  return out;
}
const pick = (o, ks) => Object.fromEntries(ks.map(k => [k, o[k]]));

// ---------------------------------------------------------------- 本体
const dsArgs = multi('--dataset').map(x => { const i = x.indexOf('='); return [x.slice(0, i), x.slice(i + 1)]; });
const replayArgs = Object.fromEntries(multi('--replay').map(x => { const i = x.indexOf('='); return [x.slice(0, i), x.slice(i + 1)]; }));
const teacherRoot = opt('--teacher');
const nominalPath = opt('--nominal-recover');
const t0 = Date.now();
const out = {schema: 'jf-sim-rl-modes-variation/v1', written_at: new Date().toISOString(), protocol: 'protocol.json variation_metrics(VM1〜VM7)',
  inputs: {}, teacher: null, modes: {}, pooled: null, judged: {}};
const groups = {};
const exact = {};
for (const [key, root] of dsArgs) {
  const ds = readDataset(root);
  groups[key] = ds;
  out.inputs[key] = {dataset: ds.name, scenes: ds.scenes.length, policy_sha256: ds.card.rl?.policy_sha256 ?? null, tags: ds.card.tags};
  if (replayArgs[key]) {
    const rp = readJson(replayArgs[key]);
    exact[key] = Object.fromEntries(rp.scenes.map(s => [s.scene, s.min_clearance_m]));
  }
}
// 離脱復活: 出来事・学習から外す割合・立て直し
const nl = nominalPath ? readJson(nominalPath) : null;
for (const [key, ds] of Object.entries(groups)) {
  for (const sc of ds.scenes) {
    sc.events = eventsOf(sc);
    sc.mask = maskFromLearn(sc.frames.map(q => q.l));
    sc.recovery = nl && key === 'recover' ? recoveryOf(sc, nl) : [];
  }
}
// 教師(前のデータ): 全 60 シーン(VM1〜VM6)と、0〜14 番の 15 シーン(VM7・同じ数での比べ)
let teacher = null, teacher15 = null;
if (teacherRoot) {
  teacher = readDataset(teacherRoot);
  for (const sc of teacher.scenes) { sc.events = []; sc.mask = sc.frames.map(() => 0); sc.recovery = []; }
  teacher15 = teacher.scenes.slice(0, 15);
  out.teacher = {dataset: teacher.name, all: groupMetrics(teacher.scenes), first15: groupMetrics(teacher15),
    note: 'RC加速教師で記録した 60 シーン(ゲート 0・±3 cm・±3° の開始)。VM1 は今日の物差し(7.6 µs)と同じ式。VM7 は 0〜14 番の 15 シーン(9,000 フレーム)'};
}
for (const [key, ds] of Object.entries(groups)) {
  const g = groupMetrics(ds.scenes, {exactMin: exact[key]});
  if (key === 'recover') {
    const rec = ds.scenes.flatMap(sc => sc.recovery);
    g.recover = {events: rec.length, events_per_scene: dist(ds.scenes.map(sc => sc.events.length)), masked_fraction: dist(ds.scenes.map(sc => sc.mask.reduce((a, b) => a + b, 0) / sc.mask.length)),
      recovered: rec.filter(r => r.recovered_s != null).length, success: rec.filter(r => r.success).length,
      success_rate: rec.length ? rec.filter(r => r.success).length / rec.length : null,
      recovery_s_success: dist(rec.filter(r => r.success).map(r => r.recovered_s)), perturb_frames: dist(rec.map(r => r.frames)),
      nominal_line: nl ? {policy_sha256: nl.policy_sha256, runs: nl.runs} : null};
  }
  out.modes[key] = g;
}
// 合わせた群: モードごとに同じシーン数(最小のモードに合わせる、先頭から)。VM7 はモードごとに同じフレーム数で合計 9,000
const keys = Object.keys(groups);
if (keys.length) {
  const nMin = Math.min(...keys.map(k => groups[k].scenes.length));
  const pooled = keys.flatMap(k => groups[k].scenes.slice(0, nMin));
  const pm = groupMetrics(pooled);
  pm.VM7_equal_frames = vm7Equal(keys.map(k => groups[k].scenes));
  pm.scenes_per_mode = nMin;
  out.pooled = pm;
  // 2 つずつ・1 つ抜いた群(参考)
  out.pooled_leave_one_out = Object.fromEntries(keys.map(k => {
    const ks = keys.filter(x => x !== k);
    const sc = ks.flatMap(x => groups[x].scenes.slice(0, nMin));
    return [k, {VM3_m: VM3(sc).mean_abs_dev, VM7_equal_frames: vm7Equal(ks.map(x => groups[x].scenes)).cells}];
  }));
}
// 判定(protocol variation_metrics.per_mode_judged・headline_judged)
const T = out.teacher;
if (T) {
  const tVM3 = T.all.VM3_m.mean_abs_dev, tVM7 = T.first15.VM7.cells;
  out.judged.reference = {teacher_VM1_us: T.all.VM1_us, teacher_VM3_m: tVM3, teacher_VM7_cells_15: tVM7,
    note: 'protocol の基準は Node で作り直した REC60(VM1 7.58・VM3 0.0039)。ここでは記録した前の教師のデータそのものから同じコードで計算した値を基準にする'};
  if (out.modes.recover) {
    const r = out.modes.recover;
    const ev = r.recover.events_per_scene.mean, mf = r.recover.masked_fraction.mean, ratio = r.VM7.cells / tVM7;
    out.judged.recover = {events_per_scene_mean: ev, masked_fraction_mean: mf, vm7_ratio: ratio,
      pass: ev >= 3.5 && mf >= .10 && mf <= .40 && ratio >= 2.0, rule: '出来事 ≥ 3.5/シーン・外す割合 0.10〜0.40・VM7 ≥ 教師の 2 倍'};
  }
  if (out.modes.speed) {
    const s = out.modes.speed;
    out.judged.speed = {vm4_mps: s.VM4.mean_abs_dev, lap1_range_s: s.VM4.lap1.range, pass: s.VM4.mean_abs_dev >= .4 && s.VM4.lap1.range >= 3.0,
      rule: 'VM4 ≥ 0.4 m/s・1 周目の時間の幅 ≥ 3 秒'};
  }
  for (const k of ['fastest', 'no_wall']) if (out.modes[k]) out.judged[k] = {report_only: true, VM3_m: out.modes[k].VM3_m.mean_abs_dev, vm7_ratio: out.modes[k].VM7.cells / tVM7};
  if (out.pooled) {
    const vm3 = out.pooled.VM3_m.mean_abs_dev, vm7 = out.pooled.VM7_equal_frames.cells;
    out.judged.headline = {modes_recorded: keys.length, judged: keys.length >= 3, vm3_combined_m: vm3, vm3_ratio: vm3 / tVM3, vm3_min_m: .039,
      vm7_cells: vm7, vm7_ratio: vm7 / tVM7, vm7_ratio_min: 3.0, pass_vm3: vm3 >= .039, pass_vm7: vm7 / tVM7 >= 3.0,
      pass: keys.length >= 3 && vm3 >= .039 && vm7 / tVM7 >= 3.0};
  }
}
out.wall_s = (Date.now() - t0) / 1000;
fs.writeFileSync(opt('--out'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.judged, null, 1));
// 図のためのデータ(軌跡・速さ・すき間・e_y の s ごとの形)
if (opt('--figdata')) {
  const fig = {course: {segments: BASE_WORLD.segments.map(s => [s.a, s.b, s.kind]), disks: BASE_WORLD.disks.filter(d => d.kind !== 'post-foot').map(d => [d.x, d.z, d.radius, d.kind]),
    surfaces: BASE_WORLD.surfaces, route: BASE_WORLD.route, bounds: BASE_WORLD.bounds}, groups: {}};
  const pack = (scenes) => scenes.map(sc => ({scene: sc.scene, start: sc.start, scenario: sc.scenario, params: sc.params, laps: sc.laps,
    x: sc.frames.map(q => +q.x.toFixed(4)), z: sc.frames.map(q => +q.z.toFixed(4)), v: sc.frames.map(q => +q.v.toFixed(4)),
    d: sc.frames.map(q => +q.dmin.toFixed(4)), s: sc.frames.map(q => +q.s.toFixed(4)), ey: sc.frames.map(q => +q.ey.toFixed(4)),
    l: sc.frames.map(q => q.l), s0: sc.s0}));
  if (teacher) fig.groups.teacher = pack(teacher.scenes);
  for (const [k, ds] of Object.entries(groups)) fig.groups[k] = pack(ds.scenes);
  fig.loop_length = L;
  fs.writeFileSync(opt('--figdata'), JSON.stringify(fig));
}
