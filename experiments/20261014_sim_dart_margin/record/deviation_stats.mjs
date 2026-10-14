#!/usr/bin/env node
// 記録したデータセットの「ずれ」の統計(記録の段、2026-10-14、sim_dart_20261014)。データは変えない(読むだけ)。
//
//   node deviation_stats.mjs --dataset-dir <data/datasets/<名前>> --teacher fast|margin [--work <付け替えの途中のファイル>]
//        [--examples 2] [--out 結果.json]
//
// シーンごとに:
//  1) 記録の操作を元の sim-core.js で再生し(状態は記録と ≤ 1e-9。relabel_scene.mjs と同じ)、乱しなしの先生(同じ先生・同じ開始・
//     同じ物理)も Node で 20 秒走らせる。どちらも先生の進みの追跡で R 上の s と横ずれ e_y を出す(dart_driver.mjs の log)
//  2) ずれ = |e_y(実際) − e_y(乱しなしの走り、同じ s)|(predict_dart.mjs・measure_teachers.mjs の deviationFromClean と同じ)。
//     符号つきのずれも残す(左 +)。乱しごとの最大のずれ = 乱しの始まり〜終わり + 45 フレーム(1.5 秒)の最大(予想と同じ)
//  3) 先生の正解(付け替えのあと)がどう戻すか: 付け替えの途中のファイル(relabel_scene.mjs の出力)の labels[t][0](= フレーム t+1 の
//     先生の舵 â_{t+1})と、記録の舵 ã_{t+1}、乱しなしの走りの同じ s での舵 δ_clean(s) を比べる。
//     Δ正解 = â_{t+1} − δ_clean(s)、Δ記録 = ã_{t+1} − δ_clean(s) を、符号つきのずれ e_{t+1} に対して回帰する(戻す向きなら傾きが負または正で一定)
//  4) 乱しの終わりから、ずれが 2 cm 未満に戻るまでのフレーム数
// --examples n: 最初の n シーンについて、フレームごとの系列(時刻・ずれ・舵 3 種・乱しの印・位置)を出力に入れる(図に使う)
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {baseWorld, activeWorldFor, createSimulation, SIM_ROOT} from '../lib/app_emulation.mjs';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i < 0 ? def : args[i + 1]; };
const dsDir = opt('--dataset-dir'), kind = opt('--teacher'), workDir = opt('--work', ''), outPath = opt('--out', null);
const nExamples = Number(opt('--examples', 2));
const exampleNames = (opt('--example-names', '') || '').split(',').filter(Boolean);
if (!dsDir || !['fast', 'margin'].includes(kind)) { console.error('使い方: node deviation_stats.mjs --dataset-dir <dir> --teacher fast|margin'); process.exit(2); }
const DART = path.join(SIM_ROOT, 'dist/dart');
const imp = p => import(pathToFileURL(path.join(DART, p)).href);
const {createDartDriver} = await imp('dart_driver.mjs');
const {createTeacherCore, previousAction} = await imp('teacher_core.mjs');
const W = (await imp(kind === 'fast' ? 'rt/weights_fast.mjs' : 'rt/weights_margin.mjs')).default;
const rc = await import(pathToFileURL(path.join(SIM_ROOT, 'dist/rc-speed-controller.js')).href);
const base = await baseWorld();

function interpByS(cs, cv, s) {
  if (s < cs[0] || s > cs.at(-1)) return null;
  let lo = 0, hi = cs.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cs[m] <= s) lo = m; else hi = m; }
  const fr = Math.min(1, Math.max(0, (s - cs[lo]) / Math.max(1e-9, cs[hi] - cs[lo])));
  return cv[lo] + (cv[hi] - cv[lo]) * fr;
}
const pct = (a, p) => { const b = a.filter(x => x != null && Number.isFinite(x)).sort((x, y) => x - y); return b.length ? b[Math.min(b.length - 1, Math.floor(p * b.length))] : null; };
const mean = a => { const b = a.filter(x => x != null && Number.isFinite(x)); return b.length ? b.reduce((p, q) => p + q, 0) / b.length : null; };
function linfit(x, y) {
  const n = x.length; if (n < 3) return {n, slope: null, r: null};
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return {n, slope: sxy / sxx, intercept: my - (sxy / sxx) * mx, r: sxy / Math.sqrt(sxx * syy)};
}

// 状態 b(記録の再生)から先生だけで n フレーム走る(relabel_scene.mjs の forkFrom と同じ分かれ方: 経路を今の目標ゲートで回した world の単周)
function forkRun(fc, b, prev0, n, report, variation, NG) {
  const k0 = b.metrics.route_index % NG;
  const {world: wf} = activeWorldFor(JSON.parse(JSON.stringify(base)), (variation.gate + k0) % NG, 0, 0);
  const cfg = {...report.config, timeLimit: 600};
  delete cfg.episodeMode;
  const fsim = createSimulation(wf, cfg);
  const s0 = b.state;
  const snap0 = fsim.reset({seed: report.seed, scenario: report.scenario, initialState: {x: s0.x, z: s0.z, yaw: s0.yaw, speed: s0.speed, steering: s0.steering, routeIndex: 0}});
  const out = {s: [], ey: [], steer: [], why: null};
  if (snap0.state.terminal) { out.why = 'invalid_spawn'; return out; }
  let p = [...prev0];
  for (let j = 0; j < n; j++) {
    const sn = fsim.snapshot();
    const st = j === 0 ? {x: s0.x, z: s0.z, yaw: s0.yaw, speed: s0.speed, steering: s0.steering, frame: s0.frame}
      : {x: sn.state.x, z: sn.state.z, yaw: sn.state.yaw, speed: sn.state.speed, steering: sn.state.steering, frame: sn.state.frame};
    if (j > 0) fc.advance(st, (k0 + sn.metrics.route_index) % NG);
    const d = fc.decide(st, p[0], p[1]);
    out.s.push(d.s); out.ey.push(d.ey); out.steer.push(d.steering);
    p = [d.a0, d.a1];
    fsim.step({steering: d.steering, speed: d.speed});
    const a = fsim.snapshot();
    if (a.metrics.collisions > 0) { out.why = 'collision'; break; }
    if (a.state.terminal) { out.why = 'terminal:' + a.state.reason; break; }
  }
  return out;
}

function sceneStats(sceneDir, wantExample) {
  const read = p => JSON.parse(fs.readFileSync(path.join(sceneDir, p), 'utf8'));
  const session = read('session.json'), report = read('evaluation-report.json');
  const rows = fs.readFileSync(path.join(sceneDir, 'frames.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const N = rows.length, variation = session.context.start_variation;
  const {world} = activeWorldFor(JSON.parse(JSON.stringify(base)), variation.gate, variation.left_offset_m, variation.left_yaw_deg);
  world.spawn = {...rows[0].pose_pre_action};
  // 1) 実際: 記録の操作の再生(状態の確かめつき)
  const sim = createSimulation(world, report.config);
  sim.reset({seed: report.seed, scenario: report.scenario});
  const actual = rc.assertRcTeacherContract(world, sim.snapshot().config, {gate: variation.gate, left_offset_m: variation.left_offset_m, left_yaw_deg: variation.left_yaw_deg});
  // 実際の走りに沿って先生を進める(relabel_scene.mjs と同じ: 「前回の行動」は実際に適用した操作。乱していない成分は先生自身の値)
  const coreA = createTeacherCore(kind, world, actual, W);
  const LA = [];
  // 乱し(記録の info.dart)。乱しの始まりで、先生だけで走り続けたときの走り(分かれ)を基準に作る
  const info = rows.map(r => r.action_info?.dart ?? null);
  const noisy = info.map(x => !!x?.noise);
  const events = [];
  for (let f = 0; f < N; f++) {
    if (!noisy[f]) continue;
    const ev = info[f].ev;
    let e = events.at(-1);
    if (!e || e.ev !== ev || e.end !== f) { e = {ev, start: f, end: f, dSteer: info[f].noise[0], speedFactor: info[f].noise[1]}; events.push(e); }
    e.end = f + 1;
  }
  const startAt = new Map(events.map(e => [e.start, e]));
  const NG = world.route.length;
  let err = 0, prev = [0, 0], cleanVsInfo = 0;
  for (let i = 0; i < N; i++) {
    const b = sim.snapshot();
    for (const k of ['x', 'z', 'yaw']) err = Math.max(err, Math.abs(b.state[k] - rows[i].pose_pre_action[k]));
    const st = {x: b.state.x, z: b.state.z, yaw: b.state.yaw, speed: b.state.speed, steering: b.state.steering, frame: b.state.frame};
    if (i === 0) coreA.begin(st, b.metrics.route_index); else coreA.advance(st, b.metrics.route_index);
    const d = coreA.decide(st, prev[0], prev[1]);
    if (info[i]) cleanVsInfo = Math.max(cleanVsInfo, Math.abs(d.steering - info[i].clean[0]), Math.abs(d.speed - info[i].clean[1]));
    LA.push({f: i, s: d.s, ey: d.ey, dmin: d.dmin, clean_steering: d.steering});
    const e = startAt.get(i);
    if (e) e._fork = forkRun(coreA.fork(), b, prev, Math.min(N, e.end + 45) - e.start + 30, report, variation, NG);
    const ap = report.actions[i];
    prev = previousAction(d, ap, noisy[i], noisy[i] && info[i].noise[1] !== 1, coreA.speedCap, coreA.maxSteer);
    sim.step(ap);
  }
  if (err > 1e-9) throw new Error(`${sceneDir}: 再生が記録と合いません(${err})`);
  // 2) 乱しなしの先生(同じ開始・同じ物理)
  const sim2 = createSimulation(world, report.config);
  sim2.reset({seed: report.seed, scenario: report.scenario});
  const actual2 = rc.assertRcTeacherContract(world, sim2.snapshot().config, {gate: variation.gate, left_offset_m: variation.left_offset_m, left_yaw_deg: variation.left_yaw_deg});
  const drvC = createDartDriver({kind, world, config: actual2, weights: W, noise: null});
  const cleanXZ = [];
  for (let i = 0; i < N; i++) {
    const b = sim2.snapshot(); if (b.state.terminal) break;
    cleanXZ.push([b.state.x, b.state.z]);
    sim2.step(drvC.action(b));
  }
  const LC = drvC.log, cs = LC.map(q => q.s), ce = LC.map(q => q.ey), cst = LC.map(q => q.clean_steering);
  const signed = LA.map(q => { const e = interpByS(cs, ce, q.s); return e == null ? null : q.ey - e; });
  const dev = signed.map(x => x == null ? null : Math.abs(x));
  const steerCleanAtS = LA.map(q => interpByS(cs, cst, q.s));
  // 乱しごとのずれ。(a) protocol の決まり: 乱しなしの走り(開始から)との差。(b) 乱しの始まりから先生だけで走り続けた分かれとの差
  // ((b) はブラウザと Node の最後の桁の差・前の乱しの影響を含まない。乱しが車をどれだけ動かしたかそのもの)
  let work = null;
  if (workDir) {
    const p = path.join(workDir, path.basename(sceneDir) + '.json');
    if (fs.existsSync(p)) work = JSON.parse(fs.readFileSync(p, 'utf8'));
  }
  const forkSigned = new Array(N).fill(null), forkSteer = new Array(N).fill(null);
  const eta = [];   // 乱しの始まりにそろえた系列(乱しの向きで符号をそろえる): [k, ずれ, 記録の舵 − 分かれの舵, 正解の舵 − 分かれの舵, 乱し中か]
  for (const e of events) {
    let m = 0, mf = null;
    for (let f = e.start; f < Math.min(N, e.end + 45); f++) if (dev[f] != null && dev[f] > m) { m = dev[f]; mf = f; }
    e.max_dev_m = m; e.max_dev_frame = mf;
    let back = null;
    for (let f = e.end; f < Math.min(N, e.end + 90); f++) if (dev[f] != null && dev[f] < .02) { back = f - e.end; break; }
    e.frames_back_below_2cm = back;
    const fk = e._fork; delete e._fork;
    if (!fk || fk.s.length < 2) { e.fork_max_dev_m = null; continue; }
    const nextStart = events.find(q => q.start > e.start)?.start ?? N;
    const sg = Math.sign(e.dSteer) || 1;
    for (let f = e.start; f < Math.min(N, e.start + 76, nextStart); f++) {
      const ey = interpByS(fk.s, fk.ey, LA[f].s), sf = interpByS(fk.s, fk.steer, LA[f].s);
      if (ey == null || sf == null) break;
      const lab = work?.labels?.[f - 1]?.[0];
      const ok = work ? work.valid_teacher?.[f - 1] : null;
      eta.push([f - e.start, sg * (LA[f].ey - ey), sg * (report.actions[f].steering - sf), lab != null && ok ? sg * (lab - sf) : null, noisy[f] ? 1 : 0]);
    }
    let m2 = 0, mf2 = null, back2 = null;
    for (let f = e.start; f < Math.min(N, e.end + 45); f++) {
      const ey = interpByS(fk.s, fk.ey, LA[f].s);
      if (ey == null) continue;
      const sd = LA[f].ey - ey;
      if (forkSigned[f] == null) { forkSigned[f] = sd; forkSteer[f] = interpByS(fk.s, fk.steer, LA[f].s); }
      if (Math.abs(sd) > m2) { m2 = Math.abs(sd); mf2 = f; }
      if (f >= e.end && back2 == null && Math.abs(sd) < .02 && mf2 != null && mf2 < f) back2 = f - e.end;
    }
    e.fork_max_dev_m = m2; e.fork_max_dev_frame = mf2; e.fork_frames_back_below_2cm = back2; e.fork_end = fk.why;
  }
  // 3) 正解の戻し方(work は上で読んだ)
  const fitX = [], fitLab = [], fitApp = [], perFrame = [];
  let noisyNext = 0, labOpposesNoise = 0, labMinusApplied = [];
  for (let t = 0; t < N - 1; t++) {
    const lab = work?.labels?.[t]?.[0];
    const ok = work?.valid_teacher?.[t];
    const e1 = forkSigned[t + 1], sc = forkSteer[t + 1], app = report.actions[t + 1].steering;
    if (lab == null || !ok || e1 == null || sc == null) continue;
    if (Math.abs(e1) >= .01) { fitX.push(e1); fitLab.push(lab - sc); fitApp.push(app - sc); }
    if (noisy[t + 1]) {
      noisyNext++;
      labMinusApplied.push(lab - app);
      if (Math.sign(lab - app) === -Math.sign(info[t + 1].noise[0])) labOpposesNoise++;
    }
  }
  const fs0 = LA.filter(q => q.f < 60 && variation.gate === 0).map(q => dev[q.f]).filter(x => x != null);
  const out = {scene: path.basename(sceneDir), name: session.display_name, seed: report.seed, k: info[0]?.p?.k ?? null, frames: N, replay_err: err,
    teacher_clean_vs_info_max: info.some(Boolean) ? cleanVsInfo : null,
    noisy_frames: noisy.filter(Boolean).length, n_events: events.length, events,
    dev_p50: pct(dev, .5), dev_p90: pct(dev, .9), dev_max: pct(dev, 1), first_straight_dev_max: fs0.length ? Math.max(...fs0) : null,
    frac_dev_ge_5cm: dev.filter(x => x != null && x >= .05).length / N, frac_dev_ge_10cm: dev.filter(x => x != null && x >= .10).length / N,
    min_dmin_actual: Math.min(...LA.map(q => q.dmin)), min_dmin_clean: Math.min(...LC.map(q => q.dmin)),
    clean_frames: LC.length, labels_available: !!work,
    steer_back: work ? {fit_label_vs_dev: linfit(fitX, fitLab), fit_applied_vs_dev: linfit(fitX, fitApp), noisy_next_frames: noisyNext,
      label_opposes_noise_frac: noisyNext ? labOpposesNoise / noisyNext : null, label_minus_applied_noisy_mean_abs: mean(labMinusApplied.map(Math.abs))} : null,
    _eta: eta, _dmin: LA.map(q => +q.dmin.toFixed(5)),
    first_straight_min_dmin: variation.gate === 0 ? Math.min(...LA.filter(q => q.f < 60).map(q => q.dmin)) : null,
    _dev: dev.map(x => x == null ? null : +x.toFixed(5)), _fit: work ? {x: fitX.map(v => +v.toFixed(5)), lab: fitLab.map(v => +v.toFixed(5)), app: fitApp.map(v => +v.toFixed(5))} : null};
  if (wantExample) {
    out.example = {t: LA.map(q => q.f / 30), signed_dev: signed, fork_signed_dev: forkSigned, applied_steer: report.actions.map(a => a.steering), clean_steer_at_s: steerCleanAtS,
      teacher_clean_at_state: info.map(x => x?.clean?.[0] ?? null),
      label_steer_next: work ? work.labels.map(l => l?.[0] ?? null) : null, valid: work ? work.valid_teacher : null, noisy,
      xz: rows.map(r => [r.pose_pre_action.x, r.pose_pre_action.z]), clean_xz: cleanXZ, s: LA.map(q => q.s)};
  }
  return out;
}

// --dataset-dir には、取り込んだデータセット(simulation/scenes/<id>)か、シーンのフォルダー(session.json があるもの)を , で並べたもの(試し用)
const direct = dsDir.includes(',') || fs.existsSync(path.join(dsDir, 'session.json'));
const scenesRoot = direct ? '' : path.join(dsDir, 'simulation', 'scenes');
const splits = direct ? {train: dsDir.split(',')} : JSON.parse(fs.readFileSync(path.join(dsDir, 'splits.json'), 'utf8'));
const maxScenes = Number(opt('--max-scenes', 0));
let ids = [...(splits.train ?? []), ...(splits.val ?? [])].filter(s => fs.existsSync(path.join(scenesRoot, s)));
if (maxScenes > 0) ids = ids.slice(0, maxScenes);
const all = [];
for (let i = 0; i < ids.length; i++) {
  const nm = JSON.parse(fs.readFileSync(path.join(scenesRoot, ids[i], 'session.json'), 'utf8')).display_name;
  const want = exampleNames.length ? exampleNames.includes(nm) : i < nExamples;
  all.push(sceneStats(path.join(scenesRoot, ids[i]), want));
  process.stderr.write(`[ずれ] ${i + 1}/${ids.length} ${all.at(-1).name} 乱し ${all.at(-1).n_events} 回 ずれ p50 ${(100 * (all.at(-1).dev_p50 ?? 0)).toFixed(1)} cm 最大 ${(100 * (all.at(-1).dev_max ?? 0)).toFixed(1)} cm\n`);
}
const evMax = all.flatMap(s => s.events.map(e => e.max_dev_m));
const devAll = all.flatMap(s => s._dev);
const dminAll = all.flatMap(s => s._dmin);
const back = all.flatMap(s => s.events.map(e => e.frames_back_below_2cm));
const evFork = all.flatMap(s => s.events.map(e => e.fork_max_dev_m)).filter(x => x != null);
const backFork = all.flatMap(s => s.events.map(e => e.fork_frames_back_below_2cm));
const fitAll = {x: all.flatMap(s => s._fit?.x ?? []), lab: all.flatMap(s => s._fit?.lab ?? []), app: all.flatMap(s => s._fit?.app ?? [])};
const summary = {dataset_dir: dsDir, teacher: kind, scenes: all.length, frames: all.reduce((a, s) => a + s.frames, 0),
  noisy_frames: all.reduce((a, s) => a + s.noisy_frames, 0), events: evMax.length,
  ev_max_p50_m: pct(evMax, .5), ev_max_p90_m: pct(evMax, .9), ev_max_max_m: pct(evMax, 1), ev_frac_5_25cm: evMax.length ? evMax.filter(x => x >= .05 && x <= .25).length / evMax.length : null,
  frame_dev_p50_m: pct(devAll, .5), frame_dev_p90_m: pct(devAll, .9), frame_dev_p99_m: pct(devAll, .99), frame_dev_max_m: pct(devAll, 1),
  fork_ev_max_p50_m: pct(evFork, .5), fork_ev_max_p90_m: pct(evFork, .9), fork_ev_max_max_m: pct(evFork, 1),
  fork_ev_frac_5_25cm: evFork.length ? evFork.filter(x => x >= .05 && x <= .25).length / evFork.length : null,
  fork_ev_frac_ge_10cm: evFork.length ? evFork.filter(x => x >= .10).length / evFork.length : null,
  fork_frames_back_below_2cm_p50: pct(backFork.filter(x => x != null), .5), fork_frames_back_below_2cm_p90: pct(backFork.filter(x => x != null), .9),
  fork_events_not_back_by_window_end: backFork.filter(x => x == null).length,
  teacher_clean_vs_info_max: Math.max(...all.map(s => s.teacher_clean_vs_info_max ?? 0)),
  dmin_p01_m: pct(dminAll, .01), dmin_p05_m: pct(dminAll, .05), dmin_p50_m: pct(dminAll, .5),
  frac_frames_dmin_lt_10cm: dminAll.filter(x => x < .10).length / Math.max(1, dminAll.length),
  frac_frames_dmin_lt_5cm: dminAll.filter(x => x < .05).length / Math.max(1, dminAll.length),
  first_straight_min_dmin_p50_m: pct(all.map(s => s.first_straight_min_dmin), .5), first_straight_min_dmin_min_m: pct(all.map(s => s.first_straight_min_dmin), 0),
  frac_frames_dev_ge_5cm: devAll.filter(x => x != null && x >= .05).length / Math.max(1, devAll.length),
  frac_frames_dev_ge_10cm: devAll.filter(x => x != null && x >= .10).length / Math.max(1, devAll.length),
  first_straight_dev_max_p50_m: pct(all.map(s => s.first_straight_dev_max), .5), first_straight_dev_max_max_m: pct(all.map(s => s.first_straight_dev_max), 1),
  frames_back_below_2cm_p50: pct(back.filter(x => x != null), .5), frames_back_below_2cm_p90: pct(back.filter(x => x != null), .9),
  events_not_back_within_3s: back.filter(x => x == null).length,
  min_dmin_actual_m: Math.min(...all.map(s => s.min_dmin_actual)), replay_err_max: Math.max(...all.map(s => s.replay_err)),
  steer_back: fitAll.x.length ? {fit_label_vs_dev: linfit(fitAll.x, fitAll.lab), fit_applied_vs_dev: linfit(fitAll.x, fitAll.app),
    label_opposes_noise_frac: (() => { const n = all.reduce((a, s) => a + (s.steer_back?.noisy_next_frames ?? 0), 0);
      return n ? all.reduce((a, s) => a + (s.steer_back?.label_opposes_noise_frac ?? 0) * (s.steer_back?.noisy_next_frames ?? 0), 0) / n : null; })(),
    label_minus_applied_noisy_mean_abs_rad: mean(all.filter(s => s.steer_back?.noisy_next_frames).map(s => s.steer_back.label_minus_applied_noisy_mean_abs))} : null};
// 乱しの始まりにそろえた平均(k = 0..75 フレーム)。符号は乱しの向きでそろえる(+ = 乱しが押した向き)
const etaAll = all.flatMap(s => s._eta ?? []);
const etaAgg = [];
for (let k = 0; k <= 75; k++) {
  const r = etaAll.filter(q => q[0] === k);
  if (!r.length) continue;
  const labs = r.map(q => q[3]).filter(x => x != null);
  etaAgg.push({k, n: r.length, dev_mean: mean(r.map(q => q[1])), dev_p25: pct(r.map(q => q[1]), .25), dev_p75: pct(r.map(q => q[1]), .75),
    applied_minus_fork_mean: mean(r.map(q => q[2])), label_minus_fork_mean: labs.length ? mean(labs) : null, noisy_frac: mean(r.map(q => q[4]))});
}
summary.event_aligned_peak = etaAgg.length ? (() => { const m = etaAgg.reduce((a, q) => q.dev_mean > a.dev_mean ? q : a); return {k: m.k, dev_mean_m: m.dev_mean}; })() : null;
const out = {schema: 'jf-sim-dart-deviation/v1', made: new Date().toISOString(), summary, event_aligned: etaAgg,
  hist: {ev_max_m: evMax.map(x => +x.toFixed(5)), fork_ev_max_m: evFork.map(x => +x.toFixed(5)), frame_dev_m: devAll, frame_dmin_m: dminAll}, fit_points: fitAll, scenes: all.map(s => { const o = {...s}; delete o._dev; delete o._fit; delete o._eta; delete o._dmin; return o; })};
if (outPath) fs.writeFileSync(outPath, JSON.stringify(out));
console.log(JSON.stringify(summary, null, 1));
