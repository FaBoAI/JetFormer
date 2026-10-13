#!/usr/bin/env node
// 選んだ RL の方策の周回を詳しく評価する(2026-10-13、RL の学習の段。design.md 2.10・3 章 / protocol.json selection・best_lap_definition・realism_metrics)
//
//   node rl/final_eval.mjs --policy-json <方策.json> [--variant main|open] --out <結果.json> [--records <フォルダー>] [--teacher] [--learned-lap]
//
// すること(すべて元の sim-core.js。--core fast で高速版):
//   1. REC60・STD・ROB24 を元のコアでやり直し(evaluateController)、同じ方策の高速版の結果と周回時間を突き合わせる(差 ≤ 1e-9 秒)
//   2. 閉ループの開始 E0〜E5 の停止から 1 周(単周、15 秒の制限)と、E0・E1 の連続 20 秒(フライングラップ)
//   3. 参考: 遅延 1 フレーム(アプリの固定遅延)での STD の単周
//   4. 各走りの指標: 周回時間・ゲート・黒シートの横断・衝突・最小すき間・最高速・横加速度・舵の速さ・舵が端の割合・スロットル PWM の 1% 点
//   5. REC60 の全フレームをまとめた現実味の指標(protocol realism_metrics)
//   6. 周の独立な確認(rl/lap_audit.mjs の verifyReport): STD の単周・連続、E0〜E5 の単周
//   --teacher: 同じ指標を RC 教師で(E0〜E2 だけ。教師の約束の範囲)。--learned-lap: 学習済み ONNX の 8.964 秒の周(保存された操作列の再生)
import fs from 'node:fs';
import path from 'node:path';
import {loadRec60, STD, ROB24, CLOSED_LOOP_STARTS, runEpisode, policyController, teacherController, replayController, evaluateController,
  summarizeRec60} from './eval_sets.mjs';
import {verifyReport, loadReport} from './lap_audit.mjs';
import {loadPolicy} from './policy_forward.mjs';
import {PATHS, sha256File, loadBaseWorld} from './env_core.mjs';

const args = process.argv.slice(2);
const opt = (name, fallback = null) => { const i = args.indexOf(name); if (i < 0) return fallback; const v = args[i + 1]; args.splice(i, 2); return v; };
const flag = name => { const i = args.indexOf(name); if (i < 0) return false; args.splice(i, 1); return true; };
const policyJson = opt('--policy-json'), variant = opt('--variant', 'main'), out = opt('--out'), recordsDir = opt('--records');
const coreKind = opt('--core', 'original'), fastEvalPath = opt('--fast-eval');
const withTeacher = flag('--teacher'), withLearned = flag('--learned-lap'), skipRob = flag('--skip-selection-sets');
if (!out || (!policyJson && !withTeacher && !withLearned)) {
  console.error('使い方: node rl/final_eval.mjs --policy-json policy.json --out result.json [--variant main] [--records dir] [--teacher] [--learned-lap]');
  process.exit(2);
}
if (recordsDir) fs.mkdirSync(recordsDir, {recursive: true});
const MAX_STEER = .52, PWM_NEUTRAL = 1498, PWM_SPAN = 498, MAX_SPEED = 7.24, G = 9.80665;
const world = loadBaseWorld();
const MU = Object.fromEntries(world.surfaces.map(s => [s.id, s.mu]));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const quantile = (a, q) => { if (!a.length) return null; const b = Float64Array.from(a).sort(), p = (b.length - 1) * q, i = Math.floor(p), f = p - i;
  return i + 1 < b.length ? b[i] * (1 - f) + b[i + 1] * f : b[i]; };
const strip = r => { if (!r) return r; const {report, collision_events, ...rest} = r; return rest; };
const stripV = v => { const {collision_events_own, ...rest} = v; return rest; };

// ---- 1 本の走り(sim のレポート)から、フレームごとの値を取り出す ----
function frameSeries(report) {
  const tr = report.trace, acts = report.actions, cfg = report.config, dt = cfg.dt;
  const s = {speed: [], lat: [], rho: [], steerCmd: [], steerCmdRate: [], steerRate: [], throttlePwm: [], moving: []};
  let yawPrev = report.world.spawn.yaw, steerPrev = 0, cmdPrev = 0;
  for (let i = 0; i < tr.length; i++) {
    const t = tr[i], a = acts[i] ?? acts[acts.length - 1];
    const omega = wrap(t.yaw - yawPrev) / dt, mu = MU[t.surface] ?? .8;
    s.speed.push(t.speed);
    s.lat.push(Math.abs(t.speed * omega));
    s.rho.push(t.speed * t.speed * Math.abs(Math.tan(t.steering)) / (cfg.wheelbase * mu * G));
    s.steerCmd.push(a.steering);
    s.steerCmdRate.push(Math.abs(a.steering - cmdPrev) / dt);
    s.steerRate.push(Math.abs(t.steering - steerPrev) / dt);
    s.throttlePwm.push(PWM_NEUTRAL - (Math.max(0, a.speed) / MAX_SPEED) * PWM_SPAN);
    s.moving.push(t.speed > .3);
    yawPrev = t.yaw; steerPrev = t.steering; cmdPrev = a.steering;
  }
  return s;
}
function mergeSeries(list) {
  const m = {speed: [], lat: [], rho: [], steerCmd: [], steerCmdRate: [], steerRate: [], throttlePwm: [], moving: []};
  for (const s of list) for (const k of Object.keys(m)) for (const v of s[k]) m[k].push(v);
  return m;
}
// protocol realism_metrics の指標(走行中 = 0.3 m/s 超)
function motionMetrics(s, extra = {}) {
  const idx = s.moving.map((m, i) => m ? i : -1).filter(i => i >= 0), pick = k => idx.map(i => s[k][i]);
  const sp = pick('speed'), lat = pick('lat');
  return {frames: s.speed.length, moving_frames: idx.length,
    max_speed: Math.max(...s.speed), speed_median: quantile(sp, .5), speed_p99: quantile(sp, .99), frac_over_4mps: sp.filter(v => v > 4).length / Math.max(1, sp.length),
    lat_acc_max: Math.max(...s.lat), lat_acc_p95: quantile(lat, .95), lat_acc_median: quantile(lat, .5),
    rho_gt1_frac: pick('rho').filter(v => v > 1 + 1e-9).length / Math.max(1, idx.length),
    steer_cmd_rate_p95: quantile(pick('steerCmdRate'), .95), steer_cmd_rate_max: Math.max(...s.steerCmdRate),
    steer_rate_p95: quantile(pick('steerRate'), .95), steer_rate_max: Math.max(...s.steerRate),
    steer_edge_frac: pick('steerCmd').filter(v => Math.abs(v) >= .494).length / Math.max(1, idx.length),
    throttle_pwm_p01: quantile(pick('throttlePwm'), .01), ...extra};
}
function runRow(r) {
  const rep = r.report, s = frameSeries(rep);
  const crossings = rep.metrics.required_crossing_events?.length ?? null;
  return {name: r.name, start: r.start, reason: r.reason, sim_reason: r.sim_reason, sim_time: r.sim_time, lap_complete: r.lap_complete ?? null,
    valid_two_laps: r.valid_two_laps ?? null, laps: r.laps, completed_laps: r.completed_laps, passed_gates: r.passed_gates,
    collisions: r.collisions, crossings_total: crossings, min_clearance: r.min_clearance,
    sim_max_lateral_acceleration: rep.metrics.max_lateral_acceleration, traction_limited_steps: rep.metrics.traction_limited_steps,
    metrics: motionMetrics(s)};
}
function saveRecord(name, report) {
  if (!recordsDir) return null;
  const p = path.join(recordsDir, name + '.json');
  fs.writeFileSync(p, JSON.stringify(report));
  return p;
}
// 1 つの controller を E0〜E5(単周)・E0/E1(連続)・STD 遅延 1 で走らせる
async function closedLoop(controller, label, {starts = CLOSED_LOOP_STARTS, continuousStarts = ['E0', 'E1'], delay1 = true, verify = true} = {}) {
  const res = {single: [], continuous: []};
  for (const c of starts) {
    const r = await runEpisode({start: c, controller, coreKind, variant, episodeMode: 'single', timeLimit: 15, record: true});
    const row = runRow(r);
    row.single_lap_s = r.lap_complete ? r.laps[0].elapsed : null;
    row.record = saveRecord(`${label}_${c.name}_single`, r.report);
    if (verify) row.verification = stripV(await verifyReport(r.report, {minClearance: 0}));
    res.single.push(row);
  }
  for (const name of continuousStarts) {
    const c = starts.find(s => s.name === name) ?? CLOSED_LOOP_STARTS.find(s => s.name === name);
    const r = await runEpisode({start: c, controller, coreKind, variant, episodeMode: 'continuous', record: true});
    const row = runRow(r);
    row.flying_lap_s = r.valid_two_laps ? r.laps[1].elapsed : null;
    row.record = saveRecord(`${label}_${c.name}_continuous`, r.report);
    if (verify) row.verification = stripV(await verifyReport(r.report, {minLaps: 2, minClearance: 0}));
    res.continuous.push(row);
  }
  if (delay1) {
    const r = await runEpisode({start: STD, controller, coreKind, variant, episodeMode: 'single', timeLimit: 15, record: true, delayFrames: 1});
    res.STD_delay1_reference = {...runRow(r), single_lap_s: r.lap_complete ? r.laps[0].elapsed : null,
      collision_event: r.report.metrics.collision_events[0] ?? null};
  }
  return res;
}
async function rec60Realism(controller, label) {
  const series = [], rows = [];
  for (const c of loadRec60()) {
    const r = await runEpisode({start: c, controller, coreKind, variant, record: true});
    series.push(frameSeries(r.report)); rows.push(strip(r));
  }
  return {summary: summarizeRec60(rows), realism: motionMetrics(mergeSeries(series))};
}

const t0 = performance.now();
const result = {schema: 'rl-bestlap-final-eval/v1', created_at: new Date().toISOString(), core: coreKind, variant,
  original_core_sha256: sha256File(PATHS.originalCore)};

if (policyJson) {
  const fn = await loadPolicy(policyJson);
  const controller = policyController(fn, path.basename(policyJson));
  result.policy = {path: policyJson, sha256: sha256File(policyJson), meta: fn.meta};
  // 1. REC60・STD・ROB24 を元のコアで(選ぶ条件と点数)
  if (!skipRob) {
    const ev = await evaluateController(controller, {coreKind, variant});
    result.selection_sets = {eligible: ev.eligible, score: ev.score, REC60: ev.REC60.summary, STD_single_lap_from_stop: ev.STD.single_lap_from_stop,
      STD_flying_lap: ev.STD.flying_lap, ROB24: ev.ROB24.summary, REC60_scenes: ev.REC60.scenes.map(r => ({name: r.name, laps: r.laps.map(l => l.elapsed),
        collisions: r.collisions, reason: r.reason, min_clearance: r.min_clearance, valid_two_laps: r.valid_two_laps}))};
    if (fastEvalPath) {
      const fast = JSON.parse(fs.readFileSync(fastEvalPath, 'utf8'));
      let maxDiff = 0, mismatch = [];
      fast.REC60.scenes.forEach((s, i) => {
        const o = result.selection_sets.REC60_scenes[i];
        if (s.laps.length !== o.laps.length || s.reason !== o.reason) mismatch.push(s.name);
        s.laps.forEach((l, k) => { if (o.laps[k] !== undefined) maxDiff = Math.max(maxDiff, Math.abs(l.elapsed - o.laps[k])); });
      });
      const stdDiff = Math.abs((fast.STD.single_lap_from_stop ?? NaN) - (ev.STD.single_lap_from_stop ?? NaN));
      result.fast_vs_original = {fast_eval: fastEvalPath, rec60_max_lap_diff_s: maxDiff, rec60_mismatched_scenes: mismatch,
        std_single_diff_s: stdDiff, fast_score: fast.score, original_score: ev.score, fast_eligible: fast.eligible, original_eligible: ev.eligible,
        rob24_fast: fast.ROB24.summary.valid_two_laps, rob24_original: ev.ROB24.summary.valid_two_laps,
        within_1e9: maxDiff <= 1e-9 && mismatch.length === 0 && (stdDiff <= 1e-9 || Number.isNaN(stdDiff))};
    }
  }
  // 2〜3. 閉ループの開始・遅延 1
  result.rl = await closedLoop(controller, `rl_${variant}`);
  // STD の単周・連続(記録と確認)
  const single = await runEpisode({start: STD, controller, coreKind, variant, episodeMode: 'single', record: true});
  const cont = await runEpisode({start: STD, controller, coreKind, variant, episodeMode: 'continuous', record: true});
  result.rl.STD = {single: {...runRow(single), single_lap_s: single.lap_complete ? single.laps[0].elapsed : null, record: saveRecord(`rl_${variant}_STD_single`, single.report),
    verification: stripV(await verifyReport(single.report, {minClearance: .03}))},
  continuous: {...runRow(cont), flying_lap_s: cont.valid_two_laps ? cont.laps[1].elapsed : null, record: saveRecord(`rl_${variant}_STD_continuous`, cont.report),
    verification: stripV(await verifyReport(cont.report, {minLaps: 2, minClearance: .03}))}};
  // 5. 現実味(REC60 の全フレーム)
  result.rl.REC60_realism = await rec60Realism(controller, 'rl');
}
if (withTeacher) {
  const teacher = await teacherController();
  result.teacher = await closedLoop(teacher, 'teacher', {starts: CLOSED_LOOP_STARTS.slice(0, 3), delay1: true});
  const single = await runEpisode({start: STD, controller: teacher, coreKind, episodeMode: 'single', record: true});
  const cont = await runEpisode({start: STD, controller: teacher, coreKind, episodeMode: 'continuous', record: true});
  result.teacher.STD = {single: {...runRow(single), single_lap_s: single.lap_complete ? single.laps[0].elapsed : null, record: saveRecord('teacher_STD_single', single.report)},
    continuous: {...runRow(cont), flying_lap_s: cont.valid_two_laps ? cont.laps[1].elapsed : null, record: saveRecord('teacher_STD_continuous', cont.report)}};
  result.teacher.REC60_realism = await rec60Realism(teacher, 'teacher');
}
if (withLearned) {
  // 作者の学習済み ONNX(shortcut-rc-finetuned)の 8.964 秒の周。保存された操作列(遅延 1 を含む適用済みの値)をそのまま流す
  const rep = loadReport(PATHS.learnedLap);
  const v = await verifyReport(rep, {minClearance: 0});
  const r = await runEpisode({start: STD, controller: replayController(rep.actions, 'learned-onnx-replay'), coreKind, episodeMode: 'single', record: true});
  result.learned_onnx = {source: PATHS.learnedLap, sha256: sha256File(PATHS.learnedLap), ...runRow(r), single_lap_s: r.lap_complete ? r.laps[0].elapsed : null,
    record: saveRecord('learned_onnx_E0_single', r.report), verification: stripV(v),
    note: '外部 ONNX(モード 4・先読み 1・前進ゲイン 1.01・遅延 1)の記録の操作列を再生。E0 のみ'};
}
result.wall_s = (performance.now() - t0) / 1000;
fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true});
fs.writeFileSync(out, JSON.stringify(result, null, 1) + '\n');
const fmt = v => v == null ? '—' : v.toFixed(6);
if (result.selection_sets) {
  const s = result.selection_sets;
  console.log(`[${variant}] 元のコア: 選べる ${s.eligible} 点数 ${s.score} / REC60 有効 ${s.REC60.valid_two_laps}/60 1 周目平均 ${fmt(s.REC60.lap1.mean)} 2 周目平均 ${fmt(s.REC60.lap2.mean)} ` +
    `最小すき間 ${s.REC60.min_clearance.toFixed(4)} / STD 単周 ${fmt(s.STD_single_lap_from_stop)} フライング ${fmt(s.STD_flying_lap)} / ROB24 ${s.ROB24.valid_two_laps}/24`);
}
if (result.fast_vs_original) console.log(`高速版との差: ${JSON.stringify(result.fast_vs_original)}`);
for (const [who, r] of [['RL', result.rl], ['教師', result.teacher]]) if (r) {
  console.log(`${who}: ` + r.single.map(x => `${x.name} ${fmt(x.single_lap_s)}`).join(' / ') + ' | 連続 ' + r.continuous.map(x => `${x.name} ${x.laps.map(l => l.elapsed.toFixed(4)).join(',')}`).join(' / ') +
    ` | 遅延 1 ${r.STD_delay1_reference?.reason} ${fmt(r.STD_delay1_reference?.single_lap_s)}`);
}
if (result.learned_onnx) console.log(`学習済み ONNX: ${fmt(result.learned_onnx.single_lap_s)} 確認 ${result.learned_onnx.verification.pass}`);
console.log(`(${result.wall_s.toFixed(1)} 秒)`);
