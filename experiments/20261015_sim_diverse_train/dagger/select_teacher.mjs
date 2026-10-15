#!/usr/bin/env node
// DAgger の先生の選びと切り替えのフレーム k(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.procedure 2・3)
//
//   node select_teacher.mjs --runs <状態集めの走行の JSON をカンマ区切り> --out <結果.json> [--every 1]
//
// 2) 先生の選び: 各走行の操作を元の sim-core.js で再生し、各フレーム t の次の状態 s_{t+1} から T_fast・T_margin に任せて
//    2 秒(60 フレーム)走らせ、衝突・本当のゲートの外し(gate_miss.mjs の決まり)・すき間 2 cm 未満が無ければ「立て直せる」と数える
//    (10/14 の relabel_scene.mjs の forkFrom と同じ作り方。先生は開始から同じ順に観測を進め、前回の行動は実際に適用した操作)。
//    全部の走行で数えた合計の多いほうの先生(同じなら T_margin)。
// 3) 切り替えのフレーム k: 選んだ先生(と参考にもう一方)で、フレーム k まで記録の操作、k から先生に任せて単周を完走できる最後の k
//    (後ろから --every ずつ戻して最初に完走できた k)。完走した走りは k = 300。
// 実車・I2C には触れない。走行の記録は読むだけ。
import fs from 'node:fs';
import {baseWorld, activeWorldFor, createSimulation} from '../lib/app_emulation.mjs';
import {createTeacherCore, previousAction} from '../web/teacher_core.mjs';
import {GATE_MISS_THRESHOLD_M} from '../web/gate_miss.mjs';
import WF from '../web/rt/weights_fast.mjs';
import WM from '../web/rt/weights_margin.mjs';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const files = opt('--runs').split(',');
const outPath = opt('--out');
const every = Number(opt('--every', 1));
const H = 10, V = 60, MIN_CLEAR = .02;
const W = {fast: WF, margin: WM};
const base = await baseWorld();
const pick = st => ({x: st.x, z: st.z, yaw: st.yaw, speed: st.speed, steering: st.steering, frame: st.frame});

function replayStates(report) {
  const sim = createSimulation(report.world, report.config);
  sim.reset({seed: report.seed, scenario: report.scenario});
  const S = [], ri = [], gc = [];
  for (let i = 0; i < report.actions.length; i++) {
    const b = sim.snapshot();
    S.push(pick(b.state)); ri.push(b.metrics.route_index); gc.push(`${b.metrics.completed_laps ?? 0}:${b.metrics.passed_gates ?? 0}`);
    sim.step(report.actions[i]);
  }
  const b = sim.snapshot();
  S.push(pick(b.state)); ri.push(b.metrics.route_index); gc.push(`${b.metrics.completed_laps ?? 0}:${b.metrics.passed_gates ?? 0}`);
  return {S, ri, gc, end: b};
}

function recoverable(report, kind) {
  const {S, ri, gc} = replayStates(report);
  const N = report.actions.length, NG = report.world.route.length, g0 = report.context.start_variation.gate;
  const core = createTeacherCore(kind, report.world, report.config, W[kind]);
  let prev = [0, 0], lastGateS = 0, ok = 0, tried = 0;
  const why = {};
  const valid = [];
  for (let i = 0; i < N; i++) {
    if (i === 0) core.begin(S[0], ri[0]); else core.advance(S[i], ri[i]);
    const d = core.decide(S[i], prev[0], prev[1]);
    if (i === 0 || gc[i] !== gc[i - 1]) lastGateS = d.s;
    if (i >= 1) {
      // t = i − 1: s_i から先生だけで V フレーム
      const fc = core.fork();
      const k0 = ri[i] % NG;
      const {world: wf} = activeWorldFor(base, (g0 + k0) % NG, 0, 0);
      const cfg = {...report.config, timeLimit: 600};
      delete cfg.episodeMode;
      const fsim = createSimulation(wf, cfg);
      const s0 = S[i];
      const snap0 = fsim.reset({seed: report.seed, scenario: report.scenario, initialState: {x: s0.x, z: s0.z, yaw: s0.yaw, speed: s0.speed, steering: s0.steering, routeIndex: 0}});
      let w = snap0.state.terminal ? 'invalid_spawn' : null, p = prev, lastS = lastGateS, lastG = 0, minD = Infinity;
      for (let j = 0; j < V && !w; j++) {
        const sn = j === 0 ? null : fsim.snapshot();
        const st = j === 0 ? s0 : pick(sn.state);
        const tri = j === 0 ? ri[i] : (k0 + sn.metrics.route_index) % NG;
        if (j > 0) fc.advance(st, tri);
        const dd = fc.decide(st, p[0], p[1]);
        minD = Math.min(minD, dd.dmin);
        const g = j === 0 ? 0 : sn.metrics.passed_gates;
        if (g !== lastG) { lastS = dd.s; lastG = g; }
        if (dd.s - lastS > GATE_MISS_THRESHOLD_M) { w = 'gate_miss'; break; }
        p = [dd.a0, dd.a1];
        fsim.step({steering: dd.steering, speed: dd.speed});
        const after = fsim.snapshot();
        if (after.metrics.collisions > 0) { w = 'collision'; break; }
        if (after.state.terminal) { w = 'terminal:' + after.state.reason; break; }
      }
      if (!w && minD < MIN_CLEAR) w = 'low_clearance';
      tried++;
      valid.push(!w);
      if (!w) ok++; else why[w] = (why[w] ?? 0) + 1;
    }
    prev = previousAction(d, report.actions[i], true, true, core.speedCap, core.maxSteer);
  }
  return {frames: N, tried, recoverable: ok, not_recoverable: why, first_unrecoverable_t: valid.indexOf(false) < 0 ? null : valid.indexOf(false)};
}

// k まで記録の操作、k から先生(遅延 0)で単周を完走できるか
function takeoverLap(report, k, kind) {
  const sim = createSimulation(report.world, report.config);
  let snap = sim.reset({seed: report.seed, scenario: report.scenario});
  const core = createTeacherCore(kind, report.world, report.config, W[kind]);
  let prev = [0, 0], f = 0;
  while (!snap.state.terminal) {
    const st = snap.state, ri = snap.metrics.route_index;
    if (f === 0) core.begin(st, ri); else core.advance(st, ri);
    const d = core.decide(st, prev[0], prev[1]);
    let a;
    if (f < k) { a = {steering: report.actions[f].steering, speed: report.actions[f].speed}; prev = previousAction(d, a, true, true, core.speedCap, core.maxSteer); }
    else { a = {steering: d.steering, speed: d.speed}; prev = [d.a0, d.a1]; }
    snap = sim.step(a);
    f++;
  }
  const rep = sim.report();
  return {reason: rep.outcome.reason, lap_s: rep.outcome.reason === 'route_complete' && rep.metrics.collisions === 0 ? rep.metrics.elapsed : null, frames: f};
}

function findK(report, kind) {
  const N = report.actions.length;
  if (report.outcome.reason === 'route_complete' && report.metrics.collisions === 0) return {k: 300, completed_run: true};
  let tried = 0;
  for (let k = N - 1; k >= 0; k -= every) {
    tried++;
    const r = takeoverLap(report, k, kind);
    if (r.lap_s != null) return {k, lap_s: r.lap_s, tried, frames_before_end: N - 1 - k};
  }
  return {k: null, tried};
}

const rows = [];
const totals = {fast: 0, margin: 0};
for (const file of files) {
  const report = JSON.parse(fs.readFileSync(file, 'utf8'));
  const row = {name: report.name, file, start_variation: report.context.start_variation, reason: report.outcome.reason, frames: report.actions.length,
    passed_gates: report.metrics.passed_gates, collisions: report.metrics.collisions};
  for (const kind of ['fast', 'margin']) {
    row[kind] = recoverable(report, kind);
    totals[kind] += row[kind].recoverable;
    console.error(`${report.name} ${kind}: 立て直せる ${row[kind].recoverable} / ${row[kind].tried}`);
  }
  rows.push(row);
}
const teacher = totals.fast > totals.margin ? 'fast' : 'margin';
for (const row of rows) {
  const report = JSON.parse(fs.readFileSync(row.file, 'utf8'));
  for (const kind of ['fast', 'margin']) {
    row[kind].takeover = findK(report, kind);
    console.error(`${row.name} ${kind}: k = ${row[kind].takeover.k}`);
  }
  row.k = row[teacher].takeover.k;
}
const out = {schema: 'jf-sim-diverse-dagger-teacher-select/v1', created_at: new Date().toISOString(), rule: 'protocol.json dagger.procedure 2・3',
  validity_frames: V, horizon: H, min_clearance_m: MIN_CLEAR, every, totals, teacher, tie_rule: '同じなら T_margin', rows};
fs.writeFileSync(outPath, JSON.stringify(out, null, 1) + '\n');
console.log(JSON.stringify({totals, teacher, k: rows.map(r => [r.name, r.k, r.frames])}));
