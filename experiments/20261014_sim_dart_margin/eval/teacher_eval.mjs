#!/usr/bin/env node
// 閉ループ評価の「先生の目」での見直し(sim_dart_20261014 の評価の段、2026-10-14)。走行の記録は読むだけ。
//
//   node eval/teacher_eval.mjs --runs <runs_list.json> --out <teacher_eval.json> [--takeover 1] [--every 3]
//
// runs_list.json = [{name, label, file, teacher: 'fast'|'margin', takeover: bool}, ...](ブラウザの 1 走行 = sim.report() + world + context)
//  1) 先生の参照の走り: 走行の world・config(開始ごとに同じ)で、T_fast と T_margin を Node の元の sim-core.js で単周(遅延 0)。
//     先生は記録と同じ web/teacher_core.mjs(前回の行動 = 先生自身の値。protocol dart.applied_and_previous_action)。
//  2) ずれ: 走行の操作を元の sim-core.js で再生し(ブラウザと同じ終わり方・フレーム数を確かめる)、各フレームの位置を
//     最速の先生の進みの追跡(features.mjs)で R 上の s と横ずれ e_y にする。先生の参照の走りも同じ追跡で (s, e_y) にし、
//     ずれ = |e_y(走行) − e_y(先生, 同じ s)|(設計の段の deviationFromClean と同じ)。T_fast からと、その腕の先生からの 2 つ。
//     もう 1 つ、線そのものからの距離 = 走行のフレーム f の位置から、先生の同じ開始の走りの線(折れ線)のうち、フレーム f − 45 〜 f + 45
//     (前後 1.5 秒)の部分のいちばん近い点までの距離(dev_path)。窓を付けるのは、コースの別の場所の線を拾わないため(ゲートを外して
//     コースの内側へ入った走りが、別の区間の線の近くを通っても小さく出ない)。窓なしの値も path_global に残す。
//     s で合わせる方は、走行が経路から大きく外れると進みの追跡が止まって小さく出るので、図と報告の「線からのずれ」には dev_path を使う。
//  3) 引き継ぎ(--takeover 1、完走しなかった走行だけ。参考): フレーム k まで走行の操作、k から先を先生(遅延 0)に任せて単周を完走できるか。
//     失敗のフレームから k を every ずつ戻して、最初に完走できた k(=「先生ならまだ立て直せた最後のフレーム」)を探す。k = 0 も確かめる。
// 実車・I2C には触れない。
import fs from 'node:fs';
import path from 'node:path';
import {createSimulation} from '../lib/app_emulation.mjs';
import {createTeacherCore} from '../web/teacher_core.mjs';
import {createFeatureTracker, commandToAction, OBS_DIM, gateStatus, LOOP_LENGTH} from '../web/rt/features.mjs';
import WF from '../web/rt/weights_fast.mjs';
import WM from '../web/rt/weights_margin.mjs';

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; };
const runsPath = opt('--runs'), outPath = opt('--out');
const doTakeover = opt('--takeover', '1') === '1', every = Number(opt('--every', '3'));
const W = {fast: WF, margin: WM};
const runs = JSON.parse(fs.readFileSync(runsPath, 'utf8'));

const startKey = sv => `${sv.gate}|${(+sv.left_offset_m).toFixed(4)}|${(+sv.left_yaw_deg).toFixed(4)}`;

// 1 フレームの (s, e_y, d_min) を出す追跡(最速の先生の特徴量。どの走りにも同じものを使う)
function makeTrack(world, cfg) {
  const tr = createFeatureTracker(world, cfg), obs = new Float64Array(OBS_DIM), aux = {};
  return (f, st, ri) => {
    if (f === 0) tr.resetProgress(st.x, st.z, ri); else tr.advanceProgress(st.x, st.z);
    tr.observe(st, ri, 0, 0, obs, aux);
    return {s: aux.s, ey: aux.lateral_error, dmin: aux.d_min};
  };
}

// フレーム k まで actions、その先は先生(kind)。k = Infinity なら全部 actions(再生)
function drive(report, actions, k, kind, {trackRows = false} = {}) {
  const sim = createSimulation(report.world, report.config);
  let snap = sim.reset({seed: report.seed, scenario: report.scenario});
  const cfg = JSON.parse(JSON.stringify(snap.config));
  const core = kind ? createTeacherCore(kind, report.world, cfg, W[kind]) : null;
  const track = trackRows ? makeTrack(report.world, cfg) : null;
  const rows = [], crossings = [], seenGate = new Set();
  const gs = {along: 0, lateral: 0, index: 0};
  let prevAlong = null, prevRi = -1;
  const g0 = report.context?.start_variation?.gate ?? 0;
  let prev = [0, 0], f = 0;
  while (!snap.state.terminal) {
    const st = snap.state, ri = snap.metrics.route_index;
    if (core) { if (f === 0) core.begin(st, ri); else core.advance(st, ri); }
    if (track) {
      rows.push({f, t: st.sim_time ?? f / 30, x: st.x, z: st.z, speed: st.speed, ...track(f, st, ri)});
      // 目標ゲートの線(前の点 → 目標の点の向きに垂直)を横切った最初の時の横の位置。|横| > 0.65 m なら sim は数えない(ゲートの外し)
      if (ri !== prevRi) {
        // sim が前の目標ゲートを数えた(このフレームで目標が進んだ): 前の目標に対する今の横の位置を、そのゲートを通った位置とする
        if (prevRi >= 0 && !seenGate.has(prevRi)) {
          gateStatus(report.world.route, prevRi, st.x, st.z, gs);
          seenGate.add(prevRi);
          crossings.push({target: prevRi, point: (g0 + prevRi + 1) % 33, t: r4(st.sim_time ?? f / 30), lateral_m: r4(gs.lateral),
            margin_m: r4(.65 - Math.abs(gs.lateral)), counted: true});
        }
        prevAlong = null; prevRi = ri;
      }
      gateStatus(report.world.route, ri, st.x, st.z, gs);
      if (prevAlong != null && prevAlong < 0 && gs.along >= 0 && !seenGate.has(ri)) {
        seenGate.add(ri);
        crossings.push({target: ri, point: (g0 + ri + 1) % 33, t: r4(st.sim_time ?? f / 30), lateral_m: r4(gs.lateral), margin_m: r4(.65 - Math.abs(gs.lateral)),
          counted: false});
      }
      prevAlong = gs.along;
    }
    let action;
    if (f >= k) {
      const d = core.decide(st, prev[0], prev[1]);
      action = {steering: d.steering, speed: d.speed};
      prev = [d.a0, d.a1];
    } else {
      if (f >= actions.length) break;
      action = {steering: actions[f].steering, speed: actions[f].speed};
      if (core) { const b = commandToAction(action.steering, action.speed, core.speedCap, core.maxSteer); prev = [b.a0, b.a1]; }
    }
    snap = sim.step(action);
    f++;
  }
  const rep = sim.report();
  return {reason: rep.outcome.reason, lap_s: rep.outcome.reason === 'route_complete' && rep.metrics.collisions === 0 ? rep.metrics.elapsed : null,
    passed_gates: rep.metrics.passed_gates, collisions: rep.metrics.collisions, frames: rep.actions.length, rows,
    crossings: (rep.metrics.required_crossing_events ?? []).length, trace: trackRows ? rep.trace : null, gate_crossings: crossings};
}

function interpByS(cs, cv, s) {
  if (!(s >= cs[0] && s <= cs.at(-1))) return null;
  let lo = 0, hi = cs.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cs[m] <= s) lo = m; else hi = m; }
  const fr = Math.min(1, Math.max(0, (s - cs[lo]) / Math.max(1e-9, cs[hi] - cs[lo])));
  return cv[lo] + (cv[hi] - cv[lo]) * fr;
}
// 先生の s は単調に増える前提(単周)。念のため単調な部分だけを使う
function monotone(rows) {
  const cs = [], ce = [];
  for (const q of rows) { if (!cs.length || q.s > cs.at(-1)) { cs.push(q.s); ce.push(q.ey); } }
  return {cs, ce};
}
const r4 = x => x == null ? null : +x.toFixed(4);
// 点 (x, z) から折れ線 xy までのいちばん近い距離
const WIN = 45;
function distToPath(xy, x, z, lo = 0, hi = Infinity) {
  let best = Infinity;
  lo = Math.max(0, Math.min(lo, xy.length - 2 - WIN));   // 先生の走りが先に終わったときは、最後の 1.5 秒の線と比べる
  for (let i = lo; i + 1 < Math.min(xy.length, hi + 1); i++) {
    const [ax, az] = xy[i], [bx, bz] = xy[i + 1], dx = bx - ax, dz = bz - az, L = dx * dx + dz * dz;
    const u = L > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / L)) : 0;
    best = Math.min(best, Math.hypot(x - ax - u * dx, z - az - u * dz));
  }
  return best;
}

const refs = new Map();   // startKey → {fast, margin}
function teacherRefs(report) {
  const key = startKey(report.context.start_variation);
  if (!refs.has(key)) {
    const out = {};
    for (const kind of ['fast', 'margin']) {
      const r = drive(report, [], 0, kind, {trackRows: true});
      out[kind] = {reason: r.reason, lap_s: r.lap_s, passed_gates: r.passed_gates, collisions: r.collisions, frames: r.frames,
        min_clearance_m: Math.min(...r.rows.map(q => q.dmin)), ...monotone(r.rows), gate_crossings: r.gate_crossings,
        xy: r.rows.map(q => [r4(q.x), r4(q.z)]), s: r.rows.map(q => r4(q.s)), ey: r.rows.map(q => r4(q.ey)), speed: r.rows.map(q => r4(q.speed))};
      console.error(`先生の参照 ${key} ${kind}: ${r.reason} ${r.lap_s?.toFixed(3)} 秒`);
    }
    refs.set(key, out);
  }
  return refs.get(key);
}

const rowsOut = [];
for (const item of runs) {
  const report = JSON.parse(fs.readFileSync(item.file, 'utf8'));
  if (report.config.episodeMode === 'continuous-laps-v1') continue;
  const ref = teacherRefs(report);
  const n = report.actions.length;
  const rep = drive(report, report.actions, Infinity, null, {trackRows: true});
  const replaySame = rep.reason === report.outcome.reason && rep.frames === n;
  const devTo = kind => rep.rows.map(q => { const e = interpByS(ref[kind].cs, ref[kind].ce, q.s); return e == null ? null : Math.abs(q.ey - e); });
  const dF = devTo('fast'), dT = devTo(item.teacher);
  // 先生の単周の線は 1 周で終わる。走行が 1 周を越えて走ったとき(ゲートを外して周が終わらない)のために、線を 2 周ぶん(同じ線をつないだもの)にする
  const loop2 = xy => xy.concat(xy.slice(1));
  const xyF = loop2(ref.fast.xy), xyT = loop2(ref[item.teacher].xy);
  const pF = rep.rows.map((q, f) => distToPath(xyF, q.x, q.z, f - WIN, f + WIN));
  const pT = rep.rows.map((q, f) => distToPath(xyT, q.x, q.z, f - WIN, f + WIN));
  const gT = rep.rows.map(q => distToPath(ref[item.teacher].xy, q.x, q.z));
  const signedT = rep.rows.map(q => { const e = interpByS(ref[item.teacher].cs, ref[item.teacher].ce, q.s); return e == null ? null : q.ey - e; });
  const last = rep.rows.length - 1;
  const lastVal = a => { for (let i = a.length - 1; i >= 0; i--) if (a[i] != null) return a[i]; return null; };
  const maxOf = a => { const b = a.filter(x => x != null); return b.length ? Math.max(...b) : null; };
  const meanOf = a => { const b = a.filter(x => x != null); return b.length ? b.reduce((p, q) => p + q, 0) / b.length : null; };
  const firstOver = (a, th) => { const i = a.findIndex(x => x != null && x > th); return i < 0 ? null : i; };
  const sv = report.context.start_variation;
  // 最初の直線を抜けた: ゲート 0 の開始で、衝突せずに x > 2.6 m に届いた
  let firstStraight = null;
  if (sv.gate === 0) {
    const i = rep.rows.findIndex(q => q.x > 2.6);
    const ce = (report.metrics.collision_events ?? [])[0];
    firstStraight = i >= 0 && (!ce || (ce.time ?? Infinity) > rep.rows[i].t);
  }
  const row = {name: report.name, label: item.label, file: item.file, teacher: item.teacher, start_variation: sv, browser_reason: report.outcome.reason,
    frames: n, replay_same: replaySame, replay_reason: rep.reason,
    dev_fast_m: {at_0p5s: dF[15] ?? null, at_1s: dF[30] ?? null, at_end: lastVal(dF), max: maxOf(dF), mean: meanOf(dF)},
    dev_teacher_m: {at_0p5s: dT[15] ?? null, at_1s: dT[30] ?? null, at_end: lastVal(dT), max: maxOf(dT), mean: meanOf(dT),
      first_frame_over_5cm: firstOver(dT, .05), first_frame_over_10cm: firstOver(dT, .10)},
    path_dev_fast_m: {at_0p5s: pF[15] ?? null, at_1s: pF[30] ?? null, at_end: lastVal(pF), max: maxOf(pF), mean: meanOf(pF)},
    path_dev_teacher_m: {at_0p5s: pT[15] ?? null, at_1s: pT[30] ?? null, at_end: lastVal(pT), max: maxOf(pT), mean: meanOf(pT),
      max_before_first_gate_miss: (() => { const m = rep.gate_crossings.find(c => c.margin_m < 0); const nf = m ? Math.round(m.t * 30) : pT.length;
        return maxOf(pT.slice(0, Math.max(1, nf))); })(),
      first_frame_over_5cm: firstOver(pT, .05), first_frame_over_10cm: firstOver(pT, .10), first_frame_over_20cm: firstOver(pT, .20)},
    first_straight_passed: firstStraight,
    // 進んだ道のり: 最速の先生の進みの追跡の s(参照の周 R に沿った長さ。ゲートを数えたかとは関係ない)の、最後 − 最初
    progress_m: r4(rep.rows.at(-1).s - rep.rows[0].s), lap_fraction: r4((rep.rows.at(-1).s - rep.rows[0].s) / LOOP_LENGTH), loop_length_m: r4(LOOP_LENGTH),
    gate_crossings: rep.gate_crossings,
    // 最初に外したゲート(線を |横| > 0.65 m で横切った)と、同じゲートでの先生の横の位置
    first_gate_miss: (() => { const m = rep.gate_crossings.find(c => c.margin_m < 0); if (!m) return null;
      const tf = ref.fast.gate_crossings.find(c => c.point === m.point), tt = ref[item.teacher].gate_crossings.find(c => c.point === m.point);
      return {...m, teacher_fast_lateral_m: tf?.lateral_m ?? null, teacher_fast_margin_m: tf?.margin_m ?? null,
        teacher_lateral_m: tt?.lateral_m ?? null, teacher_margin_m: tt?.margin_m ?? null}; })(),
    series: {t: rep.rows.map(q => r4(q.t)), s: rep.rows.map(q => r4(q.s)), dev_fast: dF.map(r4), dev_teacher: dT.map(r4), signed_teacher: signedT.map(r4),
      dmin: rep.rows.map(q => r4(q.dmin)), path_teacher: pT.map(r4), path_fast: pF.map(r4), path_global_teacher: gT.map(r4), x: rep.rows.map(q => r4(q.x)), z: rep.rows.map(q => r4(q.z))}};
  // 引き継ぎ(完走しなかった走行だけ)
  const completed = report.outcome.reason === 'route_complete' && report.metrics.collisions === 0;
  if (doTakeover && item.takeover && !completed) {
    const kinds = [...new Set(['fast', item.teacher])];
    row.takeover = {};
    for (const kind of kinds) {
      const k0 = drive(report, report.actions, 0, kind);
      let lastOk = null, tried = 0, firstFailAfter = null;
      for (let k = n - 1; k >= 0 && tried < 160; k -= every) {
        tried++;
        const r = drive(report, report.actions, k, kind);
        if (r.lap_s != null) { lastOk = k; break; }
        firstFailAfter = k;
      }
      row.takeover[kind] = {every, k0_lap_s: k0.lap_s, k0_reason: k0.reason, last_recoverable_frame: lastOk,
        first_unrecoverable_frame_checked: firstFailAfter, frames_before_end: lastOk == null ? null : n - 1 - lastOk,
        seconds_before_end: lastOk == null ? null : r4((n - 1 - lastOk) / 30), tried};
    }
  }
  rowsOut.push(row);
  console.error(`${report.name}: ${report.outcome.reason} ${n} フレーム 再生=${replaySame} 線からの距離(先生) 0.5秒 ${(row.path_dev_teacher_m.at_0p5s ?? NaN).toFixed(3)} ` +
    `終わり ${(row.path_dev_teacher_m.at_end ?? NaN).toFixed(3)} 最大 ${(row.path_dev_teacher_m.max ?? NaN).toFixed(3)}` +
    (row.takeover ? ' 引き継ぎ ' + Object.entries(row.takeover).map(([k, v]) => `${k}:k=${v.last_recoverable_frame}`).join(' ') : ''));
}
const refOut = {};
for (const [k, v] of refs) refOut[k] = Object.fromEntries(Object.entries(v).map(([kind, r]) => [kind, (({cs, ce, ...x}) => x)(r)]));
fs.writeFileSync(outPath, JSON.stringify({schema: 'sim-dart-teacher-eval/v1', created_at: new Date().toISOString(),
  note: '参考。先生は web/teacher_core.mjs(記録と同じ)、遅延 0、元の sim-core.js。ずれは最速の先生の進みの追跡の (s, e_y) で、同じ s の先生の e_y と比べた横の差',
  teacher_refs: refOut, rows: rowsOut}) + '\n');
console.log(outPath);
