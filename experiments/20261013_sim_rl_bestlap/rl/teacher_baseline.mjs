#!/usr/bin/env node
// 教師(RC 教師)をこの RL 環境で走らせた基準の値(2026-10-13、RL 環境の段)
//
//   node rl/teacher_baseline.mjs [出力フォルダー]   (既定 /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/env)
//
// 作るもの:
//   teacher_baseline.json  REC60(高速版と元のコアの両方)・STD(単周と連続)・E0〜E2・遅延 1 の STD(参考)・ROB24 のゲート 0 の 8 条件(約束の外、参考)
//                          H1 の比べる相手(教師の REC60 の 1 周目・2 周目の平均)もここで決まる(protocol acceptance_H1.teacher_rec60_means)
//   records/teacher_*.json  sim のレポート(rl/verify_lap.mjs で確かめられる形)
// 教師の約束(ゲート 0・±3 cm・±3°)の外は合否に使わない。
import fs from 'node:fs';
import path from 'node:path';
import {loadRec60, STD, ROB24, CLOSED_LOOP_STARTS, runEpisode, teacherController, teacherOutsideContract, summarizeRec60, lapStats} from './eval_sets.mjs';
import {verifyReport} from './lap_audit.mjs';
import {sha256File, PATHS, EXPECTED_SHA256} from './env_core.mjs';
import {REFERENCE_LOOP_SHA256, FEATURES_VERSION} from './features.mjs';

const outDir = process.argv[2] ?? '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/env';
fs.mkdirSync(path.join(outDir, 'records'), {recursive: true});
const teacher = await teacherController(), outside = await teacherOutsideContract();
const strip = r => { const {report, collision_events, ...rest} = r; return rest; };
const save = (name, r) => { fs.writeFileSync(path.join(outDir, 'records', name + '.json'), JSON.stringify(r.report)); return path.join(outDir, 'records', name + '.json'); };
const t0 = performance.now();
const out = {schema: 'rl-bestlap-teacher-baseline/v1', created_at: new Date().toISOString(), node: process.version,
  features_version: FEATURES_VERSION, reference_loop_sha256: REFERENCE_LOOP_SHA256,
  assets: Object.fromEntries(Object.keys(EXPECTED_SHA256).map(k => [k, sha256File(PATHS[k])]))};

// REC60: 高速版と元のコア
const rec60 = loadRec60();
for (const coreKind of ['fast', 'original']) {
  const t = performance.now(), rows = [];
  for (const c of rec60) rows.push(await runEpisode({start: c, controller: teacher, coreKind, record: c.name.endsWith('-00') || c.name.endsWith('val-59')}));
  for (const r of rows) if (r.report) save(`teacher_REC60_${r.name}_${coreKind}`, r);
  out[`REC60_${coreKind}`] = {summary: summarizeRec60(rows), wall_s: (performance.now() - t) / 1000,
    scenes: rows.map(r => ({name: r.name, start: r.start, laps: r.laps.map(l => l.elapsed), collisions: r.collisions, reason: r.reason,
      valid_two_laps: r.valid_two_laps, min_clearance: r.min_clearance, max_speed: r.max_speed}))};
}
// 高速版と元のコアの一致(周回時間の差)
out.REC60_fast_vs_original_max_lap_diff_s = Math.max(...out.REC60_fast.scenes.map((s, i) =>
  Math.max(...s.laps.map((v, k) => Math.abs(v - out.REC60_original.scenes[i].laps[k])))));
// H1 の比べる相手
out.teacher_rec60_means = {lap1_mean_s: out.REC60_original.summary.lap1.mean, lap2_mean_s: out.REC60_original.summary.lap2.mean,
  h1_lap1_threshold_s: out.REC60_original.summary.lap1.mean - .1, h1_lap2_threshold_s: out.REC60_original.summary.lap2.mean - .1};

// STD(単周 = 停止から 1 周、連続 = フライングラップ)
for (const coreKind of ['fast', 'original']) {
  const single = await runEpisode({start: STD, controller: teacher, coreKind, episodeMode: 'single', record: true});
  const cont = await runEpisode({start: STD, controller: teacher, coreKind, episodeMode: 'continuous', record: true});
  out[`STD_${coreKind}`] = {single_lap_from_stop_s: single.lap_complete ? single.laps[0].elapsed : null, single: strip(single),
    flying_lap_s: cont.valid_two_laps ? cont.laps[1].elapsed : null, continuous: strip(cont)};
  if (coreKind === 'original') {
    out.STD_original.single_record = save('teacher_STD_single', single);
    out.STD_original.continuous_record = save('teacher_STD_continuous', cont);
    out.STD_original.single_verification = strip2(await verifyReport(single.report, {minClearance: .03}));
    out.STD_original.continuous_verification = strip2(await verifyReport(cont.report, {minLaps: 2, minClearance: .03}));
  }
}
function strip2(v) { const {collision_events_own, ...rest} = v; return rest; }

// 閉ループ評価の開始のうち教師の約束の範囲(E0〜E2)、単周と E0・E1 の連続
out.closed_loop_starts = [];
for (const c of CLOSED_LOOP_STARTS.slice(0, 3)) {
  const single = await runEpisode({start: c, controller: teacher, coreKind: 'original', episodeMode: 'single', timeLimit: 15});
  const row = {name: c.name, start: c, single_lap_s: single.lap_complete ? single.laps[0].elapsed : null, reason: single.reason, min_clearance: single.min_clearance};
  if (c.name !== 'E2') { const cont = await runEpisode({start: c, controller: teacher, coreKind: 'original'}); row.continuous_laps_s = cont.laps.map(l => l.elapsed); row.continuous_valid = cont.valid_two_laps; }
  out.closed_loop_starts.push(row);
}
// 参考: 遅延 1 フレーム(アプリの固定遅延と同じ。最初は中立)での STD の単周。教師は衝突する(design.md 1 章)
const d1 = await runEpisode({start: STD, controller: teacher, coreKind: 'original', episodeMode: 'single', delayFrames: 1, record: true});
out.STD_delay1_reference = {reason: d1.reason, passed_gates: d1.passed_gates, sim_time: d1.sim_time, collisions: d1.collisions,
  collision_event: d1.report.metrics.collision_events[0] ?? null};
// 参考: ROB24 のうちゲート 0 の 8 条件(教師の約束の外。参照を標準の開始で作って使う)
out.ROB24_gate0_outside_contract = [];
for (const c of ROB24.filter(c => c.gate === 0)) {
  const r = await runEpisode({start: c, controller: outside, coreKind: 'fast'});
  out.ROB24_gate0_outside_contract.push({name: r.name, start: c, laps: r.laps.map(l => l.elapsed), reason: r.reason, valid_two_laps: r.valid_two_laps,
    min_clearance: r.min_clearance});
}
// ROB24 の開始で、障害物と重なって次のゲートに移すもの(規則の確認)
out.ROB24_spawn_check = [];
for (const c of ROB24) {
  const r = await runEpisode({start: c, controller: {name: 'stop', create: () => ({action: () => ({steering: 0, speed: 0})})}, coreKind: 'fast',
    episodeMode: 'single', timeLimit: 1 / 30, invalidSpawnRule: 'next-gate'});
  out.ROB24_spawn_check.push({name: c.name, requested_gate: c.gate, used_gate: r.start?.gate ?? null, moved: r.start_moved_gates ?? null,
    invalid: !!r.invalid_spawn});
}
out.wall_s = (performance.now() - t0) / 1000;
fs.writeFileSync(path.join(outDir, 'teacher_baseline.json'), JSON.stringify(out, null, 1) + '\n');
const f = out.REC60_fast.summary, o = out.REC60_original.summary;
console.log(`教師 REC60(高速版): 1 周目 ${f.lap1.min.toFixed(6)}〜${f.lap1.max.toFixed(6)} 平均 ${f.lap1.mean.toFixed(6)} / 2 周目 ${f.lap2.min.toFixed(6)}〜${f.lap2.max.toFixed(6)} 平均 ${f.lap2.mean.toFixed(6)} / 有効 ${f.valid_two_laps}/60 衝突 ${f.collisions} 最小すき間 ${f.min_clearance.toFixed(4)}`);
console.log(`教師 REC60(元のコア): 1 周目 平均 ${o.lap1.mean.toFixed(6)} / 2 周目 平均 ${o.lap2.mean.toFixed(6)} / 高速版との差の最大 ${out.REC60_fast_vs_original_max_lap_diff_s}`);
console.log(`教師 STD: 停止から 1 周 ${out.STD_original.single_lap_from_stop_s} 秒(高速版 ${out.STD_fast.single_lap_from_stop_s})/ フライングラップ ${out.STD_original.flying_lap_s} 秒 / 確認 ${out.STD_original.single_verification.pass}・${out.STD_original.continuous_verification.pass}`);
console.log(`遅延 1(参考): ${out.STD_delay1_reference.reason} ゲート ${out.STD_delay1_reference.passed_gates} ${out.STD_delay1_reference.sim_time.toFixed(4)} 秒`);
console.log(`ROB24 ゲート 0(約束の外・参考): 有効 ${out.ROB24_gate0_outside_contract.filter(r => r.valid_two_laps).length}/8`);
console.log(`ROB24 の開始の移動: ${JSON.stringify(out.ROB24_spawn_check.filter(r => r.moved || r.invalid))}`);
console.log(`全体 ${out.wall_s.toFixed(1)} 秒 → ${path.join(outDir, 'teacher_baseline.json')}`);
