#!/usr/bin/env node
// 記録の段: 記録の前に、同じ 60 条件をブラウザと同じ差し込みモジュール + 元の sim-core.js で Node で走らせて「予想」を作る
// (2026-10-13、design.md 4.2 節)
//
//   node ~/jetformer/experiments/20261013_sim_rl_bestlap/record/predict_rec60.mjs [--plan plan_rlbest60.json] [--out prediction.json]
//
//  - 教師: コピーした dist に置いた dist/rl/rl-teacher-bestlap.js(ブラウザが読むファイルそのもの)
//  - 手順: lib/app_emulation.mjs の runCondition(app.js の reset と tick と同じ順: 教師の操作 → sim.step)
//  - シーンごとに、周回時間・ゲート・黒シートの横断・衝突・終わり方・最小すき間(features の clearance、車体の長方形と全障害物)・
//    操作列の sha256 を保存し、RL の段の最終評価(rl/final/final_eval_main.json の REC60、元のコア)と周回時間を照合する
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runCondition, baseWorld, SIM_ROOT} from '../lib/app_emulation.mjs';

const REPORT = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013';
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i < 0 ? def : args[i + 1]; };
const planPath = opt('--plan', REPORT + '/recording/plan_rlbest60.json');
const outPath = opt('--out', REPORT + '/recording/prediction.json');
const teacherPath = path.join(SIM_ROOT, 'dist/rl/rl-teacher-bestlap.js');
const teacher = await import(pathToFileURL(teacherPath).href);
const features = await import(pathToFileURL(path.join(SIM_ROOT, 'dist/rl/features.mjs')).href);
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
const finalEval = JSON.parse(fs.readFileSync(REPORT + '/rl/final/final_eval_main.json', 'utf8'));
const refScenes = finalEval.selection_sets.REC60_scenes;
const world0 = await baseWorld();
const geometry = features.prepareGeometry(world0);

const t0 = Date.now();
const scenes = [];
for (let i = 0; i < plan.length; i++) {
  const c = plan[i];
  const hash = crypto.createHash('sha256');
  let minClear = Infinity, minClearFrame = -1, maxSpeed = 0;
  const buf = new Float64Array(2);
  const onFrame = (before, action, after) => {
    buf[0] = action.steering; buf[1] = action.speed; hash.update(Buffer.from(buf.buffer));
    const st = after.state;
    const d = features.clearance(geometry, st.x, st.z, st.yaw, after.config.vehicleLength, after.config.vehicleWidth, 0.5);
    if (d < minClear) { minClear = d; minClearFrame = st.frame; }
    if (st.speed > maxSpeed) maxSpeed = st.speed;
  };
  const {report, variation} = await runCondition(c, teacher, {world: JSON.parse(JSON.stringify(world0)), onFrame});
  const m = report.metrics;
  const laps = (m.lap_events ?? []).map(l => ({lap: l.lap_index, elapsed_s: l.elapsed, gates: l.passed_gates,
    crossings: l.required_crossing_events.length, collisions: l.collisions, start_frame: l.start_frame, end_frame: l.end_frame}));
  const ref = refScenes[i];
  const refName = ref.name.replace(/^rc20-mode8-/, 'rlbest20-mode8-');
  const lapDiff = ref.laps.length === laps.length ? Math.max(...ref.laps.map((x, k) => Math.abs(x - laps[k].elapsed_s))) : null;
  scenes.push({index: i, name: c.name, split: c['dataset-split'], scenario: c.scenario, seed: c.seed, start_variation: variation,
    frames: report.trace.length, reason: report.outcome.reason, collisions: m.collisions, completed_laps: m.completed_laps,
    passed_gates: m.passed_gates, laps, every_lap_valid: laps.length >= 2 && laps.every(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0),
    min_clearance_m: minClear, min_clearance_frame: minClearFrame, max_speed_mps: maxSpeed,
    actions_sha256: hash.digest('hex'),
    rl_stage_reference: {name_matches: refName === c.name, laps: ref.laps, min_clearance: ref.min_clearance, max_lap_abs_diff_s: lapDiff}});
  process.stderr.write(`[予想] ${i + 1}/60 ${c.name} 周 ${laps.map(l => l.elapsed_s.toFixed(4)).join(' / ')} 衝突 ${m.collisions}\n`);
}
const lap1 = scenes.map(s => s.laps[0]?.elapsed_s).filter(Number.isFinite), lap2 = scenes.map(s => s.laps[1]?.elapsed_s).filter(Number.isFinite);
const stats = a => ({n: a.length, mean: a.reduce((x, y) => x + y, 0) / a.length, min: Math.min(...a), max: Math.max(...a)});
const out = {schema: 'rl-bestlap-recording-prediction/v1', created_at: new Date().toISOString(), plan: planPath, plan_sha256: sha(planPath),
  teacher_module: teacherPath, teacher_module_sha256: sha(teacherPath), sim_core_sha256: sha(path.join(SIM_ROOT, 'dist/sim-core.js')),
  core: 'original dist/sim-core.js (copy, byte-identical to the user dist)', wall_s: (Date.now() - t0) / 1000,
  summary: {scenes: scenes.length, all_600_frames: scenes.every(s => s.frames === 600),
    all_duration_complete: scenes.every(s => s.reason === 'collection_duration_complete'), collisions: scenes.reduce((x, s) => x + s.collisions, 0),
    all_laps_valid: scenes.every(s => s.every_lap_valid), laps_per_scene: [...new Set(scenes.map(s => s.completed_laps))],
    lap1: stats(lap1), lap2: stats(lap2), min_clearance_m: Math.min(...scenes.map(s => s.min_clearance_m)),
    rl_stage_max_lap_abs_diff_s: Math.max(...scenes.map(s => s.rl_stage_reference.max_lap_abs_diff_s ?? Infinity)),
    rl_stage_names_match: scenes.every(s => s.rl_stage_reference.name_matches)},
  scenes};
fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.summary, null, 1));
