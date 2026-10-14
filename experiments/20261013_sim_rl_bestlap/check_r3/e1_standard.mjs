#!/usr/bin/env node
// 確認 r3 の 1: 取り違え・標準の物理での周(STD・REC60)・すき間・ゲート・黒シート・決定性・ユーザーの確認器・保存した記録の再生・同じ条件の比べ・荒さ
//   node check_r3/e1_standard.mjs
import fs from 'node:fs';
import {POLICY_FILES, REPORTS, PLAN, WORLD_VERSION, loadPolicy, drive, driveTeacher, hashJson, sha256File, verifier, BASE_WORLD,
  replayInflated, clearanceBounds, gateCrossings, sheetMargin, rough, save} from './common_r3.mjs';

const out = {schema: 'rl-check-r3-e1/v1', created_at: new Date().toISOString()};
const KEYS = ['selected', 'r2_5M', 'r3_s21', 'r3_s22'];
const P = Object.fromEntries(KEYS.map(k => [k, loadPolicy(k)]));

// 1) 取り違え: 書き出した policy.json と、学習の最後のチェックポイントの policy・provenance・meta
out.identity = {};
for (const s of [21, 22]) {
  const k = `r3_s${s}`, exp = REPORTS + `/rl/r3/r3-s${s}/policy.json`, src = REPORTS + `/rl/r3-s${s}/policies/step_005013504.json`;
  const prov = JSON.parse(fs.readFileSync(REPORTS + `/rl/r3/r3-s${s}/provenance.json`, 'utf8'));
  const run = JSON.parse(fs.readFileSync(REPORTS + `/rl/r3-s${s}/run.json`, 'utf8'));
  const done = JSON.parse(fs.readFileSync(REPORTS + `/rl/r3-s${s}/done.json`, 'utf8'));
  const step4 = sha256File(REPORTS + `/rl/r3-s${s}/policies/step_004014080.json`);
  out.identity[k] = {exported_sha: sha256File(exp), final_ckpt_policy_sha: sha256File(src), provenance_sha: prov.policy_json_sha256,
    same: sha256File(exp) === sha256File(src) && sha256File(src) === prov.policy_json_sha256, differs_from_4M: step4 !== sha256File(src),
    meta: P[k].meta, run: {seed: run.seed, started_at: run.started_at, dr: run.dr, reward: run.reward, reward_version: run.env.reward_version,
      init_ckpt: run.init_ckpt, init_ckpt_sha256: run.init_ckpt_sha256, dr_spec: run.env.dr_spec, clearance_terminate_m: run.env.reward.clearance_terminate_m},
    done, init_ckpt_sha_now: sha256File(run.init_ckpt)};
}
const sel = JSON.parse(fs.readFileSync(REPORTS + '/rl/selection.json', 'utf8'));
out.identity.selected_from_selection_json = sel.selected ?? sel.choice ?? null;

// 2) STD の単周・連続(決定性: 2 回)
const R1_SELECTED_STD_ACTIONS = '17c9dc00fd5caafb3b06314899685e3efa4960564293395e7ea648525b392e63';
out.std = {};
for (const k of KEYS) {
  const a = drive(P[k]), b = drive(P[k]), c = drive(P[k], {continuous: true});
  const g = gateCrossings(a), gc = gateCrossings(c);
  out.std[k] = {single: a.judge, continuous: c.judge, single_actions_sha: hashJson(a.report.actions), deterministic: hashJson(a.report.actions) === hashJson(b.report.actions),
    gate_lat_max_single: Math.max(...g.filter(x => x.lat <= .65).map(x => x.lat)), gate_misses_single: g.filter(x => x.lat > .65).length,
    gate_lat_max_cont: Math.max(...gc.filter(x => x.lat <= .65).map(x => x.lat)), gate_misses_cont: gc.filter(x => x.lat > .65).length,
    sheet_margin_single: sheetMargin(a), sheet_margin_cont: sheetMargin(c), rough_single: rough(a),
    max_speed: Math.max(...a.report.trace.map(t => t.speed)), max_lat_acc: a.report.metrics.max_lateral_acceleration};
  // 保存した記録(r3 の段の records)と同じ操作列か
  const recDir = k === 'selected' ? REPORTS + '/rl/final/records' : k === 'r2_5M' ? REPORTS + '/rl/r3/r2_5M/records' : REPORTS + `/rl/r3/${k.replace('_', '-')}/records`;
  for (const [mode, run] of [['single', a], ['continuous', c]]) {
    const f = recDir + `/rl_main_STD_${mode}.json`;
    if (!fs.existsSync(f)) { out.std[k][`record_${mode}`] = 'なし'; continue; }
    const rec = JSON.parse(fs.readFileSync(f, 'utf8'));
    out.std[k][`record_${mode}`] = {file: f, same_actions: hashJson(rec.actions) === hashJson(run.report.actions), elapsed: rec.metrics?.elapsed,
      laps: (rec.metrics?.lap_events ?? []).map(e => e.elapsed)};
  }
  console.log(k, JSON.stringify(out.std[k].single.laps), JSON.stringify(out.std[k].continuous.laps), out.std[k].single_actions_sha.slice(0, 12), out.std[k].deterministic,
    'gate', out.std[k].gate_lat_max_single.toFixed(3), 'sheet', out.std[k].sheet_margin_single.map(x => x.toFixed(3)), JSON.stringify(out.std[k].record_single?.same_actions));
}
out.std.selected.matches_r1_sha = out.std.selected.single_actions_sha === R1_SELECTED_STD_ACTIONS;
out.std.teacher = (t => ({single: t.judge}))(driveTeacher());

// 3) ユーザーの確認器: (a) 今走らせた STD 単周、(b) r3 の段が保存した記録をそのまま。物理の再生・現行コース・33 ゲート・黒シート・衝突の項目を見る
//    (RL の周には ONNX・センサーの証跡が無いので、その項目は必ず落ちる)
const physRe = /collision|route_complete|Finished|ordered gates|mandatory crossing|Mandatory crossing|world_vs_current_scene|benchmark physics|stopOnCollision|replay|Replay|Nominal start|Trace\/actions/i;
const isEvidence = x => /ONNX|model|Model|sensor|Sensor|inference|policy|RGB|ToF|longitudinal|Longitudinal|profile|controls/i.test(x) && !/replay\./.test(x);
function userCheck(rep, sha) {
  const v = verifier.verifyLearnedReport(rep, {expectedWorld: BASE_WORLD_ROTATED(rep), expectedSha256: sha});
  return {pass: v.pass, failure_count: v.failure_count, physics_failures: v.failures.filter(x => physRe.test(x) && !isEvidence(x)),
    evidence_failures_n: v.failures.filter(x => !(physRe.test(x) && !isEvidence(x))).length, replay: v.replay ?? null};
}
function BASE_WORLD_ROTATED(rep) { return rep.world; }   // STD はゲート 0(回していない)。world は scene.js から作った現行 v3 と同じかを別に比べる
out.user_verifier = {};
for (const k of ['r3_s21', 'r3_s22']) {
  const a = drive(P[k]), rep = a.report;
  rep.context = {controller: 'external', world_version: WORLD_VERSION, start_variation: {gate: 0, left_offset_m: 0, left_yaw_deg: 0}};
  const live = verifier.verifyLearnedReport(rep, {expectedWorld: BASE_WORLD, expectedSha256: P[k].sha});
  const rec = JSON.parse(fs.readFileSync(REPORTS + `/rl/r3/${k.replace('_', '-')}/records/rl_main_STD_single.json`, 'utf8'));
  const recW = rec.world ?? null;
  const recV = verifier.verifyLearnedReport({...rec, context: {...(rec.context ?? {}), controller: 'external', world_version: WORLD_VERSION,
    start_variation: {gate: 0, left_offset_m: 0, left_yaw_deg: 0}}}, {expectedWorld: BASE_WORLD, expectedSha256: P[k].sha});
  const f = v => ({pass: v.pass, failure_count: v.failure_count, physics_failures: v.failures.filter(x => physRe.test(x) && !isEvidence(x)), replay: v.replay ?? null});
  out.user_verifier[k] = {live: f(live), saved_record: f(recV), saved_record_has_world: !!recW};
  console.log('user verifier', k, 'live', live.failure_count, JSON.stringify(f(live).physics_failures), '| record', recV.failure_count, JSON.stringify(f(recV).physics_failures),
    JSON.stringify(live.replay?.outcome), live.replay?.elapsed);
}

// 4) 同じ条件の比べ(学習済み ONNX の記録の条件: 遅延 1 + 前進ゲイン 1.01)。観測の「前回の行動」の約束 2 通り
out.samecond = {};
for (const k of KEYS) {
  out.samecond[k] = {};
  for (const [lab, o] of [['delay1_applied', {delay: 1}], ['delay1_fresh', {delay: 1, prev: 'fresh'}],
    ['delay1_gain101_applied', {delay: 1, speedGain: 1.01}], ['delay1_gain101_fresh', {delay: 1, speedGain: 1.01, prev: 'fresh'}]]) {
    out.samecond[k][lab] = drive(P[k], o).judge;
  }
  console.log('samecond', k, JSON.stringify(Object.fromEntries(Object.entries(out.samecond[k]).map(([a, b]) => [a, b.ok ? +b.laps[0].toFixed(6) : b.reason]))));
}
const onnx = JSON.parse(fs.readFileSync('/home/jetson/sim/extracted/course-simulator/results/shortcut-rc-learned-lap.json', 'utf8'));
out.onnx_record = {elapsed: onnx.metrics.elapsed, delay_ticks: onnx.context?.execution?.control_delay?.ticks ?? null,
  throttle_gain: onnx.context?.policy_info?.calibration?.throttle_gain ?? null};

// 5) REC60(標準の物理・連続 20 秒)を遅延 0 と遅延 1(約束 2 通り)。すき間は太らせた車体の再生で
out.rec60 = {};
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
for (const k of KEYS) {
  const t0 = Date.now(), runs = PLAN.map(row => ({row, run: drive(P[k], {row, continuous: true})}));
  const ok = runs.filter(r => r.run.judge.ok);
  const below3 = runs.filter(r => replayInflated(r.run, .03).hit).map(r => r.row.name);
  const below5 = runs.filter(r => replayInflated(r.run, .05).hit).length;
  const bounds = clearanceBounds(runs.map(r => r.run));
  const lats = runs.flatMap(r => gateCrossings(r.run)), sheets = runs.flatMap(r => sheetMargin(r.run));
  const rg = runs.map(r => rough(r.run)), med = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
  const d1a = PLAN.filter(row => drive(P[k], {row, continuous: true, delay: 1}).judge.ok).length;
  const d1f = PLAN.filter(row => drive(P[k], {row, continuous: true, delay: 1, prev: 'fresh'}).judge.ok).length;
  const again = drive(P[k], {row: PLAN[0], continuous: true});
  out.rec60[k] = {valid: ok.length, n: runs.length, lap1_mean: mean(ok.map(r => r.run.judge.laps[0])), lap2_mean: mean(ok.map(r => r.run.judge.laps[1])),
    lap1_range: [Math.min(...ok.map(r => r.run.judge.laps[0])), Math.max(...ok.map(r => r.run.judge.laps[0]))],
    runs_hit_at_3cm: below3.length, runs_hit_at_3cm_names: below3, runs_hit_at_5cm: below5, clearance_bounds_all_runs: bounds,
    gate_lat_max: Math.max(...lats.filter(x => x.lat <= .65).map(x => x.lat)), gate_misses: lats.filter(x => x.lat > .65).length,
    sheet_crossings: sheets.length, sheet_margin_min: Math.min(...sheets),
    rough_p95_median: med(rg.map(x => x.p95)), rough_edge_frac: rg.reduce((a, x) => a + x.edge * x.n, 0) / rg.reduce((a, x) => a + x.n, 0),
    delay1_applied_valid: d1a, delay1_fresh_valid: d1f, deterministic_first: hashJson(again.report.actions) === hashJson(runs[0].run.report.actions),
    failures: runs.filter(r => !r.run.judge.ok).map(r => ({name: r.row.name, ...r.run.judge})), wall_s: (Date.now() - t0) / 1000};
  const S = out.rec60[k];
  console.log('rec60', k, `${S.valid}/${S.n}`, S.lap1_mean.toFixed(4), S.lap2_mean.toFixed(4), 'hit3cm', S.runs_hit_at_3cm, 'hit5cm', S.runs_hit_at_5cm,
    'clr', JSON.stringify(bounds), 'gate', S.gate_lat_max.toFixed(3), S.gate_misses, 'sheet', S.sheet_crossings, S.sheet_margin_min.toFixed(3),
    'rough', S.rough_p95_median.toFixed(2), S.rough_edge_frac.toFixed(3), 'd1', d1a, d1f, S.deterministic_first, S.wall_s);
  save('e1_standard.json', out);
}
// 教師の REC60(約束の中の開始だけ)
{
  const inC = r => r['start-gate'] === 0 && Math.abs(r['spawn-offset']) <= .03 + 1e-12 && Math.abs(r['spawn-yaw']) <= 3 + 1e-12;
  const runs = PLAN.filter(inC).map(row => driveTeacher({row, continuous: true}));
  const ok = runs.filter(r => r.judge.ok);
  out.rec60.teacher = {valid: ok.length, n: runs.length, lap1_mean: mean(ok.map(r => r.judge.laps[0])), lap2_mean: mean(ok.map(r => r.judge.laps[1]))};
  console.log('rec60 teacher', JSON.stringify(out.rec60.teacher));
}
save('e1_standard.json', out);
