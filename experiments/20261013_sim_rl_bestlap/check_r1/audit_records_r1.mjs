// 保存された周の記録(rl/final/records)を、RL のコードを使わずに確かめる(2026-10-13、確認の段 r1)
//
//   node check_r1/audit_records_r1.mjs
//
// 記録ごとに:
//   1) コース: 記録の world が、ユーザーの scene.js から作った現行 v3(qa/load-course-world.mjs)を app.js の式で回したものと同じか
//   2) 設定: 記録の config が、20 分データの収集の計画(qa/rc-collection-plan-60.json)を app.js の式で設定にしたものと同じか
//   3) 再生: 現行コース + 記録の操作列を **元の dist/sim-core.js** で再生し、軌跡・集計が記録と一致するか
//   4) ユーザーの qa/verify-learned-lap.mjs の verifyLearnedReport: 単周の記録で、どの項目が落ちるか(ONNX の証跡など、RL には当てはまらない項目を分ける)
//   5) 自前の積分(own_physics_r1.mjs)で再生し、sim と軌跡が一致するかと、小区間ごとの判定(すき間・ゲートの横・黒シートの実際の幅・路面・横の頭打ち)
//   6) 自前の積分で小区間を 8 分の 1 にして同じ操作列を再生(開ループ。離散化に頼った走りなら、ここで外れる)
import fs from 'node:fs';
import {core, userVerifier, SCENE_WORLD, STD_PLAN, browserConfig, activeWorld, REPORTS, writeJson, sha256File, POLICY_JSON} from './common_r1.mjs';
import {createOwnSim} from './own_physics_r1.mjs';

const dir = REPORTS + '/rl/final/records';
const files = fs.readdirSync(dir).filter(f => f.endsWith('.json')).sort();
const policySha = sha256File(POLICY_JSON);

function deepDiff(a, b, path = '', out = {max: 0, mismatches: []}) {
  if (typeof b === 'number') {
    const d = typeof a === 'number' ? Math.abs(a - b) : Infinity;
    if (!(d <= 1e-9) && out.mismatches.length < 10) out.mismatches.push(path);
    out.max = Math.max(out.max, Number.isFinite(d) ? d : Infinity);
  } else if (b && typeof b === 'object') {
    if (!a || typeof a !== 'object') { out.mismatches.push(path + ' (型)'); out.max = Infinity; return out; }
    const ka = Object.keys(a).sort().join(), kb = Object.keys(b).sort().join();
    if (ka !== kb) { out.mismatches.push(path + ' (キー)'); out.max = Infinity; return out; }
    for (const k of Object.keys(b)) deepDiff(a[k], b[k], path + '.' + k, out);
  } else if (a !== b) { out.mismatches.push(path + ' (値)'); out.max = Infinity; }
  return out;
}
// ユーザーの確認器の落ちた項目を分ける: RL の記録に当てはまらない証跡(ONNX・センサー・推論の記録・縦の応答の宣言)か、物理・判定か
function classify(msg) {
  const meta = [/schema/, /Controller is not external ONNX/, /SHA-256/, /Independent model identity/, /JetFormer ONNX/, /sensor profile/i,
    /RGB input/, /ToF dimensions/, /Missing sensor/, /longitudinal_contract/, /sensor_contract/, /inference/i, /ONNX/, /Ego body/, /control_delay/];
  return meta.some(r => r.test(msg)) ? 'metadata' : 'physics_or_judgement';
}

const rows = [];
for (const f of files) {
  const rec = JSON.parse(fs.readFileSync(`${dir}/${f}`, 'utf8'));
  const sv = rec.context?.start_variation ?? {gate: 0, left_offset_m: 0, left_yaw_deg: 0};
  const continuous = rec.config.episodeMode === 'continuous-laps-v1';
  const row = {file: f, continuous, start: sv, frames: rec.actions.length};
  // 1) コース
  const expectedWorld = activeWorld(SCENE_WORLD, sv.gate, sv.left_offset_m, sv.left_yaw_deg);
  const wd = deepDiff(rec.world, expectedWorld);
  row.world_matches_scene = wd.max <= 1e-9; row.world_mismatches = wd.mismatches;
  // 2) 設定(計画の標準の条件 + 記録の種)。sim の正規化の後で比べる
  const expectedCfg = core.createSimulation(expectedWorld, browserConfig({...STD_PLAN, seed: rec.seed}, {episodeMode: continuous ? 'continuous' : 'single'})).report().config;
  const cd = deepDiff({...rec.config, timeLimit: 0}, {...expectedCfg, timeLimit: 0});
  row.config_matches_browser = cd.max <= 1e-9; row.config_mismatches = cd.mismatches; row.timeLimit = rec.config.timeLimit;
  // 3) 元の sim-core で再生(コースは現行の scene から作ったもの。記録の world は使わない)
  const sim = core.createSimulation(expectedWorld, rec.config);
  sim.reset({seed: rec.seed, scenario: rec.scenario});
  for (const a of rec.actions) sim.step(a);
  const rep = sim.report();
  const rd = {max: 0, mismatches: []};
  for (const k of ['world_hash', 'outcome', 'metrics', 'final_state', 'actions', 'trace']) deepDiff(rep[k], rec[k], k, rd);
  row.replay_max_diff = rd.max; row.replay_mismatches = rd.mismatches;
  const m = rep.metrics;
  row.sim = {reason: rep.outcome.reason, success: rep.outcome.success, elapsed: m.elapsed, passed_gates: m.passed_gates, collisions: m.collisions,
    crossings: m.required_crossing_events.length, laps: (m.lap_events ?? []).map(l => ({elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions}))};
  // 4) ユーザーの確認器(単周だけが対象。連続は対象外と書いてある)
  if (!continuous) {
    const v = userVerifier.verifyLearnedReport(rec, {expectedWorld, expectedSha256: policySha});
    const failures = v.failures;
    row.user_verifier = {pass: v.pass, failure_count: v.failure_count,
      metadata_failures: failures.filter(x => classify(x) === 'metadata').length,
      physics_failures: failures.filter(x => classify(x) === 'physics_or_judgement'),
      replay: v.replay ?? null, sample_metadata_failures: failures.filter(x => classify(x) === 'metadata').slice(0, 6)};
  }
  // 5) 自前の積分(小区間は sim と同じ)
  for (const refine of [1, 8]) {
    const own = createOwnSim(expectedWorld, rec.config, {refine, interp: 6});
    for (const a of rec.actions) own.step(a);
    const r = own.result();
    let traceDiff = 0;
    if (refine === 1) for (let i = 0; i < Math.min(r.trace.length, rec.trace.length); i++) {
      const p = r.trace[i], q = rec.trace[i];
      traceDiff = Math.max(traceDiff, Math.abs(p.x - q.x), Math.abs(p.z - q.z), Math.abs(wrapd(p.yaw - q.yaw)), Math.abs(p.speed - q.speed), Math.abs(p.time - q.time));
    }
    const gl = r.gate_events.filter(g => g.ok).map(g => Math.abs(g.lateral));
    const out = {reason: r.reason, completed_laps: r.completed_laps, laps: r.lap_events.map(l => +l.elapsed.toFixed(6)), passed_gates: r.passed_gates,
      collisions: r.collisions_own.length, first_collision: r.collisions_own[0] ?? null,
      min_clearance_substep_m: r.stats.minClear, min_clearance_who: r.stats.minClearWho, min_clearance_interp_m: Math.min(r.stats.minClearInterp, r.stats.minClear),
      gate_missed_lines: r.gate_events.filter(g => !g.ok).length, gate_lateral_max_m: Math.max(...gl), gate_lateral_over_055: gl.filter(x => x > .55).length,
      gate_edge_min_margin_m: .65 - Math.max(...gl),
      sheet_events: r.sheet_events.map(e => ({lap: e.lap, dir: e.dir, ok: e.sim_rule_ok, x: +e.x.toFixed(4), margin_m: +e.margin_min_m.toFixed(4), slice_margin_m: e.slice_margin_min_m === null ? null : +e.slice_margin_min_m.toFixed(4)})),
      traction_limited_frac: r.stats.traction_limited_frac, max_lat_acc: r.stats.maxLatAcc, max_speed: r.stats.maxSpeed,
      surface_time_center_s: r.stats.surfaceTimeCenter, surface_time_area_weighted_s: r.stats.surfaceTimeArea, ptfe_center_off_but_body_over_25pct_s: r.stats.ptfeCenterOff_areaOn};
    if (refine === 1) { out.trace_max_diff_vs_sim = traceDiff; row.own = out; }
    else row.own_refine8_openloop = out;
  }
  rows.push(row);
  console.error(f, 'world', row.world_matches_scene, 'cfg', row.config_matches_browser, 'replay', row.replay_max_diff, 'own trace', row.own.trace_max_diff_vs_sim,
    'own', row.own.reason, row.own.laps, 'clr', row.own.min_clearance_interp_m.toFixed(4), 'r8', row.own_refine8_openloop.reason, row.own_refine8_openloop.laps,
    row.user_verifier ? `user: pass=${row.user_verifier.pass} meta=${row.user_verifier.metadata_failures} phys=${row.user_verifier.physics_failures.length}` : '');
}
function wrapd(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }
writeJson('audit_records.json', {schema: 'rl-check-r1-audit-records/v1', created_at: new Date().toISOString(), policy_json_sha256: policySha, rows});
