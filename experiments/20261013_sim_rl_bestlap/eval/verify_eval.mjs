#!/usr/bin/env node
// 閉ループ評価の 1 走行ずつの確認と要約(2026-10-14、段 E)
//
//   node eval/verify_eval.mjs [--root <closed_loop のフォルダー>] [--out <要約.json>]
//
// ブラウザのアプリが連続試験で残した 1 走行の JSON(sim.report() + world + context + inference)ごとに:
//  1) rl/lap_audit.mjs の verifyReport(best_lap_definition.verification と同じ「同等の検証」):
//     元の sim-core.js で操作列を再生して軌跡・集計が一致(差 ≤ 1e-9)、自前の積分でゲート(順番・±0.65 m)・黒シートの車幅・
//     衝突・周回時間を数え直して一致。完走したかどうかは別に数える(requireSuccess: false で、ぶつかった回も記録の正しさだけを見る)
//  2) 要約: 終わり方・周回時間・ゲート・黒シート・衝突・最小すき間・ぶつかった場所・モデルの sha256・遅延・先読み・ゲイン・推論の時間
//  3) 軌跡の CSV(<走行>.trajectory.csv): フレーム・時刻・位置・向き・速さ・舵角・ONNX の要求(舵・速さ)・実際に使った操作
// ユーザーの qa/verify-learned-lap.mjs(E0 の単周)は Python の側(analyze_eval.py)から別のプロセスで呼ぶ。
import fs from 'node:fs';
import path from 'node:path';
import {verifyReport, independentAudit} from '../rl/lap_audit.mjs';

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; };
const root = path.resolve(opt('--root', '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/eval/closed_loop'));
const outPath = opt('--out', path.join(root, 'runs_verified.json'));
const only = opt('--only', null);

const median = v => { const s = [...v].sort((a, b) => a - b); return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null; };
const rows = [];
for (const label of fs.readdirSync(root).filter(d => fs.statSync(path.join(root, d)).isDirectory()).sort()) {
  if (only && !only.split(',').includes(label)) continue;
  const dir = path.join(root, label);
  for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.json') && !f.startsWith('plan_') && !f.includes('.verify')).sort()) {
    const p = path.join(dir, f);
    let r;
    try { r = JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) { rows.push({file: p, error: '読めない JSON: ' + e.message}); continue; }
    if (r.schema !== 'course-simulator-report/v1') continue;
    const v = await verifyReport(r, {requireSuccess: false, minClearance: 0});
    // ゲートを外した(線を越えたが |横| > 0.65 m)場所。sim は外したゲートを目標のまま残すので、次の周でそこを通り直すまでゲートが数えられない
    const misses = independentAudit(r).gate_misses.slice(0, 6).map(m => ({gate: m.gate, time: m.time, lateral_m: m.lateral}));
    // 記録の正しさ(再生・自前の積分が一致)と、周の有効さ(33 ゲート・黒シート 1 回)を分ける
    const consistencyFailures = v.failures.filter(f => !/周目/.test(f));
    const continuous = r.config.episodeMode === 'continuous-laps-v1';
    const inf = r.inference ?? [];
    const info = inf.find(x => x?.prediction?.info || x?.info)?.prediction?.info ?? inf.find(x => x?.info)?.info ?? {};
    const lapEvents = continuous ? (r.metrics.lap_events ?? []) : [];
    const laps = continuous ? lapEvents.map((l, i) => ({lap: i + 1, elapsed: l.elapsed, gates: l.passed_gates, collisions: l.collisions,
      crossings: (l.required_crossing_events ?? []).length, valid: !!(v.lap_checks[i] && v.lap_checks[i].gates_ok && v.lap_checks[i].crossing_exactly_once && !v.lap_checks[i].collisions)}))
      : (r.outcome.reason === 'route_complete' ? [{lap: 1, elapsed: r.metrics.elapsed, gates: r.metrics.passed_gates, collisions: r.metrics.collisions,
        crossings: (r.metrics.required_crossing_events ?? []).length, valid: v.lap_checks.length === 1 && v.lap_checks[0].gates_ok && v.lap_checks[0].crossing_exactly_once && !v.lap_checks[0].collisions}] : []);
    const ce = (r.metrics.collision_events ?? [])[0] ?? null;
    const last = r.trace?.at(-1) ?? null;
    const completed = continuous ? (r.outcome.reason === 'collection_duration_complete' && r.metrics.collisions === 0)
      : (r.outcome.reason === 'route_complete' && r.metrics.collisions === 0);
    const lat = inf.map(x => x?.wall_latency_ms).filter(Number.isFinite), ort = inf.map(x => (x?.prediction?.info ?? x?.info)?.ort_inference_ms).filter(Number.isFinite);
    const speeds = (r.trace ?? []).map(t => t.speed);
    const row = {
      name: r.name, label, file: p, batch_status: r.batch_status, episode_mode: continuous ? 'continuous-laps-v1' : 'single-lap',
      start_variation: r.context?.start_variation, reason: r.outcome.reason, success: r.outcome.success, completed,
      sim_time: r.final_state?.sim_time ?? last?.time ?? null, frames: (r.actions ?? []).length,
      single_lap_s: !continuous && completed && laps[0]?.valid ? laps[0].elapsed : null,
      laps, flying_lap_s: continuous && laps[1]?.valid ? laps[1].elapsed : null, lap1_s: laps[0]?.valid ? laps[0].elapsed : null,
      passed_gates: r.metrics.passed_gates, route_index: r.metrics.route_index, collisions: r.metrics.collisions,
      crossings: (r.metrics.required_crossing_events ?? []).length, crossings_valid_own: v.crossings_valid,
      collision_event: ce ? {time: ce.time ?? ce.sim_time ?? null, x: ce.x ?? null, z: ce.z ?? null, obstacle: ce.obstacle ?? ce.id ?? ce.kind ?? null, raw: ce} : null,
      gates_at_end: r.metrics.passed_gates, end_pose: last ? {x: last.x, z: last.z, yaw: last.yaw, speed: last.speed} : null,
      min_clearance_m: v.min_clearance_m, gate_margin_min_m: v.gate_margin_min_m, crossing_margin_min_m: v.crossing_margin_min_m,
      max_speed: speeds.length ? Math.max(...speeds) : null, mean_speed: speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null,
      distance_m: r.metrics.distance ?? null,
      verification: {pass: v.pass, consistency_pass: consistencyFailures.length === 0, failures: v.failures.slice(0, 10), max_abs_error: v.max_abs_error,
        laps_own: v.laps_own.map(l => l.elapsed), lap_checks: v.lap_checks}, gate_misses: misses, gate_miss_count: v.gate_misses,
      model_sha256: r.context?.policy_info?.model_sha256 ?? null, policy: r.context?.policy ?? null,
      mode_id: info.mode_id ?? null, lookahead: info.lookahead ?? null,
      steer_gain: r.context?.policy_info?.calibration?.steer_gain ?? r.context?.policy_info?.steer_gain ?? null,
      throttle_gain: r.context?.policy_info?.calibration?.throttle_gain ?? r.context?.policy_info?.throttle_gain ?? null,
      control_delay_ticks: r.context?.execution?.control_delay?.ticks ?? 0, sensor_profile: r.context?.sensor_profile ?? null,
      config: {timeLimit: r.config.timeLimit, maxSpeed: r.config.maxSpeed, longitudinalModel: r.config.longitudinalModel, noiseStd: r.config.noiseStd},
      latency_ms_median: median(lat), ort_ms_median: median(ort),
    };
    rows.push(row);
    // 軌跡の CSV(図と報告の元)
    const lines = ['frame,time_s,x_m,z_m,yaw_rad,speed_mps,steering_rad,applied_steer_rad,applied_speed_mps,pred_steer_rad,pred_speed_mps'];
    (r.trace ?? []).forEach((t, i) => {
      const a = r.actions?.[i] ?? {}, q = inf[i]?.prediction?.requested ?? inf[i]?.requested ?? {};
      lines.push([t.frame, t.time, t.x, t.z, t.yaw, t.speed, t.steering, a.steering, a.speed, q.steering, q.speed].join(','));
    });
    fs.writeFileSync(p.replace(/\.json$/, '.trajectory.csv'), lines.join('\n') + '\n');
    console.error(`${r.name}: ${row.reason} ${row.single_lap_s ?? row.flying_lap_s ?? ''} gates=${row.passed_gates} coll=${row.collisions} verify=${v.pass}`);
  }
}
fs.writeFileSync(outPath, JSON.stringify({schema: 'sim-rl-bestlap-closed-loop-runs/v1', created_at: new Date().toISOString(), root, rows}, null, 1) + '\n');
console.log(outPath);
