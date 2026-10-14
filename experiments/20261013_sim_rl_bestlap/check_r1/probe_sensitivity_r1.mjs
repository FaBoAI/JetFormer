// 感度の掘り下げ: 舵の速さ・ブレーキを細かく振り、ぶつかる場所を見る(確認の段 r1)
import {loadOwnPolicy, runClosedLoop, runTeacher, PLAN60, writeJson} from './common_r1.mjs';
const policy = loadOwnPolicy();
const rows = [];
const where = r => { const e = r.report?.metrics?.collision_events?.[0]; return e ? {t: +e.time.toFixed(3), collider: e.collider, kind: e.kind, x: +e.x.toFixed(3), z: +e.z.toFixed(3), gates_before: r.report.metrics.passed_gates} : null; };
for (const [key, vals] of [['steerRate', [.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6]], ['brakeDeceleration', [.8, .9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]]]) {
  for (const v of vals) {
    const row = {param: key, value: v};
    for (const mode of ['single', 'continuous']) {
      const r = runClosedLoop(policy, {episodeMode: mode, simOverrides: {[key]: v}});
      row[mode] = {valid: r.summary.valid, reason: r.summary.reason, laps: r.summary.laps.map(l => +l.elapsed.toFixed(4)), collision: where(r)};
      try { const t = runTeacher({episodeMode: mode, simOverrides: {[key]: v}}); row['teacher_' + mode] = {valid: t.summary.valid, reason: t.summary.reason, laps: t.summary.laps.map(l => +l.elapsed.toFixed(4))}; }
      catch (e) { row['teacher_' + mode] = {error: e.message}; }
    }
    rows.push(row);
    console.error(key, v, 'RL', row.single.reason, row.single.laps, JSON.stringify(row.single.collision), '|', row.continuous.reason, row.continuous.laps, JSON.stringify(row.continuous.collision),
      '|| T', row.teacher_single.reason, row.teacher_single.laps, '|', row.teacher_continuous.reason, row.teacher_continuous.laps);
  }
}
// 遅延 1 の REC60 で落ちた条件
const d1 = [];
for (const c of PLAN60) {
  const r = runClosedLoop(policy, {plan: c, gate: c['start-gate'], offset: c['spawn-offset'], yawDeg: c['spawn-yaw'], episodeMode: 'continuous', delay: 1});
  if (!r.summary.valid) d1.push({name: c.name, scenario: c.scenario, offset: c['spawn-offset'], yaw: c['spawn-yaw'], reason: r.summary.reason, laps: r.summary.laps.map(l => +l.elapsed.toFixed(4)), collision: where(r)});
}
console.error('delay1 REC60 failures', JSON.stringify(d1));
writeJson('sensitivity.json', {schema: 'rl-check-r1-sensitivity/v1', rows, rec60_delay1_failures: d1});
