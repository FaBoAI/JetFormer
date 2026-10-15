#!/usr/bin/env node
// DAgger の記録の Node の予想(sim_diverse_train_20261015、2026-10-15): 記録の計画(record_plan.json)の各シーンを、ブラウザと同じ差し込みモジュール
// (web/dagger-teacher.js・dagger_plan.mjs)と元の sim-core.js(lib/app_emulation.mjs の runCondition = app.js の連続試験と同じ手順)で走らせ、
// 600 フレーム・衝突 0・全部の周が有効(33 ゲート・横断 1 回)・本当のゲートの外し 0(gate_miss.mjs)・k までの位置が状態集めの走行と同じかを見る。
//   node predict_dagger.mjs <record_plan.json> <状態集めのフォルダー> <出力.json>
import fs from 'node:fs';
import path from 'node:path';
import {runCondition, SIM_ROOT} from '../lib/app_emulation.mjs';
import {realGateMisses} from '../web/gate_miss.mjs';

const [planPath, srcDir, outPath] = process.argv.slice(2);
const rc = await import(SIM_ROOT + '/dist/rc-speed-controller.js');
const {makeSlot, startKey} = await import('../web/dagger-teacher.js');
const {DAGGER_PLAN} = await import('../web/dagger_plan.mjs');
const slot = makeSlot({original: rc});
const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
const rows = [];
for (const c of plan) {
  const key = startKey({gate: c['start-gate'], left_offset_m: c['spawn-offset'], left_yaw_deg: c['spawn-yaw']});
  const sc = DAGGER_PLAN.scenes[key];
  const src = JSON.parse(fs.readFileSync(path.join(srcDir, sc.source_run + '.json'), 'utf8'));
  let f = 0, maxd = 0;
  const res = await runCondition(c, slot, {onFrame: (before, action, after) => {
    if (f < sc.k) { const t = src.trace[f]; maxd = Math.max(maxd, Math.abs(after.state.x - t.x), Math.abs(after.state.z - t.z)); }
    f++;
  }});
  const rep = res.report;
  const log = globalThis.__daggerLog[key].log;
  const gm = realGateMisses(log.map(q => q.s), log.map(q => q.g));
  const laps = (rep.metrics.lap_events ?? []).map(l => ({elapsed: +l.elapsed.toFixed(4), gates: l.passed_gates, crossings: (l.required_crossing_events ?? []).length, collisions: l.collisions}));
  const ok = rep.actions.length === 600 && rep.metrics.collisions === 0 && laps.length > 0 && laps.every(l => l.gates === 33 && l.crossings === 1 && !l.collisions) && gm.misses === 0 && maxd <= 1e-6;
  rows.push({name: c.name, key, k: sc.k, frames: rep.actions.length, reason: rep.outcome.reason, collisions: rep.metrics.collisions, laps, gate_misses: gm.misses,
    min_dmin: Math.min(...log.map(q => q.dmin)), max_pos_diff_before_k_m: maxd, ok});
  console.error(`${c.name}: k=${sc.k} ${rep.outcome.reason} 衝突 ${rep.metrics.collisions} 周 ${laps.map(l => l.elapsed).join('/')} 外し ${gm.misses} 差 ${maxd.toExponential(2)} → ${ok ? '合格' : '不合格'}`);
}
fs.writeFileSync(outPath, JSON.stringify({schema: 'jf-sim-diverse-dagger-predict/v1', created_at: new Date().toISOString(), plan_sha256: DAGGER_PLAN.plan_sha256,
  teacher: DAGGER_PLAN.teacher, all_ok: rows.every(r => r.ok), rows}, null, 1) + '\n');
console.log(JSON.stringify({all_ok: rows.every(r => r.ok), ok: rows.filter(r => r.ok).length, n: rows.length}));
