#!/usr/bin/env node
// 準備の段: 記録の前に、30 シーンをブラウザと同じ差し込みモジュール(dist/dart/entry-Dn.js)+ 元の sim-core.js で Node で走らせて
// 「予想」を作り、乱しの種の引き直し(k)を決める(2026-10-14、sim_dart_20261014。protocol.json dart.recoverability.prefilter_node)
//
//   node predict_dart.mjs --dataset D1 [--plan plan_D1.json] [--out prediction_D1.json] [--write-table] [--scenes 0,1]
//
//  - 各シーン: まず乱しなし(同じ先生)で走らせ、次に k = 0 から乱しありで走らせる。衝突・20 秒走り切れない・本当のゲートの外し
//    (modes の rl/gate_miss.mjs の決まり)・1 周目が有効でない(33 ゲート・黒シートの横断 1 回・衝突 0)のどれかがあれば k を 1 つ進める
//    (k = 1..5)。5 回でだめなら乱しの大きさ × 0.7 で k = 0..5 をもう一度(deviations.md に書く)
//  - --write-table: 決めた k を web/plan_Dn.mjs(K_TABLE)に書く(そのあと install_dart.py をもう一度走らせて dist に写す)
//  - 乱しごとの最大のずれ(同じ開始・乱しなしの先生の走りからの横ずれ。R 上の s で合わせる)を、設計の段の measure_teachers.mjs と同じ式で測る
//  - 乱しなしの確かめ: D1 は乱しなしの最初の 2 シーンの操作列の sha256 が A0 の予想(ベストラップの recording/prediction.json)と同じ、
//    D2・D3 は単周 E0 の周回時間が設計の段の測り(design_probe/clearance.json の line_l3 E0 = 8.924 秒)と同じ
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runCondition, baseWorld, activeWorldFor, configFromCondition, createSimulation, SIM_ROOT} from '../lib/app_emulation.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261014_sim_dart_margin';
const REPORT = '/home/jetson/jetformer_reports/sim_dart_20261014';
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i < 0 ? def : args[i + 1]; };
const D = opt('--dataset');
if (!['D1', 'D2', 'D3'].includes(D)) throw new Error('--dataset は D1・D2・D3');
const planPath = opt('--plan', `${REPORT}/record_prep/plan_${D}.json`);
const outPath = opt('--out', `${REPORT}/record_prep/prediction_${D}.json`);
const writeTable = args.includes('--write-table');
const only = opt('--scenes', null)?.split(',').map(Number) ?? null;
const DIST = path.join(SIM_ROOT, 'dist/dart');
const imp = p => import(pathToFileURL(path.join(DIST, p)).href);
const entry = await imp(`entry-${D}.js`);
const {createDartDriver} = await imp('dart_driver.mjs');
const {realGateMisses} = await imp('gate_miss.mjs');
const {DART_MANIFEST} = await imp('dart_manifest.mjs');
const DM = DART_MANIFEST.datasets[D];
const W = (await imp(DM.kind === 'fast' ? 'rt/weights_fast.mjs' : 'rt/weights_margin.mjs')).default;
const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const world0 = await baseWorld();

// 1 条件を走らせ、フレームごとの記録を返す。override: {k, scale}(乱しありのとき)。clean: true なら乱しなしの同じ先生
async function runScene(c, {override = null, clean = false} = {}) {
  globalThis.__dartOverride = override ? {[c.seed >>> 0]: override} : {};
  const gArr = [], hash = crypto.createHash('sha256'), buf = new Float64Array(2);
  let drv = null;
  const mod = clean ? {createRcTrainingTeacher: (w, cfg, v) => {
    const actual = entry.assertRcTeacherContract(w, cfg, v);
    drv = createDartDriver({kind: DM.kind, world: w, config: actual, weights: W, noise: null});
    return {action: drv.action};
  }} : entry;
  const onFrame = (before, action) => {
    gArr.push(`${before.metrics.completed_laps ?? 0}:${before.metrics.passed_gates ?? 0}`);
    buf[0] = action.steering; buf[1] = action.speed; hash.update(Buffer.from(buf.buffer));
  };
  const {report} = await runCondition(c, mod, {world: JSON.parse(JSON.stringify(world0)), onFrame});
  const L = clean ? {log: drv.log, events: []} : globalThis.__dartLog[c.seed >>> 0];
  const log = L.log, m = report.metrics;
  const gm = realGateMisses(log.map(q => q.s), gArr);
  const laps = (m.lap_events ?? []).map(l => ({elapsed_s: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions}));
  const lap1 = laps[0] ?? null;
  const ok = m.collisions === 0 && report.outcome.reason === 'collection_duration_complete' && report.trace.length === 600 && gm.misses === 0
    && !!lap1 && lap1.gates === 33 && lap1.crossings === 1 && lap1.collisions === 0;
  return {ok, reason: report.outcome.reason, frames: report.trace.length, collisions: m.collisions,
    collision: m.collision_events?.[0] ? {t: m.collision_events[0].time, collider: m.collision_events[0].collider} : null,
    gate_misses: gm.misses, gate_stuck_max_m: gm.stuck_max_m, laps, log, events: L.events.map(e => ({...e})), actions_sha256: hash.digest('hex'),
    min_clearance_m: Math.min(...log.map(q => q.dmin)), noisy_frames: log.filter(q => q.noisy).length};
}
// 乱しなしの走りとの横ずれ(R 上の s で合わせた e_y の差)。measure_teachers.mjs の deviationFromClean と同じ
function deviationFromClean(rows, clean) {
  const cs = clean.map(q => q.s), ce = clean.map(q => q.ey);
  let j = 0; const out = [];
  for (const q of rows) {
    while (j < cs.length - 2 && cs[j + 1] < q.s) j++;
    if (q.s < cs[0] || q.s > cs.at(-1)) { out.push(null); continue; }
    const fr = (q.s - cs[j]) / Math.max(1e-9, cs[j + 1] - cs[j]);
    out.push(Math.abs(q.ey - (ce[j] + (ce[j + 1] - ce[j]) * Math.min(1, Math.max(0, fr)))));
  }
  return out;
}
const pct = (a, p) => { const b = a.filter(x => x != null).sort((x, y) => x - y); return b.length ? b[Math.min(b.length - 1, Math.floor(p * b.length))] : null; };
const med = a => pct(a, .5);

const t0 = Date.now();
const scenes = [], table = {};
const idx = only ?? plan.map((_, i) => i);
for (const i of idx) {
  const c = plan[i];
  const clean = await runScene(c, {clean: true});
  const rec = {index: i, name: c.name, seed: c.seed, scenario: c.scenario, split: c['dataset-split'],
    clean: {ok: clean.ok, reason: clean.reason, collisions: clean.collisions, gate_misses: clean.gate_misses, laps: clean.laps,
      min_clearance_m: clean.min_clearance_m, actions_sha256: clean.actions_sha256}};
  if (DM.noise) {
    const tries = [];
    let chosen = null;
    for (const scale of [1, .7]) {
      for (let k = 0; k <= 5 && !chosen; k++) {
        const r = await runScene(c, {override: {k, scale}});
        tries.push({k, scale, ok: r.ok, reason: r.reason, collisions: r.collisions, collision: r.collision, gate_misses: r.gate_misses,
          lap1: r.laps[0] ?? null, n_events: r.events.length});
        if (r.ok) chosen = {k, scale, r};
      }
      if (chosen) break;
    }
    // ブラウザの取り直し用に、選んだ k のあとで Node で通る k を 2 つまで(同じ大きさ)探しておく(protocol: ブラウザで出たら次の k で取り直す、2 回まで)
    const alt = [];
    if (chosen) for (let k = chosen.k + 1; k <= chosen.k + 8 && alt.length < 2; k++) {
      const r = await runScene(c, {override: {k, scale: chosen.scale}});
      tries.push({k, scale: chosen.scale, ok: r.ok, reason: r.reason, collisions: r.collisions, collision: r.collision, gate_misses: r.gate_misses,
        lap1: r.laps[0] ?? null, n_events: r.events.length, alt_search: true});
      if (r.ok) alt.push(k);
    }
    rec.tries = tries;
    rec.alt_k = alt;
    if (chosen) {
      const r = chosen.r;
      const dev = deviationFromClean(r.log, clean.log);
      const evMax = r.events.map(e => { let m = 0; for (let f = e.start; f < Math.min(r.log.length, e.end + 45); f++) if (dev[f] != null) m = Math.max(m, dev[f]); return m; });
      const fs0 = r.log.filter(q => q.f < 60 && c['start-gate'] === 0).map(q => dev[q.f]).filter(x => x != null);
      rec.chosen = {k: chosen.k, scale: chosen.scale, laps: r.laps, min_clearance_m: r.min_clearance_m, noisy_frac: r.noisy_frames / r.log.length,
        n_events: r.events.length, aborted_events: r.events.filter(e => e.aborted != null).length, ev_max_p50: med(evMax), ev_max_p90: pct(evMax, .9),
        ev_frac_5_25cm: evMax.length ? evMax.filter(x => x >= .05 && x <= .25).length / evMax.length : null,
        first_straight_dev_max: fs0.length ? Math.max(...fs0) : null, dev_p50: med(dev), dev_max: pct(dev, 1), first_event_frame: r.events[0]?.start ?? null,
        events: r.events, actions_sha256: r.actions_sha256, ev_max_list: evMax};
      table[c.seed >>> 0] = {k: chosen.k, scale: chosen.scale, alt: alt};
    } else {
      rec.chosen = null;
    }
  }
  scenes.push(rec);
  const ch = rec.chosen;
  process.stderr.write(`[予想 ${D}] ${scenes.length}/${idx.length} ${c.name} きれい ${clean.ok ? 'OK' : 'NG'} 周 ${clean.laps.map(l => l.elapsed_s.toFixed(3)).join('/')}`
    + (DM.noise ? ` → 乱し k=${ch?.k ?? '×'} scale=${ch?.scale ?? '-'} 試し ${rec.tries.length} ずれ p50 ${ch?.ev_max_p50?.toFixed(3)} p90 ${ch?.ev_max_p90?.toFixed(3)} 乱し ${(100 * (ch?.noisy_frac ?? 0)).toFixed(1)}%` : '') + '\n');
}

// 乱しなしの確かめ
const checks = {};
if (D === 'D1') {
  const pred = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/recording/prediction.json', 'utf8'));
  const a0 = pred.scenes ?? pred;
  checks.clean_equals_A0_prediction = scenes.filter(s => s.index < 2).map(s => ({name: s.name, sha: s.clean.actions_sha256,
    a0_sha: a0[s.index].actions_sha256, same: s.clean.actions_sha256 === a0[s.index].actions_sha256}));
} else {
  // 単周 E0(ゲート 0・ずれ 0)の周回時間
  const c = {...plan[0], 'episode-mode': 'single-lap', limit: 15, 'spawn-offset': 0, 'spawn-yaw': 0};
  const {config, scenario} = configFromCondition(c);
  const {world} = activeWorldFor(JSON.parse(JSON.stringify(world0)), 0, 0, 0);
  const sim = createSimulation(world, config); sim.reset({seed: config.seed, scenario});
  const drv = createDartDriver({kind: 'margin', world, config: sim.snapshot().config, weights: W, noise: null});
  while (!sim.snapshot().state.terminal) sim.step(drv.action(sim.snapshot()));
  const rep = sim.report();
  checks.single_lap_E0 = {reason: rep.outcome.reason, elapsed_s: rep.metrics.elapsed, collisions: rep.metrics.collisions, design_measure_s: 8.924,
    same_to_1ms: rep.outcome.reason === 'route_complete' && Math.abs(rep.metrics.elapsed - 8.924) < 5e-4};
}
const ok = scenes.filter(s => DM.noise ? !!s.chosen : s.clean.ok);
const ch = scenes.map(s => s.chosen).filter(Boolean);
const summary = {dataset: D, teacher: DM.kind, noise: DM.noise, salt: DM.salt, scenes: scenes.length, ok: ok.length,
  clean_ok: scenes.filter(s => s.clean.ok).length, clean_lap1_mean_s: scenes.reduce((a, s) => a + (s.clean.laps[0]?.elapsed_s ?? 0), 0) / scenes.length,
  clean_min_clearance_m: Math.min(...scenes.map(s => s.clean.min_clearance_m))};
if (DM.noise) Object.assign(summary, {
  k_hist: ch.reduce((a, q) => { const key = `k${q.k}_s${q.scale}`; a[key] = (a[key] ?? 0) + 1; return a; }, {}),
  first_try_ok: scenes.filter(s => s.tries[0]?.ok).length, tries_total: scenes.reduce((a, s) => a + s.tries.filter(t => !t.alt_search).length, 0),
  noisy_frac_mean: ch.reduce((a, q) => a + q.noisy_frac, 0) / Math.max(1, ch.length),
  events_total: ch.reduce((a, q) => a + q.n_events, 0), aborted_total: ch.reduce((a, q) => a + q.aborted_events, 0),
  ev_max_all_p50: med(ch.flatMap(q => q.ev_max_list)), ev_max_all_p90: pct(ch.flatMap(q => q.ev_max_list), .9),
  ev_frac_5_25cm_all: (() => { const a = ch.flatMap(q => q.ev_max_list); return a.filter(x => x >= .05 && x <= .25).length / Math.max(1, a.length); })(),
  first_straight_dev_max_p50: med(ch.map(q => q.first_straight_dev_max)), min_clearance_m: Math.min(...ch.map(q => q.min_clearance_m)),
  lap1_mean_s: ch.reduce((a, q) => a + (q.laps[0]?.elapsed_s ?? 0), 0) / Math.max(1, ch.length),
  first_event_frame_max: Math.max(...ch.map(q => q.first_event_frame ?? 1e9))});
const out = {schema: 'jf-sim-dart-prediction/v1', made: new Date().toISOString(), wall_s: (Date.now() - t0) / 1000, dataset: D, plan: planPath,
  plan_sha256: sha(planPath), entry_sha256: sha(path.join(DIST, `entry-${D}.js`)), manifest_sha256: sha(path.join(DIST, 'dart_manifest.mjs')),
  summary, checks, k_table: table, scenes: scenes.map(s => ({...s, chosen: s.chosen ? {...s.chosen, ev_max_list: undefined} : s.chosen}))};
fs.mkdirSync(path.dirname(outPath), {recursive: true});
fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
if (writeTable && DM.noise) {
  if (only) throw new Error('--scenes を付けたときは表を書きません');
  const missing = plan.filter(c => !table[c.seed >>> 0]).map(c => c.name);
  const body = `// 記録 ${D} の乱しの引き直しの表(record/predict_dart.mjs が ${out.made} に書いた。手で変えない)。{シーンの種: {k, scale}}。\n`
    + `// 予想: ${outPath}(sha256 は install.json)。予想で通らなかったシーン: ${missing.length ? missing.join(', ') : 'なし'}\n`
    + `export const K_TABLE = Object.freeze(${JSON.stringify(table, null, 1)});\n`;
  fs.writeFileSync(`${EXP}/web/plan_${D}.mjs`, body);
}
console.log(JSON.stringify({summary, checks}, null, 1));
