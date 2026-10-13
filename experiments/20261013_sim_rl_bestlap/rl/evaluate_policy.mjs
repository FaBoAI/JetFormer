#!/usr/bin/env node
// 方策を REC60・STD・ROB24 で評価して、選ぶ条件(eligible)と点数(score)を出す(2026-10-13、RL 環境の段で用意)
//
//   node rl/evaluate_policy.mjs --policy <方策のモジュール.mjs> [--policy-arg 値] [--core fast|original] [--variant main|open]
//        [--sets REC60,STD,ROB24] [--record-dir フォルダー] [--verify] --out 結果.json
//   node rl/evaluate_policy.mjs --teacher [--core original] --out 結果.json      (教師を同じ形で評価する。ROB24 はゲート 0 以外で走れない)
//
// 方策のモジュールは次を export する(PPO の段の policy_forward.mjs など):
//   export async function loadPolicy(arg) → (obs: Float64Array(79), env) => [a0, a1]
//   (obs は features.mjs の観測。移動平均での正規化などは方策の側で行う。決定的であること)
// --record-dir を付けると STD(単周・連続)と REC60 の 1 シーン目のレポートを保存し、--verify で rl/verify_lap.mjs と同じ確認をする。
// 判定は protocol.json selection.eligible のとおり: REC60 60/60 有効・最小すき間 ≥ 0.03 m・STD の単周が完走・ROB24 22/24 以上。
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {evaluateController, policyController, teacherController, runEpisode, STD} from './eval_sets.mjs';
import {verifyReport} from './lap_audit.mjs';

const args = process.argv.slice(2);
const opt = (name, fallback = null) => { const i = args.indexOf(name); if (i < 0) return fallback; const v = args[i + 1]; args.splice(i, 2); return v; };
const flag = name => { const i = args.indexOf(name); if (i < 0) return false; args.splice(i, 1); return true; };
const policyPath = opt('--policy'), policyArg = opt('--policy-arg'), coreKind = opt('--core', 'fast'), variant = opt('--variant', 'main');
const sets = opt('--sets', 'REC60,STD,ROB24').split(','), recordDir = opt('--record-dir'), out = opt('--out'), useTeacher = flag('--teacher'), verify = flag('--verify');
if (!out || (!policyPath && !useTeacher)) { console.error('使い方: node rl/evaluate_policy.mjs --policy module.mjs --out result.json [--core fast|original]'); process.exit(2); }

let controller;
if (useTeacher) controller = await teacherController();
else {
  const mod = await import(pathToFileURL(path.resolve(policyPath)).href);
  const fn = await mod.loadPolicy(policyArg);
  controller = policyController(fn, path.basename(policyPath));
}
const t0 = performance.now();
const safeSets = useTeacher ? sets.filter(s => s !== 'ROB24') : sets;
const result = await evaluateController(controller, {coreKind, variant, sets: safeSets});
const strip = r => { if (!r) return r; const {report, collision_events, ...rest} = r; return rest; };
const summary = {schema: 'rl-bestlap-policy-eval/v1', created_at: new Date().toISOString(), controller: controller.name, policy: policyPath, policy_arg: policyArg,
  core: coreKind, variant, sets: safeSets, eligible: result.eligible ?? null, score: result.score ?? null,
  REC60: result.REC60 && {summary: result.REC60.summary, scenes: result.REC60.scenes.map(strip)},
  STD: result.STD && {single_lap_from_stop: result.STD.single_lap_from_stop, flying_lap: result.STD.flying_lap, single: strip(result.STD.single), continuous: strip(result.STD.continuous)},
  ROB24: result.ROB24 && {summary: result.ROB24.summary, scenes: result.ROB24.scenes.map(strip)}};
if (recordDir) {
  fs.mkdirSync(recordDir, {recursive: true});
  summary.records = {};
  for (const [name, mode] of [['STD_single', 'single'], ['STD_continuous', 'continuous']]) {
    const r = await runEpisode({start: STD, controller, coreKind, variant, episodeMode: mode, record: true});
    const p = path.join(recordDir, name + '.json'); fs.writeFileSync(p, JSON.stringify(r.report)); summary.records[name] = p;
    if (verify) { const {collision_events_own, ...v} = await verifyReport(r.report, {minLaps: mode === 'continuous' ? 2 : null, minClearance: .03}); summary.records[name + '_verification'] = v; }
  }
}
summary.wall_s = (performance.now() - t0) / 1000;
fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true});
fs.writeFileSync(out, JSON.stringify(summary, null, 1) + '\n');
const r = summary.REC60?.summary;
console.log(`${controller.name}: eligible ${summary.eligible} score ${summary.score}` +
  (r ? ` / REC60 有効 ${r.valid_two_laps}/60 1 周目平均 ${r.lap1.mean} 2 周目平均 ${r.lap2.mean} 最小すき間 ${r.min_clearance}` : '') +
  (summary.STD ? ` / STD 単周 ${summary.STD.single_lap_from_stop} フライング ${summary.STD.flying_lap}` : '') +
  (summary.ROB24 ? ` / ROB24 ${summary.ROB24.summary.valid_two_laps}/24` : '') + ` (${summary.wall_s.toFixed(1)} 秒)`);
