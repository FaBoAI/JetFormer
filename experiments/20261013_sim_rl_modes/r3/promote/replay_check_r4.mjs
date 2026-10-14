#!/usr/bin/env node
// 昇格の段(2026-10-15、r3/promote): integrate/replay_check.mjs の写しを r4 の決まりにした版。違いは
//   - 教師に登録簿の mode.json の runtime を modeRuntime として渡す(ページの teacher.js と同じ。line のシールド v5b)。予想の閉ループは r3/promote/episode_r4.mjs
//   - 決まり(scenes.json の rules = r3/promote/make_scene_pool_r4.mjs): 有効な周 ≥ 1・止まり(フレーム 30 以降に |v| < 0.2 m/s が 30 フレーム以上。
//     教師に渡した状態の速さと sim のステップ後の速さの両方。recover はブレーキの乱しの区間 + 30 フレームを除く)・統合の段の止まり(1 秒を越えない)・
//     最小すき間・シールドの作動の割合・後ろへ動いたフレーム 0・本当のゲートの外し 0・(recover)進み ≥ 1 周
// 以下は元の説明。
// ブラウザで記録して JetFormer に取り込んだ強化学習モードのデータセットを Node で確かめる(統合の段。記録の段の RA2・RA3 と同じ考え)。
//   node integrate/replay_check.mjs --dataset <data/datasets/名前> [--out result.json]
// シーンごとに:
//   (a) 開ループの再生: 記録した操作(frames.jsonl の command_si)を元の sim-core.js に同じ開始・種で与え、記録の姿勢(state_post_action)と比べる(≤ 1e-6)
//   (b) 教師の同一性: (a) の各フレームの snapshot を同じ mode_runtime.mjs(ブラウザと同じファイル)に与え、出た操作と info.jf が記録と同じか(≤ 1e-6)
//   (c) 記録した走りの正確なすき間(lap_audit の小区間ごと)・有効な周・止まった長さ = モードの記録の決まり(scenes.json の rules)を満たすか
//   (d) 参考: 同じ開始・種で Node が閉ループで走らせた走り(記録の前の予想)とどこから違うか。ブラウザと Node は Math.tanh などの最後の桁が違うことがあり、
//       閉ループでは小さな違いが育つので、予想と記録は全部同じにはならない(だから (a)〜(c) で記録そのものを確かめる)
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {runModeEpisode, readJson, BASE_WORLD, EXP, RT_DIR} from './episode_r4.mjs';
import {stallR4} from './make_scene_pool_r4.mjs';
const {activeWorldFor, simConfig} = await import(pathToFileURL(path.join(EXP, 'snapshot/rl/env_core.mjs')).href);
const {independentAudit} = await import(pathToFileURL(path.join(EXP, 'snapshot/rl/lap_audit.mjs')).href);
const {createModeTeacher} = await import(pathToFileURL(path.join(RT_DIR, 'mode_runtime.mjs')).href);
const {LOOP_LENGTH} = await import(pathToFileURL(path.join(RT_DIR, 'features.mjs')).href);
const {realGateMisses} = await import(pathToFileURL(path.join(EXP, 'rl/gate_miss.mjs')).href);

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const root = path.resolve(opt('--dataset'));
const coll = readJson(path.join(root, 'simulation/collection.json'));
const card = readJson(path.join(root, 'dataset_card.json'));
const key = card.rl?.mode ?? coll.teacher?.mode;
const reg = '/home/jetson/jetformer/data/sim_rl/modes/' + key;
const weights = readJson(reg + '/policy.json');
const modeRuntime = readJson(reg + '/mode.json').runtime ?? null;
const rules = (fs.existsSync(reg + '/scenes.json') ? readJson(reg + '/scenes.json').rules : null) ?? {min_valid_laps: 1, max_stall_s: 1};
const core = await import(pathToFileURL(path.join(EXP, 'snapshot/sim/dist/sim-core.js')).href);
const out = {dataset: path.basename(root), mode: key, rules, scenes: []};
for (const sc of coll.scenes) {
  const dir = path.join(root, 'simulation/scenes', sc.scene);
  const sess = readJson(path.join(dir, 'session.json'));
  const rows = fs.readFileSync(path.join(dir, 'frames.jsonl'), 'utf8').split('\n').filter(Boolean).map(JSON.parse);
  const st0 = sc.plan_start ?? sess.context.start_variation;      // プランの値(丸めていない)。無ければ 6 桁に丸めた記録の値
  const params = sc.params ?? {};
  const {world, variation} = activeWorldFor(BASE_WORLD, st0.gate, st0.left_offset_m, st0.left_yaw_deg);
  const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: sess.config.seed, noiseStd: sess.config.noiseStd ?? 0});
  const sim = core.createSimulation(world, cfg);
  let snap = sim.reset({seed: cfg.seed, scenario: sess.config.noiseStd ? 'noise' : 'baseline'});
  const cfgT = JSON.parse(JSON.stringify(snap.config));
  const t = createModeTeacher(world, cfgT, variation, {mode: key, weights, params, policySha256: card.rl?.policy_sha256 ?? null, ...(modeRuntime ? {modeRuntime} : {})});
  const gateCount = [];
  let maxPose = 0, maxS = 0, maxV = 0, infoDiff = 0, firstInfo = null, n = 0;
  for (const r of rows) {
    if (snap.state.terminal) break;
    gateCount.push(`${snap.metrics.completed_laps ?? 0}:${snap.metrics.passed_gates ?? 0}`);
    const a = t.action({state: snap.state, metrics: snap.metrics, config: cfgT});
    const c = r.command_si;
    maxS = Math.max(maxS, Math.abs(a.steering - c.steering)); maxV = Math.max(maxV, Math.abs(a.speed - c.speed));
    if (JSON.stringify(a.info) !== JSON.stringify(r.action_info)) { infoDiff++; if (firstInfo === null) firstInfo = {frame: r.frame, node: a.info, rec: r.action_info}; }
    snap = sim.step({steering: c.steering, speed: c.speed});                // 記録した操作で進める(開ループ)
    const q = r.state_post_action;
    maxPose = Math.max(maxPose, Math.abs(q.x - snap.state.x), Math.abs(q.z - snap.state.z), Math.abs(q.yaw - snap.state.yaw));
    n++;
  }
  const rep = {...sim.report(), world};
  const audit = independentAudit(rep);
  const m = snap.metrics;
  const laps = (m.lap_events ?? []).filter(l => l.passed_gates === 33 && l.required_crossing_events.length === 1 && l.collisions === 0).length;
  let run = 0, stall = 0;
  rep.trace.forEach((q, i) => { if (i / 30 < 1) return; run = q.speed < 0.2 ? run + 1 : 0; stall = Math.max(stall, run); });
  const clearance = audit.min_clearance_m;
  const ruleFail = [];
  // r4 の決まり(記録そのもので)
  const tlog = t.log, tev = t.events;
  const st4 = stallR4({log: tlog.map(q => ({v: q.v})), trace: rep.trace.map(q => ({v: q.speed})), events: tev}, key);
  if (st4.stall) ruleFail.push(`stall_r3 ${st4.max_run} フレーム(${st4.where})`);
  const shFrac = rows.filter(x => x.action_info?.jf?.sh === 1).length / Math.max(1, rows.length);
  if (rules.max_shield_fraction != null && shFrac > rules.max_shield_fraction) ruleFail.push(`shield_fraction ${(100 * shFrac).toFixed(1)}%`);
  const back = rep.trace.filter(q => q.speed < -.01).length, negCmd = rows.filter(x => x.command_si.speed < 0).length;
  if (rules.no_reverse && (back > 0 || negCmd > 0)) ruleFail.push(`reverse ${back}/${negCmd}`);
  const gm = realGateMisses(tlog.map(q => q.s), gateCount);
  if (rules.no_gate_miss && gm.misses > 0) ruleFail.push(`gate_misses ${gm.misses}`);
  const progress = tlog.length ? tlog.at(-1).s - tlog[0].s : 0;
  if (rules.min_progress_m != null && !(progress >= rules.min_progress_m - 1e-9)) ruleFail.push(`progress ${progress.toFixed(2)} m`);
  if (laps < (rules.min_valid_laps ?? 1)) ruleFail.push(`valid_laps ${laps}`);
  if (stall / 30 > (rules.max_stall_s ?? 1)) ruleFail.push(`stall ${(stall / 30).toFixed(2)} s`);
  if (rules.min_clearance_m != null && !(clearance >= rules.min_clearance_m)) ruleFail.push(`clearance ${clearance.toFixed(4)} < ${rules.min_clearance_m}`);
  // (d) 予想(閉ループ)との違い
  const pred = await runModeEpisode({mode: key, weights, start: {gate: st0.gate, left_offset_m: st0.left_offset_m, left_yaw_deg: st0.left_yaw_deg,
    seed: sess.config.seed}, params, audit: true, keepLog: true, modeRuntime});
  let firstDiff = null, maxPredPose = 0;
  for (let i = 0; i < Math.min(rows.length, pred.actions.length); i++) {
    const d = Math.abs(pred.actions[i].steering - rows[i].command_si.steering) + Math.abs(pred.actions[i].speed - rows[i].command_si.speed);
    if (firstDiff === null && d > 1e-9) firstDiff = i;
    const q = rows[i].state_post_action, p = pred.trace[i];
    if (p) maxPredPose = Math.max(maxPredPose, Math.hypot(q.x - p.x, q.z - p.z));
  }
  out.scenes.push({scene: sc.scene, plan_scene: sc.plan_scene ?? null, seed: sess.config.seed, start: st0, params, frames: rows.length, replayed: n,
    open_loop_max_pose_err: maxPose, teacher_max_abs_steering: maxS, teacher_max_abs_speed: maxV, teacher_info_mismatch: infoDiff, first_info_mismatch: firstInfo,
    reason: snap.state.reason, collisions: m.collisions, valid_laps: laps, min_clearance_m: clearance, stall_s: +(stall / 30).toFixed(3),
    learn0_frames: rows.filter(x => x.action_info?.jf?.l === 0).length, shield_frames: rows.filter(x => x.action_info?.jf?.sh === 1).length,
    rule_fail: ruleFail, r4: {stall_r3_frames: st4.max_run, shield_fraction: +shFrac.toFixed(5), back_frames: back, neg_speed_cmd_frames: negCmd,
      gate_misses: gm.misses, gate_stuck_max_m: +gm.stuck_max_m.toFixed(3), progress_m: +progress.toFixed(3), events: tev.length},
    prediction: {first_action_diff_frame: firstDiff, max_pose_diff_m: maxPredPose, reason: pred.reason, collisions: pred.collisions,
      min_clearance_m: pred.min_clearance ?? null, valid_laps: pred.valid_laps},
    pass_identity: n === rows.length && maxPose <= 1e-6 && maxS <= 1e-6 && maxV <= 1e-6 && infoDiff === 0,
    pass_rules: ruleFail.length === 0 && m.collisions === 0 && snap.state.reason === 'collection_duration_complete'});
}
out.pass_identity = out.scenes.every(s => s.pass_identity);
out.pass_rules = out.scenes.every(s => s.pass_rules);
const txt = JSON.stringify(out, null, 1);
if (opt('--out')) fs.writeFileSync(opt('--out'), txt); else console.log(txt);
