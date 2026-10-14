#!/usr/bin/env node
// 閉ループ評価の診断(参考・合否に使わない。2026-10-14、段 E): JetFormer がぶつかった走行を、RL の方策(記録の教師)の目で見直す
//
//   node eval/takeover.mjs [--root <closed_loop>] [--labels SR-s0,SR-s1,...] [--every 3] [--out <結果.json>]
//
// ブラウザの 1 走行(sim.report() + world)の「実際に使った操作」を Node の元の sim-core.js(コピー、元と同じ)で再生しながら:
//  1) 付け直し(DAgger のラベル): 各フレームの状態で RL の方策(rl/best/main、遅延 0、観測は正解の状態)が出す操作と、
//     JetFormer が実際に使った操作の差(舵 [rad]・速さ [m/s])。前回の行動の欄には JetFormer が実際に使った操作を入れる
//  2) 引き継ぎ: フレーム k まで JetFormer の操作で進め、k から先を RL の方策に任せて(遅延 0、または --delay 1)、
//     単周を完走できるか。k を 0 から every おきに、ぶつかる直前まで調べる。「RL ならまだ立て直せた最後のフレーム」を出す
//  3) 確認: k = 0(最初から RL)の E0 が RL の段の値(8.091953 秒)と同じになること
// 物理・コースはブラウザの走行の report.config・report.world をそのまま使う。ユーザーのファイルは変更しない。
import fs from 'node:fs';
import path from 'node:path';
import {createSimulation} from '../lib/app_emulation.mjs';
import {createFeatureTracker, actionToCommand, commandToAction, OBS_DIM} from '../rl/features.mjs';
import {createPolicy} from '../rl/policy_forward.mjs';
import WEIGHTS from '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/best/main/policy_weights.mjs';

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; };
const root = path.resolve(opt('--root', '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/eval/closed_loop'));
const labels = opt('--labels', 'SR-s0,SR-s1,ST-s0,ST-s1,EXT').split(',');
const every = Number(opt('--every', '3'));
const rlDelay = Number(opt('--delay', '0'));
const outPath = opt('--out', path.join(root, '..', 'takeover.json'));
const SPEED_CAP = 4.5;
if (WEIGHTS.obs_dim !== OBS_DIM) throw Error('方策の入力の次元が観測と違います');

// フレーム k まで JetFormer の操作、その先は RL(k = Infinity なら最後まで JetFormer の操作で、付け直しだけを計算する)
function drive(report, k, {relabel = false} = {}) {
  const sim = createSimulation(report.world, report.config);
  sim.reset({seed: report.seed, scenario: report.scenario});
  const cfg = sim.snapshot().config;
  const tracker = createFeatureTracker(report.world, cfg), act = createPolicy(WEIGHTS);
  const obs = new Float64Array(OBS_DIM), a = [0, 0];
  let prev = {a0: 0, a1: 0}, queued = null, f = 0;
  const diffs = [];
  while (!sim.snapshot().state.terminal) {
    const snap = sim.snapshot(), st = snap.state;
    if (f === 0) tracker.resetProgress(st.x, st.z, snap.metrics.route_index); else tracker.advanceProgress(st.x, st.z);
    let action;
    const useRl = f >= k;
    if (useRl || relabel) {
      tracker.observe(st, snap.metrics.route_index, prev.a0, prev.a1, obs);
      act(obs, a);
    }
    const c = (useRl || relabel) ? actionToCommand(a[0], a[1], SPEED_CAP, cfg.maxSteer) : null;
    if (useRl) {
      if (rlDelay === 1) { action = queued ?? (f === k && f > 0 ? report.actions[f - 1] : {steering: 0, speed: 0}); queued = {steering: c.steering, speed: c.speed}; }
      else action = {steering: c.steering, speed: c.speed};
      prev = rlDelay === 1 ? commandToAction(action.steering, action.speed, SPEED_CAP, cfg.maxSteer) : {a0: c.a0, a1: c.a1};
    } else {
      if (f >= report.actions.length) break;
      action = {steering: report.actions[f].steering, speed: report.actions[f].speed};
      if (relabel) diffs.push({frame: f, d_steer: action.steering - c.steering, d_speed: action.speed - c.speed, rl_steer: c.steering, rl_speed: c.speed,
        jf_steer: action.steering, jf_speed: action.speed});
      prev = commandToAction(action.steering, action.speed, SPEED_CAP, cfg.maxSteer);
    }
    sim.step(action);
    f++;
  }
  const rep = sim.report();
  return {reason: rep.outcome.reason, success: rep.outcome.success, lap_s: rep.outcome.reason === 'route_complete' ? rep.metrics.elapsed : null,
    passed_gates: rep.metrics.passed_gates, collisions: rep.metrics.collisions, frames: rep.actions.length, diffs,
    crossings: rep.metrics.required_crossing_events.length};
}

const q = (v, p) => { const s = [...v].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : null; };
const rows = [];
for (const label of labels) {
  const dir = path.join(root, label);
  if (!fs.existsSync(dir)) continue;
  for (const f of fs.readdirSync(dir).filter(f => /-single(-[^.]+)?\.json$/.test(f) && !f.startsWith('plan_')).sort()) {
    const report = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
    if (report.config.episodeMode) continue;
    const full = drive(report, Infinity, {relabel: true});
    // 再生の確認: ブラウザと同じ終わり方・フレーム数
    const same = full.reason === report.outcome.reason && full.frames === report.actions.length;
    const n = report.actions.length;
    const takeovers = [];
    for (let k = 0; k < n; k += every) takeovers.push({k, ...(({diffs, ...x}) => x)(drive(report, k))});
    if ((n - 1) % every) takeovers.push({k: n - 1, ...(({diffs, ...x}) => x)(drive(report, n - 1))});
    const ok = takeovers.filter(t => t.reason === 'route_complete' && t.collisions === 0);
    const lastOk = ok.length ? Math.max(...ok.map(t => t.k)) : null;
    const firstFail = takeovers.find(t => !(t.reason === 'route_complete' && t.collisions === 0));
    const ad = full.diffs.map(d => Math.abs(d.d_steer)), as = full.diffs.map(d => Math.abs(d.d_speed));
    // 付け直しの差の大きさ(最初の 20 フレーム = まだ RL の線の近く、を別に出す)
    const early = full.diffs.filter(d => d.frame < 20);
    rows.push({name: report.name, label, start_variation: report.context?.start_variation, browser_reason: report.outcome.reason,
      browser_frames: n, browser_lap_s: report.outcome.reason === 'route_complete' ? report.metrics.elapsed : null, replay_same: same,
      relabel: {frames: full.diffs.length, steer_abs_mean_rad: ad.reduce((x, y) => x + y, 0) / Math.max(1, ad.length), steer_abs_p90_rad: q(ad, .9),
        speed_abs_mean_mps: as.reduce((x, y) => x + y, 0) / Math.max(1, as.length), speed_abs_p90_mps: q(as, .9),
        steer_abs_mean_first20_rad: early.reduce((x, d) => x + Math.abs(d.d_steer), 0) / Math.max(1, early.length),
        per_frame: full.diffs.map(d => [d.frame, +d.jf_steer.toFixed(5), +d.rl_steer.toFixed(5), +d.jf_speed.toFixed(4), +d.rl_speed.toFixed(4)])},
      takeover: {rl_delay: rlDelay, every, results: takeovers, last_recoverable_frame: lastOk, first_unrecoverable_frame: firstFail?.k ?? null,
        all_recoverable_until_end: !firstFail, k0_lap_s: takeovers[0]?.lap_s ?? null}});
    console.error(`${report.name}: ${report.outcome.reason} ${n} フレーム、RL が立て直せた最後 k=${lastOk}、最初に立て直せない k=${firstFail?.k ?? '-'}、` +
      `付け直しの舵の差 平均 ${(rows.at(-1).relabel.steer_abs_mean_rad).toFixed(3)} rad`);
  }
}
fs.writeFileSync(outPath, JSON.stringify({schema: 'sim-rl-bestlap-takeover/v1', created_at: new Date().toISOString(), root, rl_policy: 'rl/best/main/policy_weights.mjs',
  note: '参考(合否に使わない)。RL の方策は観測に正解の状態を使う。引き継ぎは遅延 ' + rlDelay + '。', rows}, null, 1) + '\n');
console.log(outPath);
