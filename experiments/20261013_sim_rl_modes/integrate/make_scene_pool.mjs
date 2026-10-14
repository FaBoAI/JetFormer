#!/usr/bin/env node
// 記録に使うシーンの「前もって走らせて確かめた候補の組」(scene pool)を作る(2026-10-14、sim_rl_modes の統合の段)。
//
// JetFormer の収集プラン(jetformer/sim/rl_modes.py)は、この組から開始・種・シーンの条件を選ぶだけにする(サーバーで Node を動かさない)。
// 各候補は rl/episode.mjs(元の sim-core.js + ブラウザと同じ rl/runtime/mode_runtime.mjs)で 20 秒・連続周回を走らせ、
// 次の決まりを全部満たすものだけを組に入れる(modes.json recording_common.prefilter + 確認の段の指摘):
//   - 600 フレーム・衝突 0(valid_episode)
//   - 有効な周(33 ゲート・横断 1 回・衝突 0)が 1 周以上(止まったまま・逆走のまま 20 秒が過ぎるシーンを替える。確認 no_wall r2・recover r2・speed r2)
//   - 1 秒以降に 0.2 m/s 未満が続くのは 1.0 秒まで(止まってすくむシーンを替える。確認 recover r2 の P33・Q10・Q12)
//   - 最小すき間 ≥ mode.json recording.prefilter_min_clearance_m(no_wall 0.08 m・speed 0.03 m。lap_audit の小区間ごとの正確な距離)
//   - 本当のゲートの外し(rl/gate_miss.mjs)が 0(recover だけは外しを許す。データセットの説明に書いてある性質のため)
// 開始は modes.json(登録の mode.json)の recording.start_distribution の Halton 列(記録の番号 1〜9,999)、種は 700000 + 1000·m + 候補の番号。
//
//   node integrate/make_scene_pool.mjs --mode no_wall --policy bundle/no_wall/policy.json --mode-json bundle/no_wall/mode.json \
//        --count 90 [--max-candidates 400] [--workers 2] --out pool.json
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import {Worker, isMainThread, parentPort, workerData} from 'node:worker_threads';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
export const POOL_SCHEMA = 'jf-sim-rl-scene-pool/v1';
export const STALL_SPEED = 0.2, STALL_MAX_S = 1.0, STALL_AFTER_S = 1.0;

// 走り 1 回の結果 → 組に入れるかの判定と、組に残す値
export function judgeRun(r, rules) {
  const dt = 1 / 30;
  // 0.2 m/s 未満が続いた最長のフレーム数(最初の 1 秒 = 停止から走り出す間は数えない)
  let run = 0, stall = 0;
  (r.trace ?? []).forEach((q, i) => { if (i * dt < STALL_AFTER_S) return; run = q.v < STALL_SPEED ? run + 1 : 0; stall = Math.max(stall, run); });
  const why = [];
  if (r.invalid_spawn) why.push('invalid_spawn');
  else {
    if (!r.valid_episode) why.push(`not_valid(${r.reason}・${r.frames} フレーム・衝突 ${r.collisions})`);
    if ((r.valid_laps ?? 0) < rules.min_valid_laps) why.push(`valid_laps ${r.valid_laps ?? 0} < ${rules.min_valid_laps}`);
    if (stall * dt > rules.max_stall_s) why.push(`stall ${(stall * dt).toFixed(2)} s`);
    if (rules.min_clearance_m != null && !((r.min_clearance ?? 0) >= rules.min_clearance_m)) why.push(`clearance ${(r.min_clearance ?? 0).toFixed(4)} < ${rules.min_clearance_m}`);
    if (rules.no_gate_miss && (r.gate_misses ?? 0) > 0) why.push(`gate_misses ${r.gate_misses}`);
  }
  const learn0 = (r.log ?? []).filter(q => q.l === 0).length;
  return {ok: why.length === 0, why, stall_s: +(stall * dt).toFixed(3), learn0_frames: learn0};
}

function summarize(r, cand, judge) {
  const acts = r.invalid_spawn ? [] : r.actions;
  return {c: cand.c, halton_index: cand.halton_index ?? null, start: cand.start, seed: cand.seed, params_in: cand.params, params: r.params ?? null,
    ok: judge.ok, why: judge.why, reason: r.reason ?? 'invalid_spawn', frames: r.frames ?? 0, collisions: r.collisions ?? null,
    min_clearance: r.min_clearance ?? null, valid_laps: r.valid_laps ?? 0, laps: (r.laps ?? []).map(l => +l.elapsed.toFixed(4)),
    gate_misses: r.gate_misses ?? null, stall_s: judge.stall_s, events: r.events?.length ?? 0, learn0_frames: judge.learn0_frames,
    shield_frames: r.shield_frames ?? 0, mean_speed: r.trace?.length ? +(r.trace.reduce((a, q) => a + q.v, 0) / r.trace.length).toFixed(4) : null,
    actions_sha256: crypto.createHash('sha256').update(JSON.stringify(acts)).digest('hex')};
}

if (!isMainThread) {
  const {runModeEpisode, readJson} = await import(path.join(here, '../rl/episode.mjs'));
  const {mode, policyPath, rules} = workerData;
  const weights = readJson(policyPath);
  const sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
  parentPort.on('message', async (cand) => {
    if (cand === null) { process.exit(0); }
    try {
      const r = await runModeEpisode({mode, weights, policySha256: sha, start: {...cand.start, seed: cand.seed}, params: cand.params});
      parentPort.postMessage(summarize(r, cand, judgeRun(r, rules)));
    } catch (e) {
      parentPort.postMessage({c: cand.c, error: String(e && e.stack || e)});
    }
  });
} else if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
  const mode = opt('--mode'), policyPath = path.resolve(opt('--policy')), modeJson = JSON.parse(fs.readFileSync(opt('--mode-json'), 'utf8'));
  const count = +opt('--count', 90), maxCand = +opt('--max-candidates', 600), nw = +opt('--workers', 2), out = opt('--out');
  const {haltonStarts} = await import(path.join(here, '../rl/eval_modes.mjs'));
  const {MODE_NUMBER, SPEED_LEVELS} = await import(path.join(here, '../rl/runtime/mode_runtime.mjs'));
  if (modeJson.key !== mode) throw new Error('mode.json のモードが違います');
  const dist = modeJson.recording?.start_distribution;
  if (!dist) throw new Error('mode.json に recording.start_distribution がありません');
  const rules = {min_valid_laps: 1, max_stall_s: STALL_MAX_S, min_clearance_m: modeJson.recording?.prefilter_min_clearance_m ?? null,
    no_gate_miss: mode !== 'recover', stall_speed_mps: STALL_SPEED, stall_after_s: STALL_AFTER_S};
  const starts = haltonStarts(mode, 'recording', maxCand, {from: 1, dist});
  const cands = starts.map((st, c) => {
    const params = mode === 'speed' ? {c_v: SPEED_LEVELS[c % SPEED_LEVELS.length]} : mode === 'recover' ? {perturb_class: 'auto'} : {};
    const {initial_clearance_m, ...start} = st;
    return {c, start, seed: 700000 + 1000 * MODE_NUMBER[mode] + c, params, initial_clearance_m: initial_clearance_m ?? null};
  });
  const levels = mode === 'speed' ? SPEED_LEVELS : null;
  const need = levels ? Object.fromEntries(levels.map(v => [v, Math.ceil(count / levels.length)])) : null;
  const results = new Map();
  const enough = () => {
    // 番号の若い順に途切れなく走らせた範囲だけで数える(並べ方に結果がよらない)
    const okRows = [];
    for (let c = 0; c < cands.length && results.has(c); c++) if (results.get(c).ok) okRows.push(results.get(c));
    if (!levels) return okRows.length >= count;
    return levels.every(v => okRows.filter(r => r.params?.c_v === v).length >= need[v]);
  };
  const t0 = Date.now();
  let next = 0, busy = 0, stopping = false;
  const workers = Array.from({length: Math.max(1, Math.min(nw, os.cpus().length))}, () => new Worker(fileURLToPath(import.meta.url), {workerData: {mode, policyPath, rules}}));
  await new Promise((resolve, reject) => {
    const feed = (w) => {
      if (stopping || next >= cands.length) { if (busy === 0) resolve(); return; }
      busy++; w.postMessage(cands[next++]);
    };
    for (const w of workers) {
      w.on('message', (row) => {
        busy--;
        if (row.error) { stopping = true; reject(new Error(`候補 ${row.c}: ${row.error}`)); return; }
        results.set(row.c, row);
        if (results.size % 10 === 0) process.stderr.write(`[pool ${mode}] ${results.size} 候補・合格 ${[...results.values()].filter(r => r.ok).length}・${((Date.now() - t0) / 1000).toFixed(0)} 秒\n`);
        if (enough()) stopping = true;
        feed(w);
      });
      w.on('error', reject);
      feed(w);
    }
  });
  for (const w of workers) w.postMessage(null);
  // 途切れなく走らせた範囲(番号の若い順)
  const prefix = [];
  for (let c = 0; c < cands.length && results.has(c); c++) prefix.push({...results.get(c), initial_clearance_m: cands[c].initial_clearance_m});
  let pool = prefix.filter(r => r.ok);
  if (levels) {
    // 速さの段を順に 1 つずつ(プランが連続して選ぶと段がそろう)
    const by = Object.fromEntries(levels.map(v => [v, pool.filter(r => r.params?.c_v === v)]));
    const rr = [];
    for (let k = 0; rr.length < count && levels.some(v => by[v].length > k); k++) for (const v of levels) if (by[v][k] && rr.length < count) rr.push(by[v][k]);
    pool = rr;
  } else pool = pool.slice(0, count);
  const policySha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
  const rejected = prefix.filter(r => !r.ok);
  const why = {};
  for (const r of rejected) for (const w of r.why) { const k = w.split(/[ (]/)[0]; why[k] = (why[k] ?? 0) + 1; }
  const doc = {schema: POOL_SCHEMA, mode, policy_sha256: policySha, mode_json_policy_sha256: modeJson.policy_sha256,
    created_at: new Date().toISOString(), generator: 'experiments/20261013_sim_rl_modes/integrate/make_scene_pool.mjs',
    core: 'original sim-core.js(snapshot/sim/dist/sim-core.js)', runtime: 'rl/runtime/mode_runtime.mjs(= jetformer/sim/static/rl/)',
    start_distribution: dist, seed_rule: '700000 + 1000·m + 候補の番号(m = protocol modes.mode_number)', halton_range: '記録 1〜9,999',
    rules, candidates_run: prefix.length, accepted: prefix.filter(r => r.ok).length, rejected_by: why,
    acceptance_rate: prefix.length ? +(prefix.filter(r => r.ok).length / prefix.length).toFixed(4) : null,
    wall_s: +((Date.now() - t0) / 1000).toFixed(1), count: pool.length,
    scenes: pool.map(r => ({c: r.c, start: r.start, seed: r.seed, params: r.params_in, initial_clearance_m: r.initial_clearance_m,
      min_clearance: r.min_clearance, valid_laps: r.valid_laps, laps: r.laps, gate_misses: r.gate_misses, stall_s: r.stall_s, events: r.events,
      learn0_frames: r.learn0_frames, shield_frames: r.shield_frames, mean_speed: r.mean_speed, actions_sha256: r.actions_sha256})),
    rejected: rejected.map(r => ({c: r.c, start: r.start, seed: r.seed, why: r.why}))};
  if (pool.length < count) process.stderr.write(`[pool ${mode}] 注意: 合格が ${pool.length} 本で ${count} 本に届きません(候補 ${prefix.length})\n`);
  fs.writeFileSync(out, JSON.stringify(doc, null, 1) + '\n');
  process.stderr.write(`[pool ${mode}] 書いた: ${out}(${pool.length} シーン・候補 ${prefix.length}・${doc.wall_s} 秒)\n`);
  process.exit(0);
}
