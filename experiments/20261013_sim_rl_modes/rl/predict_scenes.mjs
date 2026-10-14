#!/usr/bin/env node
// 記録のプランのシーンを Node(元の sim-core.js・同じ mode_runtime.mjs)で前もって走らせる(modes.json recording_common.prefilter)。2026-10-13
// 600 フレーム・衝突 0 にならないシーンは、プランを作る側(JetFormer、統合の段)が次の候補に替える。記録のあとの再生の照合にも、ここの操作列を使える。
//   node rl/predict_scenes.mjs --plan plan.json --policy policy.json [--out pred.json]
//   plan.json = {mode, scenes: [{name, start: {gate, left_offset_m, left_yaw_deg}, seed, params?, layout?: {id, obstacles}}], min_clearance_m?}
//   [--min-clearance 0.08]: 最小すき間がこれより小さいシーンも valid = false にする(no_wall の mode.json recording.prefilter_min_clearance_m。
//   2026-10-14 no_wall r2 で足した。plan.min_clearance_m でも指定できる。valid_without_clearance_rule に元の判定を残す)
//   [--min-valid-laps 1]: sim が数えた有効な周(33 ゲート・横断 1 回・衝突 0)がこれより少ないシーンも valid = false にする
//   (止まったまま 20 秒が過ぎるシーンを替える。speed r2 の mode.json recording.prefilter_min_valid_laps。2026-10-14 学習の段 speed r2。plan.min_valid_laps でも可)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {runModeEpisode, readJson} from './episode.mjs';

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const plan = readJson(opt('--plan')), policyPath = opt('--policy');
const weights = readJson(policyPath), sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
const minC = opt('--min-clearance') != null ? +opt('--min-clearance') : (plan.min_clearance_m ?? null);
const minLaps = opt('--min-valid-laps') != null ? +opt('--min-valid-laps') : (plan.min_valid_laps ?? null);
const out = {schema: 'jf-sim-rl-scene-prediction/v1', mode: plan.mode, policy_sha256: sha, created_at: new Date().toISOString(), min_clearance_m: minC,
  ...(minLaps != null ? {min_valid_laps: minLaps} : {}), scenes: []};
for (const sc of plan.scenes) {
  const r = await runModeEpisode({mode: plan.mode, weights, policySha256: sha, start: {...sc.start, seed: sc.seed}, params: sc.params ?? {}, layout: sc.layout ?? null});
  const acts = r.invalid_spawn ? [] : r.actions;
  const v0 = !r.invalid_spawn && r.valid_episode;
  out.scenes.push({name: sc.name, valid: v0 && (minC == null || (r.min_clearance ?? 0) >= minC) && (minLaps == null || (r.valid_laps ?? 0) >= minLaps),
    valid_without_clearance_rule: v0, ...(minLaps != null ? {valid_laps: r.valid_laps ?? 0} : {}), reason: r.reason ?? 'invalid_spawn', collisions: r.collisions ?? null,
    laps: (r.laps ?? []).map(l => l.elapsed), min_clearance: r.min_clearance ?? null, events: r.events?.length ?? 0, shield_frames: r.shield_frames ?? 0,
    // 本当のゲートの外し(rl/gate_miss.mjs)と乱しの始まりの位置(2026-10-14 recover r2: 記録のプラン・データセットの説明に書くため)
    gate_miss: r.gate_miss ?? null, gate_misses: r.gate_misses ?? null,
    event_starts: (r.events ?? []).map(e => ({ps: e.ps, pe: e.pe, type: e.type, cls: e.cls, s: e.s ?? null})),
    params: r.params, actions_sha256: crypto.createHash('sha256').update(JSON.stringify(acts)).digest('hex')});
}
const txt = JSON.stringify(out, null, 1);
if (opt('--out')) fs.writeFileSync(opt('--out'), txt); else console.log(txt);
