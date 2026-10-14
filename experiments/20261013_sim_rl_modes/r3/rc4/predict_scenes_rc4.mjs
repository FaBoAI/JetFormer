#!/usr/bin/env node
// 離脱復活 r4(2026-10-14): r3/rc/predict_scenes_rc3.mjs の写し。valid に「本当のゲートの外し 0」を足し、走行中の仕組みを r3/rc4/rt に。
// 離脱復活 r3 の記録の前の予想(protocol_recover_r3.json recording.prefilter。2026-10-14、学習の段 recover r3)。rl/predict_scenes.mjs の r3 版:
//   走行中の仕組みは r3/rc/rt(乱し v2 + 乱しのフレーム以外のシールド)、valid は valid_episode_r3
//   (valid_episode かつ 止まり(フレーム 30 以降に |v| < 0.2 m/s が 30 フレーム以上。ブレーキの乱しの区間とそのあと 30 フレームは数えない)が無い
//    かつ 20 秒で R の上を 1 周(26.12 m)以上進む)。昇格の段で記録の前の確かめに使う(統合の段の make_scene_pool.mjs と同じ止まりの決まり 1.0 秒)。
//   node r3/rc/predict_scenes_rc4.mjs --plan plan.json --policy policy.json [--out pred.json]
//   plan.json = {mode: 'recover', scenes: [{name, start: {gate, left_offset_m, left_yaw_deg}, seed, params?}]}
import fs from 'node:fs';
import crypto from 'node:crypto';
import {runModeEpisode, readJson} from './episode_rc4.mjs';
import {stallInfo} from './eval_rc4.mjs';
import {LOOP_LENGTH} from './rt/features.mjs';
import {RECOVERABLE_TABLE_SHA256} from './rt/recoverable_table.mjs';

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const plan = readJson(opt('--plan')), policyPath = opt('--policy');
if (plan.mode !== 'recover') throw new Error('predict_scenes_rc4: 離脱復活(recover)のプランだけです');
const weights = readJson(policyPath), sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
const out = {schema: 'jf-sim-rl-scene-prediction/v1', mode: plan.mode, policy_sha256: sha, created_at: new Date().toISOString(), rule: 'valid_episode_r3 かつ gate_miss_real 0(r4)',
  perturbation: 'jf-perturb-record/v2', recoverable_table_sha256: RECOVERABLE_TABLE_SHA256, scenes: []};
for (const sc of plan.scenes) {
  const r = await runModeEpisode({mode: 'recover', weights, policySha256: sha, start: {...sc.start, seed: sc.seed}, params: sc.params ?? {}});
  const acts = r.invalid_spawn ? [] : r.actions;
  const si = r.invalid_spawn ? {stall: false} : stallInfo(r.log, r.events);
  const progress = r.invalid_spawn ? 0 : r.log.at(-1).s - r.log[0].s;
  const v0 = !r.invalid_spawn && r.valid_episode;
  out.scenes.push({name: sc.name, valid: v0 && !si.stall && progress >= LOOP_LENGTH - 1e-9 && r.gate_miss === false, valid_episode_r3: v0 && !si.stall && progress >= LOOP_LENGTH - 1e-9, valid_episode: v0, stall: si.stall ?? false,
    stall_max_run_frames: si.max_run_frames ?? 0, progress_m: progress, reason: r.reason ?? 'invalid_spawn', collisions: r.collisions ?? null,
    laps: (r.laps ?? []).map(l => l.elapsed), min_clearance: r.min_clearance ?? null, events: r.events?.length ?? 0, shield_frames: r.shield_frames ?? 0,
    gate_miss: r.gate_miss ?? null, gate_misses: r.gate_misses ?? null,
    event_starts: (r.events ?? []).map(e => ({ps: e.ps, pe: e.pe, type: e.type, kind: e.kind ?? null, cls: e.cls, s: e.s ?? null, fallback: e.fallback ?? null,
      req: e.req ? {type: e.req.type, cls: e.req.cls, kind: e.req.kind} : null})),
    params: r.params, actions_sha256: crypto.createHash('sha256').update(JSON.stringify(acts)).digest('hex')});
}
const txt = JSON.stringify(out, null, 1);
if (opt('--out')) fs.writeFileSync(opt('--out'), txt); else console.log(txt);
