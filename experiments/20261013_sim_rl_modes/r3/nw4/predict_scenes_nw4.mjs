#!/usr/bin/env node
// 記録の前の予想(no_wall r4 版、2026-10-14。protocol_no_wall_r4.json recording.prefilter)。r3/nw/predict_scenes_nw3.mjs の写しに「後ろへ動いたフレーム」を足した。以下は r3 の説明。
// 記録の前の予想(no_wall r3 版、2026-10-14。protocol_no_wall_r3.json recording.prefilter)。rl/predict_scenes.mjs と同じ入出力の形で、
// 走らせ方は r3/nw/run_nw3r.mjs(元の sim-core.js・runtime_v4 = シールド v4・標準の物理)。次のどれかのシーンは valid = false(次の候補に替える):
//   - valid_episode_r3 でない(衝突・途中の終了・止まり = フレーム 30 以降に |v| < 0.2 m/s が 1 秒以上・有効な周 0)
//   - 最小すき間 < 0.08 m(mode.json recording.prefilter_min_clearance_m)
//   - シールドの作動の割合 > 0.25(recording.prefilter_max_shield_fraction)
//   node r3/nw/predict_scenes_nw3.mjs --plan plan.json --policy policy.json [--out pred.json]
//   plan.json = {mode: 'no_wall', scenes: [{name, start: {gate, left_offset_m, left_yaw_deg}, seed}]}
//   plan.json の min_clearance_m・min_valid_laps・max_shield_fraction・no_stall で値を替えられる(既定は決まりの値)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {runOne, policyFile} from './run_nw4r.mjs';

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const plan = JSON.parse(fs.readFileSync(opt('--plan'), 'utf8')), policyPath = opt('--policy');
if (plan.mode && plan.mode !== 'no_wall') throw new Error('predict_scenes_nw4: no_wall のプランだけです');
const P = policyFile('no_wall', policyPath);
const R = {min_clearance_m: plan.min_clearance_m ?? .08, min_valid_laps: plan.min_valid_laps ?? 1, max_shield_fraction: plan.max_shield_fraction ?? .25, no_stall: plan.no_stall ?? true, no_reverse: plan.no_reverse ?? true};
const out = {schema: 'jf-sim-rl-scene-prediction/v1', variant: 'no_wall-r4', mode: 'no_wall', policy_sha256: P.sha, created_at: new Date().toISOString(), rule: R, scenes: []};
for (const sc of plan.scenes) {
  const r = await runOne({mode: 'no_wall', start: {...sc.start, seed: sc.seed}, policyPath, keepPath: true});
  if (r.invalid_spawn) { out.scenes.push({name: sc.name, valid: false, why: ['invalid_spawn']}); continue; }
  const sf = r.shield_frames / Math.max(1, r.frames);
  const why = [];
  if (!(r.valid_episode)) why.push('valid_episode でない(' + r.reason + '・衝突 ' + r.collisions + ')');
  if (R.no_stall && r.stall) why.push(`止まり(${r.stall_max_frames} フレーム)`);
  if ((r.valid_laps ?? 0) < R.min_valid_laps) why.push(`有効な周 ${r.valid_laps}`);
  if (r.min_clearance < R.min_clearance_m) why.push(`最小すき間 ${(100 * r.min_clearance).toFixed(2)} cm`);
  if (sf > R.max_shield_fraction) why.push(`シールドの作動 ${(100 * sf).toFixed(1)}%`);
  // r4: 抜け出しで後ろへ動いたシーン(後ろへ下がる正解ラベル)は記録に入れない(recording.prefilter_no_reverse)
  if (R.no_reverse && ((r.back_frames ?? 0) > 0 || (r.reverse_cmd_frames ?? 0) > 0)) why.push(`後ろへ動いた(${r.back_frames} フレーム)`);
  const acts = r.path.map(p => [p[5], p[6]]);
  out.scenes.push({name: sc.name, valid: why.length === 0, why, reason: r.reason, collisions: r.collisions, valid_laps: r.valid_laps, stall: r.stall,
    stall_max_frames: r.stall_max_frames, back_frames: r.back_frames, escapes: r.escapes, min_clearance: r.min_clearance, shield_frames: r.shield_frames, shield_fraction: sf,
    laps: (r.laps ?? []).map(l => l.elapsed), actions_sha256: crypto.createHash('sha256').update(JSON.stringify(acts)).digest('hex')});
}
out.summary = {scenes: out.scenes.length, valid: out.scenes.filter(s => s.valid).length,
  replaced: out.scenes.filter(s => !s.valid).map(s => ({name: s.name, why: s.why}))};
const txt = JSON.stringify(out, null, 1);
if (opt('--out')) fs.writeFileSync(opt('--out'), txt); else console.log(txt);
console.error(JSON.stringify(out.summary));
