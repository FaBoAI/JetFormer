#!/usr/bin/env node
// no_wall r4b の下調べ(2026-10-14 20 時、電源断のあとの再開で足した): 学習の途中の点が「ゲート 16〜18 の右へずれて左を向いた開始」で止まらなくなったかを、
// 下調べの組 dev4e の点だけで見る(検証・試験の組は使わない。選びには使わない)。シールド v5 の既定の値(= 決まりの値)。
//   node probe_box_nw4b.mjs <policy.json> [--gates 15,16,17,18,19] [--all-signs] > 1 行 1 回の JSONL
import {buildDev4EdgeJobs} from './nw4_sets.mjs';
import {runOne} from './run_nw4r.mjs';

const a = process.argv.slice(2);
const pol = a[0];
const opt = (k, d = null) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
const gates = (opt('--gates') ?? '15,16,17,18,19').split(',').map(Number);
const allSigns = a.includes('--all-signs');
const J = buildDev4EdgeJobs().filter(j => gates.includes(j.spec.start.gate) && (allSigns || (j.spec.start.left_offset_m < 0 && j.spec.start.left_yaw_deg > 0)));
for (const j of J) {
  const r = await runOne({...j.spec, policyPath: pol});
  const s = j.spec.start;
  console.log(JSON.stringify({id: j.id, gate: s.gate, off: s.left_offset_m, yaw: s.left_yaw_deg, stall: r.stall, collision: r.collision, valid_laps: r.valid_laps,
    valid_episode_r3: r.valid_episode_r3, min_cm: +(100 * r.min_clearance).toFixed(2), escapes: r.escapes, back_frames: r.back_frames, policy: r.policy_sha}));
}
