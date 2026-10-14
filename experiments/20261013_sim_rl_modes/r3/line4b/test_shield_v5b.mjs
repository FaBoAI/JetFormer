#!/usr/bin/env node
// シールド v5b の試験(2026-10-14 夜、line r4b): rt4b/shield.mjs に select 'v5' を渡したときの動きが、元の v5(r3/line4/rt4/shield.mjs =
// no_wall r4 の r3/nw4/runtime_v5/shield.mjs、8ece9be4…)と同じことを、1 回まるごとの操作の列で確かめる(元の sim-core.js)。
// 組: dev4b の DEV-R3B-FAIL16(シールドがよく働く 16 回)+ LN-FAM6。方策は r3 の方策(9f10c7b7…)。値は protocol_line_r4.json の shield_params(select 'v5')。
// あわせて、no_wall r4 の値(シールドの既定 = v5 の値、select を渡さない)で 4 回を比べる。
//   node r3/line4b/test_shield_v5b.mjs [--out 結果.json]
import fs from 'node:fs';
import crypto from 'node:crypto';
import {runModeEpisode as runOld} from '../line4/episode_ln4.mjs';
import {runModeEpisode as runNew, readJson} from './episode_ln4.mjs';
import {buildJobs} from './eval_line_r4b.mjs';

const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
const pol = `${REP}/r3/line_r4/train/main/policies/step_000000000.json`;
const weights = readJson(pol), sha = crypto.createHash('sha256').update(fs.readFileSync(pol)).digest('hex');
const p4 = readJson(`${REP}/r3/protocol_line_r4.json`);
const mrV5 = p4.runtime.shield.mode_runtime;
if (mrV5.shield_params.select !== 'v5') throw new Error('protocol_line_r4 の select が v5 ではありません');
const sh = f => crypto.createHash('sha256').update(fs.readFileSync(new URL(f, import.meta.url))).digest('hex');
const out = {schema: 'jf-sim-rl-line-r4b-shield-test/v1', created_at: new Date().toISOString(), old_shield_sha256: sh('../line4/rt4/shield.mjs'),
  new_shield_sha256: sh('./rt4b/shield.mjs'), policy_sha256: sha, runs: []};
const sig = r => JSON.stringify({a: r.actions, reason: r.reason, frames: r.frames, sh: r.shield_frames, log: r.log.map(q => [q.sh, q.ph, q.st, q.sp, q.shield?.fb ?? null])});
const {jobs} = buildJobs('dev4b');
const pick = jobs.filter(j => j.test === 'DEV-R3B-FAIL16' || j.test === 'LN-FAM6');
const cases = pick.map(j => ({id: `${j.test}:${j.id}`, start: j.spec.start, params: j.spec.params, modeRuntime: mrV5}));
// no_wall の形の値(既定 = v5。c_nominal・止まり・抜け出しなどは既定のまま)を line の方策で 4 回(操作の列が同じかだけを見る)
const nwLike = {shield: true, shield_params: {}};
for (const id of ['GR-2', 'BOX17-44']) {
  const j = buildJobs('regression').jobs.find(x => x.id === id);
  if (j) cases.push({id: `nwdefault:${id}`, start: j.spec.start, params: j.spec.params, modeRuntime: nwLike});
}
for (const j of pick.slice(0, 2)) cases.push({id: `nwdefault:${j.id}`, start: j.spec.start, params: j.spec.params, modeRuntime: nwLike});
let same = 0;
for (const c of cases) {
  const a = await runOld({mode: 'line', weights, policySha256: sha, start: c.start, params: c.params, episodeMode: 'continuous', coreKind: 'original', audit: false, modeRuntime: c.modeRuntime});
  const b = await runNew({mode: 'line', weights, policySha256: sha, start: c.start, params: c.params, episodeMode: 'continuous', coreKind: 'original', audit: false, modeRuntime: c.modeRuntime});
  const ok = sig(a) === sig(b);
  same += ok;
  out.runs.push({id: c.id, same: ok, frames: a.frames, shield_frames_old: a.shield_frames, shield_frames_new: b.shield_frames,
    meta_shield_old: a.teacher_metadata?.shield?.select ?? null, meta_shield_new: b.teacher_metadata?.shield?.select ?? null});
  console.error(c.id, ok, a.shield_frames, b.shield_frames);
}
out.summary = {cases: cases.length, identical: same, pass: same === cases.length};
const o = process.argv.indexOf('--out');
if (o > 0) fs.writeFileSync(process.argv[o + 1], JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.summary));
