#!/usr/bin/env node
// 確認 r2 の 1: 取り違え・コードの変更の影響と、学習済み ONNX と同じ条件の比べ(must-fix の数字)を、自前の道具で作り直す。
//   node check_r2/e1_integrity_samecond.mjs   (作業フォルダーは experiments/20261013_sim_rl_bestlap)
// - 書き出した方策の sha256(選んだ方策は r1 から変わっていないか、r2/dr/policy.json は 500 万の policies と同じか)
// - 選んだ方策の STD の操作列が、確認 r1 の sha256 と RL の段の記録(rl/final/records)と同じか(r2 のコード変更で既定の動きが変わっていないか)
// - 遅延 1・遅延 1 + 前進ゲイン 1.01 を、ユーザーの control-delay.js の待ち行列で。観測の「前回の行動」の約束 2 通り
import fs from 'node:fs';
import {POLICIES, REPORTS, loadMlp, runPolicy, runTeacher, hashOf, writeJson, sha256File, WORLD_VERSION, userVerifier, sheetMargins, gateLaterals} from './common_r2.mjs';

const out = {schema: 'rl-check-r2-e1/v1', created_at: new Date().toISOString()};
// 1) 取り違え
out.files = {
  selected_policy_sha: sha256File(POLICIES.selected.path),
  r2_dr_policy_sha: sha256File(POLICIES.r2_5M.path),
  r2_step5M_policy_sha: sha256File(REPORTS + '/rl/r2-dr-s11/policies/step_005013504.json'),
  r2_step4M_policy_sha: sha256File(REPORTS + '/rl/r2-dr-s11/policies/step_004014080.json'),
};
out.files.r2_dr_equals_step5M = out.files.r2_dr_policy_sha === out.files.r2_step5M_policy_sha;
out.files.r2_step4M_differs = out.files.r2_step4M_policy_sha !== out.files.r2_step5M_policy_sha;
const meta = JSON.parse(fs.readFileSync(POLICIES.r2_5M.path, 'utf8')).meta;
out.files.r2_dr_meta = meta;

const sel = loadMlp('selected'), r2 = loadMlp('r2_5M');
// 2) 選んだ方策の STD が r1・RL の記録と同じ操作列か
const R1_STD_SINGLE_ACTIONS_SHA = '17c9dc00fd5caafb3b06314899685e3efa4960564293395e7ea648525b392e63';
const recSingle = JSON.parse(fs.readFileSync(REPORTS + '/rl/final/records/rl_main_STD_single.json', 'utf8'));
const recCont = JSON.parse(fs.readFileSync(REPORTS + '/rl/final/records/rl_main_STD_continuous.json', 'utf8'));
const s1 = runPolicy(sel, {mode: 'single'}), s2 = runPolicy(sel, {mode: 'single'}), c1 = runPolicy(sel, {mode: 'continuous'});
out.selected_std = {
  single: s1.summary, continuous: c1.summary,
  single_actions_sha: hashOf(s1.report.actions), r1_sha: R1_STD_SINGLE_ACTIONS_SHA, record_single_actions_sha: hashOf(recSingle.actions),
  continuous_actions_sha: hashOf(c1.report.actions), record_continuous_actions_sha: hashOf(recCont.actions),
  deterministic_repeat: hashOf(s1.report.actions) === hashOf(s2.report.actions),
};
out.selected_std.matches_r1 = out.selected_std.single_actions_sha === R1_STD_SINGLE_ACTIONS_SHA;
out.selected_std.matches_record = out.selected_std.single_actions_sha === out.selected_std.record_single_actions_sha
  && out.selected_std.continuous_actions_sha === out.selected_std.record_continuous_actions_sha;
console.log('selected STD', JSON.stringify(out.selected_std.single), JSON.stringify(out.selected_std.continuous.laps), 'r1', out.selected_std.matches_r1, 'record', out.selected_std.matches_record);

// 3) 同じ条件(遅延 1・前進ゲイン 1.01)。単周と連続 20 秒
const conds = [
  {name: '遅延 0', o: {}},
  {name: '遅延 1・前回 = 適用した行動', o: {delay: 1, prev: 'applied'}},
  {name: '遅延 1・前回 = 方策の出力', o: {delay: 1, prev: 'fresh'}},
  {name: '遅延 1 + ゲイン 1.01・前回 = 適用した行動', o: {delay: 1, prev: 'applied', speedGain: 1.01}},
  {name: '遅延 1 + ゲイン 1.01・前回 = 方策の出力', o: {delay: 1, prev: 'fresh', speedGain: 1.01}},
];
out.same_condition = {};
for (const [key, pol] of [['selected', sel], ['r2_5M', r2]]) {
  out.same_condition[key] = conds.map(c => {
    const s = runPolicy(pol, {mode: 'single', ...c.o}), k = runPolicy(pol, {mode: 'continuous', ...c.o});
    console.log(key, c.name, JSON.stringify(s.summary.laps), s.summary.reason, JSON.stringify(k.summary.laps), k.summary.reason);
    return {name: c.name, single: s.summary, continuous: k.summary};
  });
}
out.same_condition.teacher = [{name: '遅延 0', o: {}}, {name: '遅延 1', o: {delay: 1}}, {name: '遅延 1 + ゲイン 1.01', o: {delay: 1, speedGain: 1.01}}].map(c => {
  const s = runTeacher({mode: 'single', ...c.o});
  console.log('teacher', c.name, JSON.stringify(s.summary));
  return {name: c.name, single: s.summary};
});
const onnx = JSON.parse(fs.readFileSync('/home/jetson/sim/extracted/course-simulator/results/shortcut-rc-learned-lap.json', 'utf8'));
out.onnx_record = {elapsed: onnx.metrics.elapsed, delay_ticks: onnx.context?.execution?.control_delay?.ticks ?? null,
  throttle_gain: onnx.context?.policy_info?.calibration?.throttle_gain ?? null, steer_gain: onnx.context?.policy_info?.calibration?.steer_gain ?? null};

// 4) r2 の 500 万の STD 単周を、ユーザーの確認器に当てる(RL の周は ONNX の証跡が無いので、証跡の項目は落ちる。物理・判定の項目を見る)
{
  const s = runPolicy(r2, {mode: 'single'});
  const rep = s.report;
  rep.context = {controller: 'external', world_version: WORLD_VERSION, start_variation: {gate: 0, left_offset_m: 0, left_yaw_deg: 0}};
  const v = userVerifier.verifyLearnedReport(rep, {expectedWorld: s.world, expectedSha256: POLICIES.r2_5M.sha});
  const phys = /collision|route_complete|Finished|ordered gates|mandatory crossing|Mandatory crossing|world_vs_current_scene|benchmark physics|stopOnCollision|replay|Nominal start/i;
  out.r2_user_verifier = {pass: v.pass, failure_count: v.failures.length,
    physics_or_judgement_failures: v.failures.filter(x => phys.test(x) && !/ONNX|model|sensor|inference|policy/i.test(x)),
    other_failures_sample: v.failures.filter(x => !(phys.test(x) && !/ONNX|model|sensor|inference|policy/i.test(x))).slice(0, 40),
    replay: v.replay ?? null, sheet: sheetMargins(rep, s.world), gate_lateral_max: Math.max(...gateLaterals(rep, s.world).map(g => Math.abs(g.lateral)))};
  console.log('r2 user verifier', out.r2_user_verifier.failure_count, JSON.stringify(out.r2_user_verifier.physics_or_judgement_failures));
}
writeJson('e1_integrity_samecond.json', out);
