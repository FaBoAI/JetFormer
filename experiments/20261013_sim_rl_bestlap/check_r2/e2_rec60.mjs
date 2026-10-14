#!/usr/bin/env node
// 確認 r2 の 2: 記録の 60 条件(REC60、連続 20 秒)を自前の道具で走らせ、周回時間・合否・すき間・ゲートと黒シートの余裕・荒さを数え直す。
//   node check_r2/e2_rec60.mjs <方策の名前(selected|r2_5M|r2_1M|r2_3M)>
// すき間は 2 通り: 自前の厳密な距離(フレームの終わりごと。RL の段の数え方と同じ粒度)と、太らせた車体の再生(小区間ごとの上下限)。
import {PLAN60, loadMlp, runPolicy, traceClearance, inflationBounds, gateLaterals, sheetMargins, roughness, hashOf, writeJson} from './common_r2.mjs';

const key = process.argv[2] ?? 'r2_5M';
const pol = loadMlp(key);
const res = {schema: 'rl-check-r2-rec60/v1', created_at: new Date().toISOString(), policy: key, policy_sha: pol.sha, sets: {}};
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
for (const [label, o] of [['delay0', {}], ['delay1_applied', {delay: 1, prev: 'applied'}], ['delay1_fresh', {delay: 1, prev: 'fresh'}]]) {
  const runs = PLAN60.map(c => ({name: c.name, ...runPolicy(pol, {plan: c, gate: c['start-gate'], offset: c['spawn-offset'], yawDeg: c['spawn-yaw'], mode: 'continuous', ...o})}));
  const ok = runs.filter(r => r.summary.valid);
  const S = {n: runs.length, valid: ok.length, lap1_mean: mean(ok.map(r => r.summary.laps[0])), lap2_mean: mean(ok.map(r => r.summary.laps[1])),
    lap1_min: Math.min(...ok.map(r => r.summary.laps[0])), lap1_max: Math.max(...ok.map(r => r.summary.laps[0])),
    lap2_min: Math.min(...ok.map(r => r.summary.laps[1])), lap2_max: Math.max(...ok.map(r => r.summary.laps[1])),
    failures: runs.filter(r => !r.summary.valid).map(r => ({name: r.name, ...r.summary}))};
  if (label === 'delay0') {
    // すき間(フレームごとの厳密な距離)
    const cl = runs.map(r => ({name: r.name, ...traceClearance(r.report, r.world)})).sort((a, b) => a.d - b.d);
    S.clearance_frame_min = cl[0]; S.clearance_frame_lowest5 = cl.slice(0, 5).map(c => ({name: c.name, d: +c.d.toFixed(4), who: c.who, frame: c.frame}));
    S.clearance_frame_below_3cm_runs = cl.filter(c => c.d < .03).length;
    // すき間(小区間ごと。太らせた車体の再生)
    S.clearance_substep_bounds = inflationBounds(runs);
    // ゲートの横の位置と黒シートの余裕
    const lat = runs.flatMap(r => gateLaterals(r.report, r.world));
    S.gate_lines_missed = lat.filter(g => !g.ok).length;
    S.gate_lateral_abs_max = Math.max(...lat.map(g => Math.abs(g.lateral)));
    const sh = runs.flatMap(r => sheetMargins(r.report, r.world));
    S.sheet_crossings = sh.length; S.sheet_margin_min = Math.min(...sh.map(s => s.margin_m));
    // 荒さ(60 本をまとめる)
    const rr = runs.map(r => roughness(r.cmds, r.report));
    S.rough = {steer_cmd_rate_p95_median_of_runs: rr.map(x => x.steer_cmd_rate_p95).sort((a, b) => a - b)[30], steer_edge_frac_mean: mean(rr.map(x => x.steer_edge_frac)),
      speed_zero_frac_mean: mean(rr.map(x => x.speed_zero_frac)), steer_cmd_rate_max: Math.max(...rr.map(x => x.steer_cmd_rate_max))};
    // 決定性: 1 本目をもう一度
    const again = runPolicy(pol, {plan: PLAN60[0], gate: PLAN60[0]['start-gate'], offset: PLAN60[0]['spawn-offset'], yawDeg: PLAN60[0]['spawn-yaw'], mode: 'continuous'});
    S.deterministic_repeat = hashOf(again.report.actions) === hashOf(runs[0].report.actions);
  }
  res.sets[label] = S;
  console.log(key, label, `${S.valid}/${S.n}`, S.lap1_mean?.toFixed(4), S.lap2_mean?.toFixed(4),
    S.clearance_frame_min ? `clr_frame ${S.clearance_frame_min.d.toFixed(4)} ${S.clearance_frame_min.who}` : '',
    S.clearance_substep_bounds ? JSON.stringify(S.clearance_substep_bounds) : '', S.failures.map(f => f.name + ':' + f.reason).join(' '));
}
writeJson(`e2_rec60_${key}.json`, res);
