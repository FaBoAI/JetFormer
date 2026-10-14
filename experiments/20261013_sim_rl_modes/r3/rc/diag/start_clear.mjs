// 試験・検証の組の開始のすき間だけを数える(方策は走らせない。組の定義の確かめ)
import {buildJobs} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/rc/eval_rc3.mjs';
for (const set of ['validation', 'test']) {
  const J = buildJobs(set);
  const by = {};
  for (const j of J) { const c = j.spec.start.initial_clearance_m; if (c == null) continue; (by[j.test] ??= []).push([j.id, c]); }
  for (const [t, a] of Object.entries(by)) {
    const low = a.filter(([, c]) => c < .03).map(([id, c]) => `${id}:${(c * 100).toFixed(2)}cm`);
    const low35 = a.filter(([, c]) => c < .035).length;
    console.log(set, t, 'n', a.length, '3 cm 未満', low.length, low.join(' '), '3.5 cm 未満', low35, '最小', (Math.min(...a.map(x => x[1])) * 100).toFixed(2));
  }
}
