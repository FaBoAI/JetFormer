#!/usr/bin/env node
// 確認の段(review、2026-10-14): 先生(T_fast・T_margin、遅延 0)を、ユーザーの有効な版の dist/sim-core.js で E0〜E9(ゲート 0 の開始)から走らせ、
// 点 4 のゲートの線を越えた位置を線形補間で測る(evaluation.md 5 章は sim が数えた最初のフレームの位置で測ったおよその値)。
// 先生は記録・評価と同じ web/teacher_core.mjs(前回の行動 = 先生自身の値)。走行の world・config は評価の走行の記録から読む(読むだけ)。
//   node teacher_gate4.mjs > review/teacher_gate4.json
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const REL = '/home/jetson/jetformer/data/sim/releases/' + JSON.parse(fs.readFileSync('/home/jetson/jetformer/data/sim/active.json', 'utf8')).release;
const {createSimulation} = await import(pathToFileURL(REL + '/dist/sim-core.js').href);
const EXP = '/home/jetson/jetformer/experiments/20261014_sim_dart_margin';
const {createTeacherCore} = await import(pathToFileURL(EXP + '/web/teacher_core.mjs').href);
const W = {fast: (await import(pathToFileURL(EXP + '/web/rt/weights_fast.mjs').href)).default,
  margin: (await import(pathToFileURL(EXP + '/web/rt/weights_margin.mjs').href)).default};
const CL = '/home/jetson/jetformer_reports/sim_dart_20261014/eval/closed_loop/A1-s0';
const out = {note: '点 4 のゲートの線を越えた位置(線形補間)。横は曲がりの内側(点 5 の側)を負。余裕 = 0.65 m − |横|', runs: {}};
for (const e of ['E0', 'E1', 'E2', 'E3', 'E4', 'E6', 'E7', 'E8', 'E9']) {
  const rep = JSON.parse(fs.readFileSync(`${CL}/A1-s0-${e}-single.json`, 'utf8'));
  const route = rep.world.route, p3 = route[3], p4 = route[4], p5 = route[5];
  const n = Math.hypot(p4[0] - p3[0], p4[1] - p3[1]), ux = (p4[0] - p3[0]) / n, uz = (p4[1] - p3[1]) / n;
  let lx = -uz, lz = ux;
  if ((p5[0] - p4[0]) * lx + (p5[1] - p4[1]) * lz > 0) { lx = -lx; lz = -lz; }
  out.runs[e] = {};
  for (const kind of ['fast', 'margin']) {
    const sim = createSimulation(rep.world, rep.config);
    let snap = sim.reset({seed: rep.seed, scenario: rep.scenario});
    const core = createTeacherCore(kind, rep.world, JSON.parse(JSON.stringify(snap.config)), W[kind]);
    let f = 0, prev = [0, 0];
    while (!snap.state.terminal) {
      const st = snap.state, ri = snap.metrics.route_index;
      if (f === 0) core.begin(st, ri); else core.advance(st, ri);
      const d = core.decide(st, prev[0], prev[1]);
      prev = [d.a0, d.a1];
      snap = sim.step({steering: d.steering, speed: d.speed});
      f++;
    }
    const r = sim.report();
    let cross = null, pa = null, pp = null;
    for (const t of r.trace) {
      const a = (t.x - p4[0]) * ux + (t.z - p4[1]) * uz;
      if (pa !== null && pa < 0 && a >= 0 && a - pa < .5) {
        const w = -pa / (a - pa), x = pp.x + w * (t.x - pp.x), z = pp.z + w * (t.z - pp.z);
        const lat = (x - p4[0]) * lx + (z - p4[1]) * lz;
        cross = {lateral_m: +lat.toFixed(4), margin_cm: +((.65 - Math.abs(lat)) * 100).toFixed(1)};
        break;
      }
      pa = a; pp = t;
    }
    out.runs[e][kind] = {reason: r.outcome.reason, lap_s: +r.metrics.elapsed.toFixed(3), gates: r.metrics.passed_gates, gate4: cross};
  }
}
const m = k => Object.values(out.runs).map(v => v[k].gate4?.margin_cm).filter(Number.isFinite);
out.summary = {fast_margin_cm_min_max: [Math.min(...m('fast')), Math.max(...m('fast'))], margin_margin_cm_min_max: [Math.min(...m('margin')), Math.max(...m('margin'))]};
console.log(JSON.stringify(out, null, 1));
