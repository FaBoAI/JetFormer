// 準備の段の確かめ: 余裕の線の先生の単周 E0 の周回時間。設計の段の measure_teachers.mjs は「前回の行動」を commandToAction(適用した操作) で渡していた(cta)。
// この作業の決まり(protocol: 乱していないフレームの次は先生自身の値)は own。物理・world は設計の段と同じ(modes の snapshot)。2026-10-14
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const MODES = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const {activeWorldFor, simConfig} = await import(pathToFileURL(MODES + '/snapshot/rl/env_core.mjs').href);
const core = await import(pathToFileURL(MODES + '/snapshot/sim/dist/sim-core.js').href);
const {createTeacherCore, MARGIN_PARAMS} = await import('/home/jetson/jetformer/experiments/20261014_sim_dart_margin/web/teacher_core.mjs');
const {commandToAction} = await import('/home/jetson/jetformer/experiments/20261014_sim_dart_margin/web/rt/features.mjs');
const WM = (await import('/home/jetson/jetformer/experiments/20261014_sim_dart_margin/web/rt/weights_margin.mjs')).default;
const BASE = JSON.parse(fs.readFileSync(MODES + '/snapshot/sim/world-v3.json', 'utf8'));
for (const mode of ['own', 'cta']) {
  const {world} = activeWorldFor(BASE, 0, 0, 0);
  const sim = core.createSimulation(world, simConfig({episodeMode: 'single', timeLimit: 15, seed: 1}));
  let snap = sim.reset({seed: 1, scenario: 'baseline'});
  const c = createTeacherCore('margin', world, JSON.parse(JSON.stringify(snap.config)), WM);
  let prev = [0, 0];
  while (!snap.state.terminal) {
    if (snap.state.frame === 0) c.begin(snap.state, snap.metrics.route_index); else c.advance(snap.state, snap.metrics.route_index);
    const d = c.decide(snap.state, prev[0], prev[1]);
    if (mode === 'own') prev = [d.a0, d.a1]; else { const a = commandToAction(d.steering, d.speed, 4.5, .52); prev = [a.a0, a.a1]; }
    snap = sim.step({steering: d.steering, speed: d.speed});
  }
  console.log(mode, snap.state.reason, snap.metrics.elapsed);
}
console.log(JSON.stringify(simConfig({episodeMode: 'single', timeLimit: 15, seed: 1})));
