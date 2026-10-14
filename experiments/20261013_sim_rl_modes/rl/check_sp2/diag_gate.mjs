// 速さいろいろ(speed)の確認 r2: ゲートを外した回・ぶつかった回の中身(2026-10-14、確認の段)。sim の route_index の進みとゲートの横ずれを自前で数える
//   node diag_gate.mjs gate off yaw c_v [policy]
import fs from 'node:fs';
import crypto from 'node:crypto';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const HERE = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_sp2';
const core = await import(`${REL}/dist/sim-core.js`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
const RT = await import(`${HERE}/rt_current/mode_runtime.mjs`);
const {activeWorld, NOMINAL} = await import(`${HERE}/run_sp2.mjs`);
const {buildObstacles, ownClearance} = await import(`${HERE}/own_geom.mjs`);
const [g, off, yaw, cv, pol = 'speed'] = process.argv.slice(2);
const PP = {speed: '/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/speed/r2/export/policy.json', speed_r1: '/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/speed/export/policy.json',
  fastest: '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/bundle/fastest/policy.json'}[pol];
const W = JSON.parse(fs.readFileSync(PP, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(PP)).digest('hex');
const base = await loadCourseWorld(); const OB = buildObstacles(base);
const {world, variation} = activeWorld(base, +g, +off, +yaw);
const cfg = {...NOMINAL, seed: 7, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
const sim = core.createSimulation(world, cfg); let snap = sim.reset({seed: 7, scenario: 'baseline'});
const T = RT.createModeTeacher(world, snap.config, variation, {mode: pol === 'fastest' ? 'fastest' : 'speed', weights: W, policySha256: sha, params: pol === 'fastest' ? {speed_limit: 4.5 * +cv} : {c_v: +cv}});
let lastRi = 0;
const ev = [];
while (!snap.state.terminal) {
  const st = snap.state, out = T.action({state: st, metrics: snap.metrics, config: snap.config}), lg = T.log.at(-1);
  if (st.frame % 6 === 0 || snap.metrics.route_index !== lastRi) ev.push({f: st.frame, x: +st.x.toFixed(3), z: +st.z.toFixed(3), v: +st.speed.toFixed(2), ri: snap.metrics.route_index, gates: snap.metrics.passed_gates,
    s: +lg.s.toFixed(2), ey: +lg.ey.toFixed(3), epsi: +lg.epsi.toFixed(2), dmin: +ownClearance(OB, st.x, st.z, st.yaw).toFixed(3), missed: lg.missed ?? null, steer: +out.steering.toFixed(3), speed: +out.speed.toFixed(2)});
  lastRi = snap.metrics.route_index;
  snap = sim.step({steering: out.steering, speed: out.speed});
}
const rep = sim.report();
console.log(JSON.stringify({start: {g: +g, off: +off, yaw: +yaw, cv: +cv, pol}, reason: rep.final_state.reason, laps: (rep.metrics.lap_events ?? []).map(l => [l.elapsed.toFixed(2), l.distance.toFixed(1), l.start_frame, l.end_frame]),
  gates: rep.metrics.passed_gates, distance: rep.metrics.distance, collision: rep.metrics.collision_events?.[0] ?? null, gateHalfWidth: rep.config.gateHalfWidth}));
for (const e of ev) if (e.f <= +(process.env.MAXF ?? 200)) console.log(JSON.stringify(e));
