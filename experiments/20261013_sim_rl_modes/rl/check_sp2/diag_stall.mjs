// 速さいろいろ(speed)の確認 r2: 止まったままの回の中身(2026-10-14、確認の段)。ランタイムの観測の補助の値(s・e_y・向きのずれ・d_min)と指令をフレームごとに出す
//   node diag_stall.mjs gate off yaw c_v [policy]
import fs from 'node:fs';
import crypto from 'node:crypto';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const HERE = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_sp2';
const core = await import(`${REL}/dist/sim-core.js`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
const RT = await import(`${HERE}/${process.env.CHK_RT === 'web' ? 'rt_web' : 'rt_current'}/mode_runtime.mjs`);
const {activeWorld, NOMINAL} = await import(`${HERE}/run_sp2.mjs`);
const [g, off, yaw, cv, pol = 'speed'] = process.argv.slice(2);
const PP = {speed: '/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/speed/r2/export/policy.json', speed_r1: '/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/speed/export/policy.json',
  fastest: '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/bundle/fastest/policy.json'}[pol];
const W = JSON.parse(fs.readFileSync(PP, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(PP)).digest('hex');
const base = await loadCourseWorld();
const {world, variation} = activeWorld(base, +g, +off, +yaw);
const cfg = {...NOMINAL, seed: 7, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
const sim = core.createSimulation(world, cfg); let snap = sim.reset({seed: 7, scenario: 'baseline'});
const T = RT.createModeTeacher(world, snap.config, variation, {mode: pol === 'fastest' ? 'fastest' : 'speed', weights: W, policySha256: sha, params: pol === 'fastest' ? {} : {c_v: +cv}});
const rows = [];
while (!snap.state.terminal) {
  const st = snap.state, out = T.action({state: st, metrics: snap.metrics, config: snap.config}), lg = T.log.at(-1);
  rows.push({f: st.frame, x: +st.x.toFixed(3), z: +st.z.toFixed(3), yaw: +st.yaw.toFixed(3), v: +st.speed.toFixed(3), s: +lg.s.toFixed(3), ey: +lg.ey.toFixed(3), epsi: +lg.epsi.toFixed(3), dmin: +lg.dmin.toFixed(3),
    ri: snap.metrics.route_index, steer: +out.steering.toFixed(3), speed: +out.speed.toFixed(3), a0: +lg.a0.toFixed(3), a1: +lg.a1.toFixed(3)});
  snap = sim.step({steering: out.steering, speed: out.speed});
}
const rep = sim.report();
console.log(JSON.stringify({start: {g: +g, off: +off, yaw: +yaw, cv: +cv, pol}, reason: rep.final_state.reason, laps: rep.metrics.lap_events?.length, gates: rep.metrics.passed_gates, distance: rep.metrics.distance}));
for (const r of rows) if (r.f < 40 || r.f % 60 === 0) console.log(JSON.stringify(r));
