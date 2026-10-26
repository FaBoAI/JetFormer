// 反証 V-replay: 較正したシミュレーター(P_cal = profile_real_v1 と同じ値、jf-calib-v1)の 1 フレームずつの橋(2026-10-25)。
// Python(sim_ceiling.py)が天井・自動合わせを計算し、この橋はシミュレーターを 1 フレーム進めるだけ。読むだけ。
//
//   標準入力に 1 行 1 つの JSON:
//     {"op":"reset","start":<id>,"k":1.4,"vmax":5.5,"kBatt":1.0,"world":"W0"} → 状態と、運転手 PP の次の指令
//     {"op":"step","steering":<rad>,"speed":<m/s の指令>}                         → 1 フレーム進めた状態と、PP の次の指令
//     {"op":"quit"}
//   状態: t・x・z・yaw・v・steer(実際の舵角)・yaw_rate [deg/s](左 +)・s(参照ループの進み)・sec(区間)・terminal・reason・
//         collisions・collision(最初の衝突の区間と速さ)・laps(シミュレーターの周の数)
// ユーザーのシミュレーター・方策・世界は読むだけ(cap_test.mjs・pp_driver.mjs・calib_core.mjs の関数を使う)。
import readline from 'node:readline';
import fs from 'node:fs';
import {loadCalibratedCore, loadProfile} from '/home/jetson/jetformer/experiments/20261016_sim_calibration/calib_core.mjs';
import {compileProfile} from '/home/jetson/jetformer/experiments/20261016_sim_calibration/calib_runtime.mjs';
import {createLoop, createProgressTracker, SECTIONS_R} from '/home/jetson/jetformer/experiments/20261016_sim_calibration/cap_runtime.mjs';
import {speedProfile, physicsMaps, createPPDriver, REFERENCE_LOOP_JSON} from '/home/jetson/jetformer/experiments/20261016_sim_calibration/pp_driver.mjs';
import {loadWorld, activeWorldFor, makeStarts, ppPath, PHYSICS_CONFIG} from '/home/jetson/jetformer/experiments/20261016_sim_calibration/cap_test.mjs';

const {profile, sha256} = loadProfile('P_cal');
const core = await loadCalibratedCore(profile, {fast: true, profileSha256: sha256, brakeScenario: 'nominal'});
const compiled = compileProfile(profile, {brakeScenario: 'nominal'});
const loop = createLoop(JSON.parse(fs.readFileSync(REFERENCE_LOOP_JSON, 'utf8')));
const starts = makeStarts(loadWorld('W0')).starts;
const SEC_ORDER = ['HS', 'LH', 'R1', 'GM', 'DE', 'SW'];
function sectionOfS(s) {
  const u = ((s % loop.L) + loop.L) % loop.L;
  for (const [k, [a, b]] of Object.entries(SECTIONS_R)) if (u >= a && u < b) return k.startsWith('HS') ? 'HS' : k;
  for (const k of SEC_ORDER) { const [a] = k === 'HS' ? SECTIONS_R.HS_b : SECTIONS_R[k]; if (u < a) return k; }
  return 'HS';
}
let sim = null, driver = null, tracker = null, prevYaw = null, snap = null;
function state() {
  const st = snap.state, m = snap.metrics;
  const s = st.frame === 0 ? tracker.reset(st.x, st.z, 0) : tracker.advance(st.x, st.z);
  const dyaw = prevYaw === null ? 0 : Math.atan2(Math.sin(st.yaw - prevYaw), Math.cos(st.yaw - prevYaw));
  prevYaw = st.yaw;
  const pp = driver.action(st);
  const ce = m.collision_events?.[0] ?? null;
  return {t: st.sim_time, frame: st.frame, x: st.x, z: st.z, yaw: st.yaw, v: st.speed, steer: st.steering,
    yaw_rate: dyaw / PHYSICS_CONFIG.dt * 180 / Math.PI, s, sec: sectionOfS(s), terminal: st.terminal, reason: st.reason,
    collisions: m.collisions, collision: ce ? {x: ce.x, z: ce.z, sec: sectionOfS(loop.projectWindow(ce.x, ce.z, s - 1.0, s + 0.5, {s: 0, d: 0}).s),
      collider: ce.collider, kind: ce.kind} : null, laps: m.completed_laps ?? 0, pp: {steering: pp.steering, speed: pp.speed, vref: pp.info.vref}};
}
const rl = readline.createInterface({input: process.stdin});
for await (const line of rl) {
  const c = JSON.parse(line);
  if (c.op === 'quit') break;
  if (c.op === 'reset') {
    const base = loadWorld(c.world ?? 'W0');
    const st0 = starts[c.start ?? 0];
    const {world} = activeWorldFor(base, st0.gate, st0.left_offset_m, st0.left_yaw_deg);
    const config = {...PHYSICS_CONFIG, seed: 1, timeLimit: c.timeLimit ?? 90, episodeMode: 'continuous-laps-v1'};
    sim = core.createSimulation(world, config);
    sim.reset({seed: 1, scenario: 'baseline'});
    if ((c.kBatt ?? 1) !== 1) sim.calibControl({kBatt: c.kBatt});
    const pp = ppPath(), maps = physicsMaps('cal', compiled);
    driver = createPPDriver(pp.path, speedProfile(pp.path, base, maps, c.k ?? 1.0, {vmax: c.vmax ?? 4.3}), maps, {k: c.k ?? 1.0, wheelbase: config.wheelbase});
    tracker = createProgressTracker(loop, world.route);
    prevYaw = null;
    snap = sim.peek();
    process.stdout.write(JSON.stringify(state()) + '\n');
  } else if (c.op === 'step') {
    sim.step({steering: c.steering, speed: c.speed}, true);
    snap = sim.peek();
    process.stdout.write(JSON.stringify(state()) + '\n');
  }
}
process.exit(0);
