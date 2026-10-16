// 上限の試験(S4)の試験。node --test tests/cap.test.mjs
//   T7  ブラウザ用の上限つきのコア(calib_cap.html が読むファイル)= patchCapCore の出力。読み込んで走らせると、
//       Node の上限の試験(較正版のコア + calibControl({filter}) に createCapFilter を差す形)と report が完全に同じ
//   T8  ジャイロの位相の写し(cap_runtime.phaseStep・combineZone)= jetformer/safety_caps.py(3,600 サンプル)
//   T9  進み s の追跡(cap_runtime)= features.mjs の createFeatureTracker(射影の距離 ≤ 0.3 m の間。RL の走行)
//   T10 上限の層の不変条件(乱数の入力 20,000 フレーム): 出力 ≤ 入力・ブレーキは R1_app だけ・1 m/s 未満で始めない・やめて 0.5 秒は始めない
//   T11 PP の経路が (a)(a')(b)(c) を満たす。上限の層は舵を変えない
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {loadCalibratedCore, loadProfile, ORIGINAL_SIM_CORE} from '../calib_core.mjs';
import {compileProfile, governorStep, governorInitialState, GOVERNOR_V1_SIM} from '../calib_runtime.mjs';
import {createLoop, createProgressTracker, createCapFilter, phaseStep, phaseInitialState, combineZone} from '../cap_runtime.mjs';
import {patchCapCore, CAP_SPEC_O25} from '../build_cap_core.mjs';
import {DIST} from '../build_browser_core.mjs';
import {speedProfile, physicsMaps, createPPDriver, REFERENCE_LOOP_JSON} from '../pp_driver.mjs';
import * as CT from '../cap_test.mjs';
import {createFeatureTracker} from '/home/jetson/jetformer/jetformer/sim/static/rl/features.mjs';
import {createModeTeacher} from '/home/jetson/jetformer/jetformer/sim/static/rl/mode_runtime.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const REPORT = '/home/jetson/jetformer_reports/sim_calibration_20261016';
const loop = createLoop(JSON.parse(fs.readFileSync(REFERENCE_LOOP_JSON, 'utf8')));
const {profile, sha256: psha} = loadProfile('P_cal');
const CAP_FILE = path.join(DIST, '_calib', `sim-core-${profile.name}-${CAP_SPEC_O25.name}.js`);
const importText = async text => import(`data:text/javascript;base64,${Buffer.from(text).toString('base64')}`);

// 1 走行(アプリの遅れ 1 フレーム)。filterMode: 'external'(calibControl に差す)/ 'builtin'(コアが自分で差す)
async function runWith(core, filterMode, driverName, start) {
  const base = CT.loadWorld('W0');
  const {world, variation} = CT.activeWorldFor(base, start.gate, start.left_offset_m, start.left_yaw_deg);
  const config = {...CT.PHYSICS_CONFIG, seed: 1, timeLimit: 60, episodeMode: 'continuous-laps-v1'};
  const sim = core.createSimulation(world, config);
  sim.reset({seed: 1, scenario: 'baseline'});
  if (filterMode === 'external') {
    const cap = createCapFilter({loop, route: world.route, governorStep, governorInitialState, params: {...GOVERNOR_V1_SIM}, zoneMode: 'oracle', name: CAP_SPEC_O25.name});
    sim.calibControl({filter: cap.filter, filterName: cap.name});
  }
  let snap = sim.snapshot();
  const cfgN = JSON.parse(JSON.stringify(snap.config));
  let drv, isRl = false;
  if (driverName.startsWith('PP-')) {
    const k = +driverName.slice(3), pp = CT.ppPath(), maps = physicsMaps('cal', compileProfile(profile));
    drv = createPPDriver(pp.path, speedProfile(pp.path, base, maps, k), maps, {k});
  } else {
    isRl = true;
    const w = JSON.parse(fs.readFileSync('/home/jetson/jetformer/data/sim_rl/modes/speed/policy.json', 'utf8'));
    drv = createModeTeacher(world, cfgN, variation, {mode: 'speed', weights: w, params: {c_v: .75}});
  }
  const q = [];
  let applied = null;
  while (!snap.state.terminal) {
    let ap;
    if (isRl) { const o = drv.action(snap, {applied: applied ?? [0, 0]}); const lg = drv.log.at(-1); ap = {steering: o.steering, speed: o.speed, a: [lg.a0, lg.a1]}; }
    else { const o = drv.action(snap.state); ap = {steering: o.steering, speed: o.speed, a: null}; }
    q.push(ap); ap = q.length > 1 ? q.shift() : {steering: 0, speed: 0, a: [0, -1]}; applied = ap.a;
    snap = sim.step({steering: ap.steering, speed: ap.speed});
  }
  const rep = sim.report();
  return {rep, cal: sim.calibControl({})};
}

test('T7 ブラウザ用の上限つきのコア = patchCapCore の出力で、Node の上限の試験と同じ動き', async () => {
  const source = fs.readFileSync(ORIGINAL_SIM_CORE, 'utf8');
  const text = patchCapCore(source, profile, CAP_SPEC_O25, {name: profile.name, profileSha256: psha});
  assert.equal(fs.readFileSync(CAP_FILE, 'utf8'), text, 'ブラウザ用のファイルが patchCapCore の出力と違う');
  const html = fs.readFileSync(path.join(DIST, 'calib_cap.html'), 'utf8').split('\n'), idx = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8').split('\n');
  assert.equal(html.length, idx.length);
  const diff = html.map((l, i) => l === idx[i] ? null : i).filter(i => i !== null);
  assert.equal(diff.length, 2, 'calib_cap.html と index.html の差は 2 行(title と import map)');
  const capCore = await importText(fs.readFileSync(CAP_FILE, 'utf8'));
  const calCore = await loadCalibratedCore(profile, {fast: false, profileSha256: psha});
  const starts = CT.makeStarts(CT.loadWorld('W0')).starts;
  let compared = 0;
  for (const drv of ['PP-1.4', 'RL-speed-0.75']) for (const si of [0, 3]) {
    const a = await runWith(calCore, 'external', drv, starts[si]);
    const b = await runWith(capCore, 'builtin', drv, starts[si]);
    assert.equal(b.cal.filter, CAP_SPEC_O25.name, 'ブラウザ用のコアが上限の層を差していない');
    assert.equal(JSON.stringify(b.rep), JSON.stringify(a.rep), `${drv} 開始 ${si}: report が違う`);
    compared += a.rep.actions.length;
  }
  // 上限が効いていること(上限なしと違う)
  const off = await runWith(calCore, 'none', 'RL-speed-0.75', starts[0]);
  const on = await runWith(capCore, 'builtin', 'RL-speed-0.75', starts[0]);
  assert.notEqual(JSON.stringify(off.rep.trace), JSON.stringify(on.rep.trace));
  // reset のあとも上限の層が戻る
  const base = CT.loadWorld('W0'), {world} = CT.activeWorldFor(base, 0, 0, 0);
  const sim = capCore.createSimulation(world, {...CT.PHYSICS_CONFIG, seed: 1, timeLimit: 5});
  sim.reset({seed: 1}); assert.equal(sim.calibControl({}).filter, CAP_SPEC_O25.name);
  sim.reset({seed: 2}); assert.equal(sim.calibControl({}).filter, CAP_SPEC_O25.name);
  // R の経路でないコースでは上限の層を付けずに動く
  const w2 = {...world, route: [[0, 0], [1, 0], [1, 1]]};
  const warn = console.warn; let warned = 0; console.warn = () => { warned++; };
  try { const s2 = capCore.createSimulation(w2, {...CT.PHYSICS_CONFIG, seed: 1, timeLimit: 5}); s2.reset({seed: 1}); assert.equal(s2.calibControl({}).filter, null); }
  finally { console.warn = warn; }
  assert.ok(warned >= 1);
  fs.writeFileSync(path.join(REPORT, 'tests', 't7_cap_browser.json'), JSON.stringify({compared_runs: 4, frames: compared, identical: true, cap_file: CAP_FILE}, null, 1));
});

test('T8 ジャイロの位相の写し = safety_caps.py の phase_step・combine_zone', () => {
  const V = JSON.parse(fs.readFileSync(path.join(REPORT, 'tests', 't8_phase_vectors.json'), 'utf8'));
  let n = 0;
  for (const c of V.cases) {
    let st = phaseInitialState();
    c.seq.forEach(([t, gz, v], i) => {
      const r = phaseStep(st, t, gz, v); st = r.state;
      const e = c.res[i];
      assert.equal(r.out.zone, e.zone, `サンプル ${i}: 区間`);
      assert.equal(r.out.turning_right, e.tr);
      assert.equal(r.out.r1_confirmed, e.conf);
      if (e.rel === null) assert.equal(r.out.rel_psi, null); else assert.ok(Math.abs(r.out.rel_psi - e.rel) < 1e-6, `サンプル ${i}: 向き`);
      assert.equal(combineZone(r.out.zone, r.out.turning_right, false, v), e.comb);
      n++;
    });
  }
  assert.ok(n >= 3000);
});

test('T9 進み s の追跡 = features.mjs の追跡(射影の距離 ≤ 0.3 m の間)', async () => {
  const starts = CT.makeStarts(CT.loadWorld('W0')).starts;
  let n = 0;
  for (const si of [0, 5, 7, 11]) {
    const r = await CT.runEpisode({driver: 'RL-speed-0.75', gov: 'OFF', physics: 'P_cal', brake: 'nominal', world: 'W0', kBatt: 1}, starts[si], {keepTrace: true});
    const base = CT.loadWorld('W0'), {world} = CT.activeWorldFor(base, starts[si].gate, starts[si].left_offset_m, starts[si].left_yaw_deg);
    const ft = createFeatureTracker(world, CT.PHYSICS_CONFIG), tr = createProgressTracker(loop, world.route);
    for (let i = 0; i < r.trace.x.length; i++) {
      const x = r.trace.x[i], z = r.trace.z[i];
      const a = i === 0 ? ft.resetProgress(x, z, 0) : ft.advanceProgress(x, z);
      const b = i === 0 ? tr.reset(x, z, 0) : tr.advance(x, z);
      if (tr.d > .3) break;   // ここから先は前へ 3 m の探し(deviations D21)で違ってよい
      assert.ok(Math.abs(a - b) < 1e-9, `開始 ${si} フレーム ${i}: ${a} と ${b}`);
      n++;
    }
  }
  assert.ok(n > 600, `比べたフレーム ${n}`);
});

test('T10 上限の層の不変条件(乱数の入力)', () => {
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };
  const zones = ['none', 'R1_app', 'R1_turn', 'GM', 'DE', 'unknown'];
  for (const p of [{...GOVERNOR_V1_SIM}, {...GOVERNOR_V1_SIM, cap_R1_app: 2.75, cap_R1_turn: 2.75}]) {
    let st = governorInitialState(), zone = 'none', lastEnd = -Infinity, prevBrake = false;
    for (let f = 0; f < 10000; f++) {
      const t = f / 30;
      if (rnd() < .03) zone = zones[Math.floor(rnd() * zones.length)];
      const v = rnd() * 4.5, u = rnd() < .1 ? -rnd() : rnd() * 5, steer = rnd() * 2 - 1;
      const g = governorStep(st, t, zone, v, u, p, steer); st = g.state;
      assert.ok(g.out.u <= u + 1e-12, '出力が入力より大きい');
      const brake = g.out.mode === 'brake';
      if (brake) assert.equal(zone, 'R1_app', 'R1_app の外でブレーキ');
      if (g.out.brake_start) { assert.ok(v >= 1.0, '1 m/s 未満でブレーキを始めた'); assert.ok(t - lastEnd >= .5 - 1e-9, 'やめて 0.5 秒以内に始めた'); }
      if (prevBrake && !brake) lastEnd = t;
      prevBrake = brake;
    }
  }
});

test('T11 PP の経路と、上限の層が舵を変えないこと', async () => {
  const pp = CT.ppPath();
  const c = pp.checks[pp.use];
  assert.ok(c.ok && c.a_gates_within_0_5m && c.a2_sim_gate_rule && c.b_shortcut && c.c_clearance_ge_0_05);
  const cap = createCapFilter({loop, route: CT.loadWorld('W0').route, governorStep, governorInitialState, params: {...GOVERNOR_V1_SIM}});
  const out = cap.filter({frame: 0, steering: .3, speed: 4, state: {x: -.85, z: 2.1, yaw: 0, speed: 3, sim_time: 0}, config: {maxSteer: .52, dt: 1 / 30}});
  assert.equal(out.steering, undefined, '舵を返してはいけない(変えない)');
});
