// 先生の中身(web/teacher_core.mjs)が、元のモジュールとビット単位で同じ計算をすることの試験(node --test)。2026-10-14、sim_dart_20261014。
//   fast   : ベストラップの rl-teacher-bestlap.js(記録 A0 を走らせたファイルそのもの)
//   margin : modes の createModeTeacher(mode 'line'、c0 = +0.3・c_v = 1.0)。乱しありのときは extra.applied で「前回の行動」を渡す
// あわせて、乱しの予定の決定性(同じ種で同じ・塩や k で変わる)と、fork(写し)が元と同じ続きを出すことを確かめる。
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {runCondition, baseWorld, activeWorldFor, configFromCondition, createSimulation} from '../lib/app_emulation.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261014_sim_dart_margin';
const BEST_DIST = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/sim/dist/rl';
const PLAN = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/recording/plan_rlbest60.json', 'utf8'));
const PROTOCOL = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_dart_20261014/protocol.json', 'utf8'));
const {createDartDriver} = await import(pathToFileURL(EXP + '/web/dart_driver.mjs').href);
const {createDartNoise, NOISE_SPECS, applyNoise} = await import(pathToFileURL(EXP + '/web/dart_noise.mjs').href);
const {createTeacherCore, previousAction, MARGIN_PARAMS} = await import(pathToFileURL(EXP + '/web/teacher_core.mjs').href);
const {createModeTeacher} = await import(pathToFileURL(EXP + '/web/rt/mode_runtime.mjs').href);
const WF = (await import(pathToFileURL(EXP + '/web/rt/weights_fast.mjs').href)).default;
const WM = (await import(pathToFileURL(EXP + '/web/rt/weights_margin.mjs').href)).default;
const bestlap = await import(pathToFileURL(BEST_DIST + '/rl-teacher-bestlap.js').href);
const N = Number(process.env.DART_TEST_FRAMES ?? 600);

// 1 条件を最後まで(app.js と同じ順: 教師 → sim.step)。teacherFactory(world, config, variation) → {action}
async function run(c, teacherFactory, frames = N) {
  const w = await baseWorld();
  const {config, scenario} = configFromCondition(c);
  const {world, variation} = activeWorldFor(w, c['start-gate'], c['spawn-offset'], c['spawn-yaw']);
  const sim = createSimulation(world, config);
  sim.reset({seed: config.seed, scenario});
  const t = teacherFactory(world, sim.snapshot().config, variation);
  const acts = [];
  while (!sim.snapshot().state.terminal && acts.length < frames) { const a = t.action(sim.snapshot()); acts.push(a); sim.step(a); }
  return {acts, report: sim.report(), teacher: t};
}

test('乱しの決まりが protocol.json と同じ', () => {
  for (const [name, P] of Object.entries(NOISE_SPECS)) {
    const q = PROTOCOL.dart[name];
    assert.deepEqual(P.steer_offset_rad, q.steer_offset_rad); assert.deepEqual(P.duration_frames, q.duration_frames);
    assert.deepEqual(P.gap_after_end_s, q.gap_after_end_s); assert.deepEqual(P.first_start_s, q.first_start_s);
    assert.equal(P.guard_start_dmin, q.clearance_guard_m.start_if_dmin_ge); assert.equal(P.guard_abort_dmin, q.clearance_guard_m.abort_if_dmin_lt);
  }
  assert.deepEqual(NOISE_SPECS.N_fast.speed_factor, [.70, .95]); assert.deepEqual(NOISE_SPECS.N_margin.speed_factor, [.65, .95]);
});

test('fast: rl-teacher-bestlap.js とビット単位で同じ(乱しなし、2 シーン)', async () => {
  for (const i of [0, 49]) {
    const a = await run(PLAN[i], (w, c, v) => bestlap.createRcTrainingTeacher(w, c, v));
    const b = await run(PLAN[i], (w, c, v) => createDartDriver({kind: 'fast', world: w, config: bestlap.assertRcTeacherContract(w, c, v), weights: WF}));
    assert.equal(a.acts.length, b.acts.length);
    for (let f = 0; f < a.acts.length; f++) {
      assert.equal(b.acts[f].steering, a.acts[f].steering, `scene ${i} f${f} steering`);
      assert.equal(b.acts[f].speed, a.acts[f].speed, `scene ${i} f${f} speed`);
    }
  }
});

test('margin: createModeTeacher(line, c0 = +0.3)とビット単位で同じ(乱しなし・乱しあり)', async () => {
  for (const [i, withNoise] of [[0, false], [3, true], [50, true]]) {
    const noise = withNoise ? {spec: 'N_margin', seed: PLAN[i].seed, salt: 0xD3A7, k: 0, scale: 1} : null;
    const b = await run(PLAN[i], (w, c, v) => createDartDriver({kind: 'margin', world: w, config: c, weights: WM, noise}));
    // 元の createModeTeacher に、同じ乱しを外から足し、「前回の行動」を extra.applied で渡す
    const a = await run(PLAN[i], (w, c, v) => {
      const t = createModeTeacher(w, c, v, {mode: 'line', weights: WM, params: MARGIN_PARAMS});
      const nz = noise ? createDartNoise(noise.spec, noise) : null;
      let applied = null;
      return {action(snap) {
        const out = t.action(snap, applied ? {applied} : null);
        const lg = t.log.at(-1);
        const ctx = {s: lg.s, dmin: lg.dmin};
        // gate の位置は teacher_core と同じ sim の route_index で(measure_teachers.mjs の gateCtx と同じ)
        const route = w.route, n = route.length, ri = Math.min(Math.max(0, (snap.metrics.route_index ?? 0) | 0), n - 1);
        const tg = route[(ri + 1) % n], bf = route[ri], L = Math.hypot(tg[0] - bf[0], tg[1] - bf[1]), nx = (tg[0] - bf[0]) / L, nzz = (tg[1] - bf[1]) / L;
        ctx.gate_along = (snap.state.x - tg[0]) * nx + (snap.state.z - tg[1]) * nzz; ctx.gate_lateral = -(snap.state.x - tg[0]) * nzz + (snap.state.z - tg[1]) * nx;
        const nn = nz ? nz(snap.state.frame, ctx) : null;
        const cmd = applyNoise({steering: out.steering, speed: out.speed}, nn, 4.5);
        applied = previousAction({a0: lg.a0, a1: lg.a1}, cmd, !!nn, !!nn && nn.speedFactor !== 1, 4.5, .52);
        return {...cmd, clean: [out.steering, out.speed]};
      }};
    });
    assert.equal(a.acts.length, b.acts.length, `scene ${i} の長さ`);
    let noisy = 0;
    for (let f = 0; f < a.acts.length; f++) {
      assert.equal(b.acts[f].info.dart.clean[0], a.acts[f].clean[0], `scene ${i} f${f} clean steering`);
      assert.equal(b.acts[f].info.dart.clean[1], a.acts[f].clean[1], `scene ${i} f${f} clean speed`);
      assert.equal(b.acts[f].steering, a.acts[f].steering); assert.equal(b.acts[f].speed, a.acts[f].speed);
      if (b.acts[f].info.dart.noise) noisy++;
    }
    if (withNoise) assert.ok(noisy > 30, `乱したフレームが少なすぎる: ${noisy}`); else assert.equal(noisy, 0);
  }
});

test('乱しの予定は (種, 塩, k) で決まり、同じフレームで 2 回呼んでも同じ', () => {
  const ctx = {s: 1.0, dmin: .3, gate_along: -3, gate_lateral: 0};
  const seq = (o) => { const nz = createDartNoise('N_fast', o); const out = []; for (let f = 0; f < 300; f++) { const a = nz(f, {...ctx, s: 1 + f * .1}); const b = nz(f, {...ctx, s: 1 + f * .1}); assert.deepEqual(a, b); out.push(a ? a.dSteer : 0); } return out; };
  const base = seq({seed: 2026101200, salt: 0xD1A7, k: 0});
  assert.deepEqual(seq({seed: 2026101200, salt: 0xD1A7, k: 0}), base);
  assert.notDeepEqual(seq({seed: 2026101200, salt: 0xD1A7, k: 1}), base);
  assert.notDeepEqual(seq({seed: 2026101200, salt: 0xD3A7, k: 0}), base);
  assert.ok(base.some(x => x !== 0));
  assert.throws(() => { const nz = createDartNoise('N_fast', {seed: 1, salt: 2, k: 0}); nz(0, ctx); nz(2, ctx); }, /飛びました/);
});

test('fork(写し)は、元の core と同じ続きを出す(fast・margin)', async () => {
  for (const kind of ['fast', 'margin']) {
    const c = PLAN[5];
    const w = await baseWorld();
    const {config, scenario} = configFromCondition(c);
    const {world} = activeWorldFor(w, c['start-gate'], c['spawn-offset'], c['spawn-yaw']);
    const sim = createSimulation(world, config); sim.reset({seed: config.seed, scenario});
    const cfg = sim.snapshot().config, W = kind === 'fast' ? WF : WM;
    const core = createTeacherCore(kind, world, cfg, W);
    let prev = [0, 0], forked = null, forkPrev = null;
    const mainOut = [], forkOut = [];
    for (let f = 0; f < 260; f++) {
      const snap = sim.snapshot();
      if (f === 0) core.begin(snap.state, snap.metrics.route_index); else core.advance(snap.state, snap.metrics.route_index);
      if (forked) forked.advance(snap.state, snap.metrics.route_index);
      if (f === 150) { forked = core.fork(); forkPrev = [...prev]; }
      const d = core.decide(snap.state, prev[0], prev[1]);
      if (forked) { const e = forked.decide(snap.state, forkPrev[0], forkPrev[1]); forkOut.push([e.steering, e.speed, e.s]); mainOut.push([d.steering, d.speed, d.s]); forkPrev = [e.a0, e.a1]; }
      prev = [d.a0, d.a1];
      sim.step({steering: d.steering, speed: d.speed});
    }
    assert.deepEqual(forkOut, mainOut, kind + ' の fork が元と違う');
  }
});
