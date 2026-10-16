// 較正版のコアの試験(2026-10-16、シミュレーター較正 S2)。T1・T2・T4・T5。
//
//   cd ~/jetformer/experiments/20261016_sim_calibration && node --test tests/calib_core.test.mjs
//
// T1 プロファイル無しの較正版のコア(fast あり・なし)が元と同じ: 教師の REC60(60 条件)・STD(単周・連続)・ランダム操作 20 本で
//    report の JSON が完全に一致する。あわせて、忠実度の再生で P_default に使う「元の式 + 遅れ 0」の実行部も、
//    report の model.calibration を除いて完全に一致する(T1b)。
// T2 元の sha256 が違う・差分の場所が 1 か所でない・プロファイルがおかしい → 止まる
// T4 ブラウザ用のファイルが patchSimCore の出力とバイト列で同じ。calib.html と index.html の差は title と import map だけ
// T5 新しい物理の部品(惰性・ブレーキ・ESC の後退の手順・遅れ・頭打ち・アンダーステア・横の限界・電池・不感帯・こすれ)を手計算の値で
// 結果の要約は /home/jetson/jetformer_reports/sim_calibration_20261016/tests/t1_identity.json にも書く。
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {patchSimCore, loadCalibratedCore, loadProfile, origReplayProfile, ORIGINAL_SIM_CORE, ORIGINAL_SIM_CORE_SHA256, sha256, sha256File,
  REPORT, replaceOnce} from '../calib_core.mjs';
import {compileProfile} from '../calib_runtime.mjs';
import {calibPageHtml, DIST} from '../build_browser_core.mjs';

const EXP13 = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap';
const {RlEnv, loadBaseWorld, activeWorldFor} = await import(pathToFileURL(EXP13 + '/rl/env_core.mjs').href);
const {loadRec60, STD, teacherController} = await import(pathToFileURL(EXP13 + '/rl/eval_sets.mjs').href);
const original = await import(pathToFileURL(ORIGINAL_SIM_CORE).href);
const fast13 = await import(pathToFileURL(EXP13 + '/sim/fast-sim-core.mjs').href);
const nullCore = await loadCalibratedCore(null, {fast: false});
const nullFast = await loadCalibratedCore(null, {fast: true});
const origDelay0 = await loadCalibratedCore(origReplayProfile(0, 'P_default-replay'), {fast: false});
const {profile: PCAL, sha256: PCAL_SHA} = loadProfile('P_cal');
const calCore = await loadCalibratedCore(PCAL, {fast: true, profileSha256: PCAL_SHA});
const OUT = path.join(REPORT, 'tests');
fs.mkdirSync(OUT, {recursive: true});
const summary = {schema: 'jf-calib-t1/v1', created_at: new Date().toISOString(), node: process.version, sim_core_sha256: sha256File(ORIGINAL_SIM_CORE)};

// ---------------------------------------------------------------- T1
async function runTeacher(core, coreKind, start, episodeMode, teacher) {
  const env = new RlEnv({core, coreKind, baseWorld: loadBaseWorld(), variant: 'main', mode: 'eval', episodeMode, timeLimit: 20});
  assert.ok(env.reset(start), 'invalid spawn ' + start.name);
  const ctrl = teacher.create({world: env.world, config: JSON.parse(JSON.stringify(env.config)), variation: env.variation, env});
  while (!env.done) { const c = ctrl.action(env); env.stepCommand(c.steering, c.speed); }
  return env.sim.report();
}
const stripCal = r => { const c = JSON.parse(JSON.stringify(r)); delete c.model.calibration; return c; };

test('T1 REC60・STD: プロファイル無しの較正版(fast なし・あり)と元の式の実行部が、元と同じ report を出す', async () => {
  const teacher = await teacherController();
  const rec = loadRec60();
  assert.equal(rec.length, 60);
  const rows = [];
  let nSame = 0, nSameFast = 0, nSameD0 = 0, nSame13 = 0;
  const starts = [...rec.map(c => ({c, mode: 'continuous'})), {c: STD, mode: 'single'}, {c: STD, mode: 'continuous'}];
  for (const {c, mode} of starts) {
    const a = JSON.stringify(await runTeacher(original, 'original', c, mode, teacher));
    const b = JSON.stringify(await runTeacher(nullCore, 'original', c, mode, teacher));
    const f = JSON.stringify(await runTeacher(nullFast, 'fast', c, mode, teacher));
    const f13 = JSON.stringify(await runTeacher(fast13, 'fast', c, mode, teacher));
    const d0r = await runTeacher(origDelay0, 'original', c, mode, teacher);
    assert.equal(d0r.model.calibration.kind, 'orig');
    const d0 = JSON.stringify(stripCal(d0r));
    nSame += a === b; nSameFast += a === f; nSameD0 += a === d0; nSame13 += a === f13;
    rows.push({name: c.name, mode, same_null: a === b, same_null_fast: a === f, same_orig_delay0: a === d0, same_fast13: a === f13,
      report_sha256: sha256(a), frames: JSON.parse(a).final_state.frame});
  }
  summary.T1_teacher = {n: starts.length, same_null: nSame, same_null_fast: nSameFast, same_orig_delay0: nSameD0, same_fast13_reference: nSame13, rows};
  assert.equal(nSame, starts.length);
  assert.equal(nSameFast, starts.length);
  assert.equal(nSameD0, starts.length);
});

function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
test('T1 ランダム操作 20 本(設定・路面・衝突・ブレーキ・後退を含む): report と observe が元と同じ', () => {
  const base = loadBaseWorld();
  const rows = [];
  for (let k = 0; k < 20; k++) {
    const rnd = mulberry32(20261016 + k);
    const gate = Math.floor(rnd() * 33);
    const {world} = activeWorldFor(base, gate, (rnd() - .5) * .2, (rnd() - .5) * 10);
    const cfg = {dt: k % 4 === 3 ? 1 / 60 : 1 / 30, maxSpeed: 7.24, maxSteer: .52, steerRate: 1.2, timeLimit: 15, seed: k + 1,
      stopOnCollision: k % 2 === 0, frictionMultiplier: k % 5 === 1 ? .4 : 1, noiseStd: k % 5 === 2 ? .025 : 0,
      ...(k % 3 === 2 ? {} : {longitudinalModel: 'real-drive-response-estimate-v1', driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2}),
      ...(k % 4 === 1 ? {episodeMode: 'continuous-laps-v1'} : {})};
    const sims = [original, nullCore, nullFast, origDelay0].map(m => m.createSimulation(world, cfg));
    let st = 0, sp = 0, same = true;
    for (let i = 0; i < 400; i++) {
      st = Math.max(-.7, Math.min(.7, st + (rnd() - .5) * .25));
      sp = Math.max(-1.5, Math.min(5, sp + (rnd() - .5) * 1.2));
      const a = {steering: st, speed: i % 97 < 6 ? -1 : sp};
      const snaps = sims.map(s => JSON.stringify(s.step(a)));
      const obs = sims.map(s => JSON.stringify(s.observe()));
      if (!(snaps[0] === snaps[1] && snaps[0] === snaps[2] && snaps[0] === snaps[3] && obs[0] === obs[1] && obs[0] === obs[2] && obs[0] === obs[3])) same = false;
    }
    const reps = sims.map(s => s.report());
    const r0 = JSON.stringify(reps[0]);
    const ok = same && r0 === JSON.stringify(reps[1]) && r0 === JSON.stringify(reps[2]) && r0 === JSON.stringify(stripCal(reps[3]));
    rows.push({k, gate, cfg, same: ok, reason: reps[0].outcome.reason, collisions: reps[0].metrics.collisions, frames: reps[0].final_state.frame});
    assert.ok(ok, 'ランダム操作 ' + k + ' が一致しない');
  }
  summary.T1_random = {n: rows.length, same: rows.filter(r => r.same).length, rows};
});

// ---------------------------------------------------------------- T2
test('T2 元の sha256 が違う・差分の場所が 1 か所でない・プロファイルがおかしい → 止まる', () => {
  const src = fs.readFileSync(ORIGINAL_SIM_CORE, 'utf8');
  assert.throws(() => patchSimCore(src + '\n// x', null), /sha256 が違います/);
  const dup = src + '\n    actionLog.push({ frame: state.frame, steering, speed });\n';
  assert.throws(() => patchSimCore(dup, null, {expectedSha256: sha256(dup)}), /置換箇所が1つではありません/);
  const miss = src.replace('const limit = surface.mu*9.80665', 'const limit = surface.mu*9.81');
  assert.throws(() => patchSimCore(miss, null, {expectedSha256: sha256(miss)}), /置換箇所が1つではありません/);
  assert.throws(() => replaceOnce('aXbXc', 'X', 'Y'), /置換箇所が1つではありません/);
  const bad = JSON.parse(JSON.stringify(PCAL));
  bad.steering.delta_L_rad[3] = 0.1;   // 単調でない
  assert.throws(() => patchSimCore(src, bad), /単調ではありません/);
  const bad2 = JSON.parse(JSON.stringify(PCAL)); bad2.schema = 'x';
  assert.throws(() => patchSimCore(src, bad2), /知らないプロファイル/);
  assert.throws(() => loadProfile('nope'));
  summary.T2 = 'pass';
});

// ---------------------------------------------------------------- T4
test('T4 ブラウザ用の較正版のコアは patchSimCore の出力とバイト列で同じ。calib.html の差は title と import map だけ', () => {
  const coreFile = `sim-core-${PCAL.name}.js`;
  const built = fs.readFileSync(path.join(DIST, '_calib', coreFile));
  const expect = Buffer.from(patchSimCore(fs.readFileSync(ORIGINAL_SIM_CORE, 'utf8'), PCAL, {fast: false, name: PCAL.name, profileSha256: PCAL_SHA}));
  assert.ok(built.equals(expect));
  const index = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8').split('\n');
  const calib = fs.readFileSync(path.join(DIST, 'calib.html'), 'utf8').split('\n');
  assert.equal(index.length, calib.length);
  const diff = index.map((l, i) => l === calib[i] ? null : i).filter(i => i !== null);
  assert.equal(diff.length, 2);
  assert.ok(calib[diff[0]].startsWith('<title>') && calib[diff[1]].startsWith('<script type="importmap">'));
  const map = JSON.parse(calib[diff[1]].replace(/^<script type="importmap">/, '').replace(/<\/script>$/, ''));
  const map0 = JSON.parse(index[diff[1]].replace(/^<script type="importmap">/, '').replace(/<\/script>$/, ''));
  assert.deepEqual(Object.keys(map.imports).filter(k => !(k in map0.imports)), ['./sim-core.js']);
  assert.equal(map.imports['./sim-core.js'], './_calib/' + coreFile);
  for (const k of Object.keys(map0.imports)) assert.equal(map.imports[k], map0.imports[k]);
  assert.equal(calibPageHtml(index.join('\n'), coreFile, 'x').split('\n').length, index.length);
  // 写しの dist の元のファイルが、ユーザーの展開物と同じ
  assert.equal(sha256File(path.join(DIST, 'sim-core.js')), ORIGINAL_SIM_CORE_SHA256);
  summary.T4 = {core_file: coreFile, sha256: sha256(expect), diff_lines: diff.map(i => i + 1)};
});

// ---------------------------------------------------------------- T5
const EMPTY = (surface = null) => ({segments: [], disks: [], bounds: {minX: -1000, maxX: 1000, minZ: -1000, maxZ: 1000}, route: [],
  surfaces: surface ? [{id: surface.id, x: 0, z: 0, w: 1000, d: 1000, mu: surface.mu}] : [], spawn: {x: 0, z: 0, yaw: 0}});
const CFG = {dt: .01, maxSpeed: 7.24, maxSteer: .52, timeLimit: 100, stopOnCollision: false, longitudinalModel: 'real-drive-response-estimate-v1',
  driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2};
const lo = PCAL.longitudinal, stp = PCAL.steering;
async function calSim(mod = null, surface = null, extra = {}) {
  const p = mod ? mod(JSON.parse(JSON.stringify(PCAL))) : PCAL;
  const core = mod ? await loadCalibratedCore(p, {fast: true}) : calCore;
  const sim = core.createSimulation(EMPTY(surface), {...CFG, ...extra});
  return sim;
}
const run = (sim, n, a) => { for (let i = 0; i < n; i++) sim.step(a, true); return sim.peek().state; };
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} と ${b} の差 ${Math.abs(a - b)} > ${tol}`);
const brakeCmd = us => -us / 627.5;
const T5 = {};

test('T5a 惰性(直線): 中立で c0 だけ減速する', async () => {
  const sim = await calSim();
  sim.calibControl({setSpeed: 3});
  const v = run(sim, 50, {steering: 0, speed: 0}).speed;
  near(v, 3 - lo.c0 * .5, 1e-9, '惰性 0.5 秒');
  T5.coast = {v0: 3, t: .5, v, expect: 3 - lo.c0 * .5};
});

test('T5b ブレーキ: +200 µs で c0 + s_b(200 − db_b)、前進中は後退しない', async () => {
  const sim = await calSim();
  sim.calibControl({setSpeed: 3});
  const ab = lo.brake.s_b * (200 - lo.brake.db_b);
  const v = run(sim, 50, {steering: 0, speed: brakeCmd(200)}).speed;
  near(v, 3 - (lo.c0 + ab) * .5, 1e-9, 'ブレーキ 0.5 秒');
  const v2 = run(sim, 300, {steering: 0, speed: brakeCmd(200)}).speed;
  assert.equal(v2, 0);
  // 弱い要求(db_b 以下)は惰性と同じ
  const s2 = await calSim(); s2.calibControl({setSpeed: 3});
  near(run(s2, 50, {steering: 0, speed: brakeCmd(60)}).speed, 3 - lo.c0 * .5, 1e-9, '+60 µs は惰性と同じ');
  T5.brake = {ab_200us: ab, decel_total: lo.c0 + ab, v_after_0_5s: v};
});

test('T5c ESC の後退の手順: 止まってからのブレーキでは下がらず、中立 0.12 秒以上をはさむと後退する', async () => {
  const sim = await calSim();
  let s = run(sim, 100, {steering: 0, speed: brakeCmd(300)});
  assert.equal(s.speed, 0);
  run(sim, 10, {steering: 0, speed: 0});               // 中立 0.10 秒(足りない)
  s = run(sim, 50, {steering: 0, speed: brakeCmd(300)});
  assert.equal(s.speed, 0, '0.10 秒では後退しない');
  run(sim, 15, {steering: 0, speed: 0});               // 中立 0.15 秒
  s = run(sim, 50, {steering: 0, speed: brakeCmd(300)});
  near(s.speed, -lo.esc_reverse.accel * .5, 1e-9, '後退 0.5 秒で −0.75 m/s');
  s = run(sim, 100, {steering: 0, speed: brakeCmd(300)});
  near(s.speed, lo.esc_reverse.target_mps, 1e-12, '後退の目標 −0.8 m/s');
  s = run(sim, 20, {steering: 0, speed: 0});            // 前進側の指令: 1.5 m/s² で 0 へ
  near(s.speed, -.8 + 1.5 * .2, 1e-9, '後退から中立');
  T5.esc = 'pass';
});

test('T5d 遅れ: 舵は d_s = 0.05 秒遅れて動き始め、r_s = 8 rad/s で動く。速さの遅れ d_v も効く', async () => {
  const sim = await calSim();
  sim.calibControl({forceSpeed: 1});
  run(sim, 10, {steering: 0, speed: 1});
  let s = run(sim, 5, {steering: .52, speed: 1});        // フレーム 10〜14: τ − 0.05 < 0.10 なので、まだ 0 の指令
  assert.equal(s.steering, 0);
  s = run(sim, 1, {steering: .52, speed: 1});            // フレーム 15: 動き始める
  near(s.steering, stp.r_s * .01, 1e-9, '1 フレームで 0.08 rad');
  s = run(sim, 10, {steering: .52, speed: 1});
  near(s.steering, stp.delta_L_rad[5], 1e-12, '左いっぱいの頭打ち');
  // 速さの遅れ(d_v = 0.03 にしたプロファイル)
  const sv = await calSim(p => { p.longitudinal.d_v = .03; return p; });
  run(sv, 10, {steering: 0, speed: 0});
  let v = run(sv, 3, {steering: 0, speed: 3}).speed;    // フレーム 10〜12: まだ 0 の指令
  assert.equal(v, 0);
  v = run(sv, 1, {steering: 0, speed: 3}).speed;
  assert.ok(v > 0);
  T5.delay = {d_s: stp.d_s, r_s: stp.r_s};
});

test('T5e 舵の対応・アンダーステア・横の限界・路面', async () => {
  const L = .26, kL = Math.tan(stp.delta_L_rad[5]) / L, kR = Math.tan(stp.delta_R_rad[5]) / L;
  const yawRate = (sim, n, a) => { run(sim, n, a); const y0 = sim.peek().state.yaw; run(sim, 1, a); return (sim.peek().state.yaw - y0) / .01; };
  let sim = await calSim(); sim.calibControl({forceSpeed: 1});
  near(yawRate(sim, 30, {steering: .52, speed: 1}), kL, 1e-9, '左いっぱいの曲率 1.634 /m');
  sim = await calSim(); sim.calibControl({forceSpeed: 1});
  near(yawRate(sim, 30, {steering: -.52, speed: 1}), -kR, 1e-9, '右いっぱいの曲率 −1.812 /m');
  sim = await calSim(); sim.calibControl({forceSpeed: 1});
  near(yawRate(sim, 30, {steering: .26, speed: 1}), Math.tan(stp.delta_L_rad[2]) / L, 1e-9, '左 0.5');
  // アンダーステア K = 0.05(試験用のプロファイル)
  sim = await calSim(p => { p.steering.K_us = .05; return p; }); sim.calibControl({forceSpeed: 2});
  near(yawRate(sim, 30, {steering: .52, speed: 2}), 2 * kL / (1 + .05 * 4), 1e-9, 'アンダーステア');
  // 横の限界: 4 m/s・左いっぱいはカーペットの μ g / v で頭打ち、芝ではさらに低い
  sim = await calSim(); sim.calibControl({forceSpeed: 4});
  near(yawRate(sim, 30, {steering: .52, speed: 4}), PCAL.default_surface_mu * 9.80665 / 4, 1e-9, '横の限界(カーペット)');
  sim = await calSim(null, {id: 'turf', mu: .6}); sim.calibControl({forceSpeed: 4});
  near(yawRate(sim, 30, {steering: .52, speed: 4}), PCAL.surfaces_mu.turf * 9.80665 / 4, 1e-9, '横の限界(芝)');
  T5.steer = {kappa_left_full: kL, kappa_right_full: kR};
});

test('T5f 速さの対応・不感帯・電池・こすれ', async () => {
  const T = u => { const xs = [0, lo.u_db, .5, 1, 2, 3, 4, 5, 6, 7.24], ys = [0, 0, ...lo.T_values];
    let j = 0; while (j < xs.length - 2 && xs[j + 1] <= u) j++; return ys[j] + (ys[j + 1] - ys[j]) * (u - xs[j]) / (xs[j + 1] - xs[j]); };
  let sim = await calSim();
  near(run(sim, 800, {steering: 0, speed: 2}).speed, T(2), 1e-6, '定常の速さ T(2)');
  sim = await calSim(); sim.calibControl({kBatt: .8});
  near(run(sim, 800, {steering: 0, speed: 2}).speed, .8 * T(2), 1e-6, '電池 0.8');
  sim = await calSim();
  near(run(sim, 800, {steering: 0, speed: 7.24}).speed, lo.T_values[7], 1e-6, '頭打ち 5.0 m/s');
  // 不感帯: u = 0.4 < u_db は目標 0(惰性で止まり、後ろへは行かない)
  sim = await calSim(); sim.calibControl({setSpeed: 1});
  assert.equal(run(sim, 300, {steering: 0, speed: .4}).speed, 0);
  // こすれ: 2 m/s・左いっぱいで落ち着かせてから惰性 → 最初の減速 ≈ c0 + k_s v² κ
  sim = await calSim(); sim.calibControl({forceSpeed: 2});
  run(sim, 40, {steering: .52, speed: 0});
  sim.calibControl({forceSpeed: null});
  const v1 = run(sim, 1, {steering: .52, speed: 0}).speed;
  const kL = Math.tan(stp.delta_L_rad[5]) / .26;
  near((2 - v1) / .01, lo.c0 + lo.k_s * 4 * kL, .06, 'こすれの減速(1 フレームの間に v² が約 3% 下がる分を許す)');
  T5.speed = {T2: T(2), Tmax: lo.T_values[7], scrub_decel_first_frame: (2 - v1) / .01};
});

test('T5g 元の式 + 遅れ(P_cfg の再生の道具): 指令が 1/30 秒遅れて効く', async () => {
  const core = await loadCalibratedCore(origReplayProfile(1 / 30, 'P_cfg-replay'), {fast: true});
  const sim = core.createSimulation(EMPTY(), CFG);
  run(sim, 10, {steering: 0, speed: 0});
  let s = run(sim, 3, {steering: .3, speed: 2});      // フレーム 10〜12: τ − 1/30 < 0.10
  assert.equal(s.speed, 0); assert.equal(s.steering, 0);
  s = run(sim, 2, {steering: .3, speed: 2});
  assert.ok(s.speed > 0 && s.steering > 0);
  T5.orig_delay = 'pass';
});

test.after(() => {
  summary.T5 = T5;
  fs.writeFileSync(path.join(OUT, 't1_identity.json'), JSON.stringify(summary, null, 1));
});
