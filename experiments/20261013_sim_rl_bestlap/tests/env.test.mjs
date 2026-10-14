// RL 環境の一致の試験(Node)。2026-10-13、RL 環境の段
//   node --test /home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/tests/
// 確かめること:
//   - コピーした sim/dist がユーザーの dist とバイト単位で同じ。高速版・コース・計画の sha256 が手順(protocol.json)どおり
//   - 教師をこの環境で走らせると README の周回時間(1 周目 8.710〜8.758 秒、2 周目 8.359〜8.407 秒、停止から 1 周 8.931 秒)を
//     そのまま再現し、33/33 ゲート・黒シート 1 回・衝突 0。高速版と元のコアで周回時間の差が 0
//   - 学習済み ONNX の保存した操作列(results/shortcut-rc-learned-lap.json)を再生すると 8.964117 秒
//   - リセット・ステップの決定性、衝突・ゲート逃し・停止・逆走での終了
//   - 観測(79 次元)の計算(速い版のレイ = 素直な版、射影の連続性、経路の回転)
//   - 独立な周回の確認(rl/verify_lap.mjs)が正しい周を通し、壊した周・衝突した周を通さない
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const RL = path.resolve(here, '../rl');
const E = await import(path.join(RL, 'env_core.mjs'));
const F = await import(path.join(RL, 'features.mjs'));
const S = await import(path.join(RL, 'eval_sets.mjs'));
const A = await import(path.join(RL, 'lap_audit.mjs'));
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const fastCore = await E.loadCore('fast'), originalCore = await E.loadCore('original');

test('コピーした dist がユーザーの dist と同じで、手順の sha256 どおり', () => {
  const user = path.join(E.USER_SIM_ROOT, 'dist'), copy = path.join(E.EXP_ROOT, 'sim/dist');
  for (const f of ['sim-core.js', 'rc-speed-controller.js', 'speed-controller.js', 'world.js', 'scene.js', 'control-delay.js', 'sensors.js', 'ego-sensors.js', 'optimized-reference.js'])
    assert.equal(sha(path.join(copy, f)), sha(path.join(user, f)), f);
  for (const k of Object.keys(E.EXPECTED_SHA256)) assert.equal(sha(E.PATHS[k]), E.EXPECTED_SHA256[k], k);
  // 参照ループは作り直しても同じ
  execFileSync(process.execPath, [path.join(RL, 'make_reference_loop.mjs'), '--check']);
  assert.equal(F.REFERENCE_LOOP_SHA256, crypto.createHash('sha256').update(fs.readFileSync(path.join(RL, 'reference_loop.json'), 'utf8')).digest('hex'));
});

test('教師の REC60(高速版): README の 1 周目・2 周目の範囲を再現、全シーン有効・衝突 0', async () => {
  const teacher = await S.teacherController(), rows = [];
  for (const c of S.loadRec60()) rows.push(await S.runEpisode({start: c, controller: teacher, coreKind: 'fast'}));
  const s = S.summarizeRec60(rows);
  assert.equal(s.valid_two_laps, 60); assert.equal(s.collisions, 0);
  // README: 8.710〜8.758 / 8.359〜8.407(probe_result.json の元のコアの値と 1e-9 まで同じ)
  assert.ok(Math.abs(s.lap1.min - 8.709693610681859) < 1e-9 && Math.abs(s.lap1.max - 8.758448640205728) < 1e-9, JSON.stringify(s.lap1));
  assert.ok(Math.abs(s.lap2.min - 8.358862041910367) < 1e-9 && Math.abs(s.lap2.max - 8.406762681129095) < 1e-9, JSON.stringify(s.lap2));
  assert.ok(s.lap1.min >= 8.7095 && s.lap1.max <= 8.7585 && s.lap2.min >= 8.3585 && s.lap2.max <= 8.4075);
  for (const r of rows) for (const l of r.laps) { assert.equal(l.gates, 33); assert.equal(l.crossings, 1); assert.equal(l.collisions, 0); }
  // 標準の開始(rc20-mode8-train-00)
  assert.ok(Math.abs(rows[0].laps[0].elapsed - 8.75841899184614) < 1e-12 && Math.abs(rows[0].laps[1].elapsed - 8.370092214503968) < 1e-12);
});

test('教師: 高速版と元のコアで周回時間が同じ(4 シーン)、停止から 1 周は 8.930784 秒', async () => {
  const teacher = await S.teacherController(), plan = S.loadRec60();
  for (const c of [plan[0], plan[17], plan[48], plan[59]]) {
    const a = await S.runEpisode({start: c, controller: teacher, coreKind: 'fast'}), b = await S.runEpisode({start: c, controller: teacher, coreKind: 'original'});
    assert.deepEqual(a.laps.map(l => l.elapsed), b.laps.map(l => l.elapsed));
    assert.equal(a.min_clearance, b.min_clearance);
  }
  for (const coreKind of ['fast', 'original']) {
    const r = await S.runEpisode({start: S.STD, controller: teacher, coreKind, episodeMode: 'single'});
    assert.equal(r.reason, 'route_complete'); assert.equal(r.passed_gates, 33); assert.equal(r.collisions, 0);
    assert.equal(r.laps[0].elapsed, 8.93078380923196);
  }
});

test('学習済み ONNX の操作列の再生: 8.964117 秒(環境の両方のコアと独立な確認)', async () => {
  const report = JSON.parse(fs.readFileSync(E.PATHS.learnedLap, 'utf8'));
  for (const coreKind of ['fast', 'original']) {
    const r = await S.runEpisode({start: S.STD, controller: S.replayController(report.actions), coreKind, episodeMode: 'single'});
    assert.equal(r.reason, 'route_complete'); assert.equal(r.laps[0].elapsed, 8.964117142565293); assert.equal(r.frames, 269);
  }
  const v = await A.verifyReport(report, {});
  assert.ok(v.pass, v.failures.join('\n'));
  assert.equal(v.laps_own[0].elapsed, 8.964117142565293);
  assert.equal(v.max_abs_error['own_physics.x'], 0);
});

test('決定性: 同じ開始・同じ行動なら観測・報酬が同じ。リセットし直しても同じ', () => {
  const run = () => {
    const env = new E.RlEnv({core: fastCore}), rng = E.createRng(123), obs = [], rew = [];
    E.resetRandom(env, rng); obs.push(...env.obs);
    for (let i = 0; i < 400; i++) {
      const r = env.step(Math.sin(i * .07) * .4 + (rng.uniform() - .5) * .2, -.3 + (rng.uniform() - .5) * .2);
      obs.push(...env.obs); rew.push(r.reward);
      if (env.done) { E.resetRandom(env, rng); obs.push(...env.obs); }
    }
    return {obs, rew};
  };
  const a = run(), b = run();
  assert.deepEqual(a.obs, b.obs); assert.deepEqual(a.rew, b.rew);
  const env = new E.RlEnv({core: fastCore});
  env.reset({gate: 7, left_offset_m: .05, left_yaw_deg: -4}); const o1 = [...env.obs];
  env.step(.3, .2); env.step(.3, .2);
  env.reset({gate: 7, left_offset_m: .05, left_yaw_deg: -4}); assert.deepEqual([...env.obs], o1);
});

test('終了: 壁へ直進すると衝突で終わる(−10)。停止・逆走・ゲート逃しは学習用だけ', () => {
  const env = new E.RlEnv({core: fastCore});
  env.reset({}); let r;
  while (!env.done) r = env.step(0, 1);
  assert.equal(r.reason, 'collision'); assert.ok(r.terminated && !r.truncated); assert.ok(r.reward < -9.9 && r.reward > -10.2);
  assert.equal(env.summary().collisions, 1); assert.ok(env.summary().frames > 40 && env.summary().frames < 60);
  // 停止: 1 秒以降に 0.2 m/s 未満が 2 秒(60 フレーム)続く → −5
  env.reset({}); while (!env.done) r = env.step(0, -1);
  assert.equal(r.reason, 'stall'); assert.equal(env.summary().frames, 89); assert.ok(r.reward <= -5);
  // 逆走(ゲート 3 から右回りに U ターン)
  const script = (i, pre, st, v) => i < pre ? [0, 1] : [st, v];
  env.reset({gate: 3}); let i = 0; while (!env.done) { const a = script(i++, 20, .45, .8); r = env.stepCommand(a[0], a[1]); }
  assert.equal(r.reason, 'wrong_way'); assert.ok(r.reward <= -5);
  // ゲート逃し(ゲート 15 から左へ切れて、ゲートの線を ±0.65 m の外で越える)
  env.reset({gate: 15, left_offset_m: -.12}); i = 0; while (!env.done) { const a = script(i++, 5, .45, 1.2); r = env.stepCommand(a[0], a[1]); }
  assert.equal(r.reason, 'gate_miss'); assert.ok(r.reward <= -9.9);
  // 評価モードでは sim の終了だけ(同じ操作でも途中で終わらない)
  const ev = new E.RlEnv({core: fastCore, mode: 'eval'});
  ev.reset({gate: 15, left_offset_m: -.12}); i = 0; while (!ev.done) { const a = script(i++, 5, .45, 1.2); r = ev.stepCommand(a[0], a[1]); }
  assert.notEqual(r.reason, 'gate_miss'); assert.ok(ev.summary().gate_miss_frame !== null);
  ev.reset({}); while (!ev.done) r = ev.step(0, -1);
  assert.equal(r.reason, 'collection_duration_complete'); assert.ok(r.truncated && !r.terminated);
});

test('観測: 79 次元・有限、速いレイ = 素直なレイ、射影は連続、経路の回転に対応', async () => {
  const world = E.loadBaseWorld(), g = F.prepareGeometry(world);
  assert.deepEqual(g.counts, {segments: 51, circles: 49, rects: 45});
  let seed = 7; const rnd = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const a = new Float64Array(19), b = new Float64Array(19);
  for (let k = 0; k < 20000; k++) {
    const x = -5.3 + 10.6 * rnd(), z = -3.1 + 6.5 * rnd(), yaw = (rnd() * 2 - 1) * Math.PI;
    F.castRays(g, x, z, yaw, a); F.castRaysReference(g, x, z, yaw, b);
    assert.deepEqual([...a], [...b]);
  }
  for (let k = 0; k < 33; k++) assert.equal(F.routeOffset(E.activeWorldFor(world, k).world.route), k);
  // 教師の 1 周: 進み s は毎フレーム 0 以上・実走行距離 + 5 cm 以下。観測は有限
  const T = await E.loadRcTeacher();
  const env = new E.RlEnv({core: fastCore, teacherModule: T, teacherActions: true});
  for (const start of [{gate: 0}, {gate: 0, left_offset_m: .03, left_yaw_deg: 3}]) {
    env.reset(start);
    let sPrev = env.tracker.s;
    while (!env.done) {
      const c = env.teacherCommand, r = env.stepCommand(c.steering, c.speed);
      assert.equal(env.obs.length, F.OBS_DIM); assert.ok(env.obs.every(Number.isFinite));
      const ds = r.s - sPrev; sPrev = r.s;
      assert.ok(ds >= 0 && ds <= env.view.state.speed / 30 + .05, `ds ${ds}`);
      assert.ok(env.aux.projection_distance < .35);
    }
    assert.ok(Math.abs(env.tracker.s - env.s0 - env.summary().reward_parts.progress) < 1e-9);
  }
  // ほかのゲートからの開始でも、最初の s はそのゲートの近く
  for (const k of [5, 10, 16, 20, 31]) {
    env.reset({gate: k});
    const sGate = F.GATE_S[(k + 32) % 33], d = Math.abs(((env.tracker.s - sGate) % F.LOOP_LENGTH + F.LOOP_LENGTH + 13) % F.LOOP_LENGTH - 13);
    assert.ok(d < .6, `gate ${k}: s ${env.tracker.s} vs ${sGate}`);
  }
});

test('独立な周回の確認: 教師の周は通り、壊した操作・衝突は通らない', async () => {
  const teacher = await S.teacherController();
  const single = await S.runEpisode({start: S.STD, controller: teacher, coreKind: 'fast', episodeMode: 'single', record: true});
  const v1 = await A.verifyReport(single.report, {minClearance: .03});
  assert.ok(v1.pass, v1.failures.join('\n')); assert.equal(v1.laps_own[0].elapsed, 8.93078380923196); assert.equal(v1.crossings_valid, 1);
  const cont = await S.runEpisode({start: S.loadRec60()[33], controller: teacher, coreKind: 'fast', record: true});
  const v2 = await A.verifyReport(cont.report, {minLaps: 2});
  assert.ok(v2.pass, v2.failures.join('\n')); assert.equal(v2.laps_own.length, 2);
  assert.ok(Math.abs(v2.laps_own[1].elapsed - cont.laps[1].elapsed) <= 1e-9);
  // 操作を 1 つ変えると軌跡が保存したものと合わない
  const bad = structuredClone(single.report); bad.actions[100] = {...bad.actions[100], steering: bad.actions[100].steering + .01};
  const v3 = await A.verifyReport(bad, {});
  assert.ok(!v3.pass && v3.failures.some(f => f.startsWith('replay_trace')));
  // 開始姿勢を書き換えると通らない
  const moved = structuredClone(single.report); moved.world.spawn.x += .01;
  assert.ok(!(await A.verifyReport(moved, {})).pass);
  // 壁へ直進: 衝突を sim と自前の両方が同じ時刻に数える
  const crash = await S.runEpisode({start: S.STD, controller: S.replayController(Array.from({length: 600}, () => ({steering: 0, speed: 4.5}))),
    coreKind: 'original', episodeMode: 'single', record: true});
  const v4 = await A.verifyReport(crash.report, {});
  assert.ok(!v4.pass); assert.equal(v4.collisions_sim, 1); assert.equal(v4.collisions_own, 1);
  assert.ok(Math.abs(v4.collision_events_own[0].time - crash.report.metrics.collision_events[0].time) <= 1e-9);
  // 遅延 1 フレームの教師(ゲート 16 のあとで柱の脚に衝突。design.md 1 章)
  const d1 = await S.runEpisode({start: S.STD, controller: teacher, coreKind: 'original', episodeMode: 'single', delayFrames: 1, record: true});
  assert.equal(d1.reason, 'collision'); assert.equal(d1.passed_gates, 16);
  const v5 = await A.verifyReport(d1.report, {});
  assert.equal(v5.collisions_own, 1); assert.ok(Math.abs(v5.collision_events_own[0].time - 7.259719455532158) <= 1e-9);
});

test('方策の通り道(policyController → 正規化した行動 → evaluateController)で教師を走らせると周回時間がほぼ同じ', async () => {
  const T = await E.loadRcTeacher();
  // 観測は使わず、env の正解の状態から教師の操作を計算して a ∈ [−1, 1]² に直す(方策の関数の形の確認)
  const teachers = new WeakMap();
  const ctrl = S.policyController((obs, env, ctx) => {
    if (!teachers.has(ctx)) teachers.set(ctx, T.createRcTrainingTeacher(ctx.world, ctx.config, ctx.variation));
    const c = teachers.get(ctx).action(env.view), a = F.commandToAction(c.steering, c.speed, env.speedCap);
    assert.equal(obs.length, 79);
    return [a.a0, a.a1];
  }, 'teacher-as-policy');
  const out = await S.evaluateController(ctrl, {sets: ['STD']});
  assert.ok(Math.abs(out.STD.single_lap_from_stop - 8.93078380923196) < 1e-6, String(out.STD.single_lap_from_stop));
  assert.ok(Math.abs(out.STD.flying_lap - 8.370092214503968) < 1e-6, String(out.STD.flying_lap));
  // ROB24 の評価の形(教師はゲート 0 以外から走れないので、ここでは止まっているだけの方策で形と開始の規則だけを見る)
  const stop = await S.evaluateController(S.policyController(() => [0, -1], 'stop'), {sets: ['ROB24']});
  assert.equal(stop.ROB24.scenes.length, 24); assert.equal(stop.ROB24.summary.valid_two_laps, 0); assert.equal(stop.ROB24.summary.pass, false);
  assert.ok(stop.ROB24.scenes.every(r => !r.invalid_spawn && r.reason === 'collection_duration_complete'));
});

// 2026-10-13 パイロットのあとの調整(deviations.md): 進みは「今の目標ゲートの s + 0.75 m」で頭打ち、停止・逆走は −10
test('調整後の報酬: 線を越えずにコースの外へ出た走り(パイロットの方策)は、目標ゲートの先の進みを得ない', async () => {
  assert.equal(E.REWARD.progress_gate_margin, .75); assert.equal(E.REWARD.terminal.stall, -10); assert.equal(E.REWARD.terminal.wrong_way, -10);
  const fx = JSON.parse(fs.readFileSync(path.join(here, 'fixtures/pilot_offcourse_actions.json'), 'utf8'));
  const run = reward => {
    const env = new E.RlEnv({core: fastCore, reward});
    env.reset({gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1});
    for (const [a0, a1] of fx.actions) { if (env.done) break; env.step(a0, a1); }
    return {env, s: env.summary()};
  };
  const old = run(E.REWARD_PILOT), cur = run(E.REWARD);
  // 走り(物理)は同じ。ゲート 3 の線は越えず(2 ゲート)、射影の s は 7.5 m 先まで進んでいた
  assert.equal(old.s.frames, cur.s.frames); assert.equal(cur.s.passed_gates, 2); assert.equal(cur.s.reason, 'stall');
  assert.ok(old.s.reward_parts.progress > 7);
  // 新しい報酬: 進みはゲート 3 の s + 0.75 m まで。停止の罰は −10
  assert.ok(Math.abs(cur.env.sCap - (cur.env.gateS + .75)) < 1e-12);
  assert.ok(cur.s.reward_parts.progress < old.s.reward_parts.progress - 2);
  assert.equal(cur.s.reward_parts.terminal, -10); assert.ok(cur.s.return < 0 && old.s.return > 0);
  // 教師の周(REC60 の 2 シーン)は頭打ちに当たらない: 進みの合計 = 射影 s の増分
  const T = await E.loadRcTeacher();
  const env = new E.RlEnv({core: fastCore, teacherModule: T, teacherActions: true});
  for (const c of S.loadRec60().slice(0, 2)) {
    env.reset(c);
    while (!env.done) { const k = env.teacherCommand; env.stepCommand(k.steering, k.speed); assert.ok(env.tracker.s <= env.gateS + .75); }
    assert.ok(Math.abs(env.summary().progress_capped_m - env.summary().progress_m) < 1e-9);
    assert.ok(env.summary().completed_laps >= 2);
  }
});

// ---- r2(手順の外の参考の追加実験): 物理のずれのランダム化。既定の動きを変えないことと、ずれの適用が正しいことを確かめる ----
test('r2 のずれ: perturb なし・標準の perturb は今までとビット単位で同じ。ずれは元のコアの設定の上書き + 1 フレームの遅延と同じ', () => {
  const drive = (env, n = 300) => { const xs = []; for (let i = 0; i < n && !env.done; i++) { env.step(Math.sin(i * .05) * .5, .2); xs.push(env.view.state.x, env.view.state.z, ...env.obs); } return xs; };
  const a = new E.RlEnv({core: fastCore}); a.reset({gate: 3, left_offset_m: .02});
  const b = new E.RlEnv({core: fastCore}); b.reset({gate: 3, left_offset_m: .02, perturb: {physics: {}, delay: 0, steerGain: 1, speedGain: 1}});
  assert.equal(b.perturb, null);
  assert.deepEqual(drive(a), drive(b));
  // 観測は今の物理を漏らさない(リセット直後の観測が同じ)
  const c = new E.RlEnv({core: fastCore}); c.reset({gate: 3, left_offset_m: .02, perturb: {physics: {steerRate: .9, frictionMultiplier: .9, brakeDeceleration: .8}, delay: 1}});
  a.reset({gate: 3, left_offset_m: .02}); assert.deepEqual([...c.obs], [...a.obs]);
  assert.equal(c.config.steerRate, .9); assert.equal(c.config.frictionMultiplier, .9);
  // 同じことを元のコアで手で行う: 設定を上書きし、1 フレーム前の指令(最初は中立)にゲインを掛けて適用する
  const physics = {steerRate: 1.0, brakeDeceleration: .9, driveTau: .45, driveAccelerationCap: 5, frictionMultiplier: .95};
  const env = new E.RlEnv({core: originalCore, coreKind: 'original', mode: 'eval', episodeMode: 'single', timeLimit: 15});
  env.reset({perturb: {physics, delay: 1, steerGain: 1.03, speedGain: .98}});
  const {world} = E.activeWorldFor(E.loadBaseWorld(), 0, 0, 0);
  const sim = originalCore.createSimulation(world, {...E.simConfig({episodeMode: 'single', timeLimit: 15}), ...physics});
  sim.reset({seed: 1, scenario: 'baseline'});
  let queued = {steering: 0, speed: 0}, same = true;
  for (let i = 0; i < 200 && !env.done; i++) {
    const a0 = Math.sin(i * .09) * .7, a1 = .4 * Math.cos(i * .03);
    env.step(a0, a1);
    const cmd = F.actionToCommand(a0, a1, 4.5, .52), ap = queued; queued = cmd;
    const s = sim.step({steering: ap.steering * 1.03, speed: ap.speed * .98}).state;
    if (s.x !== env.view.state.x || s.z !== env.view.state.z || s.speed !== env.view.state.speed) { same = false; break; }
  }
  assert.ok(same, 'ずれの適用が元のコアの手の計算と違います');
  // 引き方は決定的で、標準の回 約 2 割・遅延 1 の回 約 4 割
  const r1 = E.createRng(5), r2 = E.createRng(5), xs = [];
  for (let i = 0; i < 1000; i++) { const p = E.sampleDr(r1, E.DR_SPECS.r2); assert.deepEqual(p, E.sampleDr(r2, E.DR_SPECS.r2)); xs.push(p); }
  const nom = xs.filter(p => p.nominal).length / 1000, d1 = xs.filter(p => p.delay === 1).length / 1000;
  assert.ok(nom > .15 && nom < .25, `標準の回の割合 ${nom}`); assert.ok(d1 > .34 && d1 < .46, `遅延 1 の割合 ${d1}`);
  for (const p of xs.filter(p => !p.nominal)) for (const k of E.PERTURB_KEYS) {
    const [lo, hi] = E.DR_SPECS.r2[k]; assert.ok(p.physics[k] >= lo && p.physics[k] <= hi);
  }
});

// ---- r3(確認 r2 のあと。手順の外の参考の追加実験): 角・辺を多めに引くランダム化と、すき間の約束 ----
test('r3: r2 の引き方はビット単位で同じ。r3 は角・辺を多めに引き、遅い舵 × 弱いブレーキが r2 の数倍出る', () => {
  // r2 の 1,000 回の引き方(r3 のコードを足す前に記録した値)
  const r2 = E.createRng(5), xs2 = [];
  for (let i = 0; i < 1000; i++) xs2.push(E.sampleDr(r2, E.DR_SPECS.r2));
  assert.equal(crypto.createHash('sha256').update(JSON.stringify(xs2)).digest('hex'), '91eb262d6312d75b35c32757cc21704d8464e7f2ed7054bf0614af5bcf89b15c');
  assert.equal(r2.state, 1313820653);
  // r3: 決定的。標準 約 2 割、角・辺 約 4 割。角・辺の回の値はどれも範囲の端の 15% の中。範囲の外には出ない
  const a = E.createRng(9), b = E.createRng(9), xs3 = [], N = 20000;
  for (let i = 0; i < N; i++) { const p = E.sampleDr(a, E.DR_SPECS.r3); assert.deepEqual(p, E.sampleDr(b, E.DR_SPECS.r3)); xs3.push(p); }
  const frac = f => xs3.filter(f).length / N;
  const nom = frac(p => p.nominal), edge = frac(p => p.edge);
  assert.ok(nom > .18 && nom < .22, `標準 ${nom}`); assert.ok(edge > .37 && edge < .43, `角・辺 ${edge}`);
  for (const p of xs3.filter(p => !p.nominal)) for (const k of E.PERTURB_KEYS) {
    const [lo, hi] = E.DR_SPECS.r3[k], v = p.physics[k], w = (hi - lo) * .15;
    assert.ok(v >= lo && v <= hi, `${k} ${v}`);
    if (p.edge) assert.ok(v <= lo + w + 1e-12 || v >= hi - w - 1e-12, `端の外 ${k} ${v}`);
  }
  // 遅い舵(≤ 1.0)× 弱いブレーキ(≤ 0.9): r2 は約 1%、r3 は約 5〜6%
  const corner3 = frac(p => E.isSlowSteerWeakBrake(p.physics)), corner2 = xs2.filter(p => E.isSlowSteerWeakBrake(p.physics)).length / 1000;
  assert.ok(corner3 > .045 && corner3 < .07, `r3 の角 ${corner3}`); assert.ok(corner2 < .03, `r2 の角 ${corner2}`);
});

test('r3: すき間の約束は REWARD_R3 の学習の回だけ。3 cm を下回ったフレームで −10 で終わり、それまでは既定の報酬と同じ', () => {
  // 壁へ 0.45 m/s でまっすぐ進む(1 フレーム 1.5 cm。速いと 1 フレームで 3 cm を飛び越えて、約束より先に衝突になる)
  const drive = env => { const rs = []; while (!env.done) { const r = env.step(0, -.8); rs.push({...r, d: env.aux.d_min, obs: [...env.obs]}); } return rs; };
  const base = new E.RlEnv({core: fastCore}); base.reset({});
  const r3 = new E.RlEnv({core: fastCore, reward: E.REWARD_R3}); r3.reset({});
  const A = drive(base), B = drive(r3);
  assert.equal(A.at(-1).reason, 'collision');
  const last = B.at(-1);
  assert.equal(last.reason, 'clearance'); assert.ok(last.terminated && !last.truncated); assert.ok(last.d < .03);
  assert.ok(B.length < A.length, `約束の終わり ${B.length} < 衝突 ${A.length}`);
  assert.ok(B.slice(0, -1).every(r => r.d >= .03));
  // 終わる前までは観測・報酬がビット単位で同じ。終わるフレームは同じ報酬 − 10
  for (let i = 0; i < B.length - 1; i++) { assert.equal(B[i].reward, A[i].reward); assert.deepEqual(B[i].obs, A[i].obs); }
  assert.ok(Math.abs(last.reward - (A[B.length - 1].reward - 10)) < 1e-12);
  assert.equal(r3.summary().reward_version, E.REWARD_R3_VERSION); assert.equal(base.summary().reward_version, E.REWARD_VERSION);
  // 評価の回(mode 'eval')では使わない(sim の判定だけ)
  const ev = new E.RlEnv({core: fastCore, mode: 'eval', reward: E.REWARD_R3}); ev.reset({});
  assert.equal(drive(ev).at(-1).reason, 'collision');
  // 既定の REWARD に約束は無い
  assert.equal(E.REWARD.clearance_terminate_m, undefined); assert.equal(E.REWARD_R3.clearance_terminate_m, .03);
});
