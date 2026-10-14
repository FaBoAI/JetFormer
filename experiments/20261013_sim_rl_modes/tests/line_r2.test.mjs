// 経路いろいろ第 2 回(protocol_line_r2.json。2026-10-14 学習の段 line r2)の試験:
// 目標の線 v2(開始のつなぎ・R の曲がりの内側の上限・なめらかさ)、v1 が変わらないこと、学習の設定、試験の組、走行中の仕組みの線の版。
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import * as modesCore from '../rl/fast_core_modes.mjs';
import {ModeEnv, trainConfig, LINE_R2_DEFAULTS} from '../rl/env_modes.mjs';
import {createRng} from '../rl/runtime/rng.mjs';
import {createLineProfile, eStarFromC, LINE_PROFILE_V2} from '../rl/runtime/line_profiles.mjs';
import {LINE_V2_TABLE, LINE_V2_TABLE_SHA256} from '../rl/runtime/line_v2_table.mjs';
import {buildJobs, loadLineR2} from '../rl/eval_modes.mjs';
import {buildLineV2Table} from '../rl/line_r2/build_line_v2_table.mjs';
import {analyse} from '../rl/line_r2/feasibility.mjs';
import {LOOP_LENGTH} from '../rl/runtime/features.mjs';
import {runModeEpisode} from '../rl/episode.mjs';
import {EXP, readJson} from './helpers.mjs';

const R2 = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/protocol_line_r2.json', 'utf8'));
const L = LOOP_LENGTH, T = LINE_V2_TABLE;
const FAMS = [{family: 'const', c0: .8}, {family: 'const', c0: -.9}, {family: 'const', c0: 0}, {family: 'inside', a: .9}, {family: 'outside', a: .9},
  {family: 'late_apex', a: .9}, {family: 'early_apex', a: .9}, {family: 'random', seed: 12345}, {family: 'random', seed: 987654321}];
const boundsAt = s => { const u = (((s % L) + L) % L) / T.ds_m, i = Math.floor(u), f = u - i, i1 = (i + 1) % T.n;
  return [T.lo[i] + (T.lo[i1] - T.lo[i]) * f, T.hi[i] + (T.hi[i1] - T.hi[i]) * f, T.kappa[i] + (T.kappa[i1] - T.kappa[i]) * f]; };

test('幅の表 v2: 作り直すと同じ sha256・protocol の値と同じ・lo ≤ hi・内側の上限で e·κ < 0.5', () => {
  const body = JSON.stringify(buildLineV2Table());
  assert.equal(crypto.createHash('sha256').update(body).digest('hex'), LINE_V2_TABLE_SHA256);
  assert.equal(LINE_V2_TABLE_SHA256, R2.target_line.line_v2_table_sha256);
  for (const k of ['r_min_m', 'slope_max', 'min_filter_m', 'bound_sigma_m', 'kappa_chord_m', 'kappa_sigma_m']) assert.equal(T.params[k], R2.target_line.params[k], k);
  for (let i = 0; i < T.n; i++) {
    assert.ok(T.lo[i] <= T.hi[i] + 1e-9, `lo > hi at ${i}`);
    assert.ok(T.hi[i] * T.kappa[i] < .5 && -T.lo[i] * T.kappa[i] < .5, `e·κ at ${i}`);
    if (i) assert.ok(Math.abs(T.hi[i] - T.hi[i - 1]) <= R2.target_line.params.slope_max * T.ds_m + 2e-5, `傾き hi at ${i}`);
  }
});

test('目標の線 v2: 区間 [lo, hi] の中・尖りなし(e·κ_R < 0.5)・線の曲率半径 ≥ 0.45 m・random は種で決まる', () => {
  for (const p of FAMS) {
    const pr = createLineProfile(p, 3.3, {version: 'v2'});
    assert.equal(pr.version, 'v2'); assert.equal(pr.needsAnchor, true);
    for (let s = 3.3; s < 3.3 + 2.2 * L; s += .037) {
      const e = pr.eStar(s), [lo, hi, k] = boundsAt(s);
      assert.ok(e >= lo - 1e-9 && e <= hi + 1e-9, `${JSON.stringify(p)} s=${s} e=${e} [${lo}, ${hi}]`);
      assert.ok(e * k < .5, `e·κ ${e * k}`);
    }
    const r = analyse(s => pr.eStar(s), {sa: 3.3, sb: 3.3 + L, standing: false});
    assert.ok(1 / r.max_abs_kappa_path >= .45, `${JSON.stringify(p)} 半径 ${1 / r.max_abs_kappa_path}`);
  }
  const a = createLineProfile({family: 'random', seed: 5}, 1, {version: 'v2'}), b = createLineProfile({family: 'random', seed: 5}, 1, {version: 'v2'});
  for (let s = 1; s < 60; s += .7) assert.equal(a.eStar(s), b.eStar(s));
});

test('目標の線 v2 の開始のつなぎ: s0 で e0、s0 + 2 m から先は線そのもの、間は単調な smoothstep。anchor は 1 回だけ', () => {
  const pr = createLineProfile({family: 'const', c0: .8}, 10.0, {version: 'v2'});
  const line = s => pr.lineAt(s);
  pr.anchor(-.15);
  assert.equal(pr.needsAnchor, false); assert.equal(pr.e0, -.15);
  assert.ok(Math.abs(pr.eStar(10.0) + .15) < 1e-12);
  assert.ok(Math.abs(pr.eStar(9.0) + .15) < 1e-12);
  for (const s of [12.0, 12.5, 20, 40]) assert.ok(Math.abs(pr.eStar(s) - line(s)) < 1e-12);
  const s1 = 11.0, w = .5 * .5 * (3 - 2 * .5);
  assert.ok(Math.abs(pr.eStar(s1) - (-.15 + (line(s1) + .15) * w)) < 1e-12);
  pr.anchor(.3);   // 2 回目は無視
  assert.equal(pr.e0, -.15);
  assert.equal(LINE_PROFILE_V2.blend_m, R2.target_line.params.blend_m);
  assert.equal(LINE_PROFILE_V2.smooth_sigma_m, R2.target_line.params.smooth_sigma_m);
});

test('目標の線 v1 は今までと同じ(版を指定しない・v1 は eStarFromC そのもの・anchor なし)', () => {
  for (const p of FAMS) {
    const a = createLineProfile(p, 2.0), b = createLineProfile(p, 2.0, {version: 'v1'});
    assert.equal(a.needsAnchor, false);
    for (let s = 2; s < 40; s += .53) { assert.equal(a.eStar(s), b.eStar(s)); assert.equal(a.eStar(s), eStarFromC(a.c(s), s)); }
  }
  assert.throws(() => createLineProfile({family: 'const'}, 0, {version: 'v9'}));
});

test('line(r2)の学習の設定 = protocol_line_r2.json training の値。cond・speed の設定は変わらない', () => {
  const c = trainConfig('line'), Rt = R2.training;
  assert.equal(c.wrapper, 'cond97'); assert.equal(c.virtual, true); assert.equal(c.gate_miss_terminal, true); assert.equal(c.line_profile, 'v2');
  assert.deepEqual(c.start_mix.map(m => [m.p, m.left_offset_m, m.left_yaw_deg, m.min_initial_clearance_m]),
    Rt.start_mix.map(m => [m.p, m.left_offset_m, m.left_yaw_deg, m.min_initial_clearance_m]));
  assert.equal(c.reward.progress_track_sigma_m, .08); assert.equal(c.reward.tracking_w, .05); assert.equal(c.reward.tracking_scale_m, .15);
  assert.equal(c.reward.clearance_barrier_w, .2); assert.equal(c.reward.clearance_barrier_start_m, .06); assert.equal(c.reward.terminal_clearance_m, .03);
  assert.deepEqual(c.cond.c_v_mix.map(q => [q.p, ...q.range]), [[.8, .9, 1.0], [.2, .6, .9]]);
  for (const [k, r] of Object.entries(Rt.tuning_ranges)) if (c.reward[k] != null) assert.ok(c.reward[k] >= r[0] && c.reward[k] <= r[1], k);
  const cond = trainConfig('cond');
  assert.equal(cond.line_profile, undefined); assert.equal(cond.reward.progress_track_sigma_m, undefined); assert.equal(cond.reward.tracking_w, .08);
  assert.equal(trainConfig('speed').line_profile, undefined);
  assert.equal(LINE_R2_DEFAULTS.line_profile, 'v2');
});

test('line(r2)の環境: 開始のつなぎ(最初の e* = 車の横ずれ)・開始のすき間・c_v の割合・進みの掛け算・3 cm 未満の終了', () => {
  const env = new ModeEnv({core: modesCore, run: 'line'}), rng = createRng(11);
  let hi = 0; const N = 200;
  for (let k = 0; k < N; k++) {
    env.resetRandom(rng);
    assert.ok(Math.abs(env.aux.e_star - env.aux.lateral_error) < 1e-12, '開始の e* = e_y');
    assert.equal(env.wrapper.line.version, 'v2');
    const comp = env.cfg.start_mix[env.start.component];
    assert.ok(env.aux.d_min >= comp.min_initial_clearance_m - 1e-12);
    if (env.cV >= .9) hi++;
  }
  assert.ok(hi / N > .72 && hi / N < .88, `c_v ≥ 0.9 の割合 ${hi / N}`);
  // 進みの掛け算: 線の上なら引かれない、線から 8 cm なら進みの半分
  assert.equal(env.reset(createRng(3), {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}), true);
  const obs = env.wrapper.observe.bind(env.wrapper);
  env.wrapper.observe = (st, a0, a1, out, aux) => { const r = obs(st, a0, a1, out, aux); aux.lateral_error = aux.e_star + .08; return r; };
  let r = null; for (let k = 0; k < 20; k++) r = env.step(0, .5);
  const P = env.stats.reward_parts;
  assert.ok(P.track_progress < 0 && Math.abs(P.track_progress + P.progress / 2) < .02 * Math.abs(P.progress), `${P.track_progress} vs ${P.progress}`);
  env.wrapper.observe = (st, a0, a1, out, aux) => { const r = obs(st, a0, a1, out, aux); aux.d_min = .02; return r; };
  r = env.step(0, .5);
  assert.equal(r.reason, 'clearance'); assert.equal(r.terminated, true);
});

test('第 2 回の試験・検証・報告の組: 回の数・開始のすき間 ≥ 0.12 m・種・第 1 回の試験の組と重ならない・形の並び', () => {
  const L2 = loadLineR2();
  const t = buildJobs('line', 'test', {protocolLineR2: L2}), v = buildJobs('line', 'validation', {protocolLineR2: L2}), f = buildJobs('line', 'fresh', {protocolLineR2: L2});
  const p = t.filter(j => j.test === 'LN-P60');
  assert.equal(p.length, 60); assert.equal(t.filter(j => j.test === 'LN-FAM6').length, 6);
  assert.equal(v.filter(j => j.test === 'LN-P60').length, 60); assert.equal(v.filter(j => j.test === 'LN-FAM6').length, 6);
  assert.equal(f.length, 120); assert.equal(f.filter(j => j.test === 'LN-FAM6').length, 0);
  assert.ok(p.every((j, i) => j.spec.start.seed === 904500 + i && j.spec.start.initial_clearance_m >= .12 && j.spec.start.halton_index >= 26001));
  assert.ok(v.filter(j => j.test === 'LN-P60').every((j, i) => j.spec.start.seed === 804500 + i && j.spec.start.initial_clearance_m >= .12));
  assert.ok(f.every((j, i) => j.spec.start.seed === 2300000 + i && j.spec.start.initial_clearance_m >= .12));
  const r1 = buildJobs('line', 'test', {}).filter(j => j.test === 'LN-P60').map(j => j.spec.start.halton_index);
  assert.ok(Math.max(...r1) < 26001);
  assert.deepEqual(p.slice(0, 15).map(j => j.spec.family), ['const_left', 'const_left', 'const_right', 'const_right', 'const_center', 'inside', 'inside',
    'outside', 'outside', 'late_apex', 'late_apex', 'early_apex', 'random', 'random', 'random']);
  assert.ok(p.every(j => j.spec.params.c_v >= .9 && j.spec.params.c_v <= 1));
});

test('走行中の仕組み: 方策の meta.line_profile = v2 なら線 v2(p と metadata に版)、無ければ v1 のまま。speed はいつも v1', async () => {
  const w1 = readJson(`${EXP}/candidates/line/policy.json`), w2 = {...w1, meta: {...w1.meta, line_profile: 'v2'}};
  const start = {gate: 7, left_offset_m: .1, left_yaw_deg: 6, seed: 4242};
  const a = await runModeEpisode({mode: 'line', weights: w1, start, params: {line: {family: 'const', c0: .8}, c_v: 1}, coreKind: 'fast'});
  const b = await runModeEpisode({mode: 'line', weights: w2, start, params: {line: {family: 'const', c0: .8}, c_v: 1}, coreKind: 'fast'});
  assert.equal(a.params.line_profile, undefined); assert.equal(b.params.line_profile, 'v2');
  assert.ok(Math.abs(b.log[0].es - b.log[0].ey) < 1e-12, 'v2 の最初の e* = e_y');
  assert.ok(Math.abs(a.log[0].es - b.log[0].es) > 1e-3, 'v1 と v2 で開始の e* が違う');
  const sp = await runModeEpisode({mode: 'speed', weights: w2, start, params: {c_v: .75}, coreKind: 'fast'});
  assert.equal(sp.params.line_profile, undefined);
});
