// 走りのばらつきの物差し: 「同じ s(参照ループ上の位置)での舵の PWM が、群の平均から平均で何 µs 離れるか」。
// 今日の実験の 60 シーン(RC 教師、ゲート 0・±3 cm・±3°)は 7.6 µs だった(ユーザーへの報告の値)。
// ここでは次の群で同じ物差しを測る(どれも連続周回 20 秒、1 周目と 2 周目を s で揃える):
//   T60   RC 教師 × 記録の 60 条件(比べるための基準。ここでの計算で 7.6 µs 前後になるかを確かめる)
//   B60   ベストラップの方策 × 記録の 60 条件
//   BW    ベストラップの方策 × 広い開始(ゲート 0〜32 のうちゲート 0 を除く 8 ゲート × ±0.2 m × ±15°)。比べるのは全部が通る s
//   ZOO   ベストラップの作業の方策のチェックポイント(種・ステップ違い・r2 の物理ランダム化)× 標準の開始
//   CAP   ベストラップの方策 × 速さの上限 4.5 / 4.0 / 3.5 / 3.0
// 舵の PWM: 1442 + a0·558(a0 ≥ 0、左)/ 1442 + a0·442(a0 < 0)(dataset_recorder の換算)。
//   node probe/zoo.mjs
import fs from 'node:fs';
import path from 'node:path';
import {features, envCore, policyForward, loadBestPolicy, runPolicyEpisode, writeJson, quantiles, BESTLAP_REPORTS} from './common.mjs';
const evalSets = await import('/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/rl/eval_sets.mjs');

const L = features.LOOP_LENGTH, DS = .05, NBIN = Math.round(2 * L / DS);   // 2 周分の s の格子(開始の s から)
const steerUs = a0 => 1442 + a0 * (a0 >= 0 ? 558 : 442);
const throttleUs = v => 1498 - v / 7.24 * 498;

// 1 エピソード → s の格子ごとの {steer_us, throttle_us, ey}(開始の s を 0 に。周をまたいでも s は増え続ける)
function profile(trace, s0, {absolute = false, skip = 0} = {}) {
  // 行動はフレームの間は一定なので、格子の点 s_b を含むフレーム [s_k, s_{k+1}) の行動・横ずれを入れる(前へ進んだ区間だけ)
  const st = new Float64Array(NBIN).fill(NaN), th = new Float64Array(NBIN).fill(NaN), ey = new Float64Array(NBIN).fill(NaN);
  const tf = new Float64Array(600).fill(NaN);   // 時刻(フレーム番号)でそろえる版の舵
  trace.forEach((q, k) => { if (k < 600) tf[k] = steerUs(q.a0); });
  for (let k = 0; k + 1 < trace.length; k++) {
    const q = trace[k], a = q.s - s0, b = trace[k + 1].s - s0;
    if (a < skip || !(b > a)) continue;
    for (let i = Math.ceil((absolute ? ((q.s % L) + L) % L : a) / DS); ; i++) {
      const sb = i * DS, base = absolute ? ((q.s % L) + L) % L : a, end = base + (b - a);
      if (sb >= end) break;
      const j = absolute ? ((i % Math.round(L / DS)) + Math.round(L / DS)) % Math.round(L / DS) : i;
      if (j < 0 || j >= NBIN || !Number.isNaN(st[j])) continue;
      st[j] = steerUs(q.a0); th[j] = throttleUs(4.5 * (q.a1 + 1) / 2); ey[j] = q.ey;
    }
  }
  return {st, th, ey, tf};
}
// 群の物差し: 全員がそろう格子で、各人と群の平均の |差| の平均
function spread(profiles, key = 'st', fromBin = 0) {
  let sum = 0, n = 0, bins = 0;
  for (let b = fromBin; b < NBIN; b++) {
    const v = profiles.map(p => p[key][b]);
    if (v.some(Number.isNaN)) continue;
    const m = v.reduce((a, x) => a + x, 0) / v.length;
    for (const x of v) { sum += Math.abs(x - m); n++; }
    bins++;
  }
  return {mean_abs_dev: n ? sum / n : null, bins_used: bins};
}
function spreadT(profiles) {
  let sum = 0, n = 0;
  for (let k = 0; k < 600; k++) {
    const v = profiles.map(p => p.tf[k]);
    if (v.some(Number.isNaN)) continue;
    const m = v.reduce((a, x) => a + x, 0) / v.length;
    for (const x of v) { sum += Math.abs(x - m); n++; }
  }
  return n ? sum / n / 1 : null;
}
// 走らせる(trace に s・a0・a1 を入れる)
async function runWith(act, start, hook = null) {
  const r = await runPolicyEpisode({act, start, hook});
  if (r.invalid_spawn) return null;
  const tr = r.env;   // eval モード
  return r;
}
async function runTeacher(start) {
  const core = await envCore.loadCore('fast'), T = await envCore.loadRcTeacher();
  const env = new envCore.RlEnv({core, coreKind: 'fast', baseWorld: envCore.loadBaseWorld(), variant: 'main', mode: 'eval', episodeMode: 'continuous',
    timeLimit: 20, teacherModule: T, teacherActions: true});
  if (!env.reset(start)) return null;
  const trace = [];
  while (!env.done) {
    const c = env.teacherCommand, a = features.commandToAction(c.steering, c.speed, 4.5, .52);
    trace.push({s: env.aux.s, a0: a.a0, a1: a.a1, ey: env.aux.lateral_error});
    env.stepCommand(c.steering, c.speed);
  }
  return {trace, s0: env.s0, summary: env.summary()};
}
// 方策の走りの trace(行動の前の s と、そのフレームの行動)
async function runPolicyTrace(act, start, hook = null) {
  const core = await envCore.loadCore('fast');
  const env = new envCore.RlEnv({core, coreKind: 'fast', baseWorld: envCore.loadBaseWorld(), variant: 'main', mode: 'eval', episodeMode: 'continuous', timeLimit: 20});
  if (!env.reset(start)) return null;
  const out = [0, 0], trace = [];
  while (!env.done) {
    act(env.obs, out);
    let a0 = Math.max(-1, Math.min(1, out[0])), a1 = Math.max(-1, Math.min(1, out[1]));
    const h = hook ? hook(env, env.view.state.frame, [a0, a1]) : null;
    if (h) { a0 = h.a0; a1 = h.a1; }
    trace.push({s: env.aux.s, a0, a1, ey: env.aux.lateral_error});
    env.step(a0, a1);
  }
  return {trace, s0: env.s0, summary: env.summary()};
}
const res = {schema: 'sim-rl-modes-probe-zoo/v1', metric: '同じ s での舵 PWM の、群の平均からの |差| の平均 [µs](速さの PWM・横ずれ e_y も同じ式)', groups: {}};
const rec60 = evalSets.loadRec60();
const {act: best} = loadBestPolicy();
async function group(name, runs, note, absolute = false) {
  const ok = runs.filter(Boolean);
  const profs = ok.map(r => profile(r.trace, r.s0, {absolute}));
  res.groups[name] = {note, n: ok.length, steer_us: spread(profs, 'st'), steer_us_time_aligned: spreadT(profs), throttle_us: spread(profs, 'th'), ey_m: spread(profs, 'ey'),
    collisions: ok.filter(r => r.summary.collisions).length,
    lap1_q: quantiles(ok.map(r => r.summary.laps?.[0]?.elapsed).filter(x => x != null), [0, .5, 1])};
  console.log(name, JSON.stringify(res.groups[name]));
}
await group('T60', await Promise.all(rec60.map(c => runTeacher(c))), 'RC 教師 × 記録の 60 条件(今日の 60 シーンと同じ開始)');
await group('B60', await Promise.all(rec60.map(c => runPolicyTrace(best, c))), 'ベストラップの方策 × 記録の 60 条件');
// 広い開始: s の始まりがそろわないので、R の上の絶対の s(周をまたいだ値は mod L)で比べる
{
  const runs = [];
  for (const g of [0, 4, 8, 12, 16, 20, 24, 28]) for (const [o, y] of [[.2, 15], [-.2, -15], [.1, -8], [-.1, 8]]) runs.push(await runPolicyTrace(best, {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 1}));
  // 各走りの開始から 3 m 以降を、絶対の s(mod L)の格子(1 周分)に入れる
  // 20 秒を走り切った走りだけ(途中で当たった走りは格子を埋めないので外す)
  const okRuns = runs.filter(r => r && r.summary.sim_reason === 'collection_duration_complete' && r.summary.collisions === 0
    && r.trace[r.trace.length - 1].s - r.s0 >= L + 3.5);   // 1 周分を埋められる走りだけ(途中で止まった・戻った走りは外す)
  const profs = okRuns.map(r => profile(r.trace, r.s0, {absolute: true, skip: 3}));
  res.groups.BW = {note: 'ベストラップの方策 × 広い開始(8 ゲート × 4 姿勢)。開始から 3 m 以降を絶対の s(mod L)で比べる(1 周分)', n: profs.length,
    steer_us: spread(profs, 'st'), throttle_us: spread(profs, 'th'), ey_m: spread(profs, 'ey'), collisions: runs.filter(r => r && r.summary.collisions).length,
    excluded: runs.length - okRuns.length, excluded_starts: runs.filter(r => !okRuns.includes(r)).map(r => r?.summary?.start)};
  console.log('BW', JSON.stringify(res.groups.BW));
}
// ZOO
{
  const R = path.join(BESTLAP_REPORTS, 'rl');
  const files = ['main-s0/policies/step_006012928.json', 'main-s0/policies/step_010010624.json', 'main-s1/policies/step_004014080.json',
    'main-s1/policies/step_006012928.json', 'main-s1/policies/step_010010624.json', 'main-s2/policies/step_010010624.json',
    'r2-dr-s11/policies/step_005013504.json'].filter(f => fs.existsSync(path.join(R, f)));
  const runs = [], per = [];
  for (const f of files) {
    const act = policyForward.createPolicy(JSON.parse(fs.readFileSync(path.join(R, f), 'utf8')));
    const r = await runPolicyTrace(act, {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1});
    runs.push(r); per.push({policy: f, reason: r.summary.sim_reason, collisions: r.summary.collisions, laps: r.summary.laps.map(l => +l.elapsed.toFixed(3)),
      min_clearance: +r.summary.min_clearance.toFixed(4)});
  }
  await group('ZOO', runs, 'ベストラップの作業のチェックポイント 7 つ × 標準の開始');
  res.groups.ZOO.members = per;
  // 教師 + ZOO の 2 つずつの比べ(舵の平均 |差|)
  const t = await runTeacher({gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1});
  const pt = profile(t.trace, t.s0);
  res.groups.ZOO.members.forEach(m => m.lap_line_note = '');
  res.groups.ZOO.vs_teacher_steer_us = runs.map((r, i) => { const p = profile(r.trace, r.s0); let s = 0, n = 0;
    for (let b = 0; b < NBIN; b++) if (!Number.isNaN(p.st[b]) && !Number.isNaN(pt.st[b])) { s += Math.abs(p.st[b] - pt.st[b]); n++; }
    return {policy: files[i], mean_abs_diff_us: n ? s / n : null}; });
}
// CAP
{
  const runs = [];
  for (const cap of [4.5, 4.0, 3.5, 3.0]) runs.push(await runPolicyTrace(best, {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1},
    (env, f, a) => { const v = Math.min(4.5 * (a[1] + 1) / 2, cap); return {a0: a[0], a1: 2 * v / 4.5 - 1}; }));
  await group('CAP', runs, 'ベストラップの方策 × 速さの上限 4.5 / 4.0 / 3.5 / 3.0(学習し直しなし)');
}
const p = writeJson('zoo.json', res);
console.log('書いた:', p);
