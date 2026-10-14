#!/usr/bin/env node
// r3(確認 r2 のあと。手順の外の参考の追加実験): 方策を「確認 r2 の道具」で測る。2026-10-13
//
// 確認 r2(rl_check_r2.md)は、RL の env を通さずに、ユーザーの元の dist/sim-core.js・dist/control-delay.js の待ち行列・scene.js のコース・
// app.js の式で方策を走らせた(check_r2/common_r2.mjs)。r3 の方策も同じ道具で測り、確認 r2 の値(r2 の 500 万・選んだ方策・教師)と
// そのまま並べられるようにする。確認 r2 のファイルは読むだけで変えない。
//
//   node rl/r3/indep_check_r3.mjs --policy 名前=path.json [--policy …] [--teacher] --sets corner,rec60x,rec60,samecond,starts,allbad,grid2d --out out.json
//
// sets(deviations.md 2026-10-13 22:40 の事前登録の評価 1〜5):
//   rec60    : 記録の 60 条件(標準の物理・連続 20 秒)を遅延 0 と遅延 1(観測の約束 2 通り)で。すき間はフレームごとの厳密な距離(本ごと)と、
//              いちばん近い本を太らせた車体で小区間ごとに再生した上下限
//   samecond : STD の単周を、遅延 1 と、遅延 1 + 前進ゲイン 1.01(学習済み ONNX の記録の条件)で。観測の約束 2 通り
//   corner   : 確認 r2 と同じ 20 マス(舵の速さ 0.85・0.9・0.95・1.0・1.1 × ブレーキ 0.8・0.9・1.0・1.2、ほかは標準・遅延 0)。STD の単周・連続と REC60 の 60 の開始
//   grid2d   : 学習の範囲全体の 25 マス(舵の速さ 0.85・1.0・1.2・1.45・1.7 × ブレーキ 0.8・1.0・1.2・1.6・2.0)を REC60 の 60 の開始で
//   rec60x   : 開始(REC60)× 学習の範囲のランダムな物理。確認 r2 と同じ引き方(mulberry32 の種 515151)と、新しい種 515152
//   allbad   : 範囲の中で悪い側をすべて重ねた組み合わせ(舵 0.85・ブレーキ 0.8・時定数 0.5・加速の上限 4・μ × 0.88・遅延 1)を STD と REC60 の開始で
//   starts   : ゲート 0 の横 ±15 cm × 向き ±10° の 25 条件と、全 33 ゲートからの開始(標準の物理・連続 20 秒)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {PLAN60, REPORTS, runPolicy, runTeacher, rng32, traceClearance, inflatedReplay} from '../../check_r2/common_r2.mjs';

const args = process.argv.slice(2);
const all = n => args.flatMap((a, i) => a === n ? [args[i + 1]] : []);
const opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const sets = opt('--sets', 'rec60,samecond,corner,grid2d,rec60x,allbad,starts').split(','), out = opt('--out');

// 方策: policy.json の重みを、確認 r2 の loadMlp と同じ式で計算する(パスを受け取る版。sha256 は結果に残す)
function loadMlpPath(path) {
  const buf = fs.readFileSync(path), w = JSON.parse(buf.toString('utf8'));
  const mean = Float64Array.from(w.obs_mean), std = Float64Array.from(w.obs_std), clip = w.obs_clip;
  const layers = w.layers.map(L => ({nIn: L.n_in, nOut: L.n_out, tanh: L.act === 'tanh', w: Float64Array.from(L.w), b: Float64Array.from(L.b)}));
  const fn = obs => {
    let x = new Float64Array(obs.length);
    for (let i = 0; i < obs.length; i++) x[i] = Math.max(-clip, Math.min(clip, (obs[i] - mean[i]) / std[i]));
    for (const L of layers) {
      const y = new Float64Array(L.nOut);
      for (let o = 0; o < L.nOut; o++) {
        let s = L.b[o];
        const off = o * L.nIn;
        for (let i = 0; i < L.nIn; i++) s += L.w[off + i] * x[i];
        y[o] = L.tanh ? Math.tanh(s) : s;
      }
      x = y;
    }
    return [Math.max(-1, Math.min(1, x[0])), Math.max(-1, Math.min(1, x[1]))];
  };
  fn.sha = crypto.createHash('sha256').update(buf).digest('hex');
  return fn;
}
const ctrls = all('--policy').map(spec => { const [key, path] = spec.split('='); const pol = loadMlpPath(path); return {key, path, sha: pol.sha, run: o => runPolicy(pol, o)}; });
if (args.includes('--teacher')) ctrls.push({key: 'teacher', path: null, sha: null, teacher: true, run: o => runTeacher(o)});

// 学習の範囲(r2・r3 で同じ。run.json の記録から)と、確認 r2 の e3_physics.mjs と同じ引き方
const DR = JSON.parse(fs.readFileSync(REPORTS + '/rl/r2-dr-s11/run.json', 'utf8')).env.dr_spec;
const PK = ['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier'];
function sample(rnd, spec) {
  const p = {phys: {}};
  for (const k of PK) p.phys[k] = spec[k][0] + (spec[k][1] - spec[k][0]) * rnd();
  p.delay = rnd() < spec.delay1_fraction ? 1 : 0;
  p.steerGain = spec.steerGain[0] + (spec.steerGain[1] - spec.steerGain[0]) * rnd();
  p.speedGain = spec.speedGain[0] + (spec.speedGain[1] - spec.speedGain[0]) * rnd();
  return p;
}
const inContract = c => c['start-gate'] === 0 && Math.abs(c['spawn-offset']) <= .03 + 1e-12 && Math.abs(c['spawn-yaw']) <= 3 + 1e-12;
const recRun = (C, c, o = {}) => C.run({plan: c, gate: c['start-gate'], offset: c['spawn-offset'], yawDeg: c['spawn-yaw'], mode: 'continuous', ...o});
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
function rec60Count(C, o = {}) {
  let ok = 0, n = 0;
  for (const c of PLAN60) { if (C.teacher && !inContract(c)) continue; n++; if (recRun(C, c, o).summary.valid) ok++; }
  return {valid: ok, n};
}

const result = {schema: 'rl-bestlap-r3-indep-check/v1', created_at: new Date().toISOString(), sets, tool: 'check_r2/common_r2.mjs(元の sim-core.js・ユーザーの遅延の待ち行列)',
  dr_spec_from_r2_run_json: DR, controllers: {}};
for (const C of ctrls) {
  const t0 = Date.now(), R = {path: C.path, sha256: C.sha};
  if (sets.includes('rec60')) {
    R.rec60 = {};
    for (const [label, o] of [['delay0', {}], ['delay1_applied', {delay: 1, prev: 'applied'}], ['delay1_fresh', {delay: 1, prev: 'fresh'}]]) {
      if (C.teacher && label !== 'delay0') continue;
      const rows = [];
      for (const c of PLAN60) {
        if (C.teacher && !inContract(c)) continue;
        const r = recRun(C, c, o);
        const row = {name: c.name, valid: r.summary.valid, reason: r.summary.reason, laps: r.summary.laps, first_collision: r.summary.first_collision};
        if (label === 'delay0') { const cl = traceClearance(r.report, r.world); row.clearance = {d: cl.d, who: cl.who, frame: cl.frame}; row._r = r; }
        rows.push(row);
      }
      const ok = rows.filter(r => r.valid), S = {n: rows.length, valid: ok.length, lap1_mean: mean(ok.map(r => r.laps[0])), lap2_mean: mean(ok.map(r => r.laps[1])),
        failures: rows.filter(r => !r.valid).map(r => ({name: r.name, reason: r.reason, laps: r.laps, first_collision: r.first_collision}))};
      if (label === 'delay0') {
        const ds = rows.map(r => r.clearance.d), lo = rows.reduce((a, b) => (b.clearance.d < a.clearance.d ? b : a));
        S.clearance_frame_min = {name: lo.name, ...lo.clearance};
        S.clearance_frame_below_3cm_runs = ds.filter(d => d < .03).length;
        S.clearance_frame_below_5cm_runs = ds.filter(d => d < .05).length;
        S.clearance_frame_median = [...ds].sort((a, b) => a - b)[Math.floor(ds.length / 2)];
        // いちばん近い本を、太らせた車体で小区間ごとに再生する(2 分探索。衝突しない幅 ⇒ すき間 > 幅)
        let a = 0, b = Math.max(.002, lo.clearance.d * 1.6);
        if (!inflatedReplay(lo._r.report, lo._r.world, b).collided) b = .2;
        for (let i = 0; i < 9; i++) { const m = (a + b) / 2; if (inflatedReplay(lo._r.report, lo._r.world, m).collided) b = m; else a = m; }
        S.clearance_substep_bounds_min_run = {no_collision_margin_m: a, collision_margin_m: b, clearance_lower_m: a, clearance_upper_m: Math.SQRT2 * b};
      }
      for (const r of rows) delete r._r;
      R.rec60[label] = S;
    }
  }
  if (sets.includes('samecond') && !C.teacher) {
    R.samecond = {};
    for (const [label, o] of [['delay0', {}], ['delay1_applied', {delay: 1, prev: 'applied'}], ['delay1_fresh', {delay: 1, prev: 'fresh'}],
      ['delay1_gain101_applied', {delay: 1, speedGain: 1.01, prev: 'applied'}], ['delay1_gain101_fresh', {delay: 1, speedGain: 1.01, prev: 'fresh'}]]) {
      const s = C.run({mode: 'single', ...o}).summary;
      R.samecond[label] = {valid: s.valid, lap: s.laps[0] ?? null, reason: s.reason, first_collision: s.first_collision};
    }
    const c = C.run({mode: 'continuous'}).summary;
    R.samecond.std_continuous = {valid: c.valid, laps: c.laps, reason: c.reason};
  }
  if (sets.includes('corner')) {
    const STEER = [.85, .9, .95, 1.0, 1.1], BRAKE = [.8, .9, 1.0, 1.2], cells = [];
    for (const sr of STEER) for (const br of BRAKE) {
      const phys = {steerRate: sr, brakeDeceleration: br};
      const s = C.run({mode: 'single', phys}).summary, c = C.run({mode: 'continuous', phys}).summary;
      cells.push({steerRate: sr, brake: br, std_single: s.valid, std_single_lap: s.laps[0] ?? null, std_cont: c.valid, ...(r => ({rec60_valid: r.valid, rec60_n: r.n}))(rec60Count(C, {phys}))});
    }
    R.corner = {steer: STEER, brake: BRAKE, cells, cells_60: cells.filter(c => c.rec60_valid === c.rec60_n).length, cells_ge57: cells.filter(c => c.rec60_valid >= 57).length,
      total_valid: cells.reduce((a, c) => a + c.rec60_valid, 0), total_n: cells.reduce((a, c) => a + c.rec60_n, 0)};
  }
  if (sets.includes('grid2d')) {
    const STEER = [.85, 1.0, 1.2, 1.45, 1.7], BRAKE = [.8, 1.0, 1.2, 1.6, 2.0], cells = [];
    for (const sr of STEER) for (const br of BRAKE) cells.push({steerRate: sr, brake: br, ...(r => ({rec60_valid: r.valid, rec60_n: r.n}))(rec60Count(C, {phys: {steerRate: sr, brakeDeceleration: br}}))});
    R.grid2d = {steer: STEER, brake: BRAKE, cells, cells_60: cells.filter(c => c.rec60_valid === c.rec60_n).length, cells_ge57: cells.filter(c => c.rec60_valid >= 57).length,
      total_valid: cells.reduce((a, c) => a + c.rec60_valid, 0), total_n: cells.reduce((a, c) => a + c.rec60_n, 0)};
  }
  if (sets.includes('rec60x')) {
    R.rec60x = {};
    for (const seed of [515151, 515152]) {
      const rnd = rng32(seed), rows = [];
      for (const c of PLAN60) {
        const p = sample(rnd, DR);   // 教師が走らない開始でも同じ数だけ引く(確認 r2 と同じ順番)
        if (C.teacher && !inContract(c)) continue;
        const s = recRun(C, c, p).summary;
        rows.push({name: c.name, p, valid: s.valid, reason: s.reason, laps: s.laps});
      }
      const isCorner = p => p.phys.steerRate < .96 && p.phys.brakeDeceleration < .84;
      R.rec60x[seed] = {n: rows.length, valid: rows.filter(r => r.valid).length, corner_draws: rows.filter(r => isCorner(r.p)).length,
        corner_valid: rows.filter(r => isCorner(r.p) && r.valid).length, failures: rows.filter(r => !r.valid).map(r => ({name: r.name, p: r.p, reason: r.reason}))};
    }
  }
  if (sets.includes('allbad')) {
    const o = {phys: {steerRate: .85, brakeDeceleration: .8, driveTau: .5, driveAccelerationCap: 4, frictionMultiplier: .88}, delay: 1};
    const s = C.run({mode: 'single', ...o}).summary, c = C.run({mode: 'continuous', ...o}).summary;
    R.allbad = {cond: o, std_single: {valid: s.valid, lap: s.laps[0] ?? null, reason: s.reason}, std_cont: {valid: c.valid, laps: c.laps, reason: c.reason}, rec60: rec60Count(C, o)};
  }
  if (sets.includes('starts') && !C.teacher) {
    const grid = [];
    for (const off of [-.15, -.075, 0, .075, .15]) for (const yaw of [-10, -5, 0, 5, 10]) { const s = C.run({mode: 'continuous', offset: off, yawDeg: yaw}).summary; grid.push({off, yaw, valid: s.valid, reason: s.reason}); }
    const gates = [];
    for (let g = 0; g < 33; g++) { const r = C.run({mode: 'continuous', gate: g}); if (r.invalid_spawn) { gates.push({gate: g, invalid_spawn: true}); continue; } gates.push({gate: g, valid: r.summary.valid, reason: r.summary.reason}); }
    R.starts = {grid_valid: grid.filter(r => r.valid).length, grid_n: grid.length, gates_valid: gates.filter(r => r.valid).length,
      gates_n: gates.filter(r => !r.invalid_spawn).length, failures: [...grid.filter(r => !r.valid), ...gates.filter(r => !r.valid && !r.invalid_spawn)]};
  }
  R.wall_s = (Date.now() - t0) / 1000;
  result.controllers[C.key] = R;
  console.log(C.key, JSON.stringify({rec60: R.rec60 && Object.fromEntries(Object.entries(R.rec60).map(([k, v]) => [k, `${v.valid}/${v.n} ${v.lap1_mean?.toFixed(4)} ${v.lap2_mean?.toFixed(4)}` +
      (v.clearance_frame_min ? ` clr ${v.clearance_frame_min.d.toFixed(4)} <3cm ${v.clearance_frame_below_3cm_runs}` : '')])),
    samecond: R.samecond && Object.fromEntries(Object.entries(R.samecond).map(([k, v]) => [k, v.lap ?? v.laps ?? v.reason])),
    corner: R.corner && `${R.corner.cells_60}/20 マスが 60/60、合計 ${R.corner.total_valid}/${R.corner.total_n}`,
    grid2d: R.grid2d && `${R.grid2d.cells_60}/25 マスが 60/60、合計 ${R.grid2d.total_valid}/${R.grid2d.total_n}`,
    rec60x: R.rec60x && Object.fromEntries(Object.entries(R.rec60x).map(([k, v]) => [k, `${v.valid}/${v.n} 角 ${v.corner_valid}/${v.corner_draws}`])),
    allbad: R.allbad && `単周 ${R.allbad.std_single.valid} 連続 ${R.allbad.std_cont.valid} REC60 ${R.allbad.rec60.valid}/${R.allbad.rec60.n}`,
    starts: R.starts && `${R.starts.grid_valid}/${R.starts.grid_n} ${R.starts.gates_valid}/${R.starts.gates_n}`, wall_s: R.wall_s}));
  if (out) fs.writeFileSync(out, JSON.stringify(result, null, 1) + '\n');   // 方策ごとに途中でも書く(電源断に備える)
}
