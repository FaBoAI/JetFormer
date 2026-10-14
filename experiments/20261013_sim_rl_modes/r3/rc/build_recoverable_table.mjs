// 離脱復活 r3(2026-10-14、学習の段 recover r3): 「立て直せる表」を作る(protocol_recover_r3.json perturbation_r3.recoverable_table のとおり)。
//   node r3/rc/build_recoverable_table.mjs --shard k/n --out rows_k.jsonl [--states-every 4] [--limit-states N] [--selftest]
// 手順(乱数を使わない決定的な手順。r3/rc/oracle_proto.mjs が元):
//   1. 参照の走り 4 つ(driver)× 開始 8 つ(STD + mulberry32(0x7ab1e001) の記録の開始の分布 7 つ、種 3,240,000 + n)を、元の sim-core.js(core_oracle.mjs、
//      物理は 1 文字も変えない写し)・標準の物理・連続 20 秒で走らせる。driver = 離脱復活 r1 の方策(7720f275…)・最速の方策(7c43cb53…)・
//      r1 + 速さの上限 3.0 m/s・r1 + 速さの上限 2.2 m/s(どれも乱しなし)。
//   2. 各回のフレーム 30〜590 を 4 フレームおきに状態を保存し(最初の衝突より前だけ)、18 通りの組(舵のパルス 弱・中・強 × 左右、わざと外す 弱・中・強 × 左右、
//      ブレーキ 弱・中・強、踏み込み 弱・中・強。中身はクラスの上限 = 悪い側)をそれぞれ差し込む。わざと外すは、その側の幅(先 1.5 m の最小)が
//      d + 0.05 m に足りなければその状態では使えない(行は ok = null)。
//   3. 立て直せる = 次の操縦のどれか 1 つで「ps から pe + 3 秒まで衝突なし・フレームの終わりのすき間 ≥ 0.02 m・最後に前へ 0.5 m/s 以上
//      (20 秒の終わりに達したときは不要)・|e_y − emid(s)| ≤ 0.20 m・|向きのずれ| ≤ 0.30 rad」:
//        (b) その driver の方策で続ける、(c) 離脱復活 r1 の方策で続ける(driver が r1 そのものなら (b) と同じなので省く)、
//        (a) oracle_proto.mjs の 24 通り(乱しの間の速さ {そのまま, 0} × 乱しのあと {最大すき間の線の pure pursuit 速さ {そのまま, 2.0, 1.5, 1.0} /
//            舵 a ∈ {±0.52, ±0.26} を t1 ∈ {0.2, 0.4} 秒ブレーキ → pure pursuit 2.0}。ブレーキ/踏み込みは乱しの間の速さが決まっているので 12 通り)。
//      乱しの間、乱していない側の指令は (b)(c) では方策、(a) では「そのまま/0」(舵のパルス・わざと外す)か最大すき間の線の pure pursuit(ブレーキ/踏み込み)。
//      わざと外すは走行中と同じく d_min < 0.10 m で途中で終える。速く見つかる順に (b)(c)(a) の順で試し、最初に立て直せたものを記録する。
//   4. 1 行 1 つの (driver, 開始, フレーム, 組) を JSONL に書く。表(マス = s 0.25 m × 速さ 0.5 m/s × 組)は merge_recoverable_table.py が作る。
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createSimulation, original as ORIG, PATCHED_SHA256, SIM_CORE_SHA256} from './core_oracle.mjs';
import {BASE_WORLD, activeWorld, project, NOMINAL, LOOP, startClearance} from '../../rl/check_rc/run_rc.mjs';
import {buildObstacles, ownClearance, nearObstacles} from '../../rl/check_nw/own_geom.mjs';
import * as RT from './rt/mode_runtime.mjs';
import {actionToCommand, prepareGeometry, loopPointAt} from './rt/features.mjs';
import {createObsWrapper, OBS_WRAPPERS} from './rt/obs_modes.mjs';
import {tableAt} from './rt/line_profiles.mjs';
import {perturbAllowed, RECORD_SPEC, TRAIN_SPEC} from './rt/perturb.mjs';

const RC = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/rc';
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const [SK, SN] = (opt('--shard', '0/1')).split('/').map(Number);
const OUT = opt('--out'), EVERY = +opt('--states-every', '4'), LIMIT = opt('--limit-states') ? +opt('--limit-states') : Infinity;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const CLS = RECORD_SPEC.classes, FPS = 30;

// ---- 参照の方策(写し。sha256 は protocol の値と照合) ----
const POL = {
  r1: {path: `${RC}/ref/recover_r1_policy.json`, want: '7720f2754972efbbd9b714008c99e63822b5421a0eef349840d3d52fb8a0456f', mode: 'recover'},
  fastest: {path: `${RC}/ref/fastest_policy.json`, want: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee', mode: 'fastest'},
};
for (const P of Object.values(POL)) {
  P.sha = sha(P.path);
  if (P.sha !== P.want) throw new Error(`方策 ${P.path} の sha256 ${P.sha} が違います`);
  P.weights = JSON.parse(fs.readFileSync(P.path, 'utf8'));
  P.act = RT.policyFromWeights(P.weights, 79);
}
export const DRIVERS = [
  {name: 'r1', pol: 'r1', limit: null},
  {name: 'fastest', pol: 'fastest', limit: null},
  {name: 'r1_cap3.0', pol: 'r1', limit: 3.0},
  {name: 'r1_cap2.2', pol: 'r1', limit: 2.2},
];
// ---- 開始(STD + mulberry32(0x7ab1e001) の記録の開始の分布 7 つ。開始のすき間 > 0) ----
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function tableStarts() {
  const out = [{gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 3240000, n: 0}];
  const u = mulberry32(0x7ab1e001);
  for (let guard = 0; out.length < 8 && guard < 1000; guard++) {
    const gate = Math.min(32, Math.floor(u() * 33)), off = -.2 + .4 * u(), yaw = -15 + 30 * u();
    if (!(startClearance(gate, off, yaw) > 0)) continue;
    out.push({gate, left_offset_m: off, left_yaw_deg: yaw, seed: 3240000 + out.length, n: out.length});
  }
  return out;
}
// ---- 18 通りの組(クラスの上限 = 悪い側) ----
export const COMBOS = [];
for (const c of ['弱', '中', '強']) {
  for (const sign of [1, -1]) COMBOS.push({key: `steer_pulse:${c}:${sign > 0 ? 'L' : 'R'}`, type: 'steer_pulse', cls: c, sign, n: CLS[c].steer_pulse.frames[1], k: CLS[c].steer_pulse.k[1]});
  for (const sign of [1, -1]) COMBOS.push({key: `excursion:${c}:${sign > 0 ? 'L' : 'R'}`, type: 'excursion', cls: c, sign, d: CLS[c].excursion.d_m[1], n: Math.max(1, Math.round(CLS[c].excursion.T_s[1] * FPS))});
  COMBOS.push({key: `brake:${c}`, type: 'speed_pulse', kind: 'brake', cls: c, n: CLS[c].speed_pulse.brake_frames[1]});
  COMBOS.push({key: `burst:${c}`, type: 'speed_pulse', kind: 'burst', cls: c, n: CLS[c].speed_pulse.burst_frames[1]});
}
// ---- 方策で動かす運転手(観測の包み + 方策。走行中の mode_runtime.mjs と同じ式) ----
function makeController(P, limit, world, cfg, geometry) {
  const spec = RT.runtimeSpec(P.mode);
  const wrapper = createObsWrapper({wrapper: spec.obs_wrapper, world, config: cfg, geometry});
  const obs = new Float64Array(OBS_WRAPPERS[spec.obs_wrapper].dim), aux = {}, pa = [0, 0];
  let prevA = [0, 0];
  return {
    start(st, ri, pA) { wrapper.reset(st, ri); prevA = [pA[0], pA[1]]; },
    advance(st, ri) { wrapper.advance(st, ri); },
    propose(st) {
      wrapper.observe(st, prevA[0], prevA[1], obs, aux); P.act(obs, pa);
      const pc = actionToCommand(pa[0], pa[1], RT.BASE_SPEED_CAP, cfg.maxSteer);
      const cmd = {steering: pc.steering, speed: limit != null && pc.speed > limit ? limit : pc.speed};
      return {pc, cmd, s: aux.s, ey: aux.lateral_error, dmin: aux.d_min};
    },
    applied(cmd, pc) { prevA = RT.appliedAction(cmd, pc, RT.BASE_SPEED_CAP, cfg.maxSteer); },
  };
}
// 最大すき間の線(room_table の emid)を前方 Ld で追う pure pursuit
const PT = {x: 0, z: 0, tx: 0, tz: 0};
function ppEmid(st, s, Ld = .6) {
  loopPointAt(s + Ld, PT); const e = tableAt('emid', s + Ld), tx = PT.x + e * PT.tz, tz = PT.z - e * PT.tx;
  const hx = Math.cos(st.yaw), hy = Math.sin(st.yaw), vx = tx - st.x, vy = -(tz - st.z);
  const alpha = Math.atan2(hx * vy - hy * vx, hx * vx + hy * vy), d = Math.hypot(vx, vy);
  return Math.max(-.52, Math.min(.52, Math.atan2(2 * NOMINAL.wheelbase * Math.sin(alpha), Math.max(d, .2))));
}
// わざと外す: 走行中(perturb.mjs)と同じ式。R を e_tgt ずらした線の前方 0.8 m を pure pursuit
const PT2 = {x: 0, z: 0, tx: 0, tz: 0};
function excursionSteer(st, s, eTgt) {
  loopPointAt(s + .8, PT2);
  const tx = PT2.x + eTgt * PT2.tz, tz = PT2.z - eTgt * PT2.tx;
  const dx = tx - st.x, dz = tz - st.z, alpha = wrap(Math.atan2(-dz, dx) - st.yaw), ld = Math.max(Math.hypot(dx, dz), .3);
  return Math.max(-.52, Math.min(.52, Math.atan2(2 * NOMINAL.wheelbase * Math.sin(alpha), ld)));
}
function room(side, s, ey) { let m = Infinity; for (let q = 0; q <= 1.5 + 1e-9; q += .25) m = Math.min(m, side > 0 ? tableAt('rhoL', s + q) - ey : tableAt('rhoR', s + q) + ey); return m; }
// 操縦の組 (a) の 24 通り(oracle_proto.mjs と同じ並び)
export const PLANS_A = [];
for (const during of ['hold', 'brake']) {
  for (const vpp of ['hold', 2.0, 1.5, 1.0]) PLANS_A.push({kind: 'a', during, t1: 0, a: null, vpp});
  for (const t1 of [.2, .4]) for (const a of [.52, .26, -.26, -.52]) PLANS_A.push({kind: 'a', during, t1, a, vpp: 2.0});
}

const OB = buildObstacles(BASE_WORLD);
// ---- 参照の走り ----
function referenceRun(D, st0) {
  const P = POL[D.pol];
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate, st0.left_offset_m, st0.left_yaw_deg);
  const base = {...NOMINAL, seed: st0.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
  const sim = createSimulation(world, base);
  sim.reset({seed: st0.seed, scenario: 'baseline'});
  const cfg = JSON.parse(JSON.stringify(sim.peek().config));
  const teacher = RT.createModeTeacher(world, cfg, variation, {mode: P.mode, weights: P.weights, policySha256: P.sha,
    params: D.limit != null ? {speed_limit: D.limit} : {}, overrides: P.mode === 'recover' ? {perturbation: false, shield: false} : {}});
  const F = [];
  let prevS = null, firstCollision = null;
  for (;;) {
    const v = sim.peek(), st = v.state, m = v.metrics;
    if (st.terminal) break;
    const f = st.frame, pr = project(st.x, st.z, st.yaw, prevS); prevS = pr.s;
    const sample = f >= 30 && f <= 590 && (f - 30) % EVERY === 0;
    const out = teacher.action({state: st, metrics: m, config: cfg});
    const lg = teacher.log.at(-1);
    F.push({f, s: lg.s, ey: lg.ey, epsi: lg.epsi, v: st.speed, ps: pr.s, ri: m.route_index, cmd: {steering: out.steering, speed: out.speed},
      prevA: f > 0 ? [teacher.log[f - 1].a0, teacher.log[f - 1].a1] : [0, 0], save: sample ? sim.__save() : null, coll: m.collisions});
    sim.step({steering: out.steering, speed: out.speed}, true);
    if (sim.peek().metrics.collisions > 0 && firstCollision == null) firstCollision = f;
  }
  const end = sim.peek().state;
  return {D, P, st0, world, variation, cfg, sim, F, firstCollision, reason: end.reason, frames: end.frame, geometry: prepareGeometry(world)};
}
// ---- 1 つの枝(乱し + 操縦)。立て直せたら true ----
function branch(R, f, combo, plan, ctlCache) {
  const {sim, F, cfg} = R, rec = F[f];
  sim.__restore(rec.save);
  let v = sim.peek(), st = v.state;
  const coll0 = v.metrics.collisions, s0 = rec.ps;
  let prevS = f > 0 ? F[f - 1].ps : null;
  let ctl = null;
  if (plan.kind === 'policy') {
    ctl = ctlCache[plan.which];
    ctl.start(st, v.metrics.route_index, rec.prevA);
  }
  const vHold0 = Math.max(0, rec.v);
  let peRel = combo.n, aborted = false, vHold = null;
  const eTgt = combo.type === 'excursion' ? rec.ey + combo.sign * combo.d : null;
  let minC = Infinity, near = nearObstacles(OB, st.x, st.z, 2.5), collided = false, lastC = Infinity;
  for (let k = 0; k < peRel + 90; k++) {
    if (st.terminal) break;
    if (k > 0 && ctl) ctl.advance(st, v.metrics.route_index);
    const pr = project(st.x, st.z, st.yaw, prevS); prevS = pr.s;
    const prop = ctl ? ctl.propose(st) : null;
    // わざと外すの打ち切り(走行中と同じ: 始まりより後のフレームで d_min < 0.10)
    if (combo.type === 'excursion' && !aborted && k > 0 && k < peRel && lastC < .10) { aborted = true; peRel = k; }
    let cmd;
    if (k < peRel) {
      if (combo.type === 'steer_pulse') cmd = {steering: combo.sign * .52 * combo.k, speed: ctl ? prop.cmd.speed : plan.during === 'brake' ? 0 : vHold0};
      else if (combo.type === 'excursion') cmd = {steering: excursionSteer(st, ctl ? prop.s : pr.s, eTgt), speed: ctl ? prop.cmd.speed : plan.during === 'brake' ? 0 : vHold0};
      else cmd = {steering: ctl ? prop.cmd.steering : ppEmid(st, pr.s), speed: combo.kind === 'brake' ? 0 : RT.BASE_SPEED_CAP};
    } else if (ctl) cmd = prop.cmd;
    else {
      if (vHold == null) vHold = Math.max(st.speed, 0);
      const kk = k - peRel;
      if (kk < Math.round(plan.t1 * 30)) cmd = {steering: plan.a, speed: 0};
      else cmd = {steering: ppEmid(st, pr.s), speed: plan.vpp === 'hold' ? vHold : plan.vpp};
    }
    if (ctl) ctl.applied(cmd, prop.pc);
    sim.step(cmd, true);
    v = sim.peek(); st = v.state;
    if (v.metrics.collisions > coll0) { collided = true; break; }
    if (k % 10 === 9) near = nearObstacles(OB, st.x, st.z, 2.5);
    lastC = ownClearance(near, st.x, st.z, st.yaw);
    if (lastC < minC) minC = lastC;
    if (minC < .02) break;
  }
  if (collided || minC < .02) return false;
  const pr = project(st.x, st.z, st.yaw, prevS);
  const forward = Math.abs(pr.epsi) < Math.PI / 2 && pr.s > s0;
  const ok = st.reason === 'collection_duration_complete' || (st.speed >= .5 && forward);
  return ok && Math.abs(pr.ey - tableAt('emid', pr.s)) <= .2 && Math.abs(pr.epsi) <= .3;
}

// ---- 本体 ----
const starts = tableStarts();
const pairs = [];
for (let d = 0; d < DRIVERS.length; d++) for (let s = 0; s < starts.length; s++) pairs.push({d, s});
const mine = pairs.filter((_, i) => i % SN === SK);
const fd = OUT ? fs.openSync(OUT, 'w') : null;
const t0 = Date.now();
let nRows = 0, nStates = 0;
// 自己試験 1: 置き換えた core と元の sim-core.js の軌跡が同じ(shard 0 だけ。r1・STD)
if (SK === 0 || args.includes('--selftest')) {
  const st0 = starts[0], {world, variation} = activeWorld(BASE_WORLD, st0.gate, st0.left_offset_m, st0.left_yaw_deg);
  const base = {...NOMINAL, seed: st0.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
  const a = ORIG.createSimulation(world, base), b = createSimulation(world, base);
  let sa = a.reset({seed: st0.seed, scenario: 'baseline'}); b.reset({seed: st0.seed, scenario: 'baseline'});
  const cfg = sa.config, P = POL.r1;
  const t = RT.createModeTeacher(world, cfg, variation, {mode: 'recover', weights: P.weights, overrides: {perturbation: false, shield: false}});
  let maxDiff = 0, n = 0;
  while (!sa.state.terminal) {
    const out = t.action(sa);
    sa = a.step({steering: out.steering, speed: out.speed}); b.step({steering: out.steering, speed: out.speed}, true);
    const q = b.peek().state;
    for (const k of ['x', 'z', 'yaw', 'speed', 'steering', 'sim_time']) maxDiff = Math.max(maxDiff, Math.abs(q[k] - sa.state[k]));
    if (q.terminal !== sa.state.terminal || q.reason !== sa.state.reason) maxDiff = Infinity;
    n++;
  }
  // 保存と戻し: 途中で保存 → 先へ進める → 戻して同じ操作で進めると同じ軌跡
  const c = createSimulation(world, base); let sc = c.reset({seed: st0.seed, scenario: 'baseline'});
  const t2 = RT.createModeTeacher(world, sc.config, variation, {mode: 'recover', weights: P.weights, overrides: {perturbation: false, shield: false}});
  const acts = [], saves = {};
  for (let f = 0; f < 300; f++) { const v = c.peek(); if (f === 150) saves[f] = c.__save(); const o = t2.action({state: v.state, metrics: v.metrics, config: sc.config}); acts.push(o); c.step(o, true); }
  const ref = {...c.peek().state};
  c.__restore(saves[150]);
  for (let f = 150; f < 300; f++) c.step(acts[f], true);
  let rd = 0; for (const k of ['x', 'z', 'yaw', 'speed', 'steering']) rd = Math.max(rd, Math.abs(c.peek().state[k] - ref[k]));
  const msg = {selftest: {core_trace_max_abs_diff: maxDiff, frames: n, save_restore_max_abs_diff: rd, patched_sha256: PATCHED_SHA256, sim_core_sha256: SIM_CORE_SHA256}};
  process.stderr.write(JSON.stringify(msg) + '\n');
  if (!(maxDiff === 0 && rd === 0)) throw new Error('自己試験に失敗しました: ' + JSON.stringify(msg));
  if (fd) fs.writeSync(fd, JSON.stringify(msg) + '\n');
}
for (const {d, s} of mine) {
  const R = referenceRun(DRIVERS[d], starts[s]);
  const ctlCache = {driver: makeController(POL[R.D.pol], R.D.limit, R.world, R.cfg, R.geometry), r1: makeController(POL.r1, null, R.world, R.cfg, R.geometry)};
  // 自己試験 2: 途中から作り直した運転手が、参照の走りと同じ指令を出す(サンプルのフレームの最初の 5 つ)
  let ctlMax = 0;
  const sampleFrames = R.F.filter(q => q.save && (R.firstCollision == null || q.f < R.firstCollision)).map(q => q.f);
  for (const f of sampleFrames.slice(0, 5)) {
    R.sim.__restore(R.F[f].save);
    const ctl = ctlCache.driver; ctl.start(R.sim.peek().state, R.sim.peek().metrics.route_index, R.F[f].prevA);
    const p = ctl.propose(R.sim.peek().state);
    ctlMax = Math.max(ctlMax, Math.abs(p.cmd.steering - R.F[f].cmd.steering), Math.abs(p.cmd.speed - R.F[f].cmd.speed));
  }
  const head = {run: {driver: R.D.name, start: s, start_spec: R.st0, reason: R.reason, frames: R.frames, first_collision_frame: R.firstCollision,
    sample_states: sampleFrames.length, controller_restart_max_abs_diff: ctlMax}};
  if (fd) fs.writeSync(fd, JSON.stringify(head) + '\n');
  process.stderr.write(JSON.stringify(head) + '\n');
  const plansFor = combo => {
    const pol = [{kind: 'policy', which: 'driver'}];
    if (!(R.D.pol === 'r1' && R.D.limit == null)) pol.push({kind: 'policy', which: 'r1'});
    const a = combo.type === 'speed_pulse' ? PLANS_A.filter(p => p.during !== 'brake') : PLANS_A;
    return [...pol, ...a];
  };
  let done = 0;
  for (const f of sampleFrames) {
    if (done++ >= LIMIT) break;
    const rec = R.F[f], sm = ((rec.s % LOOP) + LOOP) % LOOP;
    const azR = perturbAllowed(rec.s, RECORD_SPEC, LOOP), azT = perturbAllowed(rec.s, TRAIN_SPEC, LOOP);
    nStates++;
    for (let ci = 0; ci < COMBOS.length; ci++) {
      const combo = COMBOS[ci];
      let ok = 0, planIdx = -1;
      if (combo.type === 'excursion' && room(combo.sign, rec.s, rec.ey) < combo.d + .05) ok = null;
      else {
        const plans = plansFor(combo);
        for (let pi = 0; pi < plans.length; pi++) if (branch(R, f, combo, plans[pi], ctlCache)) { ok = 1; planIdx = pi; break; }
      }
      const row = {d, st: s, f, s: +sm.toFixed(4), v: +rec.v.toFixed(4), ey: +rec.ey.toFixed(4), c: ci, ok, p: planIdx, ar: azR ? 1 : 0, at: azT ? 1 : 0};
      if (fd) fs.writeSync(fd, JSON.stringify(row) + '\n');
      nRows++;
    }
  }
  process.stderr.write(`[${new Date().toISOString()}] shard ${SK}/${SN} driver ${R.D.name} 開始 ${s} 終わり: 状態 ${nStates}・行 ${nRows}・${((Date.now() - t0) / 1000).toFixed(0)} 秒\n`);
}
if (fd) { fs.writeSync(fd, JSON.stringify({done: true, rows: nRows, states: nStates, wall_s: (Date.now() - t0) / 1000}) + '\n'); fs.closeSync(fd); }
