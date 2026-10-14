// r3 の下調べ(離脱復活): 「立て直せる乱し」の見分け(oracle)の試作(2026-10-14、r3 の計画の段)。
//   node r3/rc/oracle_proto.mjs <出力.jsonl> [開始の数 3 か =0,2 のような番号の並び] [フレームの間隔 2]
// やること:
//   1. 離脱復活の方策(bundle/recover/policy.json、7720f275…)を乱しなしで、元の sim-core.js(state を返す 1 行だけ足した写し)で走らせ、
//      各フレームの state を写す(確認 r2 の escape_probe.mjs と同じやり方)。
//   2. 間隔ごとのフレーム f を ps として、乱し d(舵のパルス 左右 × 弱・中・強、ブレーキ・踏み込み × 弱・中・強。強さは各クラスの一番強い値 = 悪い側)を
//      差し込み、そのあと「操縦の組」(乱しの間の速さ {そのまま, 0} × 乱しのあと {R の pure pursuit 速さ {そのまま, 2.0, 1.5} / 舵 a を t1 秒 → pure pursuit})
//      のどれか 1 つで「立て直せる」かを調べる。立て直せる = ps から pe + 3 秒まで衝突なし・フレームの終わりのすき間 ≥ 2 cm・
//      最後に前へ 0.5 m/s 以上・R からの横ずれ ≤ 0.15 m・向きのずれ ≤ 0.3 rad。
//   3. 1 行 1 つの (開始, f, 乱し) を書く。今の記録の決まりで「始めてよい位置」か(perturbAllowed)も書く。
// 試作の版 2: pure pursuit は R ではなく最大すき間の線(room_table の emid。R は s 6.5〜7.2 m で壁から 1.2 cm)を追う。立て直し = 最後に |e_y − emid| ≤ 0.2 m。
// 方策は乱しのあと使わない(方策によらない「物理として立て直せるか」の下限の見積もり。操縦の組が弱いので「立て直せない」は目安)。
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, project, NOMINAL, LOOP} from '../../rl/check_rc/run_rc.mjs';
import {buildObstacles, ownClearance, nearObstacles} from '../../rl/check_nw/own_geom.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const CORE_PATH = `${REL}/dist/sim-core.js`;
const src = fs.readFileSync(CORE_PATH, 'utf8');
const RET = '  return { reset, step, observe, snapshot, report, demoAction, setPaused };';
if (src.split(RET).length !== 2) throw new Error('置き換えの場所が 1 つではありません');
const core = await import(`data:text/javascript;base64,${Buffer.from(src.replace(RET, '  return { reset, step, observe, snapshot, report, demoAction, setPaused, __state: () => state };')).toString('base64')}`);
const RT = await import(`${EXP}/web/rl/mode_runtime.mjs`);
const {perturbAllowed, RECORD_SPEC} = await import(`${EXP}/web/rl/perturb.mjs`);
const REF = JSON.parse(fs.readFileSync(`${EXP}/web/rl/reference_loop.json`, 'utf8'));
const {tableAt} = await import(`${EXP}/web/rl/line_profiles.mjs`);
const {loopPointAt} = await import(`${EXP}/web/rl/features.mjs`);
const weights = JSON.parse(fs.readFileSync(`${EXP}/bundle/recover/policy.json`, 'utf8'));
const CLS = RECORD_SPEC.classes;
const [outPath, nStartsArg = '3', strideArg = '2'] = process.argv.slice(2);
const OB = buildObstacles(BASE_WORLD);

function refPoint(sAbs) {
  const s = ((sAbs % LOOP) + LOOP) % LOOP;
  let lo = 0, hi = REF.n - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (REF.s[m] <= s) lo = m; else hi = m - 1; }
  const j = (lo + 1) % REF.n, segL = (lo < REF.n - 1 ? REF.s[lo + 1] : LOOP) - REF.s[lo], u = segL > 0 ? (s - REF.s[lo]) / segL : 0;
  return [REF.x[lo] + (REF.x[j] - REF.x[lo]) * u, REF.z[lo] + (REF.z[j] - REF.z[lo]) * u];
}
const PT = {x: 0, z: 0, tx: 0, tz: 0};
function purePursuit(st, sNow, Ld = .6) {
  loopPointAt(sNow + Ld, PT); const e = tableAt('emid', sNow + Ld), tx = PT.x + e * PT.tz, tz = PT.z - e * PT.tx;
  const hx = Math.cos(st.yaw), hy = Math.sin(st.yaw), vx = tx - st.x, vy = -(tz - st.z);
  const alpha = Math.atan2(hx * vy - hy * vx, hx * vx + hy * vy), d = Math.hypot(vx, vy);
  return Math.max(-.52, Math.min(.52, Math.atan2(2 * NOMINAL.wheelbase * Math.sin(alpha), Math.max(d, .2))));
}
// 開始: STD と、記録の開始の分布から決まった 2 つ(種 3,190,000〜。試験・検証・記録・確認の種と重ならない)
const STARTS = [{gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 3190000}, {gate: 11, left_offset_m: .1, left_yaw_deg: -8, seed: 3190001},
  {gate: 22, left_offset_m: -.12, left_yaw_deg: 6, seed: 3190002}, {gate: 5, left_offset_m: 0.05, left_yaw_deg: 10, seed: 3190003}]
  .map((st, i) => ({...st, idx: i})).filter((st, i) => nStartsArg.includes(',') || nStartsArg.startsWith('=') ? nStartsArg.replace('=', '').split(',').map(Number).includes(i) : i < +nStartsArg);

function nominalRun(st0) {
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate, st0.left_offset_m, st0.left_yaw_deg);
  const base = {...NOMINAL, seed: st0.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
  const sim = core.createSimulation(world, base);
  let snap = sim.reset({seed: st0.seed, scenario: 'baseline'});
  const cfg = snap.config;
  const t = RT.createModeTeacher(world, cfg, variation, {mode: 'recover', weights, policySha256: 'x', overrides: {perturbation: false}});
  const states = [];
  let prevS = null;
  while (!snap.state.terminal) {
    const st = snap.state, pr = project(st.x, st.z, st.yaw, prevS); prevS = pr.s;
    const out = t.action({state: st, metrics: snap.metrics, config: cfg});
    states.push({S: {...sim.__state()}, s: pr.s, ey: pr.ey, epsi: pr.epsi, v: st.speed, cmd: {steering: out.steering, speed: out.speed}});
    snap = sim.step({steering: out.steering, speed: out.speed});
  }
  return {world, base, seed: st0.seed, states};
}
function branch(ctx, S, s0, plan, H) {
  const sim = core.createSimulation(ctx.world, ctx.base);
  sim.reset({seed: ctx.seed, scenario: 'baseline'});
  Object.assign(sim.__state(), {x: S.x, z: S.z, yaw: S.yaw, speed: S.speed, steering: S.steering, sim_time: S.sim_time, frame: S.frame, surface: S.surface, terminal: false, reason: null});
  let prevS = s0, snap = sim.snapshot(), minC = Infinity, near = null;
  for (let k = 0; k < H; k++) {
    const st = snap.state;
    if (st.terminal) break;
    const pr = project(st.x, st.z, st.yaw, prevS); prevS = pr.s;
    snap = sim.step(plan(k, st, pr));
    if (k % 15 === 0) near = nearObstacles(OB, snap.state.x, snap.state.z, 2.5);
    if (k % 2 === 1) { const c = ownClearance(near, snap.state.x, snap.state.z, snap.state.yaw); if (c < minC) minC = c; }
    if (minC < .02) break;
  }
  const st = snap.state, collided = (snap.metrics.collisions ?? 0) > 0;
  const pr = project(st.x, st.z, st.yaw, prevS);
  const forward = Math.abs(pr.epsi) < Math.PI / 2 && pr.s > s0;
  const ok = !collided && minC >= .02 && (st.reason === 'collection_duration_complete' || (st.speed >= .5 && forward));
  return {ok, recovered: ok && Math.abs(pr.ey - tableAt('emid', pr.s)) <= .2 && Math.abs(pr.epsi) <= .3, collided, minC};
}
// 乱し: 各クラスの悪い側(フレーム数・k の上限)
const DIST = [];
for (const c of ['弱', '中', '強']) {
  for (const sign of [1, -1]) DIST.push({type: 'steer_pulse', cls: c, sign, n: CLS[c].steer_pulse.frames[1], k: CLS[c].steer_pulse.k[1]});
  DIST.push({type: 'speed_pulse', cls: c, kind: 'brake', n: CLS[c].speed_pulse.brake_frames[1]});
  DIST.push({type: 'speed_pulse', cls: c, kind: 'burst', n: CLS[c].speed_pulse.burst_frames[1]});
}
// 操縦の組(24 通り): 乱しの間の速さ {そのまま, 0} × 乱しのあと {最大すき間の線の pure pursuit 速さ {そのまま, 2.0, 1.5, 1.0} / 舵 a を t1 秒ブレーキ → pure pursuit 2.0}
const T1 = [.2, .4], A = [.52, .26, -.26, -.52];
const PLANS = [];
for (const during of ['hold', 'brake']) {
  for (const vpp of ['hold', 2.0, 1.5, 1.0]) PLANS.push({during, t1: 0, a: null, v1: null, vpp});
  for (const t1 of T1) for (const a of A) PLANS.push({during, t1, a, v1: 'brake', vpp: 2.0});
}
function oracle(ctx, rec, d) {
  const n = d.n, H = n + 90, S = rec.S, vHold0 = Math.max(0, rec.v);
  let nOk = 0, first = null;
  for (let i = 0; i < PLANS.length; i++) {
    const P = PLANS[i];
    // 舵のパルス: 乱しの間は舵が固定、速さは操縦の組。ブレーキ/踏み込み: 速さが固定、舵は R の pure pursuit
    if (d.type === 'speed_pulse' && P.during === 'brake') continue;
    let vHold = null;
    const plan = (k, st, pr) => {
      if (k < n) {
        if (d.type === 'steer_pulse') return {steering: d.sign * .52 * d.k, speed: P.during === 'brake' ? 0 : vHold0};
        return {steering: purePursuit(st, pr.s), speed: d.kind === 'brake' ? 0 : 4.5};
      }
      if (vHold == null) vHold = Math.max(st.speed, 0);
      const kk = k - n;
      if (kk < Math.round(P.t1 * 30)) return {steering: P.a, speed: P.v1 === 'brake' ? 0 : vHold};
      return {steering: purePursuit(st, pr.s), speed: P.vpp === 'hold' ? vHold : P.vpp};
    };
    const o = branch(ctx, S, rec.s, plan, H);
    if (o.recovered) { nOk++; first = i; break; }
  }
  return {recoverable: nOk > 0, plan_index: first};
}
const fd = fs.openSync(outPath, 'w');
const t0 = Date.now();
for (const st0 of STARTS) {
  const si = st0.idx;
  const ctx = nominalRun(st0);
  for (let f = 30; f < ctx.states.length - 5; f += +strideArg) {
    const rec = ctx.states[f], sm = ((rec.s % LOOP) + LOOP) % LOOP;
    const allowed = perturbAllowed(rec.s, RECORD_SPEC, LOOP);
    for (const d of DIST) {
      const o = oracle(ctx, rec, d);
      fs.writeSync(fd, JSON.stringify({start: si, f, s: +sm.toFixed(3), v: +rec.v.toFixed(3), ey: +rec.ey.toFixed(3), allowed_v1: allowed,
        type: d.type, cls: d.cls, sign: d.sign ?? null, kind: d.kind ?? null, ...o}) + '\n');
    }
  }
  process.stderr.write(`開始 ${si} 終わり ${((Date.now() - t0) / 1000).toFixed(0)} 秒\n`);
}
fs.closeSync(fd);
