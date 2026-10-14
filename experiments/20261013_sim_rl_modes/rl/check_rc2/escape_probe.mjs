// 離脱復活の確認 r2: 「乱しのあとの衝突は、どう操縦しても避けられないか」を、学習の段より広い操縦の組で崩しにかかる(2026-10-14、確認の段)。
//   node rl/check_rc2/escape_probe.mjs <rows.jsonl を含むフォルダ> <set> <出力.json> [最大件数]
// 学習の段の probe_exit_escape.mjs(丸めた記録の再生・高速版・ps から 1.5 秒・一定の指令 20 通り)と違うところ:
//   - 元の sim-core.js(1 行足して state を返すだけの写し)で、方策 + 乱しを ps まで**そのまま走らせ直し**、ps の state を正確に写す(再生の誤差 0)
//   - 乱しの間: 舵は乱しの値のまま(差し込み口が決める)、速さ {方策のまま, 0}
//   - 乱しのあと: 「舵 a × 速さ {0, そのまま} を t1 秒」→ そのあと自前の pure pursuit(R の前方 0.6 m、速さ 2.0 m/s)で pe から 3 秒まで。
//     t1 ∈ {0, 0.2, 0.4, 0.7, 1.0} 秒、a ∈ {±0.52, ±0.26, 0}。t1 = 0 は pe から pure pursuit(速さ {2.0, 1.5, そのまま})
//   - 逃げ道 = ps から pe + 3 秒まで衝突なし・最後に 0.5 m/s 以上で前へ進んでいる(止まり・逆走でない)。
//     さらに「戻った」= 最後に R からの横ずれ ≤ 0.15 m・向きのずれ ≤ 0.3 rad(protocol の nominal line ではなく R に対する粗い目安)
import fs from 'node:fs';
import path from 'node:path';
import {BASE_WORLD, activeWorld, project, NOMINAL, LOOP} from '../check_rc/run_rc.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const CORE_PATH = `${REL}/dist/sim-core.js`;
const src = fs.readFileSync(CORE_PATH, 'utf8');
const RET = '  return { reset, step, observe, snapshot, report, demoAction, setPaused };';
if (src.split(RET).length !== 2) throw new Error('置き換えの場所が 1 つではありません');
const core = await import(`data:text/javascript;base64,${Buffer.from(src.replace(RET, '  return { reset, step, observe, snapshot, report, demoAction, setPaused, __state: () => state };')).toString('base64')}`);
const RT = await import(`${EXP}/web/rl/mode_runtime.mjs`);
const REF = JSON.parse(fs.readFileSync(`${EXP}/web/rl/reference_loop.json`, 'utf8'));
const weights = JSON.parse(fs.readFileSync(`${EXP}/bundle/recover/policy.json`, 'utf8'));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

const [dir, setName, outPath, maxN = '9999'] = process.argv.slice(2);
const rows = [];
for (const fn of fs.readdirSync(dir).filter(f => /^rows_\d+\.jsonl$/.test(f)).sort())
  for (const line of fs.readFileSync(path.join(dir, fn), 'utf8').split('\n')) { if (line.trim()) { const r = JSON.parse(line); if (r.set === setName && !r.invalid_spawn) rows.push(r); } }
rows.sort((a, b) => +a.id.slice(1) - +b.id.slice(1));

// 対象: 衝突した回で、衝突の直前の出来事が舵のパルス(衝突が次の出来事の始まりより前)
const targets = [];
for (const r of rows) {
  if (!r.collisions) continue;
  const evs = r.events.filter(e => e.ps < r.frames);
  const e = evs.at(-1);
  if (!e || e.type !== 'steer_pulse') continue;
  targets.push({r, e});
}

function refPoint(sAbs) {
  const s = ((sAbs % LOOP) + LOOP) % LOOP;
  let lo = 0, hi = REF.n - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (REF.s[m] <= s) lo = m; else hi = m - 1; }
  const j = (lo + 1) % REF.n, segL = (lo < REF.n - 1 ? REF.s[lo + 1] : LOOP) - REF.s[lo], u = segL > 0 ? (s - REF.s[lo]) / segL : 0;
  return [REF.x[lo] + (REF.x[j] - REF.x[lo]) * u, REF.z[lo] + (REF.z[j] - REF.z[lo]) * u];
}
function purePursuit(st, sNow, Ld = .6) {
  const [tx, tz] = refPoint(sNow + Ld), hx = Math.cos(st.yaw), hy = Math.sin(st.yaw), vx = tx - st.x, vy = -(tz - st.z);
  const alpha = Math.atan2(hx * vy - hy * vx, hx * vx + hy * vy), d = Math.hypot(vx, vy);
  return Math.max(-.52, Math.min(.52, Math.atan2(2 * NOMINAL.wheelbase * Math.sin(alpha), Math.max(d, .2))));
}

function setup(r) {
  const st0 = r.spec.start, seed = (st0.seed ?? 1) >>> 0;
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const base = {...NOMINAL, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
  return {world, variation, base, seed};
}
// 方策 + 乱しで ps まで走らせ、ps の state と、乱しの間の(実際の)指令を返す
function runToPs(r, e) {
  const {world, variation, base, seed} = setup(r);
  const sim = core.createSimulation(world, base);
  let snap = sim.reset({seed, scenario: 'baseline'});
  const cfg = snap.config;
  const t = RT.createModeTeacher(world, cfg, variation, {mode: 'recover', weights, policySha256: 'x', params: r.spec.classes ? {perturb_class: r.spec.classes} : {}});
  let S = null, sAtPs = null, prevS = null;
  const inj = [];
  while (!snap.state.terminal) {
    const st = snap.state;
    const pr = project(st.x, st.z, st.yaw, prevS); prevS = pr.s;
    const out = t.action({state: st, metrics: snap.metrics, config: cfg});
    if (st.frame === e.ps) { S = {...sim.__state()}; sAtPs = pr.s; }
    if (st.frame >= e.ps) inj.push({steering: out.steering, speed: out.speed, l: out.info.jf.l});
    snap = sim.step({steering: out.steering, speed: out.speed});
  }
  return {S, sAtPs, inj, cfg};
}
// ps の state から plan で進める。戻り値 {collided, ok, recovered, end}
function branch(r, S, sAtPs, inj, plan, H) {
  const {world, base, seed} = setup(r);
  const sim = core.createSimulation(world, base);
  sim.reset({seed, scenario: 'baseline'});
  Object.assign(sim.__state(), {x: S.x, z: S.z, yaw: S.yaw, speed: S.speed, steering: S.steering, sim_time: S.sim_time, frame: S.frame, surface: S.surface,
    terminal: false, reason: null});
  let prevS = sAtPs, snap = sim.snapshot(), last = null;
  for (let k = 0; k < H; k++) {
    const st = snap.state;
    if (st.terminal) break;
    const pr = project(st.x, st.z, st.yaw, prevS); prevS = pr.s;
    last = {pr, v: st.speed};
    snap = sim.step(plan(k, st, pr, inj));
  }
  const st = snap.state, collided = (snap.metrics.collisions ?? 0) > 0;
  const pr = project(st.x, st.z, st.yaw, prevS);
  const forward = Math.abs(pr.epsi) < Math.PI / 2 && pr.s > sAtPs;
  const ok = !collided && (st.reason === 'collection_duration_complete' || (st.speed >= .5 && forward));
  return {collided, ok, recovered: ok && Math.abs(pr.ey) <= .15 && Math.abs(pr.epsi) <= .3, ey: pr.ey, epsi: pr.epsi, v: st.speed, reason: st.reason};
}

const res = [];
const T1 = [.2, .4, .7, 1.0], A = [.52, .26, 0, -.26, -.52];
for (const {r, e} of targets.slice(0, +maxN)) {
  const {S, sAtPs, inj} = runToPs(r, e);
  if (!S) { res.push({id: r.id, skipped: 'ps に届かない'}); continue; }
  const n = e.pe - e.ps, H = n + 90;
  const plans = [];
  for (const brake of [false, true]) {
    for (const vpp of ['hold', 2.0, 1.5]) plans.push({brake, t1: 0, a: null, v1: null, vpp});
    for (const t1 of T1) for (const a of A) for (const v1 of ['brake', 'hold']) plans.push({brake, t1, a, v1, vpp: 2.0});
  }
  // 確かめ: ps の state を写した sim に、元の走りの指令(方策 + 乱し)をそのまま与えると、元と同じフレームで衝突すること
  const rep = branch(r, S, sAtPs, inj, (k, st) => inj[k] ? {steering: inj[k].steering, speed: inj[k].speed} : {steering: 0, speed: 0}, inj.length + 1);
  const replayOk = rep.collided;
  let best = null, nOk = 0, nRec = 0;
  for (const P of plans) {
    let vHold = null;
    const plan = (k, st, pr, inj) => {
      if (k < n) return {steering: inj[k].steering, speed: P.brake ? 0 : inj[k].speed};
      if (vHold == null) vHold = st.speed;
      const kk = k - n;
      if (kk < Math.round(P.t1 * 30)) return {steering: P.a, speed: P.v1 === 'brake' ? 0 : vHold};
      return {steering: purePursuit(st, pr.s), speed: P.vpp === 'hold' ? vHold : P.vpp};
    };
    const o = branch(r, S, sAtPs, inj, plan, H);
    if (o.ok) nOk++;
    if (o.recovered) nRec++;
    if (o.ok && (!best || (o.recovered && !best.recovered))) best = {...P, ...o};
  }
  const pulseSteer = inj[0]?.steering ?? null;
  res.push({id: r.id, seed: r.spec.start.seed, ev_n: e.n, cls: e.cls, sign: e.sign, k: e.k, frames: e.frames, ps: e.ps, pe: e.pe,
    s_mod: +(((sAtPs % LOOP) + LOOP) % LOOP).toFixed(3), v0: +S.speed.toFixed(3), pulse_steer: pulseSteer,
    collision_frame: r.frames, replay_collided: replayOk, plans: plans.length, plans_ok: nOk, plans_recovered: nRec, escape: nOk > 0, escape_recovered: nRec > 0, best});
  process.stderr.write(`${res.length}/${Math.min(targets.length, +maxN)}\r`);
}
const R = res.filter(x => !x.skipped);
const exit = R.filter(x => x.s_mod >= 8 && x.s_mod < 8.25), other = R.filter(x => !(x.s_mod >= 8 && x.s_mod < 8.25));
const sum = g => ({n: g.length, escape: g.filter(x => x.escape).length, escape_recovered: g.filter(x => x.escape_recovered).length,
  v0_median: g.map(x => x.v0).sort((a, b) => a - b)[Math.floor(g.length / 2)] ?? null});
const summary = {set: setName, targets: R.length, replay_collided: R.filter(x => x.replay_collided).length, all: sum(R), black_sheet_exit: sum(exit), exit_right: sum(exit.filter(x => x.sign === -1)), other: sum(other)};
fs.writeFileSync(outPath, JSON.stringify({schema: 'jf-sim-rl-check-recover-escape/v1', created_at: new Date().toISOString(), rule: '冒頭の説明', summary, events: res}, null, 1));
console.log(JSON.stringify(summary, null, 1));
