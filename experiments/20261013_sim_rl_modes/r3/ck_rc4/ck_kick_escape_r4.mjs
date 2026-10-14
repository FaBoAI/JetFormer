// 離脱復活 r4 の確認(2026-10-15、確認 r3 の ck_kick_escape.mjs の写し。方策を r4 に、キックのフレームは行の kick.frame(位置で狙ったキックも)):
// 落ちたキックは「キックのあと、どう操縦しても避けられない」のか(学習の段の「物理として避けにくい」を崩しにかかる。2026-10-14 21:00)
//   node ck_kick_escape.mjs <rows のフォルダ> <組,組,...> <出力.json>
// 1. 候補の方策 + 走行中の仕組み(シールドあり)で、キックのフレーム kf まで元の sim-core.js(state を返す 1 行だけ足した写し)を走らせ、
//    各フレームの指令をそのまま残す(kf の指令はキックの前に決まる = protocol・学習の段と同じ順)
// 2. 操縦ごとに新しい sim で、残した指令を kf まで再生 → キック → kf の指令で 1 ステップ(ここまで元の回と同じ。同じフレームでぶつかれば「同じステップ」)
//    → kf+1 から操縦: 舵 a ∈ {±0.52, ±0.26, 0} × 速さの指令 {0, 1.0, 1.5, キックの瞬間の速さ} を t1 ∈ {0.1, 0.2, 0.4, 0.7, 1.0} 秒、
//    そのあと R の前方 0.6 m の pure pursuit(速さ 2.0)。t1 = 0 は kf+1 から pure pursuit(速さ {1.0, 1.5, 2.0, キックの瞬間の速さ})
// 3. 逃げ道 = kf+1 から 3 秒(90 フレーム)衝突なし・最後に前へ 0.3 m/s 以上
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {BASE_WORLD, activeWorld, project, NOMINAL, LOOP, CAND, POLICY_PATHS} from './ck_run.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const CORE_PATH = `${REL}/dist/sim-core.js`;
const src = fs.readFileSync(CORE_PATH, 'utf8');
const RET = '  return { reset, step, observe, snapshot, report, demoAction, setPaused };';
if (src.split(RET).length !== 2) throw new Error('置き換えの場所が 1 つではありません');
const core = await import(`data:text/javascript;base64,${Buffer.from(src.replace(RET, '  return { reset, step, observe, snapshot, report, demoAction, setPaused, __state: () => state };')).toString('base64')}`);
const RT = await import(`${CAND}/runtime/mode_runtime.mjs`);
const REF = JSON.parse(fs.readFileSync(`${CAND}/runtime/reference_loop.json`, 'utf8'));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

const [dir, sets, outPath] = process.argv.slice(2);
const want = new Set(sets.split(','));
const rows = [];
for (const fn of fs.readdirSync(dir).filter(f => /^rows_.*\.jsonl$/.test(f)).sort())
  for (const line of fs.readFileSync(path.join(dir, fn), 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const r = JSON.parse(line);
    if (want.has(r.set) && r.kick && r.kick.allowed_rt && r.collisions > 0
      && !(process.env.CK_SKIP_SAME_STEP === '1' && r.frames <= r.kick.frame + 1)) rows.push(r);   // 同じステップの衝突は飛ばせる(操縦の前に決まる)
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
const weights = JSON.parse(fs.readFileSync(POLICY_PATHS.r4, 'utf8')), wsha = sha(POLICY_PATHS.r4);

function newSim(r) {
  const st0 = r.spec.start, seed = (st0.seed ?? 1) >>> 0;
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const sim = core.createSimulation(world, {...NOMINAL, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  const snap = sim.reset({seed, scenario: 'baseline'});
  return {sim, snap, world, variation};
}
function applyKick(sim, kick) {
  const s = sim.__state();
  if (kick.yaw) s.yaw = wrap(s.yaw + kick.yaw);
  if (kick.lateral) { s.x += kick.lateral * -Math.sin(s.yaw); s.z += kick.lateral * -Math.cos(s.yaw); }
}
// 1. 方策で kf まで(指令を残す)
function recordPrefix(r) {
  const {sim, world, variation} = newSim(r);
  let snap = sim.reset({seed: (r.spec.start.seed ?? 1) >>> 0, scenario: 'baseline'});
  const teacher = RT.createModeTeacher(world, snap.config, variation, {mode: 'recover', weights, policySha256: wsha, params: r.spec.params ?? {}, overrides: {perturbation: false}});
  const kf = r.kick.frame, A = [];
  while (!snap.state.terminal && snap.state.frame <= kf) {
    const out = teacher.action({state: snap.state, metrics: snap.metrics, config: snap.config});
    A.push({steering: out.steering, speed: out.speed});
    if (snap.state.frame === kf) applyKick(sim, r.spec.kick);
    snap = sim.step({steering: out.steering, speed: out.speed});
  }
  // 方策で続けた元の回(確かめ: 行と同じフレームでぶつかるか)
  let collF = snap.state.reason === 'collision' ? snap.state.frame : null;
  while (!snap.state.terminal && snap.state.frame < kf + 1 + 90) {
    const out = teacher.action({state: snap.state, metrics: snap.metrics, config: snap.config});
    snap = sim.step({steering: out.steering, speed: out.speed});
    if (snap.state.reason === 'collision') collF = snap.state.frame;
  }
  return {A, kf, policyCollF: collF};
}
function tryManeuver(r, A, kf, m) {
  const {sim} = newSim(r);
  let snap = sim.reset({seed: (r.spec.start.seed ?? 1) >>> 0, scenario: 'baseline'});
  for (let f = 0; f <= kf; f++) {
    if (snap.state.terminal) return {ok: false, why: 'pre_terminal'};
    if (f === kf) applyKick(sim, r.spec.kick);
    snap = sim.step(A[f]);
  }
  if (snap.state.terminal) return {ok: false, why: snap.state.reason === 'collision' ? 'same_step' : snap.state.reason};
  const vKick = snap.state.speed;
  let prevS = project(snap.state.x, snap.state.z, snap.state.yaw).s;
  for (let i = 0; i < 90; i++) {
    const st = snap.state, pr = project(st.x, st.z, st.yaw, prevS); prevS = pr.s;
    const t = i / 30;
    let cmd;
    if (t < m.t1) cmd = {steering: m.a, speed: m.v === 'keep' ? vKick : m.v};
    else cmd = {steering: purePursuit(st, pr.s), speed: m.vpp === 'keep' ? Math.max(1.0, vKick) : m.vpp};
    snap = sim.step(cmd);
    if (snap.state.terminal) return {ok: snap.state.reason !== 'collision' && snap.state.reason === 'collection_duration_complete', why: snap.state.reason, at: i};
  }
  const st = snap.state, pr = project(st.x, st.z, st.yaw, prevS);
  return {ok: st.speed >= .3, why: st.speed >= .3 ? 'escaped' : 'stopped', final_v: st.speed, final_ey: pr.ey, final_epsi: pr.epsi};
}
const MAN = [];
for (const a of [.52, .26, 0, -.26, -.52]) for (const v of [0, 1.0, 1.5, 'keep']) for (const t1 of [.1, .2, .4, .7, 1.0]) MAN.push({a, v, t1, vpp: 2.0});
for (const vpp of [1.0, 1.5, 2.0, 'keep']) MAN.push({a: null, v: null, t1: 0, vpp});

const out = {created_at: new Date().toISOString(), maneuvers: MAN.length, cases: []};
for (const r of rows) {
  const {A, kf, policyCollF} = recordPrefix(r);
  const res = MAN.map(m => ({m, ...tryManeuver(r, A, kf, m)}));
  const esc = res.filter(x => x.ok);
  out.cases.push({set: r.set, id: r.id, kind: r.spec.kind, kick: r.spec.kick, row_frames: r.frames, row_reason: r.reason, replay_policy_collision_frame: policyCollF,
    same_step: res.every(x => x.why === 'same_step'), escapes: esc.length,
    why_counts: res.reduce((o, x) => { o[x.why] = (o[x.why] ?? 0) + 1; return o; }, {}),
    collision_after_frames_max: Math.max(...res.filter(x => x.why === 'collision').map(x => x.at ?? -1), -1), escape_examples: esc.slice(0, 4).map(x => ({...x.m, final_v: x.final_v, final_ey: x.final_ey})),
    kick_s: r.kick.rt_s % LOOP, kick_v: r.kick.v, clr_after_kick: r.kick.clr_after});
  console.log(r.set, r.id, 'escapes', esc.length, '/', MAN.length, 'same_step', out.cases.at(-1).same_step, 'replay_coll', policyCollF, 'row_frames', r.frames);
}
out.summary = {cases: out.cases.length, same_step: out.cases.filter(c => c.same_step).length,
  escapable: out.cases.filter(c => !c.same_step && c.escapes > 0).length, no_escape_found: out.cases.filter(c => !c.same_step && c.escapes === 0).length};
fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.summary));
