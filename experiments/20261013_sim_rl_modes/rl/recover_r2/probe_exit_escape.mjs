// 黒シートの出口(s 8.00〜8.25 m)の右へのパルスは、どんな操縦でも避けられないか(2026-10-14、学習の段 recover r2)。
//
// 確認 recover r1 の走り(check_recover_r1/runs、離脱復活の方策、試験の組の作り直し + 新しい開始)のうち、黒シートの出口で始まった
// 右への舵のパルスの各出来事について:
//   1) 記録された操作(適用した指令)を高速版(元の sim-core.js とビット単位で同じ。build.md)で ps の前まで再生し、記録の位置と一致を確かめる
//   2) ps から「乱しの間の速さの指令 {記録のまま, 0 = ブレーキ}」×「乱しのあと(pe から)の一定の指令: 舵 {+0.52, +0.39, +0.26, +0.13, 0}
//      × 速さ {0, そのときの速さ}」の 20 通りを走らせ、ps から 1.5 秒のあいだ衝突しない組があるか(= 逃げ道があるか)を調べる
//      (乱しの舵 −0.52·k は n フレームそのまま。逃げ道があっても戻れるとは限らないので、これは「成功」の上限)
//   3) ps の瞬間に速さを c 倍にした状態(高速版の state の書き換え)で同じ 20 通りを試し、逃げ道ができる c を探す
//      (c·v = 「その速さで出口に来ていれば避けられた」目安)
//   node rl/recover_r2/probe_exit_escape.mjs <runs のフォルダ> <出力.json>
import fs from 'node:fs';
import path from 'node:path';
import {activeWorldFor, simConfig} from '../../snapshot/rl/env_core.mjs';
import {BASE_WORLD, loadCore} from '../episode.mjs';
import {LOOP_LENGTH} from '../runtime/features.mjs';

//   第 3 引数 'other': 黒シートの出口の外で始まり、そのあと衝突した舵のパルス(左右とも)を同じ方法で調べる(一定の指令の 20 通りは
//   曲がりの多い所では弱い探し方なので、「逃げ道が見つからない」は目安。見つかったものは確かに逃げられる)
const [runsDir, outPath, which = 'exit-right'] = process.argv.slice(2);
const core = await loadCore('fast');
const rows = [];
for (const fn of fs.readdirSync(runsDir).filter(f => /^rows_\d+\.jsonl$/.test(f)).sort()) {
  for (const line of fs.readFileSync(path.join(runsDir, fn), 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const r = JSON.parse(line);
    if (r.set === 'REPLAY-P100' || r.set === 'FRESH-P') rows.push(r);
  }
}
const targets = [];
for (const r of rows) r.events.forEach((e, k) => {
  const sm = ((e.s % LOOP_LENGTH) + LOOP_LENGTH) % LOOP_LENGTH;
  const atExit = sm >= 8.0 && sm < 8.25;
  const nxt = r.events[k + 1]?.ps ?? r.frames, collided = r.collisions > 0 && r.frames <= nxt;
  if (which === 'exit-right' ? (e.type === 'steer_pulse' && e.sign === -1 && atExit) : (e.type === 'steer_pulse' && !atExit && collided)) targets.push({r, e, k});
});

function makeSim(r) {
  const st = r.spec.start;
  const {world} = activeWorldFor(BASE_WORLD, st.gate ?? 0, st.left_offset_m ?? 0, st.left_yaw_deg ?? 0);
  const sim = core.createSimulation(world, simConfig({episodeMode: 'continuous', timeLimit: 20, seed: (st.seed ?? 1) >>> 0}));
  sim.reset({seed: (st.seed ?? 1) >>> 0, scenario: 'baseline'});
  return sim;
}
// ps まで記録の操作で進め、ps からは plan(f, state) の指令で H フレーム。戻り値 {collided, frame}
function rollout(r, ps, plan, H, speedScale = 1) {
  const sim = makeSim(r), pf = r.pf;
  for (let f = 0; f < ps; f++) sim.step({steering: pf.st[f], speed: pf.sp[f]}, true);
  let q = sim.peek();
  const pos = {dx: Math.abs(q.state.x - pf.x[ps]), dz: Math.abs(q.state.z - pf.z[ps])};
  if (speedScale !== 1) q.state.speed *= speedScale;
  for (let f = ps; f < ps + H; f++) {
    q = sim.peek();
    if (q.state.terminal) break;
    sim.step(plan(f, q.state), true);
  }
  q = sim.peek();
  return {collided: (q.metrics.collisions ?? 0) > 0, end_frame: q.state.frame, reason: q.state.reason, pos};
}

const STEERS = which === 'exit-right' ? [.52, .39, .26, .13, 0] : [.52, .26, 0, -.26, -.52];   // 出口の外は左右どちらへも
const res = [];
for (const {r, e} of targets) {
  const ps = e.ps, pe = e.pe ?? ps, H = 45, pf = r.pf;
  const pulse = pf.st[ps];
  const recorded = rollout(r, ps, f => ({steering: pf.st[f] ?? 0, speed: pf.sp[f] ?? 0}), H);
  const combos = (scale) => {
    const out = [];
    for (const brakeInPulse of [false, true]) for (const s of STEERS) for (const keep of [false, true]) {
      let vHold = null;
      const plan = (f, state) => {
        if (f < pe) return {steering: pulse, speed: brakeInPulse ? 0 : (pf.sp[f] ?? 0)};
        if (vHold == null) vHold = state.speed;
        return {steering: s, speed: keep ? vHold : 0};
      };
      const o = rollout(r, ps, plan, H, scale);
      out.push({brakeInPulse, steer: s, keep, collided: o.collided});
    }
    return out;
  };
  const c1 = combos(1);
  const escape1 = c1.filter(c => !c.collided);
  let cEscape = null;
  if (!escape1.length) for (const c of [.9, .8, .7, .6, .5, .4, .3, .2]) { if (combos(c).some(x => !x.collided)) { cEscape = c; break; } }
  else cEscape = 1;
  res.push({set: r.set, id: r.id, n: e.n, cls: e.cls, k: e.k, frames: e.frames, ps, pe, s_mod: +(e.s % LOOP_LENGTH).toFixed(3), v0: pf.v[ps],
    recorded_collided: recorded.collided, replay_pos_err: Math.max(recorded.pos.dx, recorded.pos.dz),
    escape_at_v0: escape1.length > 0, escapes_at_v0: escape1.slice(0, 4), c_escape: cEscape, v_escape: cEscape == null ? null : cEscape * pf.v[ps]});
  process.stderr.write(`${res.length}/${targets.length}\r`);
}
const n = res.length, cls = c => res.filter(x => x.cls === c);
const summary = {targets: n, replay_pos_err_max: Math.max(...res.map(x => x.replay_pos_err)), recorded_collided: res.filter(x => x.recorded_collided).length,
  escape_possible_at_v0: res.filter(x => x.escape_at_v0).length,
  escape_possible_at_v0_among_collided: res.filter(x => x.recorded_collided && x.escape_at_v0).length,
  by_class: Object.fromEntries(['弱', '中', '強'].map(c => [c, {n: cls(c).length, collided: cls(c).filter(x => x.recorded_collided).length, escape_at_v0: cls(c).filter(x => x.escape_at_v0).length,
    v0_median: cls(c).map(x => x.v0).sort((a, b) => a - b)[Math.floor(cls(c).length / 2)] ?? null,
    v_escape_median: cls(c).filter(x => x.v_escape != null).map(x => x.v_escape).sort((a, b) => a - b)[Math.floor(cls(c).filter(x => x.v_escape != null).length / 2)] ?? null,
    no_escape_even_c02: cls(c).filter(x => x.c_escape == null).length}])),
  v0_range: [Math.min(...res.map(x => x.v0)), Math.max(...res.map(x => x.v0))], which,
  v_escape_quartiles_collided: (() => { const a = res.filter(x => x.recorded_collided && x.v_escape != null).map(x => x.v_escape).sort((p, q) => p - q);
    return a.length ? [a[Math.floor(a.length / 4)], a[Math.floor(a.length / 2)], a[Math.floor(3 * a.length / 4)]] : null; })(),
  v0_quartiles_collided: (() => { const a = res.filter(x => x.recorded_collided).map(x => x.v0).sort((p, q) => p - q);
    return a.length ? [a[Math.floor(a.length / 4)], a[Math.floor(a.length / 2)], a[Math.floor(3 * a.length / 4)]] : null; })()};
fs.writeFileSync(outPath, JSON.stringify({schema: 'jf-sim-rl-recover-exit-escape/v1', created_at: new Date().toISOString(), source: runsDir,
  rule: 'ps まで記録の操作を再生(高速版)。乱しの間 {記録の速さ, 0} × 乱しのあと 一定の指令 舵 {0.52, 0.39, 0.26, 0.13, 0} × 速さ {0, そのとき} の 20 通りで ps から 1.5 秒衝突しないか。c は ps の速さの倍率',
  summary, events: res}, null, 1));
console.log(JSON.stringify(summary, null, 1));
