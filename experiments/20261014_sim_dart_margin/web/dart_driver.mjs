// DART の運転(先生の中身 + 乱し)。ブラウザの差し込みモジュールと Node の予想が同じこのファイルを使う(純粋な ESM)。2026-10-14、sim_dart_20261014。
//
//   const d = createDartDriver({kind, world, config, weights, noise: {spec, seed, salt, k, scale} | null})
//   d.action(snapshot) → {steering, speed, info: {dart: {v, f, clean: [δ, v], noise: [dδ, 倍率] | null, ev}}}
// 毎フレーム: 先生のきれいな操作を計算 → 乱しの予定を見て、乱しが有れば足す → 適用する操作を返す。
// 次のフレームの観測の「前回の行動」は実際に適用した操作(乱していない成分は先生自身の値)。
// info に jf は出さない(jetformer/sim/label_mask.py が学習から外す印と混ざらないように。protocol.json relabel.label_mask_safety)。
// 同じフレームで 2 回呼ばれたら同じ値を返す。フレームが飛んだら例外(記録が先生と合わなくなるので止める)。
import {createTeacherCore, previousAction} from './teacher_core.mjs';
import {createDartNoise, applyNoise} from './dart_noise.mjs';

export const DART_DRIVER_VERSION = 'jf-sim-dart-driver/v1';

export function createDartDriver({kind, world, config, weights, noise = null}) {
  const core = createTeacherCore(kind, world, config, weights);
  const nz = noise ? createDartNoise(noise.spec, noise) : null;
  let lastF = -1, memo = null, prevA = [0, 0];
  const log = [];
  function action(snapshot) {
    const st = snapshot.state, f = st.frame, ri = snapshot.metrics?.route_index ?? 0;
    if (f === lastF && memo) return JSON.parse(JSON.stringify(memo));
    if (f === 0 && lastF === -1) core.begin(st, ri);
    else if (f === lastF + 1) core.advance(st, ri);
    else throw new Error(`DART の先生: フレームが飛びました(前 ${lastF} → 今 ${f})。記録を先生と合わせられないので止めます。`);
    const clean = core.decide(st, prevA[0], prevA[1]);
    const n = nz ? nz(f, clean) : null;
    const cmd = applyNoise(clean, n, core.speedCap);
    prevA = previousAction(clean, cmd, !!n, !!n && n.speedFactor !== 1, core.speedCap, core.maxSteer);
    const info = {dart: {v: 1, f, clean: [clean.steering, clean.speed], noise: n ? [n.dSteer, n.speedFactor] : null, ev: n ? n.ev : -1}};
    memo = {steering: cmd.steering, speed: cmd.speed, info};
    log.push({f, g: `${snapshot.metrics?.completed_laps ?? 0}:${snapshot.metrics?.passed_gates ?? 0}`, s: clean.s, dmin: clean.dmin, ey: clean.ey, gate_along: clean.gate_along, gate_lateral: clean.gate_lateral,
      clean_steering: clean.steering, clean_speed: clean.speed, steering: cmd.steering, speed: cmd.speed, noisy: !!n, ev: n ? n.ev : -1});
    lastF = f;
    return JSON.parse(JSON.stringify(memo));
  }
  return {action, core, get log() { return log; }, get events() { return nz ? nz.events : []; }, noiseParams: nz ? nz.params : null};
}
